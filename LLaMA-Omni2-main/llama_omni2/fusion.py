"""Gate fusion module (Eq. 1-4) bridging the LLM hidden states and MTTS.

The pretrained LLaMA-Omni2 checkpoint already ships this function, split across
`LLMSpeechGenerator.input_proj` and `LLMSpeechGenerator.gate`
(`llama_omni2/model/speech_generator/speech_generator.py`).  `GateFusionModule`
is the same computation in one module, and `from_pretrained_generator` /
`write_back_to_generator` move weights between the two layouts so Stage II can
warm-start from the pretrained fusion and save a checkpoint that the untouched
inference path still loads.

Layout note: `ffn`'s hidden width is `llm_hidden_size * 2` (7168 for the 7B),
not `tts_embed_size * 2`, because that is what `input_proj` uses.
"""

import torch
import torch.nn as nn


class GateFusionModule(nn.Module):
    def __init__(self, llm_hidden_size, tts_embed_size):
        super().__init__()

        # Eq 1: 2-layer FFN to map LLM hidden states to TTS embedding dimension
        self.ffn = nn.Sequential(
            nn.Linear(llm_hidden_size, llm_hidden_size * 2),
            nn.ReLU(),
            nn.Linear(llm_hidden_size * 2, tts_embed_size)
        )

        # Eq 3: Gate linear layer (maps concatenated 2d representation to d)
        # W_g in R^{2d x d} and b_g in R^d
        self.gate_linear = nn.Linear(tts_embed_size * 2, tts_embed_size)

    def forward(self, llm_hidden_states, text_embeddings):
        """
        llm_hidden_states: (batch, seq_len, llm_hidden_size) -> Output from frozen LLM
        text_embeddings: (batch, seq_len, tts_embed_size) -> Embeddings of the text tokens
        """
        # 1. Project hidden states (Eq 1)
        e_hidden = self.ffn(llm_hidden_states)

        # 2. Concatenate [e_hidden || e_emb]
        concat_reps = torch.cat([e_hidden, text_embeddings], dim=-1)

        # 3. Calculate gate scores with sigmoid (Eq 3)
        g = torch.sigmoid(self.gate_linear(concat_reps))

        # 4. Element-wise fusion (Eq 4)
        c = g * e_hidden + (1.0 - g) * text_embeddings

        return c

    # ---------------------------------------------------------------- interop

    @classmethod
    def from_pretrained_generator(cls, speech_generator, dtype=torch.float32):
        """Build a module warm-started from a pretrained `LLMSpeechGenerator`.

        `input_proj` is `Linear -> ReLU -> Linear` and `gate` is
        `Linear -> Sigmoid`, so the mapping is index-for-index:

            ffn[0]      <- input_proj[0]      (llm_hidden -> 2 * llm_hidden)
            ffn[2]      <- input_proj[2]      (2 * llm_hidden -> tts_embed)
            gate_linear <- gate[0]            (2 * tts_embed -> tts_embed)
        """
        llm_hidden_size = speech_generator.input_proj[0].in_features
        tts_embed_size = speech_generator.input_proj[2].out_features

        module = cls(llm_hidden_size, tts_embed_size)
        module.ffn[0].load_state_dict(speech_generator.input_proj[0].state_dict())
        module.ffn[2].load_state_dict(speech_generator.input_proj[2].state_dict())
        module.gate_linear.load_state_dict(speech_generator.gate[0].state_dict())
        return module.to(dtype=dtype)

    @torch.no_grad()
    def write_back_to_generator(self, speech_generator):
        """Copy trained weights back into `input_proj` / `gate` in place.

        Inverse of `from_pretrained_generator`, so the saved checkpoint stays
        loadable by `inference.py` with no code change.
        """
        pairs = [
            (speech_generator.input_proj[0], self.ffn[0]),
            (speech_generator.input_proj[2], self.ffn[2]),
            (speech_generator.gate[0], self.gate_linear),
        ]
        for dst, src in pairs:
            dst.weight.copy_(src.weight.to(dst.weight.dtype))
            dst.bias.copy_(src.bias.to(dst.bias.dtype))

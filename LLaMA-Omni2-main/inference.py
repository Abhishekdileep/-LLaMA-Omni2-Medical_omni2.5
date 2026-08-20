"""
Single-process LLaMA-Omni2 speech-to-speech inference.

Collapses the original three-process serving stack (controller, model worker,
Gradio web server) into one script with no HTTP, no dispatch, and no chunked
streaming. Intended for offline analysis, where the whole response is wanted at
once rather than token by token.

Audio file(s) in -> text on stdout + wav on disk. Single turn only: each input
file is an independent request, and nothing carries over between them. The
prompt is always exactly one user message.

Pipeline
--------
main()
  |
  |-- OmniModel(model_path)
  |     |-- load_pretrained_model(model_path)   -> (tokenizer, model on CUDA)
  |     '-- build_unit_tokenizer(unit_vocab_size) -> BertTokenizer over unit ids
  |
  |-- SpeechDecoder(vocoder_dir, hop_len, load_onnx)
  '-- load_wav(prompt_speech, 16000)            -> speaker prompt for the vocoder

  for each --audio path (independent request):
      OmniModel.generate(audio_path, temperature, top_p, max_new_tokens)
        |-- prepare_inputs(audio_path)
        |     |-- load_speech(path)      -> (T, 128) log-mel
        |     '-- process_messages(msgs) -> input_ids with SPEECH_TOKEN_INDEX
        |-- model.generate(...) on a worker thread, filling two streamers
        '-- drain both streamers to completion -> (text, unit_ids)
      |
      units_to_wav(vocoder, unit_ids, prompt_speech_16k) -> 24 kHz waveform
      torchaudio.save(...)

  transcript.json written to --out-dir

Example
-------
    python omni2_cli.py \\
        --model-path /path/to/LLaMA-Omni2-7B \\
        --vocoder-dir /path/to/cosyvoice \\
        --audio sample0.wav sample1.wav \\
        --out-dir ./out

Author: Abhishek
Date: 20 August 2026
"""
import argparse
import json
import os
import sys

import torch
import torchaudio
import whisper
from transformers import AutoTokenizer, AutoConfig, TextIteratorStreamer
from threading import Thread
import csv
from llama_omni2.constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN
from llama_omni2.model import *
from llama_omni2.serve.flow_inference import SpeechDecoder

from os.path import dirname
ROOT_DIR = dirname(dirname(__file__))
sys.path.append(os.path.join(ROOT_DIR))
sys.path.append(os.path.join(ROOT_DIR, "third_party/Matcha-TTS"))
from cosyvoice.utils.file_utils import load_wav


# ---------------------------------------------------------------- model side



def load_pretrained_model(model_path):
    """Load the Omni2 checkpoint and its text tokenizer onto the GPU.

    Parameters
    ----------
    model_path : str
        Directory holding the checkpoint; must also contain ``tts_tokenizer/``.

    Returns
    -------
    tokenizer : transformers.PreTrainedTokenizer
    model : Omni2Speech2SQwen2ForCausalLM
        In bfloat16, already moved to CUDA.
    """
    config = AutoConfig.from_pretrained(model_path)
    config.tts_tokenizer = os.path.join(model_path, "tts_tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = Omni2Speech2SQwen2ForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=torch.bfloat16)
    model.cuda()
    return tokenizer, model


class OmniModel:
    """Holds the loaded model and both tokenizers for the lifetime of a run.

    Parameters
    ----------
    model_path : str
        Checkpoint directory, passed through to ``load_pretrained_model``.

    Attributes
    ----------
    tokenizer : text tokenizer
    model : Omni2Speech2SQwen2ForCausalLM
    unit_tokenizer : tokenizer over TTS unit ids
    """

    def __init__(self, model_path):
        self.tokenizer, self.model = load_pretrained_model(model_path)
        from transformers import BertTokenizer
        with open("unit_vocab.txt", "w") as f:
            for i in range(self.model.config.unit_vocab_size + 1):
                f.write(str(i) + "\n")
        tokenizer = BertTokenizer(vocab_file="unit_vocab.txt")
        os.remove("unit_vocab.txt")
        self.unit_tokenizer = tokenizer

    def load_speech(self, path, add_noise=True, snr_db=-5):
        speech = whisper.load_audio(path)
        speech = whisper.pad_or_trim(speech)
        speech = torch.from_numpy(speech)
        if add_noise:
            speech_rms = torch.sqrt(torch.mean(speech ** 2))
            noise_rms = speech_rms * (10 ** (-snr_db / 20))
            noise = torch.randn_like(speech)
            # Normalize noise to desired RMS
            noise = noise / torch.sqrt(torch.mean(noise ** 2))
            noise = noise * noise_rms
            # Add noise
            speech = speech + noise
        speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)
        return speech

    def process_messages(self, messages):
        """Apply the chat template and swap the speech placeholder for its index.

        Parameters
        ----------
        messages : list of dict
            Alternating ``{"role": "user"/"assistant", "content": ...}``. Length
            must be odd, i.e. the sequence ends on a user turn.

        Returns
        -------
        torch.Tensor, shape (L,)
            Token ids, with every DEFAULT_SPEECH_TOKEN replaced by
            SPEECH_TOKEN_INDEX so the model knows where to splice speech.
        """
        assert len(messages) % 2 == 1, "Number of history messages must be odd"
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt")[0]
        input_ids[input_ids == self.tokenizer.convert_tokens_to_ids(
            DEFAULT_SPEECH_TOKEN)] = SPEECH_TOKEN_INDEX
        return input_ids

    def get_input_params(self, history):
        """Turn the raw history list into the three tensors ``generate`` needs.

        Entries are selected by position: ``i % 3 == 0`` is user audio,
        ``i % 3 == 1`` is assistant text, ``i % 3 == 2`` (assistant audio) is
        skipped. Break that ordering and the wrong entries get read.

        Parameters
        ----------
        history : list of dict
            See the module-level pipeline sketch for the expected layout.

        Returns
        -------
        input_ids : torch.Tensor, shape (1, L)
        speech_tensors : torch.Tensor, shape (n_user_turns, T_max, 128)
            Zero-padded across turns.
        speech_lengths : torch.LongTensor, shape (n_user_turns,)
        """
        messages = []
        speech_list = []
        for i, turn in enumerate(history):
            if i % 3 == 0:
                messages.append({"role": "user", "content": DEFAULT_SPEECH_TOKEN})
                speech_list.append(self.load_speech(turn["content"]["path"]))
            elif i % 3 == 1:
                messages.append({"role": "assistant", "content": turn["content"]})
            else:
                continue
        input_ids = self.process_messages(messages).unsqueeze(0)
        speech_tensors = torch.nn.utils.rnn.pad_sequence(
            speech_list, batch_first=True, padding_value=0)
        speech_lengths = torch.LongTensor([len(speech) for speech in speech_list])
        return input_ids, speech_tensors, speech_lengths

    @torch.inference_mode()
    def generate(self, history, temperature=0.0, top_p=0.7, max_new_tokens=512):
        """Run generation to completion for the current history.

        Blocking: unlike the original worker, nothing is yielded mid-generation.

        Parameters
        ----------
        history : list of dict
            Conversation so far, ending on a user audio turn.
        temperature : float, default 0.0
            Values <= 0.001 switch generation to greedy decoding.
        top_p : float, default 0.7
        max_new_tokens : int, default 512
            Clamped to 1024.

        Returns
        -------
        text : str
            Response with the trailing ``<|im_end|>`` stripped.
        unit_ids : list of int
            TTS units for the vocoder; empty if none were produced.
        """
        input_ids, speech_tensors, speech_lengths = self.get_input_params(history)
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        speech_tensors = speech_tensors.to(dtype=torch.bfloat16, device='cuda', non_blocking=True)
        speech_lengths = speech_lengths.to(device='cuda', non_blocking=True)

        do_sample = True if temperature > 0.001 else False
        max_new_tokens = min(int(max_new_tokens), 1024)

        # NOTE: model.generate emits units only through `streamer_unit`; there is
        # no return-value path for them. So the streamer objects stay, but they
        # are drained to exhaustion here instead of being yielded onward.
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=False, timeout=15)
        streamer_unit = TextIteratorStreamer(
            self.unit_tokenizer, skip_prompt=False, skip_special_tokens=False, timeout=15)

        thread = Thread(target=self.model.generate, kwargs=dict(
            inputs=input_ids,
            speech=speech_tensors,
            speech_lengths=speech_lengths,
            do_sample=do_sample,
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            streamer_unit=streamer_unit,
            use_cache=True,
        ))
        thread.start()

        generated_text = ""
        unit_ids = []
        stop_str = "<|im_end|>"
        for new_text in streamer:
            generated_text += new_text
            # snapshot before any flush; keep the longest seen
            snapshot = list(streamer_unit.token_cache)
            if len(snapshot) > len(unit_ids):
                unit_ids = snapshot
            if generated_text.endswith(stop_str):
                streamer_unit.end()
                generated_text = generated_text[:-len(stop_str)]
        thread.join()

        unit_ids = list(map(int, unit_ids))
        return generated_text, unit_ids

# ---------------------------------------------------------------- vocoder side
# (from gradio_web_server.py, minus Gradio)
def units_to_wav(vocoder, unit_ids, prompt_speech_16k, device="cuda"):
    """Synthesise the full unit sequence into a waveform in one shot.

    Parameters
    ----------
    vocoder : SpeechDecoder
    unit_ids : list of int
        Complete unit sequence for one assistant turn.
    prompt_speech_16k : torch.Tensor
        Speaker prompt at 16 kHz; fixes the output voice.
    device : str, default "cuda"

    Returns
    -------
    torch.Tensor or None
        24 kHz waveform, or None if the decoder returned nothing.
    """
    session = vocoder.init_prompt(prompt_speech_16k)
    units_tensor = torch.LongTensor(unit_ids).to(device)
    tts_speech, session = vocoder.process_unit_chunk(units_tensor, session, finalize=True)
    return tts_speech


# ---------------------------------------------------------------- entry point
def main():
    """Parse arguments, load the stack once, then run every ``--audio`` turn.

    Each turn appends to a shared history, so later turns are conditioned on
    earlier ones. Writes ``turn{i}.wav`` and ``transcript.json`` to ``--out-dir``.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--vocoder-dir", type=str, default=None)
    parser.add_argument("--hop-len", type=int, default=10)
    parser.add_argument("--load-onnx", action="store_true")
    parser.add_argument("--prompt-speech", type=str,
                        default="llama_omni2/inference/prompt_en.wav")
    parser.add_argument("--audio", type=str, nargs="+", required=True,
                        help="One or more input wavs; treated as consecutive turns.")
    parser.add_argument("--out-dir", type=str, default="./out")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    omni = OmniModel(args.model_path)
    vocoder = SpeechDecoder(model_dir=args.vocoder_dir,
                            hop_len=args.hop_len,
                            load_onnx=args.load_onnx)
    prompt_speech_16k = load_wav(args.prompt_speech, 16000)
    import time
    csv_path = os.path.join(args.out_dir, "inference_time.csv")

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["turn", "inference_time_seconds"])
    history = []
    transcript = []
    for turn_idx, audio_path in enumerate(args.audio):
        start_time = time.time()
        history.append({"role": "user", "content": {"path": audio_path}})

        text, unit_ids = omni.generate(
            history,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        history.append({"role": "assistant", "content": text})


        wav_path = None
        if unit_ids:
            tts_speech = units_to_wav(vocoder, unit_ids, prompt_speech_16k)
            if tts_speech is not None:
                wav_path = os.path.join(args.out_dir, f"turn{turn_idx}.wav")
                elapsed_time = time.time() - start_time
                with open(csv_path,"a",newline="",) as f:
                    writer = csv.writer(f)
                    writer.writerow([turn_idx, elapsed_time])
                torchaudio.save(wav_path, tts_speech.cpu(), 24000, format="wav")
                # third history slot, matching the i%3 layout get_input_params expects
                history.append({"role": "assistant",
                                "content": {"path": wav_path, "type": "audio/wav"}})

        print(f"[turn {turn_idx}] in={audio_path}")
        print(f"[turn {turn_idx}] text={text}")
        print(f"[turn {turn_idx}] wav={wav_path}  units={len(unit_ids)}")
        transcript.append({"input": audio_path, "text": text,
                           "wav": wav_path, "num_units": len(unit_ids)})

    with open(os.path.join(args.out_dir, "transcript.json"), "w") as f:
        json.dump(transcript, f, indent=2)


if __name__ == "__main__":
    main()

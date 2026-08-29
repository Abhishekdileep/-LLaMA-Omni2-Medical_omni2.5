# LLaMA-Omni2 + CosyVoice

## Installation

### LLaMA-Omni2

Follow the official LLaMA-Omni2 installation instructions:
<https://github.com/ictnlp/LLaMA-Omni2>

### CosyVoice

Follow the official CosyVoice installation instructions:
<https://github.com/QwenAudio/CosyVoice>

The Gradio setup is already mentioned in the LLaMA-Omni2 repository.

---

## Model Weights
I haven't put that folder in here 
The model weights are stored in `models/`. For example:

```text
Please see how the LLaMA-Omni2's / CosyVoice installation and weights put it 

models/
├── LLaMA-Omni2-7B-Bilingual/
└── cosy2_decoder/
```

---

## Inference - the actual "MY PERSONAL Contribution" 

### Individual audio files

Use `inference.py`:

```bash
python LLaMA-Omni2-main/inference.py \
    --model-path ./models/LLaMA-Omni2-7B-Bilingual/ \
    --vocoder-dir ./models/cosy2_decoder/ \
    --audio Recording.wav Recording_2.wav \
    --out-dir ./out
```

### Multiple audio files

Use `LLaMA-Omni2-main/run.sh`:

```bash
bash LLaMA-Omni2-main/run.sh
```

Provide the exact paths required by the script.

---

## Generative Audio

For generative audio, use the following CosyVoice scripts:

| Dataset | Script |
| --- | --- |
| MedQA | `cosy_voice/run_for_medqa.py` |
| ChatDoctor | `cosy_voice/run_for_chatdoctor.py` |

### Different accents

To change the accent, edit the prompt input in:

- `cosy_voice/run_for_medqa.py`
- `cosy_voice/run_for_chatdoctor.py`

---

## Training

Three stages. Each one freezes everything the previous stage produced, so they must
be run in order. All commands are run from `LLaMA-Omni2-main/`.

```
stage 1a   whisper* -> speech adaptor -> LLM            CE on the text response
stage 1b   text -> MTTS                                 CE on the speech units
stage 2    LLM hidden states -> gate fusion -> MTTS     CE on the speech units
           (*  = frozen)
```

### Stage 1a -> speech adaptor + LLM

```bash
python llama_omni2/train_stage1a.py --model_name_or_path models/LLaMA-Omni2-7B-Bilingual --data_path examples/questions.json --output_dir models/stage1_output --bf16 True --num_train_epochs 3 --per_device_train_batch_size 1 --learning_rate 5e-5 --save_steps 500
```

Whisper is frozen; the speech adaptor and the last `--llm_trainable_layers` (default 1)
LLM blocks are trained. One layer is what fits on a 24 GB card alongside the 7B in bf16.

Loss is computed only on the assistant turns. Pass `--gradient_checkpointing True` for
long or chunked audio: the adaptor sits *below* the LLM, so its gradient runs back
through all 28 layers, and a ~900-frame input OOMs on 24 GB without it.

The output is a complete 2S checkpoint -- `speech_generator.*` is carried through from
the source -- so it loads in `inference.py` and can be passed straight to stage 2.

### Stage 1b -> MTTS (text to speech units)

```bash
python llama_omni2/train_stage1b.py --data_path examples/stage1b_data.json --output_dir models/stage_1b --bf16 True --num_train_epochs 10 --save_steps 500 --logging_steps 10
```

MTTS is warm-started from the checkpoint's own `speech_generator.model.*`, whose 6561
unit embeddings are already trained, and uses the checkpoint's `tts_tokenizer`. Both
matter: a tokenizer built from raw `Qwen/Qwen2.5-0.5B` puts `<0>` at 151665 instead of
151666 and shifts every unit id handed to CosyVoice by one.

Training data is `{"text", "units"}` pairs; `llama_omni2/convert_answers.py` builds them
from `answer.json`.

### Stage 2 -> gate fusion + MTTS

Stage 1b feeds MTTS the text embeddings alone. Stage 2 replaces them with the fused
representation of Eq. 1-4 -- the LLM hidden state projected into the TTS space and gated
against the text embedding -- so MTTS sees the semantics the text tokens do not carry.
Whisper, the adaptor and the LLM stay frozen; the gate fusion and MTTS train together.

Two passes. The first caches the frozen LLM's hidden states; this is required, not an
optimisation, because a trainable MTTS plus a co-resident 7B does not fit in 24 GB:

```bash
python llama_omni2/precompute_stage2_hidden.py --model_name_or_path models/stage1_output --data_path examples/stage2_data.json --cache_dir cache/stage2_hidden
```

```bash
python llama_omni2/train_stage2.py --model_name_or_path models/stage1_output --mtts_init models/stage_1b --data_path examples/stage2_data.json --hidden_cache_dir cache/stage2_hidden --output_dir models/stage2_output --bf16 True --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --num_train_epochs 3 --learning_rate 2e-5 --fusion_lr 1e-4
```

| | with the 7B resident | with the cache |
| --- | --- | --- |
| peak GPU | ~26 GB (OOM) | **8.8 GB measured** |

Build the data with `python llama_omni2/build_stage2_data.py`, which joins
`questions.json` to `answer.json` to carry the input audio through.

Check the stack before a long run:

```bash
python llama_omni2/verify_stage2.py
```

It confirms the fusion warm-start is bit-exact against the checkpoint's `input_proj`/`gate`,
that the tokenizers line up, and -- most importantly -- that the cached hidden states match
what the live `generate` path feeds to `generate_units`. That last check compares tensors
rather than loss, because the shipped gate is nearly closed (mean 0.001), so cross-entropy
cannot detect a misaligned cache.

---

## MedQA fine-tune

```bash
bash scripts/run_medqa_finetune.sh
```

Chains all three stages plus a before/after evaluation on a held-out split.

**Audio is chunked.** A MedQA question plus its five options synthesises to ~57 s at the
median, and only **3.5%** of the dataset fits in whisper's 30 s window in one piece.
Questions are therefore split at sentence boundaries into <=28 s chunks, each fed as its
own `<speech>` token in the same user turn -- `prepare_inputs_labels_for_speech_and_text`
already splices one 300-frame block per token, so this needs no model change.
`inference.py` accepts comma-separated paths (`--audio q_00.wav,q_01.wav`) for the same
reason.

Targets come from MedQA's own `answer_idx` and `answer` fields. MedQA carries no written
explanation for its answers, so none is invented; the target is the letter plus the option
text, formatted so `bleuscore.py` can parse it.

The steps individually:

```bash
# 0. corpus: question audio, gold answer audio, gold units (CosyVoice env)
python scripts/build_medqa_corpus.py --split train --num 400 --seed 0
python scripts/project_manifest.py --manifest ../data/medqa_corpus/train/medqa_corpus.json

# 1-3. see run_medqa_finetune.sh; the re-cache between 1a and 2 is mandatory,
#      because stage 1a changes the LLM the cached hidden states came from.

# 4. evaluate: one model load for the whole split
python scripts/eval_medqa.py --model-path models/stage2_output --manifest ../data/medqa_corpus/test/medqa_corpus.json --out out/medqa/predictions.json
python bleuscore.py --predictions out/medqa/predictions.json --manifest ../data/medqa_corpus/test/medqa_corpus.json
```

## Accuracy, Inference Time and Mel-Spectrogram Analysis

For inference time and mel-spectrogram analysis, use:

```text
llama_omni2/summarize_runs.py
```

For accuracy, `bleuscore.py` takes either layout:

```bash
python bleuscore.py --predictions out/medqa/predictions.json --manifest ../data/medqa_corpus/test/medqa_corpus.json
```

```bash
python bleuscore.py --pred-dir medqa_in --manifest medqa_in/samples.json --start 52 --end 80 --index-offset 51
```

The second form reproduces the older `run.sh` layout, one `<name>/transcript.json` per wav.

The basic evaluation flow:

```text
Correct Answers
      ↓
Reference Transcripts
      ↓
Generated Transcripts
      ↓
bleuscore.py
```

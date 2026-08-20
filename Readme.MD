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

The model weights are stored in `models/`. For example:

```text
models/
├── LLaMA-Omni2-7B-Bilingual/
└── cosy2_decoder/
```

---

## Inference

### Individual audio files

Use `inference.py`:

```bash
python inference.py \
    --model-path ./models/LLaMA-Omni2-7B-Bilingual/ \
    --vocoder-dir ./models/cosy2_decoder/ \
    --audio Recording.wav Recording_2.wav \
    --out-dir ./out
```

### Multiple audio files

Use `llama_omni2/run.sh`:

```bash
bash llama_omni2/run.sh
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

## Accuracy, Inference Time and Mel-Spectrogram Analysis

For inference time and mel-spectrogram analysis, use:

```text
llama_omni2/summarize_runs.py
```

For accuracy, first generate the transcripts of the correct answers, then
compare them against the generated text transcripts using:

```text
llama_omni/bleuscore.py
```

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

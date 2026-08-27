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

#### Stage 1 -> Frozen whisper + speech adaptor (🔥) + LLM (Omni2.5) 🔥

- `python `LLaMA-Omni2-main/llama_omni2/train_stage1.py --output_dir ./models/stage1_output --num_train_epochs 3 --per_device_train_batch_size 1 --learning_rate 5e-5 --save_steps 500 --bf16 True`

This code is just to run the stage one , explained in the training section of the paper . Please refer the original paper for info 
here we are taking the frozen whisper , and speech adaptor + LLM , (since i old have 24gb gpu , i only finetune last layer of the omni2.5 7b bilingual model on float16 precision) 

#### Stage 2 -> TTS Qwen2.5 0.5b Text model(🔥) 
- `python `LLaMA-Omni2-main/llama_omni2/train_stage1b.py --data_path examples/stage1b_data.json --output_dir models/stage_1b/ --bf16 True --num_train_epochs 10 
   --save_steps 500 --logging_steps 10`


This code is to run the TTS model. The training inputs are similar to answer.json (question / output_id pairs) --> we need this as a text response pair
ie text reponse to speech unit mapping , which cosyvoice2 takes and makes into audio waveform
we initialize with omni2.5 0.5b weights, again as per paper, plus since i don't have a big gpu we use the omni2.5 last layer  


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

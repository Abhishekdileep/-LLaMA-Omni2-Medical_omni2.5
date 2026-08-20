import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio
import json
import random
import pandas as pd 
import os


def split_medical_record(text):
    instruction_marker = "### Instruction:"
    input_marker = "### Input:"
    response_marker = "### Response:"

    instruction_start = text.index(instruction_marker) + len(instruction_marker)
    input_start = text.index(input_marker)
    response_start = text.index(response_marker)

    return {
        "instruction": text[instruction_start:input_start].strip(),
        "input": text[input_start + len(input_marker):response_start].strip(),
        "response": text[response_start + len(response_marker):].strip(),
    }

os.makedirs('./chatdoctor/', exist_ok=True)
path = '/DATA/carer.ai/data/chatdoctor/chatdoctor_text.csv'
df = pd.read_csv(path)
samples = [] 
for j,i in enumerate(df.iterrows()):
    samples.append(split_medical_record(i[1]['text']))
    if j > 20 : 
        break 

cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B')
for i, sentence in enumerate(samples):
    for  j in cosyvoice.inference_zero_shot(sentence['input'], '希望你以后能够做的比我还好呦。', './asset/zero_shot_prompt.wav'):
        torchaudio.save('./chatdoctor/zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)


with open("./chatdoctor/samples.json", "w", encoding="utf-8") as file:
    json.dump(samples, file, ensure_ascii=False, indent=2)
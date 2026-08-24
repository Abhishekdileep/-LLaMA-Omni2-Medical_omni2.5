import sys
import torch
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio
import json
import random


def load_random_samples(path, num_samples=30):
    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return random.sample(records, min(num_samples, len(records)))

import os
os.makedirs('./medqa/', exist_ok=True)
with open('./medqa/samples.json', "w+", encoding="utf-8") as file:
        json.dump('', file, ensure_ascii=False, indent=2)
samples = load_random_samples('/DATA/carer.ai/data/medqa/data/questions/en_5/dev.jsonl', num_samples=50)
cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B')
for i, sentence in enumerate(samples):
    complete_sentence = f"{sentence['question']} Answer A: {sentence['A']}, Answer B: {sentence['B']}, Answer C: {sentence['C']}, Answer D: {sentence['D']}, Answer E: {sentence['E']}" 
    tt_audio = []
    for  j in cosyvoice.inference_zero_shot(complete_sentence, '希望你以后能够做的比我还好呦。', './asset/zero_shot_prompt.wav'):
        tt_audio.append(j['tts_speech'])
    tt_audio = torch.cat(tt_audio, dim=1)
    torchaudio.save('./medqa/zero_shot_{}.wav'.format(i+50),tt_audio, cosyvoice.sample_rate)
    with open('./medqa/samples.json', "a+", encoding="utf-8") as file:
        json.dump(sentence, file, ensure_ascii=False, indent=2)

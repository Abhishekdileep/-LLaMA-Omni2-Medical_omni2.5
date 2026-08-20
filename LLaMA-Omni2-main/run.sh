for audio_file in ../CosyVoice/medqa/zero_shot_{62..80}.wav; do
    name=$(basename "$audio_file" .wav)

    python inference.py \
        --model-path models/LLaMA-Omni2-7B-Bilingual  \
        --vocoder-dir models/cosy2_decoder \
        --audio "$audio_file" \
        --out-dir "./out/$name"
done

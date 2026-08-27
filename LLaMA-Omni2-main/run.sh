for audio_file in ../CosyVoice/medqa_in/zero_shot_{52..80}.wav; do
    name=$(basename "$audio_file" .wav)

    python inference.py \
        --model-path models/LLaMA-Omni2-7B-Bilingual  \
        --vocoder-dir models/cosy2_decoder \
        --audio "$audio_file" \
        --out-dir "./medqa_in/$name"
done

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
SPEECH_TOKEN_INDEX = -200
DEFAULT_SPEECH_TOKEN = "<speech>"
# Prompt appended to the user turn for MedQA. Shared by inference.py,
# train_stage1a.py and precompute_stage2_hidden.py -- the Stage II hidden-state cache
# is only valid for the prompt it was built under, so these must not drift apart.
MCQ_INSTRUCTION = (
    "Answer the multiple-choice question. "
    "Output only the correct option letter and answer. "
    "Provide explanations or reasoning. "
    "Format: A. answer, explaination. "
)

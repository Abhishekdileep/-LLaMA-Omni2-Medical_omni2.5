"""Summarise LLaMA-Omni2 zero-shot runs.

Point this at the folder holding zero_shot_1, zero_shot_2, ... and it does
two things:

  1. reads every inference_time.csv and prints per-run and overall averages
  2. renders one mel spectrogram PNG next to every .wav it finds

It touches nothing else. No model, no vocoder, no GPU.

    python summarize_runs.py /DATA/carer.ai/LLaMA-Omni2-main/chatdoctor
"""

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchaudio


# How the mel is computed. These mirror CosyVoice 2's 24 kHz analysis config.
# hop 480 at 24 kHz gives 50 frames per second, which is twice the 25 Hz
# speech-token rate the paper describes. Check these against your local
# cosyvoice config before using the plots for anything quantitative.
SAMPLE_RATE = 24000
N_FFT = 1920
HOP_LENGTH = 480
N_MELS = 80
F_MIN = 0
F_MAX = 8000

TOP_DB = 80.0


@dataclass
class RunSummary:
    """Timing result for one zero_shot_N folder."""
    name: str
    n_turns: int
    mean_seconds: float | None

    def mean_text(self):
        if self.mean_seconds is None:
            return "-"
        return f"{self.mean_seconds:.3f}"


def warn(message):
    print(f"  warn: {message}", file=sys.stderr)


# ------------------------------------------------------------------ ordering

def natural_key(path):
    """Sort key that puts zero_shot_2 before zero_shot_10.

    Plain sorting compares the names as text, so "10" lands before "2".
    This pulls the number off the end and sorts on that instead.
    """
    trailing_number = re.search(r"(\d+)$", path.name)
    if trailing_number:
        return (int(trailing_number.group(1)), path.name)
    return (-1, path.name)


def find_run_dirs(root, pattern):
    """All matching subdirectories of root, in natural order."""
    return sorted((p for p in root.glob(pattern) if p.is_dir()), key=natural_key)


# -------------------------------------------------------------------- timing

def read_seconds(csv_path):
    """Every inference_time_seconds value in one CSV.

    A row that will not parse is skipped with a warning, so one bad line
    does not cost you the rest of the scan.
    """
    seconds = []
    with csv_path.open(newline="") as f:
        rows = csv.DictReader(f)
        for line_number, row in enumerate(rows, start=2):
            try:
                seconds.append(float(row["inference_time_seconds"]))
            except (KeyError, ValueError, TypeError):
                warn(f"{csv_path}:{line_number} unparseable, skipped")
    return seconds


def summarise_run(run_dir):
    """Read one run folder's timings into a RunSummary."""
    csv_path = run_dir / "inference_time.csv"

    if not csv_path.exists():
        warn(f"no inference_time.csv in {run_dir.name}")
        return RunSummary(run_dir.name, 0, None), []

    seconds = read_seconds(csv_path)
    if not seconds:
        return RunSummary(run_dir.name, 0, None), []

    mean = sum(seconds) / len(seconds)
    return RunSummary(run_dir.name, len(seconds), mean), seconds


# ------------------------------------------------------------------ spectrogram

def build_mel_transform():
    """The mel filterbank, built once and reused for every file."""
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        n_mels=N_MELS,
        f_min=F_MIN,
        f_max=F_MAX,
        power=2.0,
    )


def load_mono_24k(wav_path):
    """Read a wav as a single channel at SAMPLE_RATE.

    Files written by the inference script are already mono at 24 kHz, so
    normally neither branch fires. They are here so an odd file does not
    silently land on a wrong frequency axis.
    """
    waveform, sample_rate = torchaudio.load(str(wav_path))

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != SAMPLE_RATE:
        warn(f"{wav_path.name} is {sample_rate} Hz, resampling to {SAMPLE_RATE}")
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)

    return waveform


def compute_mel_db(waveform, mel_transform):
    """Mel spectrogram in decibels, shape (N_MELS, n_frames)."""
    mel_power = mel_transform(waveform)[0]
    return torchaudio.functional.amplitude_to_DB(
        mel_power, multiplier=10.0, amin=1e-10, db_multiplier=0.0, top_db=TOP_DB)


def save_mel_plot(mel_db, out_path, title):
    """Write one spectrogram PNG. Width scales with clip length."""
    duration_seconds = mel_db.shape[1] * HOP_LENGTH / SAMPLE_RATE
    width = max(4, duration_seconds * 0.8)

    figure, axes = plt.subplots(figsize=(width, 3), dpi=140)
    image = axes.imshow(
        mel_db.numpy(),
        origin="lower",
        aspect="auto",
        extent=[0, duration_seconds, 0, N_MELS],
        cmap="magma",
    )
    axes.set_xlabel("time (s)")
    axes.set_ylabel("mel bin")
    axes.set_title(title, fontsize=9)
    figure.colorbar(image, ax=axes, label="dB")
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)


def render_all_wavs(run_dir, mel_transform):
    """Render turnN.wav -> turnN_mel.png for every wav in one run folder."""
    for wav_path in sorted(run_dir.glob("*.wav"), key=natural_key):
        out_path = wav_path.with_name(wav_path.stem + "_mel.png")

        waveform = load_mono_24k(wav_path)
        mel_db = compute_mel_db(waveform, mel_transform)
        save_mel_plot(mel_db, out_path, f"{run_dir.name}/{wav_path.name}")

        print(f"{run_dir.name}/{wav_path.name} -> {out_path.name}  "
              f"mel{tuple(mel_db.shape)}")


# ------------------------------------------------------------------ reporting

def print_table(summaries):
    print()
    print(f"{'run':<16}{'turns':>7}{'mean_s':>12}")
    for run in summaries:
        print(f"{run.name:<16}{run.n_turns:>7}{run.mean_text():>12}")


def print_overall(all_seconds):
    print()
    if not all_seconds:
        print("no timing rows found")
        return
    mean = sum(all_seconds) / len(all_seconds)
    print(f"turns measured : {len(all_seconds)}")
    print(f"overall mean   : {mean:.3f} s")
    print(f"min / max      : {min(all_seconds):.3f} / {max(all_seconds):.3f} s")


def write_summary_csv(summaries, out_path):
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "n_turns", "mean_seconds"])
        for run in summaries:
            mean = "" if run.mean_seconds is None else f"{run.mean_seconds:.6f}"
            writer.writerow([run.name, run.n_turns, mean])
    print(f"\nsummary written to {out_path}")


# ---------------------------------------------------------------- entry point

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", help="Directory containing the zero_shot_* folders")
    parser.add_argument("--pattern", default="zero_shot_*")
    parser.add_argument("--skip-png", action="store_true",
                        help="Only compute timing stats")
    return parser.parse_args()


def main():
    args = parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")

    run_dirs = find_run_dirs(root, args.pattern)
    if not run_dirs:
        sys.exit(f"no directories matching {args.pattern!r} under {root}")

    mel_transform = None if args.skip_png else build_mel_transform()

    summaries = []
    all_seconds = []

    for run_dir in run_dirs:
        summary, seconds = summarise_run(run_dir)
        summaries.append(summary)
        all_seconds.extend(seconds)

        if mel_transform is not None:
            render_all_wavs(run_dir, mel_transform)

    print_table(summaries)
    print_overall(all_seconds)
    write_summary_csv(summaries, root / "inference_time_summary_noisy.csv")


if __name__ == "__main__":
    main()

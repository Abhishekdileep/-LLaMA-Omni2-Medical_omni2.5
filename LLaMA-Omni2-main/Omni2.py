#!/usr/bin/env python3
"""
omni2.py - one script, subcommands, for LLaMA-Omni2.

    ./omni2.py download    # whisper + cosy2_decoder + LLaMA-Omni2, then hash-verify
    ./omni2.py verify      # hash-verify only, no downloading
    ./omni2.py serve       # tmux session: controller | gradio | worker
    ./omni2.py status      # session + port state
    ./omni2.py stop        # kill the tmux session

Verification is per-file: SHA256 against the LFS oid for LFS files, git blob
SHA-1 for plain files, size for anything the Hub does not give a hash for.
Every failure is kept with a reason and written to models/verify_report.json.
"""

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

DEFAULT_MODELS_DIR = os.environ.get("OMNI2_MODELS_DIR", "models")
DEFAULT_MODEL_NAME = os.environ.get("OMNI2_MODEL_NAME", "LLaMA-Omni2-7B-Bilingual")

WHISPER_NAME = "large-v3"
COSY_REPO = "ICTNLP/cosy2_decoder"

SESSION = os.environ.get("OMNI2_TMUX_SESSION", "llama-omni2")
CONTROLLER_PORT = 10000
GRADIO_PORT = 8000
WORKER_PORT = 40000

OK = "OK"
UNVERIFIED = "UNVERIFIED"          # file present, Hub gave no hash to check against
MISSING = "MISSING"
SIZE_MISMATCH = "SIZE_MISMATCH"
HASH_MISMATCH = "HASH_MISMATCH"
METADATA_UNAVAILABLE = "METADATA_UNAVAILABLE"

REDOWNLOAD = {MISSING, SIZE_MISMATCH, HASH_MISMATCH}


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------

def log(msg):
    print(f"[omni2] {msg}", flush=True)


def die(msg, code=1):
    print(f"[omni2] FATAL: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_blob_sha1(path, chunk=1 << 20):
    """Git object id of a blob: sha1(b'blob <size>\\0' + content)."""
    size = path.stat().st_size
    h = hashlib.sha1()
    h.update(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def lfs_sha256(sibling):
    """hub returns BlobLfsInfo on new versions, a dict on old ones."""
    lfs = getattr(sibling, "lfs", None)
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        return lfs.get("sha256") or lfs.get("oid")
    return getattr(lfs, "sha256", None)


def port_open(port, host="127.0.0.1", timeout=1.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_port(port, timeout, label):
    log(f"waiting for {label} on port {port} (timeout {timeout}s)")
    start = time.time()
    while time.time() - start < timeout:
        if port_open(port):
            log(f"{label} is up on port {port} ({time.time() - start:.0f}s)")
            return True
        time.sleep(2)
    log(f"WARNING: {label} did not open port {port} within {timeout}s")
    return False


def run(cmd, **kwargs):
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, **kwargs)


# ----------------------------------------------------------------------------
# huggingface CLI resolution - your commands, unchanged where possible
# ----------------------------------------------------------------------------

def hf_download_cmd():
    """
    Prefer the exact command you gave:
        huggingface-cli download --resume-download <repo> --local-dir <dir>
    huggingface-cli is removed in huggingface_hub 1.x, so fall back to `hf
    download` (resume is the default there, no flag exists) and say so loudly.
    """
    if shutil.which("huggingface-cli"):
        return ["huggingface-cli", "download", "--resume-download"]
    if shutil.which("hf"):
        log("NOTE: huggingface-cli not on PATH, using `hf download` instead "
            "(--resume-download is the default there and is not passed)")
        return ["hf", "download"]
    die("neither `huggingface-cli` nor `hf` found on PATH; "
        "install huggingface_hub[cli] first")


# ----------------------------------------------------------------------------
# verification
# ----------------------------------------------------------------------------

def verify_hf_repo(repo_id, local_dir):
    """Return (results, note). results = [{file, status, detail}]."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return ([], "huggingface_hub not importable, cannot verify")

    local_dir = Path(local_dir)
    try:
        info = HfApi().repo_info(repo_id=repo_id, files_metadata=True)
    except Exception as exc:  # network down, gated repo, auth, rate limit
        note = f"could not fetch file metadata for {repo_id}: {exc}"
        results = []
        if not local_dir.exists():
            results.append({"file": str(local_dir), "status": MISSING,
                            "detail": "local dir does not exist"})
        else:
            files = [p for p in local_dir.rglob("*")
                     if p.is_file() and ".cache/huggingface" not in str(p)]
            if not files:
                results.append({"file": str(local_dir), "status": MISSING,
                                "detail": "local dir is empty"})
            for p in files:
                empty = p.stat().st_size == 0
                results.append({
                    "file": str(p.relative_to(local_dir)),
                    "status": SIZE_MISMATCH if empty else METADATA_UNAVAILABLE,
                    "detail": "zero-byte file" if empty
                              else "no reference hash available",
                })
        return (results, note)

    results = []
    for sib in info.siblings:
        rel = sib.rfilename
        path = local_dir / rel
        if not path.exists():
            results.append({"file": rel, "status": MISSING,
                            "detail": "not present on disk"})
            continue

        actual_size = path.stat().st_size
        expected_size = getattr(sib, "size", None)
        if expected_size is not None and actual_size != expected_size:
            results.append({"file": rel, "status": SIZE_MISMATCH,
                            "detail": f"{actual_size} bytes on disk, "
                                      f"{expected_size} expected"})
            continue

        expected_sha = lfs_sha256(sib)
        if expected_sha:
            actual = sha256_file(path)
            if actual != expected_sha:
                results.append({"file": rel, "status": HASH_MISMATCH,
                                "detail": f"sha256 {actual[:16]}… != "
                                          f"{expected_sha[:16]}… (lfs oid)"})
            else:
                results.append({"file": rel, "status": OK,
                                "detail": "sha256 matches lfs oid"})
            continue

        expected_blob = getattr(sib, "blob_id", None)
        if expected_blob:
            actual = git_blob_sha1(path)
            if actual != expected_blob:
                results.append({"file": rel, "status": HASH_MISMATCH,
                                "detail": f"git blob sha1 {actual[:16]}… != "
                                          f"{expected_blob[:16]}…"})
            else:
                results.append({"file": rel, "status": OK,
                                "detail": "git blob sha1 matches"})
            continue

        results.append({"file": rel, "status": UNVERIFIED,
                        "detail": f"no hash from Hub, size {actual_size} "
                                  f"bytes accepted"})

    return (results, None)


def verify_whisper(speech_encoder_dir):
    """whisper embeds the expected sha256 in the download URL path."""
    try:
        import whisper
    except ImportError:
        return ([], "whisper not importable, cannot verify")

    url = whisper._MODELS[WHISPER_NAME]
    expected = url.split("/")[-2]
    path = Path(speech_encoder_dir) / f"{WHISPER_NAME}.pt"

    if not path.exists():
        return ([{"file": str(path), "status": MISSING,
                  "detail": "not present on disk"}], None)

    actual = sha256_file(path)
    if actual != expected:
        return ([{"file": str(path), "status": HASH_MISMATCH,
                  "detail": f"sha256 {actual[:16]}… != {expected[:16]}…"}], None)
    return ([{"file": str(path), "status": OK,
              "detail": "sha256 matches whisper manifest"}], None)


def failures(results):
    return [r for r in results if r["status"] not in (OK, UNVERIFIED)]


def print_report(section, results, note=None):
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no files"
    print(f"\n=== {section}: {summary}")
    if note:
        print(f"    note: {note}")
    for r in results:
        if r["status"] != OK:
            print(f"    [{r['status']}] {r['file']}")
            print(f"        {r['detail']}")


def write_report(models_dir, report):
    path = Path(models_dir) / "verify_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    log(f"full report written to {path}")
    return path


# ----------------------------------------------------------------------------
# download
# ----------------------------------------------------------------------------

def purge_bad_files(local_dir, bad_results):
    """Delete corrupt files and their hub download metadata so resume refetches."""
    local_dir = Path(local_dir)
    for r in bad_results:
        if r["status"] not in REDOWNLOAD:
            continue
        target = local_dir / r["file"]
        meta = local_dir / ".cache" / "huggingface" / "download" / (r["file"] + ".metadata")
        for p in (target, meta):
            try:
                if p.is_file():
                    p.unlink()
                    log(f"removed {p}")
            except OSError as exc:
                log(f"could not remove {p}: {exc}")


def download_hf_repo(repo_id, local_dir, attempts):
    base = hf_download_cmd()
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    results, note = [], None
    for attempt in range(1, attempts + 1):
        log(f"--- {repo_id} -> {local_dir} (attempt {attempt}/{attempts})")
        proc = run(base + [repo_id, "--local-dir", str(local_dir)])
        if proc.returncode != 0:
            log(f"downloader exited {proc.returncode}")

        log(f"verifying {repo_id} (hashing every file, this reads the whole download)")
        results, note = verify_hf_repo(repo_id, local_dir)
        bad = failures(results)
        if not bad:
            log(f"{repo_id} verified clean")
            return results, note

        log(f"{len(bad)} file(s) failed verification")
        if attempt < attempts:
            purge_bad_files(local_dir, bad)
            time.sleep(5 * attempt)

    return results, note


def download_whisper(speech_encoder_dir, attempts):
    """
    Your snippet, unchanged:
        model = whisper.load_model("large-v3", download_root="models/speech_encoder/")
    load_model checksums the file itself and refetches on mismatch. It also
    materialises the weights in RAM (~3 GB) as a side effect.
    """
    Path(speech_encoder_dir).mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        log(f"--- whisper {WHISPER_NAME} -> {speech_encoder_dir} "
            f"(attempt {attempt}/{attempts})")
        try:
            import whisper
            model = whisper.load_model(WHISPER_NAME, download_root=str(speech_encoder_dir))
            del model
        except ImportError:
            return ([{"file": f"{WHISPER_NAME}.pt", "status": MISSING,
                      "detail": "openai-whisper is not installed"}], None)
        except Exception as exc:
            log(f"whisper download/load failed: {exc}")

        results, note = verify_whisper(speech_encoder_dir)
        if not failures(results):
            log("whisper verified clean")
            return results, note
        if attempt < attempts:
            bad = Path(speech_encoder_dir) / f"{WHISPER_NAME}.pt"
            if bad.is_file():
                bad.unlink()
                log(f"removed {bad}")
            time.sleep(5 * attempt)
    return results, note


# ----------------------------------------------------------------------------
# tmux
# ----------------------------------------------------------------------------

def tmux(*args, **kwargs):
    return subprocess.run(["tmux", *args], **kwargs)


def session_exists():
    return tmux("has-session", "-t", SESSION,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def send(window, command, dry_run):
    target = f"{SESSION}:{window}"
    print(f"    {window}: {command}")
    if not dry_run:
        tmux("send-keys", "-t", target, command, "C-m")


def serve(models_dir, model_name, env_cmd, dry_run, worker_timeout):
    if not shutil.which("tmux"):
        die("tmux not found on PATH")
    if session_exists():
        die(f"tmux session '{SESSION}' already exists. "
            f"`tmux attach -t {SESSION}` or `./omni2.py stop`")

    cwd = os.getcwd()
    vocoder_dir = Path(models_dir) / "cosy2_decoder"
    model_path = Path(models_dir) / model_name

    for p in (vocoder_dir, model_path, Path(models_dir) / "speech_encoder"):
        if not p.exists():
            die(f"{p} does not exist. Run `./omni2.py download` first.")

    controller_cmd = (
        f"python -m llama_omni2.serve.controller "
        f"--host 0.0.0.0 --port {CONTROLLER_PORT}"
    )
    gradio_cmd = (
        f"python -m llama_omni2.serve.gradio_web_server "
        f"--controller http://localhost:{CONTROLLER_PORT} --port {GRADIO_PORT} "
        f"--vocoder-dir {vocoder_dir}"
    )
    worker_cmd = (
        f"python -m llama_omni2.serve.model_worker "
        f"--host 0.0.0.0 --controller http://localhost:{CONTROLLER_PORT} "
        f"--port {WORKER_PORT} --worker http://localhost:{WORKER_PORT} "
        f"--model-path {model_path} --model-name {model_name}"
    )

    if dry_run:
        print(f"\nwould create tmux session '{SESSION}' in {cwd}:")

    if not dry_run:
        tmux("new-session", "-d", "-s", SESSION, "-n", "controller", "-c", cwd)
    if env_cmd:
        send("controller", env_cmd, dry_run)
    send("controller", controller_cmd, dry_run)

    if dry_run:
        print("    [wait for controller port]")
    else:
        wait_for_port(CONTROLLER_PORT, 60, "controller")

    if not dry_run:
        tmux("new-window", "-t", SESSION, "-n", "gradio", "-c", cwd)
    if env_cmd:
        send("gradio", env_cmd, dry_run)
    send("gradio", gradio_cmd, dry_run)

    if not dry_run:
        tmux("new-window", "-t", SESSION, "-n", "worker", "-c", cwd)
    if env_cmd:
        send("worker", env_cmd, dry_run)
    send("worker", worker_cmd, dry_run)

    if dry_run:
        return 0

    wait_for_port(GRADIO_PORT, 120, "gradio web server")
    wait_for_port(WORKER_PORT, worker_timeout, "model worker")

    print()
    log(f"tmux attach -t {SESSION}    # windows: controller, gradio, worker")
    log(f"open http://localhost:{GRADIO_PORT}/")
    return 0


def status(model_name):
    if not shutil.which("tmux"):
        die("tmux not found on PATH")
    if session_exists():
        tmux("list-windows", "-t", SESSION)
    else:
        print(f"tmux session '{SESSION}' is not running")
    for label, port in (("controller", CONTROLLER_PORT),
                        ("gradio", GRADIO_PORT),
                        (f"worker ({model_name})", WORKER_PORT)):
        print(f"port {port:<6} {'open' if port_open(port) else 'closed':<6} {label}")
    return 0


def stop():
    if not session_exists():
        log(f"no tmux session '{SESSION}'")
        return 0
    tmux("kill-session", "-t", SESSION)
    log(f"killed tmux session '{SESSION}'")
    return 0


# ----------------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------------

def cmd_download(args):
    models_dir = Path(args.models_dir)
    speech_dir = models_dir / "speech_encoder"
    report = {"generated": datetime.now(timezone.utc).isoformat(),
              "models_dir": str(models_dir), "sections": {}}

    sections = []
    if not args.skip_whisper:
        res, note = download_whisper(speech_dir, args.attempts)
        sections.append((f"whisper {WHISPER_NAME}", res, note))

    res, note = download_hf_repo(COSY_REPO, models_dir / "cosy2_decoder", args.attempts)
    sections.append((COSY_REPO, res, note))

    llama_repo = f"ICTNLP/{args.model_name}"
    res, note = download_hf_repo(llama_repo, models_dir / args.model_name, args.attempts)
    sections.append((llama_repo, res, note))

    return finish(sections, report, models_dir)


def cmd_verify(args):
    models_dir = Path(args.models_dir)
    report = {"generated": datetime.now(timezone.utc).isoformat(),
              "models_dir": str(models_dir), "sections": {}}

    sections = []
    if not args.skip_whisper:
        res, note = verify_whisper(models_dir / "speech_encoder")
        sections.append((f"whisper {WHISPER_NAME}", res, note))

    res, note = verify_hf_repo(COSY_REPO, models_dir / "cosy2_decoder")
    sections.append((COSY_REPO, res, note))

    llama_repo = f"ICTNLP/{args.model_name}"
    res, note = verify_hf_repo(llama_repo, models_dir / args.model_name)
    sections.append((llama_repo, res, note))

    return finish(sections, report, models_dir)


def finish(sections, report, models_dir):
    total_bad = 0
    for name, results, note in sections:
        print_report(name, results, note)
        bad = failures(results)
        total_bad += len(bad)
        report["sections"][name] = {
            "note": note,
            "counts": {s: sum(1 for r in results if r["status"] == s)
                       for s in {r["status"] for r in results}},
            "problems": bad,
            "files": results,
        }
    report["total_problems"] = total_bad
    write_report(models_dir, report)

    print()
    if total_bad:
        log(f"{total_bad} problem file(s). NOT safe to serve. See verify_report.json")
        return 2
    log("all files verified. Safe to run `./omni2.py serve`")
    return 0


# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLaMA-Omni2 download / verify / serve")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    common.add_argument("--model-name", default=DEFAULT_MODEL_NAME)

    d = sub.add_parser("download", parents=[common], help="download then hash-verify")
    d.add_argument("--attempts", type=int, default=3,
                   help="download+verify rounds per model (default 3)")
    d.add_argument("--skip-whisper", action="store_true")
    d.set_defaults(func=cmd_download)

    v = sub.add_parser("verify", parents=[common], help="hash-verify only")
    v.add_argument("--skip-whisper", action="store_true")
    v.set_defaults(func=cmd_verify)

    s = sub.add_parser("serve", parents=[common], help="launch the 3 servers in tmux")
    s.add_argument("--env-cmd", default=os.environ.get("OMNI2_ENV_CMD", ""),
                   help="command sent to each tmux window first, "
                        "e.g. 'conda activate llama-omni2'")
    s.add_argument("--worker-timeout", type=int, default=900,
                   help="seconds to wait for the model worker port (default 900)")
    s.add_argument("--dry-run", action="store_true",
                   help="print the tmux plan and commands, create nothing")
    s.set_defaults(func=lambda a: serve(a.models_dir, a.model_name, a.env_cmd,
                                        a.dry_run, a.worker_timeout))

    st = sub.add_parser("status", parents=[common])
    st.set_defaults(func=lambda a: status(a.model_name))

    sp = sub.add_parser("stop")
    sp.set_defaults(func=lambda a: stop())

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()

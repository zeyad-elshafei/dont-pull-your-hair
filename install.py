#!/usr/bin/env python3
"""Install Tricho.

Creates a .venv next to this script, installs the pinned dependencies,
downloads the MediaPipe model files, and seeds config.json on first run.

    python install.py          # or: bash install.sh / install.bat

Safe to re-run: an existing .venv, models and config.json are reused.
"""

import json
import os
import platform
import shutil
import ssl
import subprocess
import sys
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(BASE_DIR, ".venv")
REQUIREMENTS = os.path.join(BASE_DIR, "requirements.txt")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

MIN_PYTHON = (3, 10)
MIN_MODEL_BYTES = 100_000

MODELS = {
    "hair_segmenter.tflite":
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "hair_segmenter/float32/latest/hair_segmenter.tflite",
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task",
}


# --------------------------------------------------------------------- helpers
def venv_python():
    if os.name == "nt":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")


def running_in_venv():
    venv = os.path.normcase(os.path.abspath(VENV_DIR))
    exe = os.path.normcase(os.path.abspath(sys.executable))
    return exe.startswith(venv + os.sep)


def python_ok(exe):
    try:
        return subprocess.call(
            [exe, "-c",
             "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    except OSError:
        return False


def ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def run_cmd(script):
    exe = venv_python()
    if os.name == "nt":
        return f'"{exe}" {script}'
    return f"{exe} {script}"


# ----------------------------------------------------------------------- steps
def check_python():
    if sys.version_info < MIN_PYTHON:
        sys.exit(f"Error: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required; "
                 f"found {platform.python_version()}.")
    if sys.platform == "darwin" and platform.machine() != "arm64":
        print("Warning: MediaPipe publishes arm64-only wheels for macOS; "
              "this Intel Mac is not supported and installation may fail.")


def ensure_venv():
    vp = venv_python()
    if os.path.exists(vp):
        if python_ok(vp):
            return
        print("Existing .venv uses an unsupported Python; recreating it ...")
        shutil.rmtree(VENV_DIR, ignore_errors=True)

    print(f"Creating virtual environment in {VENV_DIR} ...")
    try:
        subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
    except subprocess.CalledProcessError:
        sys.exit("Could not create the virtual environment. On Debian/Ubuntu, "
                 "install the venv package first: sudo apt install python3-venv")


def install_pip():
    print("Upgrading pip ...", flush=True)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "--version"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("pip is missing; bootstrapping it with ensurepip ...", flush=True)
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "--upgrade", "pip"])


def install_requirements():
    if not os.path.exists(REQUIREMENTS):
        sys.exit(f"Error: {REQUIREMENTS} not found next to install.py")
    print("Installing dependencies (this can take a few minutes) ...",
          flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "-r", REQUIREMENTS])


def download(url, dest):
    tmp = dest + ".part"
    ctx = ssl_context()
    with urllib.request.urlopen(url, timeout=120, context=ctx) as resp, \
            open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        written = 0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
            written += len(chunk)
            if total:
                pct = written * 100 // total
                print(f"\r  {pct:3d}%  {written / 1e6:5.1f} / "
                      f"{total / 1e6:.1f} MB", end="", flush=True)
    print()

    if os.path.getsize(tmp) < MIN_MODEL_BYTES:
        os.remove(tmp)
        raise RuntimeError(f"download of {os.path.basename(dest)} looks too "
                           f"small; check your network/proxy")
    os.replace(tmp, dest)


def download_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, url in MODELS.items():
        dest = os.path.join(MODELS_DIR, name)
        if os.path.exists(dest) and os.path.getsize(dest) >= MIN_MODEL_BYTES:
            print(f"Model {name} already present")
            continue
        print(f"Downloading {name} ...")
        download(url, dest)


def ensure_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                json.load(f)
            print("config.json already present (left unchanged)")
        except (OSError, json.JSONDecodeError):
            print("Warning: config.json is not valid JSON; the app will fall "
                  "back to its built-in defaults.")
        return

    from detector import DEFAULTS
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(DEFAULTS, f, indent=2)
        f.write("\n")
    print("Created config.json with default settings")


def verify():
    print("Verifying imports ...", flush=True)
    env = dict(os.environ, PYGAME_HIDE_SUPPORT_PROMPT="1")
    subprocess.check_call(
        [sys.executable, "-c",
         "import mediapipe, cv2, numpy, desktop_notifier, PySide6, pygame"],
        env=env)
    print("All dependencies import successfully")


def print_done():
    print()
    print("Installation complete.", flush=True)
    print(f"  Start the app:        {run_cmd('gui.py')}")
    print(f"  Command-line mode:    {run_cmd('detect.py')}")


# ------------------------------------------------------------------------ main
def main(argv):
    if any(a in ("-h", "--help") for a in argv[1:]):
        print(__doc__)
        return 0

    check_python()

    if not running_in_venv():
        ensure_venv()
        print(f"Switching to the virtual environment: {venv_python()}",
              flush=True)
        return subprocess.call([venv_python(), os.path.abspath(__file__)]
                               + argv[1:])

    try:
        install_pip()
        install_requirements()
        download_models()
        ensure_config()
        verify()
    except subprocess.CalledProcessError as e:
        sys.exit(f"\nInstallation failed (command exited {e.returncode}). "
                 f"If the environment is broken, delete {VENV_DIR} and retry.")
    except (OSError, RuntimeError) as e:
        sys.exit(f"\nInstallation failed: {e}")

    print_done()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

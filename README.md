# Demucs MLX

Music source separation on Apple Silicon, powered by [MLX](https://github.com/ml-explore/mlx).

A clean MLX port of Meta's [Hybrid Transformer Demucs](https://github.com/facebookresearch/demucs) (HTDemucs) for **inference-only** on Mac M1/M2/M3/M4. Separates any song into 4 stems: **drums**, **bass**, **other**, and **vocals**.

<p align="center">
  <img src="https://img.shields.io/badge/Apple%20Silicon-MLX-black?logo=apple" alt="Apple Silicon MLX">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License">
  <img src="https://img.shields.io/badge/python-3.10+-green" alt="Python 3.10+">
</p>

## Performance

Benchmarked on Apple M4 Max with MLX 0.21:

| Audio length | Processing time | Speed |
|:---:|:---:|:---:|
| 10s | 0.4s | **26x** realtime |
| 30s | 1.0s | **30x** realtime |
| 1 min | 1.8s | **33x** realtime |
| 3 min | 5.3s | **34x** realtime |
| 6 min | 10.7s | **34x** realtime |

**11x faster** than PyTorch CPU on the same machine. A 7-minute track separates in about 12 seconds.

Numerical accuracy: max difference vs PyTorch is **< 1 part per million** (0.8 µ). The output is effectively identical.

## Quick start

### 1. Set up a Python environment

Demucs MLX requires **Python 3.10 or later**. We recommend a dedicated conda environment to avoid dependency conflicts (especially with PyTorch):

```bash
# Create and activate a conda environment
conda create -n demucs python=3.11 -y
conda activate demucs
```

> **No conda?** You can also use `python3 -m venv .venv && source .venv/bin/activate`. Just make sure `python --version` shows 3.10+.

### 2. Install

```bash
git clone https://github.com/andrade0/demucs-mlx.git
cd demucs-mlx
make install
```

This installs the `demucs_mlx` package in editable mode with all dependencies (MLX, PyTorch CPU, etc.) and puts the `demucs-mlx` command in your PATH (`~/.local/bin`).

Alternatively, install it as a regular Python package into your active environment:

```bash
pip install .            # or: pip install git+https://github.com/iDoka/demucs-mlx.git
demucs-mlx song.mp3      # console script
python -m demucs_mlx song.mp3
```

### 3. Separate a song

```bash
demucs-mlx song.mp3
```

The first run downloads the pretrained model (~80 MB, cached in `~/.cache/demucs_mlx/`, or in the directory given by `--model-dir`). Output goes to `separated/htdemucs/<song>/` with 4 WAV files: `drums.wav`, `bass.wav`, `other.wav`, `vocals.wav`.

> If you see a PATH warning during install, add this to your `~/.zshrc`:
> ```bash
> export PATH="$HOME/.local/bin:$PATH"
> ```

### Usage

Full reference (`demucs-mlx --help`):

```
$ demucs-mlx --help

usage: demucs-mlx [-h] [-n NAME] [--model-dir DIR] [-o DIR]
                  [--stems STEM [STEM ...]] [--mp3 | --m4a] [--float32]
                  [--shifts N] [--overlap F] [--no-split]
                  input

Separate a song into stems (drums, bass, other, vocals; plus guitar and
piano with htdemucs_6s) using Meta's HTDemucs on Apple Silicon with MLX.

Input is decoded with soundfile, then macOS afconvert, then librosa as a
fallback, so WAV, FLAC, OGG, MP3, M4A/AAC and misnamed containers all work.
Mono input is duplicated to stereo; input that is not 44.1 kHz is resampled
with librosa before separation (stems are written at 44.1 kHz).

positional arguments:
  input                 input audio file: WAV, FLAC, OGG, MP3, M4A/AAC or
                        anything soundfile, afconvert or librosa can decode

options:
  -h, --help            show this help message and exit

model:
  -n, --name NAME       pretrained model: htdemucs, htdemucs_ft, htdemucs_6s
                        or a raw checkpoint signature; see 'models' below
                        (default: htdemucs)
  --model-dir DIR       directory where model checkpoints are looked up and
                        downloaded to; created if missing, '~' is expanded;
                        see 'model files' below (default: ~/.cache/demucs_mlx)

output:
  -o, --output DIR      base output directory; stems are written to
                        DIR/<song>/ (default: ./separated/<model>, i.e.
                        ./separated/<model>/<song>/)
  --stems STEM [STEM ...]
                        save only these stems, space separated: drums bass
                        other vocals (+ guitar piano for htdemucs_6s). With
                        htdemucs_ft this also skips the unneeded models
                        (default: all stems of the model)
  --mp3                 save stems as MP3 (LAME VBR -q:a 2, ~190 kbps);
                        requires ffmpeg in PATH
  --m4a                 save stems as M4A/AAC using Apple's AudioToolbox
                        encoder (ffmpeg -c:a aac_at); requires ffmpeg built
                        with AudioToolbox, e.g. Homebrew's
  --float32             write 32-bit float WAV instead of 16-bit PCM (no
                        clipping, 2x file size); WAV output only

quality / performance:
  --shifts N            run the model N times on the input shifted by a random
                        0-0.5 s offset and average the results: slightly
                        better quality, N times slower; 0 disables shifting
                        (default: 1)
  --overlap F           overlap between consecutive ~7.8 s segments as a
                        fraction in [0, 1); higher values give smoother
                        segment boundaries but take longer, e.g. 0.5 is ~1.5x
                        slower than 0.25 (default: 0.25)
  --no-split            process the input in one pass instead of overlapping
                        segments; HTDemucs only accepts up to one training
                        segment (~7.8 s, ~7.3 s with --shifts >= 1), so this
                        is only useful for short clips

models (-n NAME):
  htdemucs      default; 4 stems, 1 checkpoint (~80 MB), fastest
  htdemucs_ft   4 stems; bag of 4 fine-tuned specialists (~4 x 80 MB),
                one per stem: best quality, ~4x slower. With --stems only
                the specialists for the requested stems are loaded, e.g.
                --stems vocals runs a single model.
  htdemucs_6s   6 stems: drums bass other vocals guitar piano (~55 MB);
                piano is noticeably weaker than the other stems
  <signature>   a raw checkpoint id from demucs_mlx/remote/files.txt
                (HTDemucs checkpoints only, e.g. 955717e8)

model files (--model-dir):
  Checkpoints are downloaded from dl.fbaipublicfiles.com on first use and
  stored under human-readable names, verified by sha256 on every load:
    htdemucs.th  htdemucs_6s.th
    htdemucs_ft_drums.th  htdemucs_ft_bass.th
    htdemucs_ft_other.th  htdemucs_ft_vocals.th
  Files put there by hand are used as-is (after the sha256 check), so a
  directory can be pre-populated for offline use. Cache entries left by
  older versions (<signature>-<sha256>.th) are renamed on first use.

output layout:
  <DIR>/<song>/<stem>.<wav|mp3|m4a>, where <DIR> is -o DIR or
  ./separated/<model>; <song> is the input file name without extension.
  Existing files are overwritten.

examples:
  demucs-mlx song.mp3
  demucs-mlx song.mp3 --stems vocals
  demucs-mlx song.mp3 --stems vocals drums -o my_stems/
  demucs-mlx song.mp3 -n htdemucs_ft --stems vocals --model-dir ./models
  demucs-mlx song.mp3 -n htdemucs_6s --stems guitar piano --m4a
  demucs-mlx song.mp3 --shifts 3 --overlap 0.5 --float32
  python -m demucs_mlx song.mp3          (same as demucs-mlx)
```

### Examples

```bash
# Extract only vocals
demucs-mlx song.mp3 --stems vocals

# Extract vocals and drums
demucs-mlx song.mp3 --stems vocals drums

# Custom output directory
demucs-mlx song.mp3 -o my_stems/

# Better quality (3 shifts), slower
demucs-mlx song.mp3 --shifts 3

# Use the fine-tuned model (4 specialist models)
demucs-mlx song.mp3 -n htdemucs_ft

# Fine-tuned vocals only: loads and runs just the vocals specialist
demucs-mlx song.mp3 -n htdemucs_ft --stems vocals

# Keep model files in a project directory instead of ~/.cache/demucs_mlx
demucs-mlx song.mp3 --model-dir ./models

# Save as float32 WAV
demucs-mlx song.mp3 --float32

# Save stems as MP3
demucs-mlx song.wav --mp3
```

### Use from Python

```python
import mlx.core as mx
import soundfile as sf
from demucs_mlx.pretrained import load_model
from demucs_mlx.apply import apply_model

# Load model (downloads on first use). For htdemucs_ft this returns a
# BagOfModels of 4 specialists; pass stems=[...] to load only some of them.
model = load_model("htdemucs", cache_dir="./models")

# Load audio
wav, sr = sf.read("song.mp3", dtype="float32")
mix = mx.array(wav.T[None])  # [1, channels, samples]

# Separate
sources = apply_model(model, mix, shifts=1, split=True)
mx.eval(sources)

# sources shape: [1, 4, channels, samples]
# stems: drums, bass, other, vocals
```

## How it works

HTDemucs is a hybrid model that processes audio in two parallel branches:

1. **Frequency branch** — STFT spectrogram through a Conv2d U-Net
2. **Time branch** — Raw waveform through a Conv1d U-Net

Both branches meet at a **Cross-Transformer** bottleneck that lets them exchange information, then decode back to 4 separate source waveforms.

```
                    ┌─────────────────┐
   Waveform ──────►│  STFT (PyTorch)  │
       │            └────────┬────────┘
       │                     │
       ▼                     ▼
  ┌─────────┐         ┌─────────┐
  │  Time   │         │  Freq   │
  │ Encoder │         │ Encoder │
  │ (Conv1d)│         │ (Conv2d)│
  └────┬────┘         └────┬────┘
       │                   │
       └───────┬───────────┘
               ▼
     ┌─────────────────┐
     │ Cross-Transformer│
     │   (5+5 layers)   │
     └────────┬─────────┘
              │
       ┌──────┴──────┐
       ▼              ▼
  ┌─────────┐   ┌─────────┐
  │  Time   │   │  Freq   │
  │ Decoder │   │ Decoder │
  └────┬────┘   └────┬────┘
       │              │
       │         ┌────┴────┐
       │         │  iSTFT  │
       │         └────┬────┘
       └──────┬───────┘
              ▼
      4 separated stems
  (drums, bass, other, vocals)
```

### MLX-specific design choices

- **Layout convention**: Tensors use PyTorch's `[B, C, T]` / `[B, C, H, W]` format externally; transpose at every Conv boundary since MLX expects channels-last
- **STFT/iSTFT**: Runs through PyTorch as a bridge (MLX doesn't support complex tensors). This adds ~5% overhead but keeps the code simple
- **Weight loading**: Downloads the original PyTorch checkpoint, converts Conv weights (transpose axes) and splits MultiheadAttention projections, then loads into the MLX model
- **No training code**: Inference-only, all batch norm / dropout / weight init stripped out

## Choosing a model

Models download automatically on first use (~80 MB per checkpoint) into `~/.cache/demucs_mlx/`, or into `--model-dir DIR`. Files get human-readable names (`htdemucs.th`, `htdemucs_6s.th`, `htdemucs_ft_drums.th` … `htdemucs_ft_vocals.th`) and are verified by sha256 on every load, so a directory can also be filled by hand for offline use. Use the `-n` flag to switch models.

### Available models

| Model | Sources | Description | Status |
|-------|---------|-------------|--------|
| `htdemucs` | 4 (drums, bass, other, vocals) | Default HTDemucs — fast and reliable | Supported |
| `htdemucs_ft` | 4 (drums, bass, other, vocals) | Bag of 4 fine-tuned models, one specialist per stem — best quality, ~4x slower | Supported |
| `htdemucs_6s` | 6 (drums, bass, other, vocals, guitar, piano) | 6-source variant — separates guitar and piano too | Supported |

### Which model should I use?

- **General use?** Start with `htdemucs` (the default). It's the standard model and works well on most music.
- **Best quality?** Use `htdemucs_ft` — four models, each fine-tuned for one stem; every stem is taken from its specialist. It runs all four models (~4x slower), unless you pass `--stems`: `-n htdemucs_ft --stems vocals` runs only the vocals specialist, as fast as the default model.
- **Need guitar or piano stems?** Try `htdemucs_6s` — it outputs 6 separate stems instead of 4. The piano stem is noticeably weaker than the others (as in the original Demucs).

```bash
# Default model (4 stems)
demucs-mlx song.mp3

# Fine-tuned model (better vocals)
demucs-mlx song.mp3 -n htdemucs_ft

# 6-source model (adds guitar + piano)
demucs-mlx song.mp3 -n htdemucs_6s
```

## Requirements

- **macOS** with Apple Silicon (M1, M2, M3, or M4)
- **Python 3.10+** (3.11 recommended)
- **RAM**: 8 GB minimum, 16 GB+ recommended for long tracks without chunking
- A conda or virtual environment (recommended to avoid dependency conflicts)

### Dependencies

Installed automatically by `make install`:

| Package | Why |
|---------|-----|
| `mlx` >= 0.17 | Apple's ML framework (GPU acceleration) |
| `torch` | STFT/iSTFT + weight loading (CPU only) |
| `numpy` | Array operations |
| `soundfile` | Audio file I/O (WAV, MP3, FLAC, OGG) |
| `pyyaml` | Model config parsing |
| `tqdm` | Progress bars |

`librosa` is also installed — it resamples input that isn't 44.1 kHz and serves as a fallback decoder for files soundfile can't read.

> **Note on PyTorch**: PyTorch is only used for STFT/iSTFT and loading pretrained weights. It runs on CPU — all the heavy computation (convolutions, transformer, U-Net) runs on Apple GPU through MLX. A future version may remove the PyTorch dependency entirely.

## Project structure

```
demucs-mlx/
├── Makefile                 # make install / make uninstall
├── pyproject.toml           # package metadata + `demucs-mlx` entry point
└── demucs_mlx/
    ├── cli.py               # CLI entry point (demucs-mlx / python -m demucs_mlx)
    ├── __main__.py          # enables `python -m demucs_mlx`
    ├── htdemucs.py          # Main model (hybrid U-Net + transformer)
    ├── hdemucs.py           # Encoder/decoder layers
    ├── transformer.py       # Cross-attention transformer
    ├── demucs.py            # DConv residual branches
    ├── apply.py             # Inference pipeline (chunking, shifts)
    ├── pretrained.py        # Model download & weight loading
    ├── weight_convert.py    # PyTorch → MLX weight conversion
    ├── spec.py              # STFT/iSTFT bridge
    ├── utils.py             # Tensor utilities
    └── remote/              # Model download configs (shipped with the package)
        ├── files.txt
        └── htdemucs.yaml
```

## License

MIT License — same as the [original Demucs](https://github.com/facebookresearch/demucs).

## Credits

- [Demucs](https://github.com/facebookresearch/demucs) by Meta Research — the original PyTorch model and pretrained weights
- [MLX](https://github.com/ml-explore/mlx) by Apple — the framework that makes this fast on Apple Silicon
- HTDemucs paper: [Rouard, Massa, Défossez (2023)](https://arxiv.org/abs/2211.08553)

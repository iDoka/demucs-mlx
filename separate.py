#!/usr/bin/env python3
"""Separate music sources with HTDemucs on Apple Silicon (MLX).

Usage:
    demucs-mlx song.mp3
    demucs-mlx song.wav --stems vocals drums
    demucs-mlx song.mp3 -o output_dir/
"""

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import tempfile
import time


_MAGIC_EXTENSIONS = {
    b"RIFF": ".wav",
    b"fLaC": ".flac",
    b"OggS": ".ogg",
    b"ID3": ".mp3",
}


def _sniff_extension(path):
    # Return a likely-correct extension based on file magic bytes, or None
    # if the existing extension already matches what we see. Useful when a
    # file is misnamed (e.g. an MP4/AAC file with an .mp3 extension), since
    # CoreAudio dispatches by extension.
    try:
        with open(path, "rb") as f:
            head = f.read(12)
    except OSError:
        return None
    if len(head) >= 8 and head[4:8] == b"ftyp":
        return ".m4a"
    for magic, ext in _MAGIC_EXTENSIONS.items():
        if head.startswith(magic):
            return ext
    return None


def load_audio(path, dtype="float32"):
    import soundfile as sf

    soundfile_error = None
    try:
        wav, sr = sf.read(path, dtype=dtype)
    except Exception as exc:
        soundfile_error = exc
        wav = sr = None

    if wav is None and sys.platform == "darwin":
        afconvert = shutil.which("afconvert")
        if afconvert:
            # CoreAudio uses the extension to pick a parser, so if the file
            # is misnamed (e.g. an MP4 container with a .mp3 suffix) we copy
            # it to a temp file with the sniffed extension first.
            sniffed = _sniff_extension(path)
            current_ext = os.path.splitext(path)[1].lower()
            input_path = path
            staged_input = None
            if sniffed and sniffed != current_ext:
                fd, staged_input = tempfile.mkstemp(suffix=sniffed)
                os.close(fd)
                shutil.copyfile(path, staged_input)
                input_path = staged_input

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav",
                                                 delete=False) as tmp:
                    tmp_path = tmp.name
                subprocess.run(
                    [afconvert, "-f", "WAVE", "-d", "LEI16",
                     input_path, tmp_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                wav, sr = sf.read(tmp_path, dtype=dtype)
            except Exception:
                pass
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
                if staged_input and os.path.exists(staged_input):
                    os.remove(staged_input)

    if wav is None:
        try:
            import librosa

            wav, sr = librosa.load(path, sr=None, mono=False, dtype=dtype)
            if wav.ndim > 1:
                wav = wav.T
        except Exception as fallback_error:
            raise RuntimeError(
                f"Failed to decode audio file {path!r} with soundfile "
                f"({soundfile_error}) and fallback decoder "
                f"({fallback_error}). Try converting the file to WAV first."
            ) from (soundfile_error or fallback_error)

    if wav.ndim == 1:
        wav = wav[:, None]
    if wav.shape[1] == 1:
        # HTDemucs expects stereo input; duplicate the mono channel.
        wav = wav.repeat(2, axis=1)
    return wav, sr


def main():
    parser = argparse.ArgumentParser(
        prog="demucs-mlx",
        description="Separate a song into stems (drums, bass, other, vocals) "
                    "using HTDemucs on Apple Silicon with MLX.",
        epilog="Examples:\n"
               "  demucs-mlx song.mp3\n"
               "  demucs-mlx song.mp3 --stems vocals\n"
               "  demucs-mlx song.mp3 --stems vocals drums -o my_stems/\n"
               "  demucs-mlx song.mp3 --shifts 3 --float32\n"
               "  demucs-mlx song.mp3 -n htdemucs_ft\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input",
                        help="input audio file (WAV, MP3, FLAC, OGG, etc.)")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("-n", "--name", default="htdemucs",
                             metavar="NAME",
                             help="model name: htdemucs, htdemucs_ft, "
                                  "htdemucs_6s (default: htdemucs)")

    output_group = parser.add_argument_group("output")
    output_group.add_argument("-o", "--output", default=None,
                              metavar="DIR",
                              help="output directory "
                                   "(default: ./separated/<model>/<song>/)")
    output_group.add_argument("--stems", nargs="+", default=None,
                              metavar="STEM",
                              help="which stems to save, choose from: "
                                   "drums bass other vocals (default: all)")
    output_group.add_argument("--mp3", action="store_true",
                              help="save stems as MP3 instead of WAV")
    output_group.add_argument("--m4a", action="store_true",
                              help="save stems as M4A/AAC instead of WAV")
    output_group.add_argument("--float32", action="store_true",
                              help="save as float32 WAV instead of int16 WAV")

    quality_group = parser.add_argument_group("quality")
    quality_group.add_argument("--shifts", type=int, default=1,
                               metavar="N",
                               help="random shifts for better quality, "
                                    "slower (default: 1)")
    quality_group.add_argument("--overlap", type=float, default=0.25,
                               metavar="F",
                               help="overlap between chunks, 0.0 to 1.0 "
                                    "(default: 0.25)")
    quality_group.add_argument("--no-split", action="store_true",
                               help="process the whole track at once "
                                    "instead of chunking (uses more memory)")

    args = parser.parse_args()

    if args.mp3 and args.float32:
        parser.error("--float32 is only supported for WAV output")

    import mlx.core as mx
    import numpy as np
    import soundfile as sf
    from demucs_mlx.pretrained import load_model
    from demucs_mlx.apply import apply_model

    # Load audio
    print(f"Loading audio: {args.input}")
    wav, sr = load_audio(args.input, dtype="float32")
    if wav.ndim == 1:
        wav = wav[:, None]
    # wav: [T, C] → [1, C, T]
    wav = wav.T[None, :, :]
    print(f"Audio: {wav.shape[2] / sr:.1f}s, {wav.shape[1]}ch, {sr}Hz")

    # Load model
    print(f"Loading model: {args.name}")
    t0 = time.time()
    model = load_model(args.name)
    print(f"Model loaded in {time.time() - t0:.1f}s "
          f"({len(model.sources)} sources: {model.sources})")

    # Resample if needed
    if sr != model.samplerate:
        print(f"Resampling {sr}Hz → {model.samplerate}Hz...")
        import librosa
        wav_np = wav[0]  # [C, T]
        resampled = []
        for ch in range(wav_np.shape[0]):
            resampled.append(librosa.resample(
                wav_np[ch], orig_sr=sr, target_sr=model.samplerate))
        wav = np.stack(resampled)[None, :, :]
        sr = model.samplerate

    mix = mx.array(wav)

    # Separate
    print(f"Separating with {args.shifts} shift(s)...")
    t0 = time.time()
    sources = apply_model(
        model, mix, shifts=args.shifts,
        split=not args.no_split, overlap=args.overlap,
        progress=True)
    mx.eval(sources)
    sep_time = time.time() - t0
    print(f"\nSeparated in {sep_time:.1f}s "
          f"({wav.shape[2] / sr / sep_time:.2f}x realtime)")

    # Convert to numpy and free MLX arrays + model from GPU memory before saving
    sources_np = np.array(sources[0])  # [S, C, T]
    stem_names = args.stems or model.sources
    all_sources = model.sources
    del mix, sources, model
    mx.clear_cache()
    basename = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.output or os.path.join("separated", args.name)
    out_dir = os.path.join(out_dir, basename)
    os.makedirs(out_dir, exist_ok=True)

    use_ffmpeg = args.mp3 or args.m4a
    if use_ffmpeg:
        ext = "mp3" if args.mp3 else "m4a"
        ffmpeg_extra = ["-q:a", "2"] if args.mp3 else ["-c:a", "aac_at", "-q:a", "100"]
    else:
        ext = "wav"

    subtype = 'FLOAT' if args.float32 else 'PCM_16'

    # Collect stems to save
    stems_to_save = [
        (i, src_name)
        for i, src_name in enumerate(all_sources)
        if src_name in stem_names
    ]

    def encode_stem(i, src_name):
        stem = sources_np[i].T  # [T, C]
        out_path = os.path.join(out_dir, f"{src_name}.{ext}")
        if use_ffmpeg:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                sf.write(tmp_path, stem, sr, subtype=subtype)
                subprocess.run(
                    ["ffmpeg", "-y", "-i", tmp_path] + ffmpeg_extra + [out_path],
                    check=True, capture_output=True,
                )
            finally:
                os.unlink(tmp_path)
        else:
            sf.write(out_path, stem, sr, subtype=subtype)
        return out_path

    if use_ffmpeg:
        time.sleep(0.5)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(stems_to_save)) as executor:
            futures = {executor.submit(encode_stem, i, src_name): src_name
                       for i, src_name in stems_to_save}
            for future in concurrent.futures.as_completed(futures):
                print(f"Saved: {future.result()}")
    else:
        for i, src_name in stems_to_save:
            print(f"Saved: {encode_stem(i, src_name)}")

    print(f"\nDone! Output in: {out_dir}")


if __name__ == "__main__":
    main()

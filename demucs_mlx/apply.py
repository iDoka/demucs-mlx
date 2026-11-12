# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – Apply HTDemucs model to audio for source separation.
# Handles chunking with overlapping windows and the shift trick.

import random
import typing as tp

import mlx.core as mx
import numpy as np

from .htdemucs import HTDemucs
from .utils import center_trim


def apply_model(model: HTDemucs,
                mix: mx.array,
                shifts: int = 1,
                split: bool = True,
                overlap: float = 0.25,
                transition_power: float = 1.,
                progress: bool = False,
                segment: tp.Optional[float] = None) -> mx.array:
    """Apply HTDemucs model to separate sources from a mixture.

    Args:
        model: HTDemucs model.
        mix: Input mixture ``[B, C, T]``.
        shifts: Number of random time shifts for augmentation (0 = disabled).
        split: If True, split into overlapping chunks.
        overlap: Overlap ratio between chunks (0.25 = 25%).
        transition_power: Power for the triangular overlap weighting.
        progress: Show progress bar.
        segment: Override model segment length (seconds).

    Returns:
        Separated sources ``[B, S, C, T]``.
    """
    mx.eval(model.parameters())

    batch, channels, length = mix.shape

    if shifts:
        max_shift = int(0.5 * model.samplerate)
        out = mx.zeros((batch, len(model.sources), channels, length))
        for shift_idx in range(shifts):
            offset = random.randint(0, max_shift)
            # Pad and shift
            padded = mx.pad(mix, [(0, 0), (0, 0), (max_shift, max_shift)])
            shifted = padded[..., offset:offset + length + max_shift - offset]
            res = apply_model(
                model, shifted, shifts=0, split=split,
                overlap=overlap, transition_power=transition_power,
                progress=progress, segment=segment)
            out = out + res[..., max_shift - offset:max_shift - offset + length]
        return out / shifts

    if split:
        if segment is None:
            segment = float(model.segment)
        assert segment > 0.

        segment_length = int(model.samplerate * segment)
        stride = int((1 - overlap) * segment_length)

        out = mx.zeros((batch, len(model.sources), channels, length))
        sum_weight = mx.zeros((length,))

        # Triangular window
        w_left = mx.arange(1, segment_length // 2 + 1, dtype=mx.float32)
        w_right = mx.arange(segment_length - segment_length // 2, 0, -1,
                            dtype=mx.float32)
        weight = mx.concatenate([w_left, w_right])
        assert weight.shape[0] == segment_length
        weight = (weight / mx.max(weight)) ** transition_power

        offsets = list(range(0, length, stride))
        if progress:
            try:
                import tqdm
                offsets = tqdm.tqdm(offsets, unit_scale=stride / model.samplerate,
                                   ncols=120, unit='seconds')
            except ImportError:
                pass

        for offset in offsets:
            actual_len = min(segment_length, length - offset)
            chunk_out = _apply_padded(model, mix, offset, actual_len, segment)
            w = weight[:actual_len]

            # Accumulate
            out = _add_weighted(out, chunk_out, w, offset, actual_len)
            sum_weight = _add_weight(sum_weight, w, offset, actual_len)

        # Normalize
        out = out / mx.maximum(sum_weight[None, None, None, :],
                               mx.array(1e-8))
        return out

    else:
        # Direct application (no splitting)
        return _apply_padded(model, mix, 0, length, segment)


def _apply_padded(model: HTDemucs, mix: mx.array, offset: int, length: int,
                  segment: tp.Optional[float]) -> mx.array:
    """Run the model on ``mix[..., offset:offset + length]``.

    Like demucs' ``TensorChunk.padded``, the window is grown symmetrically
    to the model's valid length, using the surrounding audio as context
    and zeros only beyond the ends of ``mix``; the output is center-trimmed
    back to ``length`` samples.
    """
    if segment is not None:
        valid_length = int(segment * model.samplerate)
    elif hasattr(model, 'valid_length'):
        valid_length = model.valid_length(length)
    else:
        valid_length = length

    total = mix.shape[-1]
    start = offset - (valid_length - length) // 2
    end = start + valid_length
    chunk = mix[..., max(0, start):min(total, end)]
    pad_left, pad_right = max(0, -start), max(0, end - total)
    if pad_left or pad_right:
        chunk = mx.pad(chunk, [(0, 0), (0, 0), (pad_left, pad_right)])

    out = model(chunk)
    mx.eval(out)
    return center_trim(out, length)


def _add_weighted(out: mx.array, chunk: mx.array, weight: mx.array,
                  offset: int, length: int) -> mx.array:
    """Add weighted chunk to output accumulator (immutable update)."""
    # Weight shape: [length] → broadcast to [1, 1, 1, length]
    weighted = weight[None, None, None, :] * chunk
    # Since mx.array is immutable, we do slice assignment via concatenation
    before = out[..., :offset]
    middle = out[..., offset:offset + length] + weighted
    after = out[..., offset + length:]
    return mx.concatenate([before, middle, after], axis=-1)


def _add_weight(sum_w: mx.array, weight: mx.array,
                offset: int, length: int) -> mx.array:
    """Add weight to sum accumulator (immutable update)."""
    before = sum_w[:offset]
    middle = sum_w[offset:offset + length] + weight
    after = sum_w[offset + length:]
    return mx.concatenate([before, middle, after], axis=0)

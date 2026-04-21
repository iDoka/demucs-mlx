# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – HTDemucs (Hybrid Transformer Demucs) inference model.
# Spectrogram + time-domain hybrid model with CrossTransformer bottleneck.

import math
import typing as tp

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

from .demucs import DConv
from .hdemucs import ScaledEmbedding, HEncLayer, HDecLayer
from .spec import spectro, ispectro
from .transformer import CrossTransformerEncoder
from .utils import apply_conv1d, pad1d


class HTDemucs(nn.Module):
    """Hybrid Transformer Demucs for music source separation.

    Combines a frequency-domain U-Net (spectrogram) with a time-domain U-Net,
    connected by a CrossTransformerEncoder bottleneck.

    Input/output: ``[B, audio_channels, T]`` waveform.
    Output: ``[B, S, audio_channels, T]`` separated sources.
    """

    def __init__(
        self,
        sources: tp.List[str],
        # Channels
        audio_channels: int = 2,
        channels: int = 48,
        channels_time: tp.Optional[int] = None,
        growth: int = 2,
        # STFT
        nfft: int = 4096,
        wiener_iters: int = 0,
        end_iters: int = 0,
        wiener_residual: bool = False,
        cac: bool = True,
        # Main structure
        depth: int = 4,
        rewrite: bool = True,
        # Frequency branch
        multi_freqs: tp.Optional[tp.List[float]] = None,
        multi_freqs_depth: int = 3,
        freq_emb: float = 0.2,
        emb_scale: float = 10.,
        emb_smooth: bool = True,
        # Convolutions
        kernel_size: int = 8,
        time_stride: int = 2,
        stride: int = 4,
        context: int = 1,
        context_enc: int = 0,
        # Normalization
        norm_starts: int = 4,
        norm_groups: int = 4,
        # DConv residual branch
        dconv_mode: int = 1,
        dconv_depth: int = 2,
        dconv_comp: int = 8,
        dconv_init: float = 1e-3,
        # Transformer
        bottom_channels: int = 0,
        t_layers: int = 5,
        t_emb: str = "sin",
        t_hidden_scale: float = 4.0,
        t_heads: int = 8,
        t_dropout: float = 0.,
        t_max_positions: int = 10000,
        t_norm_in: bool = True,
        t_norm_in_group: bool = False,
        t_group_norm: bool = False,
        t_norm_first: bool = True,
        t_norm_out: bool = True,
        t_max_period: float = 10000.,
        t_weight_decay: float = 0.,
        t_lr: tp.Optional[float] = None,
        t_layer_scale: bool = True,
        t_gelu: bool = True,
        t_weight_pos_embed: float = 1.0,
        t_sin_random_shift: int = 0,
        t_cape_mean_normalize: bool = True,
        t_cape_augment: bool = True,
        t_cape_glob_loc_scale: list = [5000., 1., 1.4],
        t_sparse_self_attn: bool = False,
        t_sparse_cross_attn: bool = False,
        t_mask_type: str = "diag",
        t_mask_random_seed: int = 42,
        t_sparse_attn_window: int = 500,
        t_global_window: int = 100,
        t_sparsity: float = 0.95,
        t_auto_sparsity: bool = False,
        t_cross_first: bool = False,
        # Weight init (not used at inference)
        rescale: float = 0.1,
        # Metadata
        samplerate: int = 44100,
        segment: float = 10.,
        use_train_segment: bool = True,
    ):
        super().__init__()
        self.cac = cac
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.bottom_channels = bottom_channels
        self.channels = channels
        self.samplerate = samplerate
        self.segment = segment
        self.use_train_segment = use_train_segment
        self.nfft = nfft
        self.wiener_iters = wiener_iters
        self.end_iters = end_iters
        self.wiener_residual = wiener_residual
        self.hop_length = nfft // 4
        self.freq_emb = None
        self.freq_emb_scale = 0.

        if (self.wiener_iters != 0 or self.end_iters != 0
                or self.wiener_residual):
            raise NotImplementedError(
                "The MLX port does not implement Wiener filtering. "
                "Only checkpoints with wiener_iters=0, end_iters=0, and "
                "wiener_residual=False are supported."
            )

        self.encoder = []
        self.decoder = []
        self.tencoder = []
        self.tdecoder = []

        chin = audio_channels
        chin_z = chin * 2 if cac else chin
        chout = channels_time or channels
        chout_z = channels
        freqs = nfft // 2

        for index in range(depth):
            norm = index >= norm_starts
            freq = freqs > 1
            stri = stride
            ker = kernel_size
            if not freq:
                assert freqs == 1
                ker = time_stride * 2
                stri = time_stride

            pad = True
            last_freq = False
            if freq and freqs <= kernel_size:
                ker = freqs
                pad = False
                last_freq = True

            kw = {
                'kernel_size': ker,
                'stride': stri,
                'freq': freq,
                'pad': pad,
                'norm': norm,
                'rewrite': rewrite,
                'norm_groups': norm_groups,
                'dconv_kw': {
                    'depth': dconv_depth,
                    'compress': dconv_comp,
                    'init': dconv_init,
                    'gelu': True,
                },
            }
            kwt = dict(kw)
            kwt['freq'] = False
            kwt['kernel_size'] = kernel_size
            kwt['stride'] = stride
            kwt['pad'] = True
            kw_dec = dict(kw)

            if last_freq:
                chout_z = max(chout, chout_z)
                chout = chout_z

            enc = HEncLayer(chin_z, chout_z, dconv=bool(dconv_mode & 1),
                            context=context_enc, **kw)
            if freq:
                tenc = HEncLayer(chin, chout, dconv=bool(dconv_mode & 1),
                                 context=context_enc, empty=last_freq, **kwt)
                self.tencoder.append(tenc)
            self.encoder.append(enc)

            if index == 0:
                chin = audio_channels * len(sources)
                chin_z = chin * 2 if cac else chin

            dec = HDecLayer(chout_z, chin_z, dconv=bool(dconv_mode & 2),
                            last=(index == 0), context=context, **kw_dec)
            if freq:
                tdec = HDecLayer(chout, chin, dconv=bool(dconv_mode & 2),
                                 empty=last_freq, last=(index == 0),
                                 context=context, **kwt)
                self.tdecoder.insert(0, tdec)
            self.decoder.insert(0, dec)

            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)
            if freq:
                if freqs <= kernel_size:
                    freqs = 1
                else:
                    freqs //= stride
            if index == 0 and freq_emb:
                self.freq_emb = ScaledEmbedding(
                    freqs, chin_z, smooth=emb_smooth, scale=emb_scale)
                self.freq_emb_scale = freq_emb

        # Bottom channel projections
        transformer_channels = int(channels * growth ** (depth - 1))
        if bottom_channels:
            self.channel_upsampler = nn.Conv1d(transformer_channels, bottom_channels, 1)
            self.channel_downsampler = nn.Conv1d(bottom_channels, transformer_channels, 1)
            self.channel_upsampler_t = nn.Conv1d(transformer_channels, bottom_channels, 1)
            self.channel_downsampler_t = nn.Conv1d(bottom_channels, transformer_channels, 1)
            transformer_channels = bottom_channels

        # CrossTransformer
        self.crosstransformer = None
        if t_layers > 0:
            self.crosstransformer = CrossTransformerEncoder(
                dim=transformer_channels,
                emb=t_emb,
                hidden_scale=t_hidden_scale,
                num_heads=t_heads,
                num_layers=t_layers,
                cross_first=t_cross_first,
                dropout=t_dropout,
                max_positions=t_max_positions,
                norm_in=t_norm_in,
                norm_in_group=t_norm_in_group,
                group_norm=t_group_norm,
                norm_first=t_norm_first,
                norm_out=t_norm_out,
                max_period=t_max_period,
                weight_decay=t_weight_decay,
                lr=t_lr,
                layer_scale=t_layer_scale,
                gelu=t_gelu,
                sin_random_shift=t_sin_random_shift,
                weight_pos_embed=t_weight_pos_embed,
                cape_mean_normalize=t_cape_mean_normalize,
                cape_augment=t_cape_augment,
                cape_glob_loc_scale=t_cape_glob_loc_scale,
                sparse_self_attn=t_sparse_self_attn,
                sparse_cross_attn=t_sparse_cross_attn,
                mask_type=t_mask_type,
                mask_random_seed=t_mask_random_seed,
                sparse_attn_window=t_sparse_attn_window,
                global_window=t_global_window,
                auto_sparsity=t_auto_sparsity,
                sparsity=t_sparsity,
            )

    # ── STFT / iSTFT (via PyTorch bridge) ────────────────────────────────────

    def _spec(self, x: mx.array):
        """Compute STFT via PyTorch.

        Args:
            x: ``[B, C, T]`` waveform (mx.array).

        Returns:
            Complex numpy array ``[B, C, Fr, T']``.
        """
        hl = self.hop_length
        nfft = self.nfft
        assert hl == nfft // 4

        x_np = np.array(x)
        le = int(math.ceil(x_np.shape[-1] / hl))
        pad_amount = hl // 2 * 3

        # Pad in numpy (reflect)
        *other, length = x_np.shape
        x_flat = x_np.reshape(-1, length)
        padded = []
        for row in x_flat:
            t = torch.from_numpy(row)
            t = torch.nn.functional.pad(
                t.unsqueeze(0),
                (pad_amount, pad_amount + le * hl - length),
                mode='reflect',
            ).squeeze(0)
            padded.append(t.numpy())
        x_np = np.stack(padded).reshape(*other, -1)

        z = spectro(x_np, nfft, hl)  # complex numpy [*other, freqs, frames]
        # Remove last freq bin and crop time
        z = z[..., :-1, :]
        assert z.shape[-1] == le + 4, (z.shape, le)
        z = z[..., 2:2 + le]
        return z  # complex numpy array

    def _ispec(self, z_np, length: int):
        """Inverse STFT via PyTorch.

        Args:
            z_np: Complex numpy array ``[B, S, C, Fr, T']``.
            length: Target output waveform length.

        Returns:
            mx.array ``[B, S, C, T]``.
        """
        hl = self.hop_length
        # Pad freq (add back last freq bin) and time (add 2 on each side)
        *batch_dims, freqs, frames = z_np.shape

        # Pad freq: add one zero freq bin
        z_np = np.concatenate([z_np, np.zeros((*batch_dims, 1, frames),
                                              dtype=z_np.dtype)], axis=-2)
        # Pad time: add 2 frames on each side
        z_np = np.concatenate([
            np.zeros((*batch_dims, freqs + 1, 2), dtype=z_np.dtype),
            z_np,
            np.zeros((*batch_dims, freqs + 1, 2), dtype=z_np.dtype),
        ], axis=-1)

        pad_amount = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad_amount

        # Flatten batch dims for ispectro
        flat_shape = z_np.shape
        flat = z_np.reshape(-1, flat_shape[-2], flat_shape[-1])
        x_flat = ispectro(flat, hl, length=le)
        x_np = x_flat.reshape(*batch_dims, -1)

        # Trim padding
        x_np = x_np[..., pad_amount:pad_amount + length]
        return mx.array(x_np)

    def _magnitude(self, z_np) -> mx.array:
        """Convert complex spectrogram to real channels.

        For CaC mode: real and imaginary parts become separate channels.

        Args:
            z_np: Complex numpy array ``[B, C, Fr, T]``.

        Returns:
            mx.array ``[B, C*2, Fr, T]`` for CaC.
        """
        if self.cac:
            # view_as_real: complex → [..., 2]
            z_real = np.stack([z_np.real, z_np.imag], axis=-1)
            # Shape: [B, C, Fr, T, 2] → permute to [B, C, 2, Fr, T]
            z_real = np.transpose(z_real, (*range(z_real.ndim - 3), -1, -3, -2))
            # Reshape: [B, C*2, Fr, T]
            shape = list(z_real.shape)
            new_shape = shape[:-4] + [shape[-4] * shape[-3]] + shape[-2:]
            z_real = z_real.reshape(new_shape)
            return mx.array(z_real.astype(np.float32))
        else:
            return mx.array(np.abs(z_np).astype(np.float32))

    def _mask(self, z_np, m: mx.array):
        """Apply CaC mask: convert predicted channels back to complex spectrogram.

        Args:
            z_np: Original complex numpy ``[B, C, Fr, T]`` (unused for CaC).
            m: Predicted mask/spectrogram mx.array ``[B, S, C_cac, Fr, T]``.

        Returns:
            Complex numpy array ``[B, S, C, Fr, T]``.
        """
        if self.cac:
            m_np = np.array(m)
            B, S, C_cac, Fr, T = m_np.shape
            C = C_cac // 2
            # [B, S, C_cac, Fr, T] → [B, S, C, 2, Fr, T] → [B, S, C, Fr, T, 2]
            out = m_np.reshape(B, S, C, 2, Fr, T)
            out = np.transpose(out, (0, 1, 2, 4, 5, 3))
            # Combine to complex
            return out[..., 0] + 1j * out[..., 1]
        else:
            raise NotImplementedError("Only CaC mode supported for MLX port")

    def valid_length(self, length: int) -> int:
        """Return valid input length for the model."""
        if not self.use_train_segment:
            return length
        training_length = int(self.segment * self.samplerate)
        if training_length < length:
            raise ValueError(
                f"Input length {length} exceeds training length {training_length}")
        return training_length

    def __call__(self, mix: mx.array) -> mx.array:
        """Separate sources from a mixture.

        Args:
            mix: ``[B, audio_channels, T]`` input mixture.

        Returns:
            ``[B, S, audio_channels, T]`` separated sources.
        """
        length = mix.shape[-1]
        length_pre_pad = None

        if self.use_train_segment:
            training_length = int(self.segment * self.samplerate)
            if mix.shape[-1] < training_length:
                length_pre_pad = mix.shape[-1]
                pad_widths = [(0, 0)] * (mix.ndim - 1) + [
                    (0, training_length - length_pre_pad)]
                mix = mx.pad(mix, pad_widths)

        # STFT (via PyTorch bridge) → complex numpy
        z = self._spec(mix)
        mag = self._magnitude(z)
        x = mag

        B, C, Fq, T = x.shape

        # Normalize freq branch
        mean = mx.mean(x, axis=(1, 2, 3), keepdims=True)
        std = mx.sqrt(mx.mean((x - mean) ** 2, axis=(1, 2, 3), keepdims=True))
        x = (x - mean) / (1e-5 + std)

        # Normalize time branch
        xt = mix
        meant = mx.mean(xt, axis=(1, 2), keepdims=True)
        stdt = mx.sqrt(mx.mean((xt - meant) ** 2, axis=(1, 2), keepdims=True))
        xt = (xt - meant) / (1e-5 + stdt)

        # Encoder
        saved = []      # freq skip connections
        saved_t = []    # time skip connections
        lengths = []    # freq branch lengths
        lengths_t = []  # time branch lengths

        for idx in range(len(self.encoder)):
            encode = self.encoder[idx]
            lengths.append(x.shape[-1])
            inject = None

            if idx < len(self.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt

            x = encode(x, inject)

            if idx == 0 and self.freq_emb is not None:
                frs = mx.arange(x.shape[-2])
                emb = self.freq_emb(frs)  # [Fr, C]
                emb = emb.T[None, :, :, None]  # [1, C, Fr, 1]
                emb = mx.broadcast_to(emb, x.shape)
                x = x + self.freq_emb_scale * emb

            saved.append(x)

        # CrossTransformer bottleneck
        if self.crosstransformer is not None:
            if self.bottom_channels:
                b, c, f, t = x.shape
                x = x.reshape(b, c, f * t)
                x = apply_conv1d(self.channel_upsampler, x)
                x = x.reshape(b, -1, f, t)
                xt = apply_conv1d(self.channel_upsampler_t, xt)

            x, xt = self.crosstransformer(x, xt)

            if self.bottom_channels:
                x = x.reshape(b, -1, f * t)
                x = apply_conv1d(self.channel_downsampler, x)
                x = x.reshape(b, -1, f, t)
                xt = apply_conv1d(self.channel_downsampler_t, xt)

        # Decoder
        for idx in range(len(self.decoder)):
            decode = self.decoder[idx]
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))

            offset = self.depth - len(self.tdecoder)
            if idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    assert pre.shape[2] == 1, pre.shape
                    pre = pre[:, :, 0, :]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip_t = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip_t, length_t)

        assert len(saved) == 0
        assert len(lengths_t) == 0
        assert len(saved_t) == 0

        # Reconstruct freq output
        S = len(self.sources)
        x = x.reshape(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        # Inverse STFT (via PyTorch bridge)
        x_np = np.array(x)
        z_out = self._mask(z, x)

        if self.use_train_segment:
            x_audio = self._ispec(z_out, training_length)
        else:
            x_audio = self._ispec(z_out, length)

        # Reconstruct time output
        if self.use_train_segment:
            xt = xt.reshape(B, S, -1, training_length)
        else:
            xt = xt.reshape(B, S, -1, length)
        xt = xt * stdt[:, None] + meant[:, None]

        # Combine
        result = xt + x_audio

        if length_pre_pad is not None:
            result = result[..., :length_pre_pad]
        return result

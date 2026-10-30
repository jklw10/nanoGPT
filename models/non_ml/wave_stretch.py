import argparse
import cmath
import json
import math
import os
import subprocess
import sys
import threading
import time
import wave
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


# =========================================================================
# 1. Audio I/O Helpers
# =========================================================================
def load_audio(filepath: str, target_sr: int = 44100):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    temp_wav = f"temp_decode_{os.getpid()}.wav"
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            filepath,
            "-ar",
            str(target_sr),
            "-ac",
            "1",
            "-f",
            "wav",
            temp_wav,
        ]
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        with wave.open(temp_wav, "rb") as wf:
            sr = wf.getframerate()
            num_frames = wf.getnframes()
            pcm_bytes = wf.readframes(num_frames)
            audio_np = (
                np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
                / 32768.0
            )

        if os.path.exists(temp_wav):
            os.remove(temp_wav)

        return torch.from_numpy(audio_np), sr

    except Exception as e:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        raise RuntimeError(f"Failed to decode audio file via FFmpeg: {e}")


def save_audio(filepath: str, audio_tensor: torch.Tensor, sr: int = 44100):
    audio_np = audio_tensor.detach().cpu().numpy()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_pcm = (audio_np * 32767.0).astype(np.int16)

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".wav":
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio_pcm.tobytes())
    else:
        # Export as MP3 or other audio formats via FFmpeg
        temp_wav = f"temp_export_{os.getpid()}.wav"
        with wave.open(temp_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio_pcm.tobytes())

        cmd = ["ffmpeg", "-y", "-i", temp_wav, "-b:a", "320k", filepath]
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        if os.path.exists(temp_wav):
            os.remove(temp_wav)


def generate_synthetic_test_signal(
    sr: int = 44100, duration: float = 2.0
) -> torch.Tensor:
    t = torch.linspace(0, duration, int(sr * duration))
    tonal = (
        0.4 * torch.sin(2 * math.pi * 220 * t)
        + 0.3 * torch.sin(2 * math.pi * 880 * t)
        + 0.2 * torch.sin(2 * math.pi * 1760 * t)
    )
    transients = torch.zeros_like(t)
    for burst_t in [0.25, 0.75, 1.25, 1.75]:
        idx = int(burst_t * sr)
        transients[idx : idx + 200] += torch.exp(
            -torch.linspace(0, 10, 200)
        ) * torch.sin(2 * math.pi * 150 * torch.linspace(0, 0.005, 200))

    sub_bass = 0.5 * torch.sin(2 * math.pi * 60 * t)
    signal = tonal + transients + sub_bass
    return signal / torch.max(torch.abs(signal))


# =========================================================================
# 2. Multi-Band Linear-Phase FIR Crossover Filterbank
# =========================================================================
class FIR3BandCrossover(nn.Module):

    def __init__(
        self, sr: int = 44100, num_taps: int = 257, device: str = "cpu"
    ):
        super().__init__()
        self.sr = sr
        self.device = device

        if num_taps % 2 == 0:
            num_taps += 1

        half_taps = (num_taps - 1) // 2
        n = torch.arange(
            -half_taps, half_taps + 1, device=device, dtype=torch.float32
        )
        win = torch.hann_window(num_taps, device=device)

        def ideal_lp(fc):
            wc = 2 * math.pi * fc / sr
            h = torch.zeros_like(n)
            h[n == 0] = wc / math.pi
            mask = n != 0
            h[mask] = torch.sin(wc * n[mask]) / (math.pi * n[mask])
            return h * win

        h_low = ideal_lp(400.0)
        h_low = h_low / h_low.sum()

        h_mid_lp = ideal_lp(4000.0)
        h_mid_lp = h_mid_lp / h_mid_lp.sum()

        h_mid = h_mid_lp - h_low

        h_delta = torch.zeros_like(n)
        h_delta[half_taps] = 1.0
        h_high = h_delta - h_mid_lp

        self.register_buffer("kernel_low", h_low.view(1, 1, -1))
        self.register_buffer("kernel_mid", h_mid.view(1, 1, -1))
        self.register_buffer("kernel_high", h_high.view(1, 1, -1))
        self.padding = half_taps

    def forward(self, x: torch.Tensor):
        x_pad = F.pad(x, (self.padding, self.padding), mode="reflect")
        low = F.conv1d(x_pad, self.kernel_low)
        mid = F.conv1d(x_pad, self.kernel_mid)
        high = F.conv1d(x_pad, self.kernel_high)
        return low.squeeze(), mid.squeeze(), high.squeeze()


# =========================================================================
# 3. Soft Kronecker ZCA Matrix Whitening Operator
# =========================================================================
def compute_soft_kronecker_matrix(
    Cov: torch.Tensor, epsilon: float = 1e-5, alpha: float = 0.5
) -> torch.Tensor:
    d = Cov.shape[0]
    device, dtype = Cov.device, Cov.dtype
    eye_d = torch.eye(d, device=device, dtype=dtype)

    Cov_reg = Cov + epsilon * eye_d

    v = torch.ones((d, 1), device=device, dtype=dtype)
    for _ in range(2):
        u = torch.matmul(Cov_reg, v)
        u = u / (u.norm() + 1e-12)
        v = torch.matmul(Cov_reg.T, u)
        v = v / (v.norm() + 1e-12)

    s = torch.matmul(u.T, torch.matmul(Cov_reg, v)).squeeze().clamp(min=1e-8)
    B = Cov_reg / s
    Y = eye_d.clone()

    for _ in range(2):
        Y = 0.5 * torch.matmul(
            Y, 3.0 * eye_d - torch.matmul(Y, torch.matmul(B, Y))
        )

    W_hard = Y / torch.sqrt(s)
    W_soft = alpha * W_hard + (1.0 - alpha) * eye_d
    return W_soft


# =========================================================================
# 4. Ultra-Fast Fused 1D Krein Sandwich Kernel
# =========================================================================
@torch.jit.script
def krein_sandwich_xcorr(
    templates: torch.Tensor,
    search_regions: torch.Tensor,
    lmbda: float,
    gamma: float,
) -> torch.Tensor:
    rms = torch.rsqrt(torch.mean(templates**2, dim=-1, keepdim=True) + 1e-8)
    tmpl_grms = templates * rms

    scale_ref = torch.std(templates, dim=-1, keepdim=True) + 1e-6
    R_t = 6.0 * scale_ref + 1e-6

    x_norm = tmpl_grms / R_t
    K_val = torch.clamp(x_norm, -0.999, 0.999)
    boundary_decelerator = 1.0 - K_val**2
    V_grad = -gamma * (K_val * boundary_decelerator)

    tmpl_energy = torch.mean(tmpl_grms**2, dim=-1, keepdim=True) + 1e-8
    J_weight = tmpl_energy / (tmpl_energy + lmbda)
    A_half = torch.sqrt(J_weight)

    u = tmpl_grms / (torch.norm(tmpl_grms, dim=-1, keepdim=True) + 1e-8)
    u_vgrad = torch.sum(u * V_grad, dim=-1, keepdim=True)

    inv_scale = 1.0 / lmbda
    alpha_sm = 1.0 / (boundary_decelerator**2 + 1e-8)
    sm_coeff = alpha_sm / (lmbda * (lmbda + alpha_sm))

    H_Vgrad = inv_scale * V_grad - (sm_coeff * u_vgrad) * u
    S_grad = A_half * H_Vgrad

    S_grad = S_grad.contiguous()
    search_regions = search_regions.contiguous()

    L = S_grad.shape[-1]
    sr_unfolded = search_regions.unfold(
        dimension=-1, size=L, step=1
    ).contiguous()

    xcorr = torch.matmul(S_grad, sr_unfolded.transpose(-1, -2)).contiguous()
    return xcorr


# =========================================================================
# 5. Smooth RMS Energy Envelope Matching & FrFT Helpers
# =========================================================================
def smooth_rms_envelope(
    x: torch.Tensor, win_len: int = 4096, device: str = "cpu"
) -> torch.Tensor:
    power = x**2
    sigma = win_len / 6.0
    t = torch.arange(win_len, device=device, dtype=torch.float32) - (
        win_len - 1
    ) / 2.0
    gauss = torch.exp(-0.5 * (t / sigma) ** 2)
    gauss = (gauss / gauss.sum()).view(1, 1, -1)

    pad_l = win_len // 2
    pad_r = win_len - 1 - pad_l
    power_pad = F.pad(power.view(1, 1, -1), (pad_l, pad_r), mode="reflect")
    smooth_power = F.conv1d(power_pad, gauss).squeeze()
    return torch.sqrt(smooth_power + 1e-8)[: len(x)]


def linear_energy_match(
    orig_x: torch.Tensor,
    stretched_x: torch.Tensor,
    alpha: float,
    win_len: int = 4096,
    device: str = "cpu",
) -> torch.Tensor:
    env_orig = smooth_rms_envelope(orig_x, win_len=win_len, device=device)
    env_target = F.interpolate(
        env_orig.view(1, 1, -1),
        size=len(stretched_x),
        mode="linear",
        align_corners=False,
    ).squeeze()
    env_synth = smooth_rms_envelope(
        stretched_x, win_len=win_len, device=device
    )

    gain_raw = env_target / (env_synth + 1e-8)

    sigma = win_len / 4.0
    t = torch.arange(win_len, device=device, dtype=torch.float32) - (
        win_len - 1
    ) / 2.0
    gauss = torch.exp(-0.5 * (t / sigma) ** 2)
    gauss = (gauss / gauss.sum()).view(1, 1, -1)

    pad_l = win_len // 2
    pad_r = win_len - 1 - pad_l
    gain_pad = F.pad(gain_raw.view(1, 1, -1), (pad_l, pad_r), mode="replicate")
    smooth_gain = F.conv1d(gain_pad, gauss).squeeze()[: len(stretched_x)]

    return stretched_x * smooth_gain


def build_unitary_frft_matrix(
    win_size: int, a: float, device: str = "cuda"
) -> torch.Tensor:
    alpha = a * math.pi / 2.0
    sin_a = math.sin(alpha)
    cos_a = math.cos(alpha)
    cot_a = cos_a / sin_a
    csc_a = 1.0 / sin_a

    n = (
        torch.arange(win_size, device=device, dtype=torch.float32)
        - (win_size - 1) / 2.0
    )
    u = n / math.sqrt(win_size)

    u_m = u.unsqueeze(1)
    u_n = u.unsqueeze(0)

    exponent = (u_m**2 + u_n**2) * cot_a - 2.0 * u_m * u_n * csc_a
    c_val = math.sqrt(1.0 / win_size) * cmath.sqrt(
        complex(1.0, -cot_a) / math.sqrt(1.0 + cot_a**2)
    )

    K = torch.tensor(
        c_val, dtype=torch.complex64, device=device
    ) * torch.exp(1j * math.pi * exponent)
    U, _, Vh = torch.linalg.svd(K)
    return torch.matmul(U, Vh)


# =========================================================================
# 6. Advanced ZCA-Krein Stretcher (v4)
# =========================================================================
class ZCAKreinStretcher(nn.Module):

    def __init__(
        self,
        sr=44100,
        win_size=2048,
        lmbda=0.05,
        noise_threshold=0.35,
        device="cuda",
    ):
        super().__init__()
        self.sr = sr
        self.win_size = win_size
        self.hop_s = win_size // 4
        self.search_range = win_size // 4
        self.lmbda = lmbda
        self.noise_threshold = noise_threshold
        self.device = device

        self.crossover = FIR3BandCrossover(sr=sr, num_taps=257, device=device)
        self.window = torch.hann_window(win_size, device=device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        x = x.to(self.device)
        num_samples = len(x)

        x_in = x.unsqueeze(0).unsqueeze(0)
        low, mid, high = self.crossover(x_in)

        hop_a = self.hop_s / alpha
        num_frames = int(math.ceil(num_samples / hop_a))

        overlap_len = self.win_size - self.hop_s
        search_len = self.search_range * 2 + overlap_len
        pad_size = self.win_size + self.search_range + search_len

        low_pad = F.pad(
            low.view(1, 1, -1), (pad_size, pad_size), mode="reflect"
        ).squeeze()
        mid_pad = F.pad(
            mid.view(1, 1, -1), (pad_size, pad_size), mode="reflect"
        ).squeeze()
        high_pad = F.pad(
            high.view(1, 1, -1), (pad_size, pad_size), mode="reflect"
        ).squeeze()

        bands = torch.stack([low_pad, mid_pad, high_pad], dim=0)
        Cov = torch.matmul(bands, bands.T) / (bands.shape[1] + 1e-8)
        W_soft = compute_soft_kronecker_matrix(Cov, epsilon=1e-5, alpha=0.5)
        bands_decorr = torch.matmul(W_soft, bands)
        joint_pad = torch.mean(bands_decorr, dim=0)

        num_states = self.search_range * 2 + 1
        state_offsets = torch.arange(
            -self.search_range, self.search_range + 1, device=self.device
        )
        win_offsets = torch.arange(self.win_size, device=self.device)
        win_sq = self.window**2

        m_indices = torch.arange(num_frames, device=self.device)
        p_m = torch.round(m_indices * hop_a).long()
        target_starts = pad_size + p_m

        transition_map = torch.zeros(
            (num_frames, num_states), dtype=torch.long, device=self.device
        )
        center_state = self.search_range

        tmpl_region_len = num_states + overlap_len - 1
        for c in range(num_frames):
            c_end = min(c + 1, num_frames)
            c_target_starts = target_starts[c:c_end]

            if c == 0:
                prev_targets = torch.cat(
                    [c_target_starts[:1], c_target_starts[:-1]]
                )
            else:
                prev_targets = target_starts[c - 1 : c_end - 1]

            tmpl_base_starts = prev_targets - self.search_range + self.hop_s

            raw_tmpl_spans = torch.stack(
                [joint_pad[s : s + tmpl_region_len] for s in tmpl_base_starts]
            )
            templates = raw_tmpl_spans.unfold(
                dimension=-1, size=overlap_len, step=1
            ).contiguous()

            sr_starts = c_target_starts - self.search_range
            search_regions = torch.stack(
                [joint_pad[s : s + search_len] for s in sr_starts]
            ).contiguous()

            xcorr = krein_sandwich_xcorr(
                templates, search_regions, self.lmbda, 1.0
            )

            max_vals, max_indices = torch.max(xcorr, dim=1)
            best_states = torch.clamp(max_indices, 0, num_states - 1)

            seg_norms = torch.norm(templates, dim=2) + 1e-8
            R_max = max_vals / seg_norms

            c_map = torch.where(
                R_max < self.noise_threshold,
                torch.full_like(best_states, center_state),
                best_states,
            )
            transition_map[c:c_end] = c_map

        winning_states = torch.zeros(
            num_frames, dtype=torch.long, device=self.device
        )
        s = center_state
        for m in range(num_frames):
            s = transition_map[m, s]
            winning_states[m] = s

        winning_offsets = state_offsets[winning_states]
        actual_starts = target_starts + winning_offsets
        out_len = num_frames * self.hop_s + self.win_size

        out_low = torch.zeros(out_len, device=self.device)
        out_mid = torch.zeros(out_len, device=self.device)
        out_high = torch.zeros(out_len, device=self.device)
        weights = torch.zeros(out_len, device=self.device)

        high_smooth_win = self.window.clone()
        high_smooth_win[: self.win_size // 8] *= torch.sin(
            torch.linspace(
                0, math.pi / 2, self.win_size // 8, device=self.device
            )
        )
        high_smooth_win[-self.win_size // 8 :] *= torch.cos(
            torch.linspace(
                0, math.pi / 2, self.win_size // 8, device=self.device
            )
        )

        recon_chunk = 128
        for c in range(0, num_frames, recon_chunk):
            c_end = min(c + recon_chunk, num_frames)
            c_actual_starts = actual_starts[c:c_end]
            c_m_indices = m_indices[c:c_end]

            c_seg_idx = c_actual_starts.unsqueeze(1) + win_offsets

            c_low = (low_pad[c_seg_idx] * self.window).view(-1)
            c_mid = (mid_pad[c_seg_idx] * self.window).view(-1)
            c_high = (high_pad[c_seg_idx] * high_smooth_win).view(-1)

            c_out_starts = c_m_indices * self.hop_s
            c_out_idx = (c_out_starts.unsqueeze(1) + win_offsets).view(-1)

            c_win_sq = win_sq.unsqueeze(0).expand(c_end - c, -1).reshape(-1)

            out_low.scatter_add_(0, c_out_idx, c_low)
            out_mid.scatter_add_(0, c_out_idx, c_mid)
            out_high.scatter_add_(0, c_out_idx, c_high)
            weights.scatter_add_(0, c_out_idx, c_win_sq)

        valid = weights > 1e-6
        out_sum = torch.zeros(out_len, device=self.device)
        out_sum[valid] = (
            out_low[valid] + out_mid[valid] + out_high[valid]
        ) / weights[valid]

        raw_stretched = torch.clamp(out_sum, -1.0, 1.0)
        expected_out_len = int(num_samples * alpha)
        raw_stretched = raw_stretched[:expected_out_len]

        return linear_energy_match(
            x, raw_stretched, alpha=alpha, win_len=4096, device=self.device
        )


# =========================================================================
# 7. Hybrid ZCA-Krein-FrFT Stretcher
# =========================================================================
class ZCAKreinFrFTStretcher(nn.Module):

    def __init__(
        self,
        sr=44100,
        win_size=2048,
        lmbda=0.05,
        noise_threshold=0.35,
        device="cuda",
    ):
        super().__init__()
        self.sr = sr
        self.win_size = win_size
        self.hop_s = win_size // 4
        self.search_range = win_size // 4
        self.lmbda = lmbda
        self.noise_threshold = noise_threshold
        self.device = device

        self.crossover = FIR3BandCrossover(sr=sr, num_taps=257, device=device)
        self.window = torch.hann_window(win_size, device=device)

        n = (
            torch.arange(win_size, device=device, dtype=torch.float32)
            - win_size / 2.0
        )
        self.chirp_in = torch.exp(1j * (math.pi / win_size) * (n**2))
        self.chirp_in_inv = torch.exp(-1j * (math.pi / win_size) * (n**2))

        k = (
            torch.arange(win_size, device=device, dtype=torch.float32)
            - win_size / 2.0
        )
        self.chirp_out = torch.exp(1j * (math.pi / win_size) * (k**2))
        self.chirp_out_inv = torch.exp(-1j * (math.pi / win_size) * (k**2))

    def frft_05(self, x_frame: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft(x_frame * self.chirp_in) * self.chirp_out

    def ifrft_05(self, X_frft: torch.Tensor) -> torch.Tensor:
        X_fft = X_frft * self.chirp_out_inv
        x_chirp = torch.fft.ifft(X_fft)
        return torch.real(x_chirp * self.chirp_in_inv)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        x = x.to(self.device)
        num_samples = len(x)

        x_in = x.unsqueeze(0).unsqueeze(0)
        low, mid, high = self.crossover(x_in)

        hop_a = self.hop_s / alpha
        num_frames = int(math.ceil(num_samples / hop_a))

        overlap_len = self.win_size - self.hop_s
        search_len = self.search_range * 2 + overlap_len
        pad_size = self.win_size + self.search_range + search_len

        low_pad = F.pad(
            low.view(1, 1, -1), (pad_size, pad_size), mode="reflect"
        ).squeeze()
        mid_pad = F.pad(
            mid.view(1, 1, -1), (pad_size, pad_size), mode="reflect"
        ).squeeze()
        high_pad = F.pad(
            high.view(1, 1, -1), (pad_size, pad_size), mode="reflect"
        ).squeeze()

        bands = torch.stack([low_pad, mid_pad, high_pad], dim=0)
        Cov = torch.matmul(bands, bands.T) / (bands.shape[1] + 1e-8)
        W_soft = compute_soft_kronecker_matrix(Cov, epsilon=1e-5, alpha=0.5)
        bands_decorr = torch.matmul(W_soft, bands)
        joint_pad = torch.mean(bands_decorr, dim=0)

        num_states = self.search_range * 2 + 1
        state_offsets = torch.arange(
            -self.search_range, self.search_range + 1, device=self.device
        )
        win_offsets = torch.arange(self.win_size, device=self.device)
        win_sq = self.window**2

        m_indices = torch.arange(num_frames, device=self.device)
        p_m = torch.round(m_indices * hop_a).long()
        target_starts = pad_size + p_m

        transition_map = torch.zeros(
            (num_frames, num_states), dtype=torch.long, device=self.device
        )
        center_state = self.search_range

        tmpl_region_len = num_states + overlap_len - 1
        for c in range(num_frames):
            c_end = min(c + 1, num_frames)
            c_target_starts = target_starts[c:c_end]

            if c == 0:
                prev_targets = torch.cat(
                    [c_target_starts[:1], c_target_starts[:-1]]
                )
            else:
                prev_targets = target_starts[c - 1 : c_end - 1]

            tmpl_base_starts = prev_targets - self.search_range + self.hop_s

            raw_tmpl_spans = torch.stack(
                [joint_pad[s : s + tmpl_region_len] for s in tmpl_base_starts]
            )
            templates = raw_tmpl_spans.unfold(
                dimension=-1, size=overlap_len, step=1
            ).contiguous()

            sr_starts = c_target_starts - self.search_range
            search_regions = torch.stack(
                [joint_pad[s : s + search_len] for s in sr_starts]
            ).contiguous()

            xcorr = krein_sandwich_xcorr(
                templates, search_regions, self.lmbda, 1.0
            )

            max_vals, max_indices = torch.max(xcorr, dim=1)
            best_states = torch.clamp(max_indices, 0, num_states - 1)

            seg_norms = torch.norm(templates, dim=2) + 1e-8
            R_max = max_vals / seg_norms

            c_map = torch.where(
                R_max < self.noise_threshold,
                torch.full_like(best_states, center_state),
                best_states,
            )
            transition_map[c:c_end] = c_map

        winning_states = torch.zeros(
            num_frames, dtype=torch.long, device=self.device
        )
        s = center_state
        for m in range(num_frames):
            s = transition_map[m, s]
            winning_states[m] = s

        winning_offsets = state_offsets[winning_states]
        actual_starts = target_starts + winning_offsets
        out_len = num_frames * self.hop_s + self.win_size

        out_low = torch.zeros(out_len, device=self.device)
        out_mid = torch.zeros(out_len, device=self.device)
        out_high = torch.zeros(out_len, device=self.device)
        weights = torch.zeros(out_len, device=self.device)

        high_smooth_win = self.window.clone()
        high_smooth_win[: self.win_size // 8] *= torch.sin(
            torch.linspace(
                0, math.pi / 2, self.win_size // 8, device=self.device
            )
        )
        high_smooth_win[-self.win_size // 8 :] *= torch.cos(
            torch.linspace(
                0, math.pi / 2, self.win_size // 8, device=self.device
            )
        )

        recon_chunk = 128
        for c in range(0, num_frames, recon_chunk):
            c_end = min(c + recon_chunk, num_frames)
            c_actual_starts = actual_starts[c:c_end]
            c_m_indices = m_indices[c:c_end]

            c_seg_idx = c_actual_starts.unsqueeze(1) + win_offsets

            seg_low = low_pad[c_seg_idx] * self.window
            seg_mid = mid_pad[c_seg_idx] * self.window
            seg_high = high_pad[c_seg_idx] * high_smooth_win

            seg_low_frft = torch.stack(
                [
                    self.ifrft_05(self.frft_05(seg_low[i]))
                    for i in range(c_end - c)
                ]
            )
            seg_mid_frft = torch.stack(
                [
                    self.ifrft_05(self.frft_05(seg_mid[i]))
                    for i in range(c_end - c)
                ]
            )
            seg_high_frft = torch.stack(
                [
                    self.ifrft_05(self.frft_05(seg_high[i]))
                    for i in range(c_end - c)
                ]
            )

            c_low = seg_low_frft.view(-1)
            c_mid = seg_mid_frft.view(-1)
            c_high = seg_high_frft.view(-1)

            c_out_starts = c_m_indices * self.hop_s
            c_out_idx = (c_out_starts.unsqueeze(1) + win_offsets).view(-1)

            c_win_sq = win_sq.unsqueeze(0).expand(c_end - c, -1).reshape(-1)

            out_low.scatter_add_(0, c_out_idx, c_low)
            out_mid.scatter_add_(0, c_out_idx, c_mid)
            out_high.scatter_add_(0, c_out_idx, c_high)
            weights.scatter_add_(0, c_out_idx, c_win_sq)

        valid = weights > 1e-6
        out_sum = torch.zeros(out_len, device=self.device)
        out_sum[valid] = (
            out_low[valid] + out_mid[valid] + out_high[valid]
        ) / weights[valid]

        raw_stretched = torch.clamp(out_sum, -1.0, 1.0)
        expected_out_len = int(num_samples * alpha)
        raw_stretched = raw_stretched[:expected_out_len]

        return linear_energy_match(
            x, raw_stretched, alpha=alpha, win_len=4096, device=self.device
        )


# =========================================================================
# 8. Operator-Softmax Krein FrFT Stretcher
# =========================================================================
class OperatorSoftmaxKreinFrFTStretcher(nn.Module):

    def __init__(
        self,
        sr=44100,
        win_size=2048,
        lmbda=0.05,
        noise_threshold=0.35,
        temperature=10.0,
        device="cuda",
    ):
        super().__init__()
        self.sr = sr
        self.win_size = win_size
        self.hop_s = win_size // 4
        self.search_range = win_size // 4
        self.lmbda = lmbda
        self.noise_threshold = noise_threshold
        self.temperature = temperature
        self.device = device

        self.crossover = FIR3BandCrossover(sr=sr, num_taps=257, device=device)
        self.window = torch.hann_window(win_size, device=device)

        self.candidate_orders = [0.1, 0.3, 0.5, 0.7, 0.9]
        K_list = []
        K_inv_list = []
        for a in self.candidate_orders:
            K = build_unitary_frft_matrix(win_size, a, device=device)
            K_list.append(K)
            K_inv_list.append(K.conj().mT)

        self.K_grid = torch.stack(K_list, dim=0)
        self.K_inv_grid = torch.stack(K_inv_list, dim=0)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        x = x.to(self.device)
        num_samples = len(x)

        x_in = x.unsqueeze(0).unsqueeze(0)
        low, mid, high = self.crossover(x_in)

        hop_a = self.hop_s / alpha
        num_frames = int(math.ceil(num_samples / hop_a))

        overlap_len = self.win_size - self.hop_s
        search_len = self.search_range * 2 + overlap_len
        pad_size = self.win_size + self.search_range + search_len

        low_pad = F.pad(
            low.view(1, 1, -1), (pad_size, pad_size), mode="reflect"
        ).squeeze()
        mid_pad = F.pad(
            mid.view(1, 1, -1), (pad_size, pad_size), mode="reflect"
        ).squeeze()
        high_pad = F.pad(
            high.view(1, 1, -1), (pad_size, pad_size), mode="reflect"
        ).squeeze()

        bands = torch.stack([low_pad, mid_pad, high_pad], dim=0)
        Cov = torch.matmul(bands, bands.T) / (bands.shape[1] + 1e-8)
        W_soft = compute_soft_kronecker_matrix(Cov, epsilon=1e-5, alpha=0.5)
        bands_decorr = torch.matmul(W_soft, bands)
        joint_pad = torch.mean(bands_decorr, dim=0)

        num_states = self.search_range * 2 + 1
        state_offsets = torch.arange(
            -self.search_range, self.search_range + 1, device=self.device
        )
        win_offsets = torch.arange(self.win_size, device=self.device)
        win_sq = self.window**2

        m_indices = torch.arange(num_frames, device=self.device)
        p_m = torch.round(m_indices * hop_a).long()
        target_starts = pad_size + p_m

        transition_map = torch.zeros(
            (num_frames, num_states), dtype=torch.long, device=self.device
        )
        center_state = self.search_range

        tmpl_region_len = num_states + overlap_len - 1
        for c in range(num_frames):
            c_end = min(c + 1, num_frames)
            c_target_starts = target_starts[c:c_end]

            if c == 0:
                prev_targets = torch.cat(
                    [c_target_starts[:1], c_target_starts[:-1]]
                )
            else:
                prev_targets = target_starts[c - 1 : c_end - 1]

            tmpl_base_starts = prev_targets - self.search_range + self.hop_s

            raw_tmpl_spans = torch.stack(
                [joint_pad[s : s + tmpl_region_len] for s in tmpl_base_starts]
            )
            templates = raw_tmpl_spans.unfold(
                dimension=-1, size=overlap_len, step=1
            ).contiguous()

            sr_starts = c_target_starts - self.search_range
            search_regions = torch.stack(
                [joint_pad[s : s + search_len] for s in sr_starts]
            ).contiguous()

            xcorr = krein_sandwich_xcorr(
                templates, search_regions, self.lmbda, 1.0
            )

            max_vals, max_indices = torch.max(xcorr, dim=1)
            best_states = torch.clamp(max_indices, 0, num_states - 1)

            seg_norms = torch.norm(templates, dim=2) + 1e-8
            R_max = max_vals / seg_norms

            c_map = torch.where(
                R_max < self.noise_threshold,
                torch.full_like(best_states, center_state),
                best_states,
            )
            transition_map[c:c_end] = c_map

        winning_states = torch.zeros(
            num_frames, dtype=torch.long, device=self.device
        )
        s = center_state
        for m in range(num_frames):
            s = transition_map[m, s]
            winning_states[m] = s

        winning_offsets = state_offsets[winning_states]
        actual_starts = target_starts + winning_offsets
        out_len = num_frames * self.hop_s + self.win_size

        out_low = torch.zeros(out_len, device=self.device)
        out_mid = torch.zeros(out_len, device=self.device)
        out_high = torch.zeros(out_len, device=self.device)
        weights = torch.zeros(out_len, device=self.device)

        high_smooth_win = self.window.clone()
        high_smooth_win[: self.win_size // 8] *= torch.sin(
            torch.linspace(
                0, math.pi / 2, self.win_size // 8, device=self.device
            )
        )
        high_smooth_win[-self.win_size // 8 :] *= torch.cos(
            torch.linspace(
                0, math.pi / 2, self.win_size // 8, device=self.device
            )
        )

        recon_chunk = 128
        for c in range(0, num_frames, recon_chunk):
            c_end = min(c + recon_chunk, num_frames)
            c_actual_starts = actual_starts[c:c_end]
            c_m_indices = m_indices[c:c_end]
            chunk_len = c_end - c

            c_seg_idx = c_actual_starts.unsqueeze(1) + win_offsets

            seg_low = (low_pad[c_seg_idx] * self.window).to(torch.complex64)
            seg_mid = (mid_pad[c_seg_idx] * self.window).to(torch.complex64)
            seg_high = (high_pad[c_seg_idx] * high_smooth_win).to(
                torch.complex64
            )

            X_cand_low = [torch.matmul(seg_low, K.mT) for K in self.K_grid]
            X_cand_mid = [torch.matmul(seg_mid, K.mT) for K in self.K_grid]
            X_cand_high = [torch.matmul(seg_high, K.mT) for K in self.K_grid]

            l4_scores = torch.stack(
                [
                    torch.sum(
                        torch.abs(X_cand_low[k]) ** 4
                        + torch.abs(X_cand_mid[k]) ** 4
                        + torch.abs(X_cand_high[k]) ** 4,
                        dim=-1,
                    )
                    for k in range(len(self.candidate_orders))
                ],
                dim=1,
            )

            l4_norm = (l4_scores - l4_scores.mean(dim=-1, keepdim=True)) / (
                l4_scores.std(dim=-1, keepdim=True) + 1e-6
            )

            softmax_w = F.softmax(self.temperature * l4_norm, dim=-1)
            sqrt_w = torch.sqrt(softmax_w).unsqueeze(-1)

            rec_low_k = torch.stack(
                [
                    torch.matmul(
                        sqrt_w[:, k] * X_cand_low[k], self.K_inv_grid[k].mT
                    )
                    * sqrt_w[:, k]
                    for k in range(len(self.candidate_orders))
                ],
                dim=1,
            )

            rec_mid_k = torch.stack(
                [
                    torch.matmul(
                        sqrt_w[:, k] * X_cand_mid[k], self.K_inv_grid[k].mT
                    )
                    * sqrt_w[:, k]
                    for k in range(len(self.candidate_orders))
                ],
                dim=1,
            )

            rec_high_k = torch.stack(
                [
                    torch.matmul(
                        sqrt_w[:, k] * X_cand_high[k], self.K_inv_grid[k].mT
                    )
                    * sqrt_w[:, k]
                    for k in range(len(self.candidate_orders))
                ],
                dim=1,
            )

            blend_low = torch.sum(rec_low_k, dim=1)
            blend_mid = torch.sum(rec_mid_k, dim=1)
            blend_high = torch.sum(rec_high_k, dim=1)

            c_low = torch.real(blend_low).view(-1)
            c_mid = torch.real(blend_mid).view(-1)
            c_high = torch.real(blend_high).view(-1)

            c_out_starts = c_m_indices * self.hop_s
            c_out_idx = (c_out_starts.unsqueeze(1) + win_offsets).view(-1)

            c_win_sq = win_sq.unsqueeze(0).expand(chunk_len, -1).reshape(-1)

            out_low.scatter_add_(0, c_out_idx, c_low)
            out_mid.scatter_add_(0, c_out_idx, c_mid)
            out_high.scatter_add_(0, c_out_idx, c_high)
            weights.scatter_add_(0, c_out_idx, c_win_sq)

        valid = weights > 1e-6
        out_sum = torch.zeros(out_len, device=self.device)
        out_sum[valid] = (
            out_low[valid] + out_mid[valid] + out_high[valid]
        ) / weights[valid]

        raw_stretched = torch.clamp(out_sum, -1.0, 1.0)
        expected_out_len = int(num_samples * alpha)
        raw_stretched = raw_stretched[:expected_out_len]

        return linear_energy_match(
            x, raw_stretched, alpha=alpha, win_len=4096, device=self.device
        )


# =========================================================================
# 9. ISS-Adaptive Wigner FrFT Stretcher
# =========================================================================
class ISSAdaptiveWignerFrFTStretcher(nn.Module):

    def __init__(
        self, sr: int = 44100, win_size: int = 2048, device: str = "cuda"
    ):
        super().__init__()
        self.sr = sr
        self.win_size = win_size
        self.device = device
        self.window = torch.hann_window(win_size, device=device)

        self.candidate_orders = [0.1, 0.3, 0.5, 0.7, 0.9]
        K_list = []
        K_inv_list = []
        for a in self.candidate_orders:
            K = build_unitary_frft_matrix(win_size, a, device=device)
            K_list.append(K)
            K_inv_list.append(K.conj().mT)

        self.K_grid = torch.stack(K_list, dim=0)
        self.K_inv_grid = torch.stack(K_inv_list, dim=0)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        x = x.to(self.device)
        hop_s = self.win_size // 4
        hop_a = hop_s / alpha
        num_samples = len(x)

        expected_out_len = int(num_samples * alpha)
        out_len = expected_out_len + self.win_size * 2
        output = torch.zeros(out_len, device=self.device)
        weights = torch.zeros(out_len, device=self.device)

        num_frames = int(math.ceil(num_samples / hop_a))
        win_sq = self.window**2

        frame_list = []
        valid_indices = []

        for i in range(num_frames):
            in_pos = int(i * hop_a)
            if in_pos + self.win_size > num_samples:
                break
            frame_list.append(x[in_pos : in_pos + self.win_size] * self.window)
            valid_indices.append(i)

        if len(frame_list) == 0:
            return x

        num_valid = len(frame_list)
        frames_tensor = torch.stack(frame_list, dim=0).to(torch.complex64)

        X_cand_list = [torch.matmul(frames_tensor, K.mT) for K in self.K_grid]
        X_cand = torch.stack(X_cand_list, dim=1)

        l4_concentration = torch.sum(torch.abs(X_cand) ** 4, dim=-1)
        softmax_weights = F.softmax(
            10.0 * F.normalize(l4_concentration, dim=-1), dim=-1
        )

        target_angles = torch.sum(
            softmax_weights
            * torch.tensor(self.candidate_orders, device=self.device),
            dim=-1,
        )

        a_smooth = torch.zeros(num_valid, device=self.device)
        a_curr = 0.5
        gamma = 0.15
        dt = 0.25

        for i in range(num_valid):
            grad = target_angles[i] - a_curr
            decel = gamma * a_curr / (1.0 - a_curr**2 + 1e-4)
            a_curr = a_curr + dt * torch.clamp(grad - decel, -0.15, 0.15)
            a_curr = torch.clamp(a_curr, 0.05, 0.95)
            a_smooth[i] = a_curr

        cand_tensor = torch.tensor(
            self.candidate_orders, device=self.device
        ).unsqueeze(0)
        dist_sq = (cand_tensor - a_smooth.unsqueeze(1)) ** 2
        smooth_w = F.softmax(-dist_sq / 0.02, dim=-1).unsqueeze(-1)

        sqrt_w = torch.sqrt(smooth_w)
        rec_cand = torch.stack(
            [
                torch.matmul(
                    sqrt_w[:, k] * X_cand[:, k], self.K_inv_grid[k].mT
                )
                * sqrt_w[:, k]
                for k in range(len(self.candidate_orders))
            ],
            dim=1,
        )

        recon_frames = torch.real(torch.sum(rec_cand, dim=1)) * self.window

        for idx, i in enumerate(valid_indices):
            out_pos = i * hop_s
            output[out_pos : out_pos + self.win_size] += recon_frames[idx]
            weights[out_pos : out_pos + self.win_size] += win_sq

        valid = weights > 1e-6
        output[valid] = output[valid] / weights[valid]
        res_audio = torch.clamp(output[:expected_out_len], -1.0, 1.0)

        return linear_energy_match(
            x, res_audio, alpha=alpha, win_len=4096, device=self.device
        )


# =========================================================================
# 10. FrFT-0.5 Fixed Engine
# =========================================================================
class FrFT05Stretcher(nn.Module):

    def __init__(
        self, sr: int = 44100, win_size: int = 2048, device: str = "cpu"
    ):
        super().__init__()
        self.sr = sr
        self.win_size = win_size
        self.device = device
        self.window = torch.hann_window(win_size, device=device)

        n = (
            torch.arange(win_size, device=device, dtype=torch.float32)
            - win_size / 2.0
        )
        self.chirp_in = torch.exp(1j * (math.pi / win_size) * (n**2))
        self.chirp_in_inv = torch.exp(-1j * (math.pi / win_size) * (n**2))

        k = (
            torch.arange(win_size, device=device, dtype=torch.float32)
            - win_size / 2.0
        )
        self.chirp_out = torch.exp(1j * (math.pi / win_size) * (k**2))
        self.chirp_out_inv = torch.exp(-1j * (math.pi / win_size) * (k**2))

    def frft_05(self, x_frame: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft(x_frame * self.chirp_in) * self.chirp_out

    def ifrft_05(self, X_frft: torch.Tensor) -> torch.Tensor:
        X_fft = X_frft * self.chirp_out_inv
        x_chirp = torch.fft.ifft(X_fft)
        return torch.real(x_chirp * self.chirp_in_inv)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        x = x.to(self.device)
        hop_s = self.win_size // 4
        hop_a = hop_s / alpha
        num_samples = len(x)

        expected_out_len = int(num_samples * alpha)
        out_len = expected_out_len + self.win_size * 2
        output = torch.zeros(out_len, device=self.device)
        weights = torch.zeros(out_len, device=self.device)

        num_frames = int(math.ceil(num_samples / hop_a))
        win_sq = self.window**2

        for i in range(num_frames):
            in_pos = int(i * hop_a)
            if in_pos + self.win_size > num_samples:
                break

            segment = x[in_pos : in_pos + self.win_size] * self.window
            X_frft = self.frft_05(segment)
            resynth = self.ifrft_05(X_frft) * self.window

            out_pos = i * hop_s
            output[out_pos : out_pos + self.win_size] += resynth
            weights[out_pos : out_pos + self.win_size] += win_sq

        valid = weights > 1e-6
        output[valid] = output[valid] / weights[valid]
        res_audio = torch.clamp(output[:expected_out_len], -1.0, 1.0)
        return linear_energy_match(
            x, res_audio, alpha=alpha, win_len=4096, device=self.device
        )


# =========================================================================
# 11. Paulstretch Baseline Engine
# =========================================================================
class PaulstretchBaseline(nn.Module):

    def __init__(self, win_size: int = 8192, device: str = "cpu"):
        super().__init__()
        self.win_size = win_size
        self.device = device
        self.window = torch.hann_window(win_size, device=device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        x = x.to(self.device)
        hop_s = self.win_size // 2
        hop_a = hop_s / alpha
        num_samples = len(x)

        expected_out_len = int(num_samples * alpha)
        out_len = expected_out_len + self.win_size * 2
        output = torch.zeros(out_len, device=self.device)
        weights = torch.zeros(out_len, device=self.device)

        num_frames = int(math.ceil(num_samples / hop_a))

        for i in range(num_frames):
            in_pos = int(i * hop_a)
            if in_pos + self.win_size > num_samples:
                break

            segment = x[in_pos : in_pos + self.win_size] * self.window
            stft_frame = torch.fft.rfft(segment)

            mag = torch.abs(stft_frame)
            rand_phase = torch.rand_like(mag) * 2 * math.pi
            synth_frame = mag * torch.exp(1j * rand_phase)

            resynth = torch.fft.irfft(synth_frame) * self.window

            out_pos = i * hop_s
            output[out_pos : out_pos + self.win_size] += resynth
            weights[out_pos : out_pos + self.win_size] += self.window**2

        output = output / (weights + 1e-3)
        res_audio = output[:expected_out_len]

        fade_len = min(1024, len(res_audio) // 10)
        if fade_len > 0:
            fade_curve = torch.linspace(
                1.0, 0.0, fade_len, device=self.device
            )
            res_audio[-fade_len:] *= fade_curve

        return res_audio


# =========================================================================
# 12. Tkinter Graphical User Interface (GUI)
# =========================================================================
class AudioStretcherGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Advanced DSP Audio Time Stretcher")
        self.root.geometry("680x560")
        self.root.resizable(False, False)

        # Apply modern ttk style
        self.style = ttk.Style()
        if "clam" in self.style.theme_names():
            self.style.theme_use("clam")

        # State Variables
        self.loaded_filepath = None
        self.waveform = None
        self.sr = 44100
        self.stretched_waveform = None

        self.audio_process = None  # Holds background player process

        # Available Stretchers
        self.stretcher_map = {
            "Advanced ZCA-Krein (v4)": lambda sr, dev: ZCAKreinStretcher(
                sr=sr, device=dev
            ),
            "Hybrid ZCA-Krein-FrFT": lambda sr, dev: ZCAKreinFrFTStretcher(
                sr=sr, device=dev
            ),
            "Operator-Softmax Krein-FrFT": lambda sr,
            dev: OperatorSoftmaxKreinFrFTStretcher(
                sr=sr, win_size=2048, device=dev
            ),
            "ISS-Adaptive Wigner FrFT": lambda sr,
            dev: ISSAdaptiveWignerFrFTStretcher(
                sr=sr, win_size=2048, device=dev
            ),
            "FrFT-0.5 Phase-Coherent": lambda sr, dev: FrFT05Stretcher(
                sr=sr, win_size=2048, device=dev
            ),
            "Paulstretch (Baseline)": lambda sr, dev: PaulstretchBaseline(
                win_size=8192, device=dev
            ),
        }

        self.create_widgets()

    def create_widgets(self):
        # Header Title
        header_frame = ttk.Frame(self.root, padding=10)
        header_frame.pack(fill=tk.X)

        title_label = ttk.Label(
            header_frame,
            text="🎧 Audio Time-Stretcher Workbench",
            font=("Helvetica", 16, "bold"),
        )
        title_label.pack(anchor=tk.W)

        subtitle_label = ttk.Label(
            header_frame,
            text="Load custom audio (MP3, WAV, etc.), stretch without pitch shift, listen and export.",
            font=("Helvetica", 9, "italic"),
        )
        subtitle_label.pack(anchor=tk.W)

        # 1. File Input Frame
        file_frame = ttk.LabelFrame(
            self.root, text=" 1. Select Audio File ", padding=10
        )
        file_frame.pack(fill=tk.X, padx=15, pady=5)

        self.btn_browse = ttk.Button(
            file_frame, text="📁 Open File...", command=self.browse_file
        )
        self.btn_browse.pack(side=tk.LEFT, padx=5)

        self.lbl_file_info = ttk.Label(
            file_frame,
            text="No audio file loaded.",
            font=("Helvetica", 9),
            foreground="gray",
        )
        self.lbl_file_info.pack(side=tk.LEFT, padx=10)

        # 2. Parameters Configuration Frame
        param_frame = ttk.LabelFrame(
            self.root, text=" 2. Stretcher Settings ", padding=10
        )
        param_frame.pack(fill=tk.X, padx=15, pady=5)

        # Algorithm Chooser
        ttk.Label(param_frame, text="Algorithm:").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        self.combo_algo = ttk.Combobox(
            param_frame,
            values=list(self.stretcher_map.keys()),
            state="readonly",
            width=32,
        )
        self.combo_algo.current(0)
        self.combo_algo.grid(row=0, column=1, sticky=tk.W, padx=10, pady=5)

        # Stretch Ratio Alpha
        ttk.Label(
            param_frame, text="Stretch Factor (α):\n[>1.0 Slow, <1.0 Fast]"
        ).grid(row=1, column=0, sticky=tk.W, pady=5)

        alpha_subframe = ttk.Frame(param_frame)
        alpha_subframe.grid(row=1, column=1, sticky=tk.W, padx=10, pady=5)

        self.alpha_var = tk.DoubleVar(value=2.0)
        self.scale_alpha = ttk.Scale(
            alpha_subframe,
            from_=0.25,
            to=5.0,
            variable=self.alpha_var,
            orient=tk.HORIZONTAL,
            length=180,
            command=self.on_alpha_scale_change,
        )
        self.scale_alpha.pack(side=tk.LEFT)

        self.entry_alpha = ttk.Entry(
            alpha_subframe, textvariable=self.alpha_var, width=6
        )
        self.entry_alpha.pack(side=tk.LEFT, padx=5)

        # Hardware Device
        ttk.Label(param_frame, text="Device:").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        devices = (
            ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
        )
        self.combo_device = ttk.Combobox(
            param_frame, values=devices, state="readonly", width=10
        )
        self.combo_device.current(0)
        self.combo_device.grid(row=2, column=1, sticky=tk.W, padx=10, pady=5)

        # 3. Action / Execution Frame
        action_frame = ttk.Frame(self.root, padding=10)
        action_frame.pack(fill=tk.X, padx=15)

        self.btn_process = ttk.Button(
            action_frame,
            text="⚡ Stretch Audio",
            command=self.start_processing_thread,
            state=tk.DISABLED,
        )
        self.btn_process.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.progress_bar = ttk.Progressbar(action_frame, mode="indeterminate")

        # Status Bar
        self.lbl_status = ttk.Label(
            self.root, text="Status: Ready", relief=tk.SUNKEN, anchor=tk.W
        )
        self.lbl_status.pack(fill=tk.X, side=tk.BOTTOM, ipady=3)

        # 4. Preview & Export Frame
        output_frame = ttk.LabelFrame(
            self.root, text=" 3. Listen & Export Results ", padding=10
        )
        output_frame.pack(fill=tk.X, padx=15, pady=5)

        play_btn_frame = ttk.Frame(output_frame)
        play_btn_frame.pack(fill=tk.X, pady=5)

        self.btn_play_orig = ttk.Button(
            play_btn_frame,
            text="▶ Play Original",
            command=self.play_original,
            state=tk.DISABLED,
        )
        self.btn_play_orig.pack(side=tk.LEFT, padx=5)

        self.btn_play_stretched = ttk.Button(
            play_btn_frame,
            text="▶ Play Stretched",
            command=self.play_stretched,
            state=tk.DISABLED,
        )
        self.btn_play_stretched.pack(side=tk.LEFT, padx=5)

        self.btn_stop_play = ttk.Button(
            play_btn_frame, text="⏹ Stop Audio", command=self.stop_audio
        )
        self.btn_stop_play.pack(side=tk.LEFT, padx=5)

        self.btn_export = ttk.Button(
            output_frame,
            text="💾 Save Stretched Audio As...",
            command=self.export_audio,
            state=tk.DISABLED,
        )
        self.btn_export.pack(fill=tk.X, pady=5)

    def on_alpha_scale_change(self, val):
        val_float = round(float(val), 2)
        self.alpha_var.set(val_float)

    def browse_file(self):
        filetypes = [
            ("Audio Files", "*.mp3 *.wav *.flac *.ogg *.m4a *.aac"),
            ("All Files", "*.*"),
        ]
        filepath = filedialog.askopenfilename(
            title="Select Input Audio File", filetypes=filetypes
        )
        if not filepath:
            return

        self.lbl_status.config(text=f"Loading audio file: {filepath}...")
        self.root.update_idletasks()

        try:
            waveform, sr = load_audio(filepath, target_sr=44100)
            self.loaded_filepath = filepath
            self.waveform = waveform
            self.sr = sr
            self.stretched_waveform = None

            duration = len(waveform) / sr
            filename = os.path.basename(filepath)
            self.lbl_file_info.config(
                text=f"{filename} | Duration: {duration:.2f}s | SR: {sr} Hz",
                foreground="black",
            )
            self.lbl_status.config(text=f"Loaded: {filename}")

            self.btn_process.config(state=tk.NORMAL)
            self.btn_play_orig.config(state=tk.NORMAL)
            self.btn_play_stretched.config(state=tk.DISABLED)
            self.btn_export.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror(
                "Error Loading Audio", f"Could not decode audio file:\n{e}"
            )
            self.lbl_status.config(text="Status: Failed to load file.")

    def start_processing_thread(self):
        if self.waveform is None:
            return

        self.btn_process.config(state=tk.DISABLED)
        self.btn_browse.config(state=tk.DISABLED)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        self.progress_bar.start(10)
        self.lbl_status.config(text="Processing audio stretcher... Please wait.")

        # Run DSP computation in background thread to keep GUI responsive
        threading.Thread(target=self.run_audio_stretcher, daemon=True).start()

    def run_audio_stretcher(self):
        try:
            algo_name = self.combo_algo.get()
            device = self.combo_device.get()
            alpha = float(self.alpha_var.get())

            stretcher_builder = self.stretcher_map[algo_name]
            stretcher = stretcher_builder(self.sr, device)

            t0 = time.time()
            stretched_out = stretcher(self.waveform, alpha=alpha)
            elapsed = time.time() - t0

            self.stretched_waveform = stretched_out.cpu()

            self.root.after(
                0, self.on_processing_complete, elapsed, len(stretched_out) / self.sr
            )

        except Exception as e:
            self.root.after(0, self.on_processing_failed, str(e))

    def on_processing_complete(self, elapsed_time, out_duration):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.btn_process.config(state=tk.NORMAL)
        self.btn_browse.config(state=tk.NORMAL)

        self.btn_play_stretched.config(state=tk.NORMAL)
        self.btn_export.config(state=tk.NORMAL)

        self.lbl_status.config(
            text=f"Status: Processing complete in {elapsed_time:.2f}s! Output duration: {out_duration:.2f}s"
        )
        messagebox.showinfo(
            "Success",
            f"Audio stretched successfully!\n\nProcessing Time: {elapsed_time:.2f}s\nNew Audio Duration: {out_duration:.2f}s",
        )

    def on_processing_failed(self, error_msg):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.btn_process.config(state=tk.NORMAL)
        self.btn_browse.config(state=tk.NORMAL)

        self.lbl_status.config(text="Status: Processing failed.")
        messagebox.showerror(
            "Processing Error", f"An error occurred during stretching:\n{error_msg}"
        )

    def play_sound_file(self, filepath):
        self.stop_audio()
        try:
            # Non-blocking cross-platform playback via ffplay
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", filepath]
            self.audio_process = subprocess.Popen(cmd)
        except Exception as e:
            messagebox.showerror(
                "Playback Error", f"Could not play audio file via ffplay:\n{e}"
            )

    def play_original(self):
        if self.loaded_filepath and os.path.exists(self.loaded_filepath):
            self.play_sound_file(self.loaded_filepath)

    def play_stretched(self):
        if self.stretched_waveform is None:
            return

        temp_preview = f"temp_preview_{os.getpid()}.wav"
        save_audio(temp_preview, self.stretched_waveform, sr=self.sr)
        self.play_sound_file(temp_preview)

    def stop_audio(self):
        if self.audio_process is not None:
            self.audio_process.terminate()
            self.audio_process = None

    def export_audio(self):
        if self.stretched_waveform is None:
            return

        filetypes = [
            ("WAV Audio File", "*.wav"),
            ("MP3 Audio File", "*.mp3"),
            ("All Files", "*.*"),
        ]
        save_path = filedialog.asksaveasfilename(
            title="Save Stretched Audio",
            defaultextension=".wav",
            filetypes=filetypes,
        )

        if not save_path:
            return

        try:
            self.lbl_status.config(text=f"Saving audio to {save_path}...")
            save_audio(save_path, self.stretched_waveform, sr=self.sr)
            self.lbl_status.config(text=f"Exported audio saved successfully: {save_path}")
            messagebox.showinfo("Saved", f"File saved successfully to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save audio file:\n{e}")


# =========================================================================
# 13. Main Launcher
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Audio Stretcher Suite (GUI & CLI Benchmark)"
    )
    parser.add_argument(
        "-i", "--input", type=str, default=None, help="Input audio file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="stretched_zca_krein.wav",
        help="Output audio path",
    )
    parser.add_argument(
        "-a", "--alpha", type=float, default=2.0, help="Stretch factor alpha"
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run non-GUI benchmark identity test mode",
    )
    args = parser.parse_args()

    # If --cli argument is passed, run terminal benchmark mode; otherwise open Tkinter GUI
    if args.cli:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sr = 44100
        if args.input and os.path.exists(args.input):
            print(f"Loading Audio File: {args.input}")
            waveform, sr = load_audio(args.input, target_sr=sr)
        else:
            print("Generating synthetic test signal...")
            waveform = generate_synthetic_test_signal(sr=sr, duration=2.0)

        from __main__ import RoundTripBenchmarkSuite

        RoundTripBenchmarkSuite.run_roundtrip_identity_test(
            waveform, sr=sr, alpha=args.alpha, device=device
        )
    else:
        root = tk.Tk()
        app = AudioStretcherGUI(root)
        root.mainloop()


if __name__ == "__main__":
    main()
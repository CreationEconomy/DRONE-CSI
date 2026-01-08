from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TSCNNConfig:
    """
    Drone-CSI-Sense (Deep Learning Edition) 설정값.
    - 입력: I/Q 128개(raw) -> 진폭 64개(채널)로 변환 후, 20프레임 윈도우로 모델 입력
    - 모델 입력 텐서: (Batch, Channels, Length) = (N, 64, 20)
    """

    iq_len: int = 128
    channels: int = 64
    window_size: int = 20
    stride_train: int = 2  # 5 → 2: 윈도우 수 2.5배 증가 (데이터 증강 효과)
    stride_infer: int = 1


def iq128_to_amp64(iq_128: Sequence[int | float]) -> Optional[np.ndarray]:
    """
    I/Q 128개(=64쌍) raw 배열을 진폭 64개로 변환합니다.
    - 입력 예: [I0, Q0, I1, Q1, ...] 혹은 [Q0, I0, ...] 등 순서가 달라도
      진폭(sqrt(a^2+b^2))은 동일하므로 짝수/홀수 페어로 계산합니다.
    - 출력: shape (64,), dtype float32
    """
    arr = np.asarray(iq_128, dtype=np.float32)
    if arr.ndim != 1 or int(arr.shape[0]) != 128:
        return None
    a = arr[0::2]
    b = arr[1::2]
    amp = np.sqrt(a * a + b * b).astype(np.float32)
    if int(amp.shape[0]) != 64:
        return None
    return amp


def build_windows(
    frames: np.ndarray, window_size: int = 20, stride: int = 5
) -> np.ndarray:
    """
    연속 프레임(진폭 64) 시퀀스를 슬라이딩 윈도우로 잘라 모델 입력 형태로 만듭니다.
    - 입력 frames: shape (T, 64)
    - 출력 windows: shape (N, 64, window_size)
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2:
        raise ValueError(f"frames must be 2D (T, C). got shape={frames.shape}")
    t, c = int(frames.shape[0]), int(frames.shape[1])
    if c != 64:
        raise ValueError(f"frames must have 64 channels. got C={c}")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if t < window_size:
        return np.empty((0, c, window_size), dtype=np.float32)

    windows = []
    for start in range(0, t - window_size + 1, stride):
        chunk = frames[start : start + window_size]  # (window, 64)
        windows.append(chunk.T)  # (64, window)

    return np.stack(windows, axis=0).astype(np.float32) if windows else np.empty((0, c, window_size), dtype=np.float32)


class TimeSeries1DCNN(nn.Module):
    """
    지시서 스펙(1D-CNN):
    - 입력: (N, 64, 20)
    - Conv1: 64->32, k=3, p=1 + BN + ReLU + MaxPool(k=2) => (N,32,10)
    - Conv2: 32->16, k=3, p=1 + BN + ReLU + MaxPool(k=2) => (N,16,5)
    - Flatten => (N,80)
    - FC: 80->32 + Dropout(0.5)
    - FC: 32->2 (logits)
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def save_tscnn_checkpoint(
    path: str,
    model: nn.Module,
    config: TSCNNConfig,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "config": asdict(config),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_tscnn_checkpoint(
    path: str, device: str | torch.device = "cpu"
) -> Tuple[TimeSeries1DCNN, TSCNNConfig, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    cfg = TSCNNConfig(**ckpt.get("config", {}))
    model = TimeSeries1DCNN().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    extra = dict(ckpt.get("extra", {}))
    return model, cfg, extra


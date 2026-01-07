import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


MAC_RE = re.compile(r"(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}")
CSI_BRACKET_RE = re.compile(r"\[(.*?)\]")


@dataclass(frozen=True)
class ParsedCSI:
    mac: Optional[str]
    rssi: Optional[int]
    csi_raw: np.ndarray  # shape: (N,), dtype: int32
    raw_line: str


def _safe_int(s: str) -> Optional[int]:
    try:
        return int(str(s).strip())
    except Exception:
        return None


def parse_csi_line(line: str) -> Optional[ParsedCSI]:
    """
    esp32-csi-tool 로그 한 줄에서 CSI 배열을 파싱합니다.
    기대 포맷 예: "CSI_DATA,<...>,<mac>,<rssi>,..., [I0 Q0 I1 Q1 ...]"
    """
    if not line or "CSI_DATA" not in line:
        return None

    parts = line.split(",")

    # mac: 흔히 parts[2]에 들어가지만, 포맷이 바뀌어도 잡히도록 regex도 사용
    mac = parts[2].strip() if len(parts) > 2 else None
    if not mac or not MAC_RE.fullmatch(mac):
        m = MAC_RE.search(line)
        mac = m.group(0).upper() if m else None
    else:
        mac = mac.upper()

    # rssi: 흔히 parts[3]
    rssi = _safe_int(parts[3]) if len(parts) > 3 else None
    if rssi is not None and not (-127 <= rssi <= 0):
        # RSSI로 보기엔 범위가 이상하면 None 처리(포맷 차이 가능)
        rssi = None

    csi_match = CSI_BRACKET_RE.search(line)
    if not csi_match:
        return None

    tokens = csi_match.group(1).split()
    if not tokens:
        return None

    try:
        csi_raw = np.array([int(x) for x in tokens], dtype=np.int32)
    except Exception:
        return None

    return ParsedCSI(mac=mac, rssi=rssi, csi_raw=csi_raw, raw_line=line)


def csi_raw_to_complex(csi_raw: np.ndarray) -> Optional[np.ndarray]:
    """
    [I0, Q0, I1, Q1, ...] → complex array [I0+jQ0, I1+jQ1, ...]
    """
    if csi_raw is None or len(csi_raw) == 0:
        return None
    if len(csi_raw) % 2 != 0:
        return None
    i = csi_raw[0::2].astype(np.float32)
    q = csi_raw[1::2].astype(np.float32)
    return i + 1j * q


def complex_to_amp_db_rel(csi_complex: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    |H|를 '패킷 내 최대값'으로 정규화한 dB.
    출력: 0 dB(최대) ~ 음수 dB
    """
    amp = np.abs(csi_complex).astype(np.float32)
    amp_max = float(np.max(amp)) if amp.size else 1.0
    return 20.0 * np.log10((amp + eps) / (amp_max + eps))


def enforce_fixed_length(arr: np.ndarray, target_len: int) -> Optional[np.ndarray]:
    """
    길이가 target_len이면 그대로, 아니면 None(스킵용).
    """
    if arr is None:
        return None
    if int(arr.shape[0]) != int(target_len):
        return None
    return arr


def majority_label(labels: np.ndarray) -> Optional[str]:
    """
    문자열 라벨 배열에서 최빈 라벨을 반환합니다.
    - "unknown"/""는 무시합니다.
    """
    if labels is None or len(labels) == 0:
        return None
    labels = np.asarray(labels)
    mask = (labels != "") & (labels != "unknown")
    labels = labels[mask]
    if labels.size == 0:
        return None
    uniq, cnt = np.unique(labels, return_counts=True)
    return str(uniq[int(np.argmax(cnt))])


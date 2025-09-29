from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
from scipy.signal import butter, filtfilt

# ----------------------------
# Data model
# ----------------------------

@dataclass
class TorqueDataPoint:
    time: float     # seconds
    torque: float   # Nm (raw), or normalized if you choose

# ----------------------------
# Cleaning ops
# ----------------------------

def apply_zero_rest(
    data: List[TorqueDataPoint],
    sampling_rate_hz: int,
    rest_window_s: float = 0.5
) -> List[TorqueDataPoint]:
    """
    Subtract mean of an initial 'rest' window from the torque signal.
    """
    if not data:
        return []
    n_rest = max(1, int(rest_window_s * sampling_rate_hz))
    arr = np.array([p.torque for p in data], dtype=float)
    rest_mean = np.mean(arr[:min(n_rest, len(arr))])
    arr = arr - rest_mean
    return [TorqueDataPoint(time=p.time, torque=float(arr[i])) for i, p in enumerate(data)]

def apply_hampel(
    data: List[TorqueDataPoint],
    window_s: float,
    sampling_rate_hz: int,
    n_sigmas: float = 3.0
) -> List[TorqueDataPoint]:
    """
    Hampel filter on torque: replace outliers with local median.
    window_s: half-window size in seconds (similar to many JS impls that use a frac of a second)
    """
    if not data:
        return []
    x = np.array([p.torque for p in data], dtype=float)
    w = max(1, int(window_s * sampling_rate_hz))
    x_pad = np.pad(x, (w, w), mode="edge")
    med = np.empty_like(x)
    mad = np.empty_like(x)
    for i in range(len(x)):
        seg = x_pad[i:i + 2*w + 1]
        m = np.median(seg)
        med[i] = m
        mad[i] = np.median(np.abs(seg - m)) + 1e-12  # avoid div by zero
    threshold = n_sigmas * 1.4826 * mad
    out = np.copy(x)
    out[np.abs(x - med) > threshold] = med[np.abs(x - med) > threshold]
    return [TorqueDataPoint(time=data[i].time, torque=float(out[i])) for i in range(len(data))]

def apply_butterworth(
    data: List[TorqueDataPoint],
    sampling_rate_hz: int,
    cutoff_hz: float,
    order: int = 4
) -> List[TorqueDataPoint]:
    """
    Zero-phase (filtfilt) low-pass Butterworth.
    """
    if not data:
        return []
    x = np.array([p.torque for p in data], dtype=float)
    nyq = 0.5 * sampling_rate_hz
    wc = min(0.99, cutoff_hz / nyq)
    b, a = butter(order, wc, btype='low', analog=False)
    y = filtfilt(b, a, x, method="pad")
    return [TorqueDataPoint(time=data[i].time, torque=float(y[i])) for i in range(len(data))]

def apply_moving_average(
    data: List[TorqueDataPoint],
    sampling_rate_hz: int,
    window_s: float
) -> List[TorqueDataPoint]:
    """
    Simple centered moving average with reflection padding.
    """
    if not data:
        return []
    x = np.array([p.torque for p in data], dtype=float)
    w = max(1, int(window_s * sampling_rate_hz))
    if w % 2 == 0:
        w += 1
    x_pad = np.pad(x, (w//2, w//2), mode="reflect")
    kernel = np.ones(w) / w
    y = np.convolve(x_pad, kernel, mode="valid")
    return [TorqueDataPoint(time=data[i].time, torque=float(y[i])) for i in range(len(data))]

def process_data_with_cleaning(
    raw: List[TorqueDataPoint],
    sampling_rate_hz: int = 1000,
    cutoff_hz: float = 20.0,
    order: int = 4,
    ma_window_s: float = 0.15,
    hampel_window_s: float = 0.4
) -> List[TorqueDataPoint]:
    """
    Mirrors your TS chain:
      zeroRest -> Hampel(0.4 s) -> Butterworth(20 Hz, order 4) -> MovingAverage(0.15 s).
    """
    zeroed = apply_zero_rest(raw, sampling_rate_hz)
    no_outliers = apply_hampel(zeroed, window_s=hampel_window_s, sampling_rate_hz=sampling_rate_hz)
    smoothed = apply_butterworth(no_outliers, sampling_rate_hz, cutoff_hz, order)
    movavg = apply_moving_average(smoothed, sampling_rate_hz, ma_window_s)
    return movavg

# ----------------------------
# Onset/End detection
# ----------------------------

def _is_local_minimum(x: np.ndarray, i: int, neighborhood: int) -> bool:
    v = x[i]
    s = max(0, i - neighborhood)
    e = min(len(x) - 1, i + neighborhood)
    return np.all(v <= x[s:i]) and np.all(v <= x[i+1:e+1])

def _is_rising(x: np.ndarray, idx: int, win: int) -> bool:
    # non-decreasing over last 'win' samples ending at idx
    for k in range(1, win + 1):
        j = idx - k
        if j < 0:
            break
        if x[j] > x[j + 1]:
            return False
    return True

def _is_falling(x: np.ndarray, idx: int, win: int) -> bool:
    # non-increasing over next 'win' samples starting at idx
    for k in range(1, win + 1):
        j = idx + k
        if j >= len(x):
            break
        if x[j] > x[j - 1]:
            return False
    return True

def detect_onset_index(
    cleaned: List[TorqueDataPoint],
    sampling_rate_hz: Optional[int] = 1000,
    start_threshold_nm: Optional[float] = None,
    start_threshold_pct_of_peak: float = 0.05,
    min_rise_window_ms: int = 20,
    walk_back_max_ms: int = 300,
    local_min_neighborhood: int = 3
) -> Dict[str, float | int]:
    """
    Find first crossing above start threshold (Nm or %peak), guard with rising trend,
    then walk backward to nearest local minimum (within walk_back_max_ms).
    Returns dict: onsetIdx, onsetTime, startThresholdUsed
    """
    if not cleaned:
        return {"onsetIdx": 0, "onsetTime": 0.0, "startThresholdUsed": 0.0}
    t = np.array([p.time for p in cleaned], dtype=float)
    y = np.array([p.torque for p in cleaned], dtype=float)

    peak = float(np.max(y))
    thr = start_threshold_nm if start_threshold_nm is not None else start_threshold_pct_of_peak * peak

    rise_win = max(1, int((min_rise_window_ms / 1000.0) * sampling_rate_hz))
    # first crossing with rising guard
    cross_idx = -1
    for i in range(len(y)):
        if y[i] >= thr and _is_rising(y, i, rise_win):
            cross_idx = i
            break

    if cross_idx < 0:
        min_idx = int(np.argmin(y))
        return {"onsetIdx": min_idx, "onsetTime": float(t[min_idx]), "startThresholdUsed": float(thr)}

    max_back = min(cross_idx, int((walk_back_max_ms / 1000.0) * sampling_rate_hz))
    onset_idx = cross_idx
    for back in range(0, max_back + 1):
        i = cross_idx - back
        if _is_local_minimum(y, i, max(1, local_min_neighborhood)):
            onset_idx = i
            break

    return {"onsetIdx": int(onset_idx), "onsetTime": float(t[onset_idx]), "startThresholdUsed": float(thr)}

def detect_end_index(
    cleaned: List[TorqueDataPoint],
    start_idx: int,
    sampling_rate_hz: Optional[int] = 1000,
    end_threshold_nm: Optional[float] = None,
    end_threshold_pct_of_peak: float = 0.25,
    min_fall_window_ms: int = 20,
    walk_fwd_max_ms: int = 300,
    local_min_neighborhood: int = 3
) -> Dict[str, float | int]:
    """
    After onset: find the peak; then find first crossing below end threshold
    (Nm or % of peak) with falling guard; walk forward to nearest local minimum.
    Returns dict: endIdx, endTime, endThresholdUsed
    """
    if not cleaned:
        return {"endIdx": 0, "endTime": 0.0, "endThresholdUsed": 0.0}
    t = np.array([p.time for p in cleaned], dtype=float)
    y = np.array([p.torque for p in cleaned], dtype=float)

    # peak after onset
    peak_idx = start_idx + int(np.argmax(y[start_idx:]))
    peak_val = float(y[peak_idx])

    thr = end_threshold_nm if end_threshold_nm is not None else end_threshold_pct_of_peak * peak_val

    fall_win = max(1, int((min_fall_window_ms / 1000.0) * sampling_rate_hz))
    cross_idx = -1
    for i in range(min(peak_idx + 1, len(y) - 1), len(y)):
        if y[i] <= thr and _is_falling(y, i, fall_win):
            cross_idx = i
            break

    if cross_idx < 0:
        # fallback: global minimum after peak
        post = y[peak_idx:]
        min_rel = int(np.argmin(post))
        min_idx = peak_idx + min_rel
        return {"endIdx": int(min_idx), "endTime": float(t[min_idx]), "endThresholdUsed": float(thr)}

    max_fwd = min(len(y) - 1 - cross_idx, int((walk_fwd_max_ms / 1000.0) * sampling_rate_hz))
    end_idx = cross_idx
    for fwd in range(0, max_fwd + 1):
        i = cross_idx + fwd
        if _is_local_minimum(y, i, max(1, local_min_neighborhood)):
            end_idx = i
            break

    return {"endIdx": int(end_idx), "endTime": float(t[end_idx]), "endThresholdUsed": float(thr)}

def detect_active_window_indices(
    cleaned: List[TorqueDataPoint],
    sampling_rate_hz: int = 1000,
    start_threshold_nm: Optional[float] = None,
    start_threshold_pct_of_peak: float = 0.05,
    end_threshold_nm: Optional[float] = None,
    end_threshold_pct_of_peak: float = 0.25,
    min_rise_window_ms: int = 20,
    min_fall_window_ms: int = 20,
    walk_back_max_ms: int = 300,
    walk_fwd_max_ms: int = 300,
    local_min_neighborhood: int = 3
) -> Dict[str, float | int]:
    """
    Full window detection using symmetric onset/end logic.
    """
    onset = detect_onset_index(
        cleaned,
        sampling_rate_hz=sampling_rate_hz,
        start_threshold_nm=start_threshold_nm,
        start_threshold_pct_of_peak=start_threshold_pct_of_peak,
        min_rise_window_ms=min_rise_window_ms,
        walk_back_max_ms=walk_back_max_ms,
        local_min_neighborhood=local_min_neighborhood
    )
    end = detect_end_index(
        cleaned,
        start_idx=onset["onsetIdx"],
        sampling_rate_hz=sampling_rate_hz,
        end_threshold_nm=end_threshold_nm,
        end_threshold_pct_of_peak=end_threshold_pct_of_peak,
        min_fall_window_ms=min_fall_window_ms,
        walk_fwd_max_ms=walk_fwd_max_ms,
        local_min_neighborhood=local_min_neighborhood
    )
    start_idx = int(onset["onsetIdx"])
    end_idx = max(int(end["endIdx"]), start_idx)
    start_time = float([p.time for p in cleaned][start_idx]) if cleaned else 0.0
    end_time = float([p.time for p in cleaned][end_idx]) if cleaned else 0.0

    return {
        "startIdx": start_idx,
        "endIdx": end_idx,
        "startTime": start_time,
        "endTime": end_time,
        "startThresholdUsed": float(onset["startThresholdUsed"]),
        "endThresholdUsed": float(end["endThresholdUsed"])
    }

def slice_active_window_using_detectors(
    cleaned: List[TorqueDataPoint],
    **window_kwargs
) -> List[TorqueDataPoint]:
    """
    Convenience: detect indices then slice cleaned data.
    """
    if not cleaned:
        return []
    w = detect_active_window_indices(cleaned, **window_kwargs)
    return cleaned[w["startIdx"]: w["endIdx"] + 1]

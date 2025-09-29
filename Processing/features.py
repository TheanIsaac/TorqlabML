from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, NamedTuple
import numpy as np
from scipy.special import erfinv
import math

# Reuse your earlier dataclass
@dataclass
class TorqueDataPoint:
    time: float   # seconds
    torque: float # Nm (or Nm/kg if you normalize upstream)

# Result structures for the new plateau detector
class PlateauResult(NamedTuple):
    start_index: int
    end_index: int
    start_time: float
    end_time: float
    duration: float
    gradient: float
    regression_line: List[float]
    intercept: float
    r_squared: float
    stderr: float
    upper_ci: List[float]
    lower_ci: List[float]
    plateau: List[TorqueDataPoint]
    rmse: float
    max_yank: float
    mean_yank: float
    cov: float
    method: str

class CIRegressionResult(NamedTuple):
    slope: float
    intercept: float
    regression_line: List[float]
    r_squared: float
    stderr: float
    upper_ci: List[float]
    lower_ci: List[float]
    trimmed: List[TorqueDataPoint]
    trimmed_start_index: int
    trimmed_end_index: int

# ----------------------------
# Statistical utilities (matching TypeScript implementation)
# ----------------------------

def mean(values: List[float]) -> float:
    """Calculate mean of values."""
    return sum(values) / len(values) if values else 0.0

def variance(values: List[float]) -> float:
    """Calculate population variance."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / len(values)

def linreg(x: List[float], y: List[float]) -> Dict[str, float]:
    """Linear regression matching TypeScript implementation."""
    n = len(x)
    if n < 2:
        return {"slope": 0.0, "intercept": 0.0, "stderr": 0.0, "rSquared": 0.0}
    
    x_arr = np.array(x)
    y_arr = np.array(y)
    
    # Calculate slope and intercept
    x_mean = np.mean(x_arr)
    y_mean = np.mean(y_arr)
    
    numerator = np.sum((x_arr - x_mean) * (y_arr - y_mean))
    denominator = np.sum((x_arr - x_mean) ** 2)
    
    if denominator == 0:
        return {"slope": 0.0, "intercept": y_mean, "stderr": 0.0, "rSquared": 0.0}
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    # Calculate R-squared
    y_pred = slope * x_arr + intercept
    ss_res = np.sum((y_arr - y_pred) ** 2)
    ss_tot = np.sum((y_arr - y_mean) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Calculate standard error
    if n > 2:
        mse = ss_res / (n - 2)
        stderr = math.sqrt(mse)
    else:
        stderr = 0.0
    
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "stderr": float(stderr),
        "rSquared": float(r_squared)
    }

# ----------------------------
# New PlateauDetector (matching TypeScript implementation)
# ----------------------------

# Constants from TypeScript
VARIANCE_THRESHOLD = 100.0
TORQUE_THRESHOLD = 4.0  
PEAK_PERCENTAGE = 0.40

class PlateauDetector:
    def __init__(self, data: List[TorqueDataPoint], sampling_rate: int, min_plateau_duration: float):
        self.data = data
        self.sf = sampling_rate
        self.min_duration = min_plateau_duration
    
    def detect_plateau(self, confidence_level: float) -> Optional[PlateauResult]:
        """Main plateau detection method matching TypeScript implementation."""
        if confidence_level >= 1 or confidence_level <= 0.49:
            raise ValueError("Invalid confidence level")
        
        n = len(self.data)
        if n < 5:
            return None
        
        # 1) Compute moving variance
        window = max(1, int(self.sf / 3))
        var_arr = []
        for i in range(n):
            slice_vals = [self.data[j].torque for j in range(i, min(i + window, n))]
            var_val = variance(slice_vals) if len(slice_vals) > 1 else 0.0
            var_arr.append(var_val)
        
        # 2) Find two highest peaks in variance (not used in current logic but kept for compatibility)
        peaks_data = [(i, v) for i, v in enumerate(var_arr)]
        peaks_data.sort(key=lambda x: x[1], reverse=True)
        peaks = sorted([peaks_data[0][0], peaks_data[1][0]] if len(peaks_data) >= 2 else [peaks_data[0][0]])
        
        if len(peaks) < 1:
            return None
        
        # 3) Find longest run where torque >= PEAK_PERCENTAGE * maxTorque
        start_search = 0
        region = self.data[start_search:]
        torques = [p.torque for p in region]
        max_torque = max(torques)
        MAX_GAP = max(1, int(self.sf * 0.05))  # 50ms gap
        
        best_run = [0, -1]
        best_run_length = 0
        run_start = 0
        gap = 0
        
        for i in range(len(torques)):
            if torques[i] >= PEAK_PERCENTAGE * max_torque:
                gap = 0
            else:
                gap += 1
                if gap > MAX_GAP:
                    run_end = i - gap
                    run_length = run_end - run_start + 1
                    if run_length > best_run_length:
                        best_run = [run_start, run_end]
                        best_run_length = run_length
                    run_start = i + 1
                    gap = 0
        
        # Check final run
        final_end = len(torques) - 1
        final_length = final_end - run_start + 1
        if final_length > best_run_length:
            best_run = [run_start, final_end]
            best_run_length = final_length
        
        if best_run[1] < best_run[0]:
            return None
        
        rough_start = start_search + best_run[0]
        rough_end = start_search + best_run[1]
        
        # 4) Within that region, find longest run within variance & torque thresholds
        # EXACTLY matching TypeScript logic
        sub_torques = [p.torque for p in self.data]
        segs = []
        in_seg = False
        seg_start = rough_start
        
        for i in range(rough_start, rough_end + 1):
            v = var_arr[i]
            t = sub_torques[i]
            # Must satisfy BOTH conditions: variance <= threshold AND torque >= threshold

            if v <= VARIANCE_THRESHOLD and t >= TORQUE_THRESHOLD:
                if not in_seg:
                    in_seg = True
                    seg_start = i
            else:
                # Either variance too high OR torque too low - end current segment
                if in_seg:
                    segs.append([seg_start, i - 1])
                    in_seg = False
        
        # Close final segment if still in one
        if in_seg:
            segs.append([seg_start, rough_end])
        
        if not segs:
            return None
        
        # Pick longest segment - EXACTLY matching TypeScript sort logic
        segs.sort(key=lambda x: (x[1] - x[0]), reverse=True)
        p_start, p_end = segs[0]
        
        # Duration check
        duration = self.data[p_end].time - self.data[p_start].time
        if duration < self.min_duration:
            return None
        
        # 5) Regression and confidence intervals
        segment = self.data[p_start:p_end + 1]
        try:
            ci_result = self._regress_and_trim_ci(segment, confidence_level)
        except Exception as e:
            return None
        
        # Map trimmed indices back to original
        trimmed_global_start = p_start + ci_result.trimmed_start_index
        trimmed_global_end = p_start + ci_result.trimmed_end_index
        
        final_segment = self.data[trimmed_global_start:trimmed_global_end + 1]
        
        # Calculate additional metrics
        rmse = self._calculate_rmse(final_segment, ci_result.regression_line)
        yank_result = self._calculate_yank(final_segment)
        cov = self._calculate_cov(final_segment)
        
        return PlateauResult(
            start_index=trimmed_global_start,
            end_index=trimmed_global_end,
            start_time=self.data[trimmed_global_start].time,
            end_time=self.data[trimmed_global_end].time,
            duration=self.data[trimmed_global_end].time - self.data[trimmed_global_start].time,
            gradient=ci_result.slope,
            regression_line=ci_result.regression_line,
            intercept=ci_result.intercept,
            r_squared=ci_result.r_squared,
            stderr=ci_result.stderr,
            upper_ci=ci_result.upper_ci,
            lower_ci=ci_result.lower_ci,
            plateau=ci_result.trimmed,
            rmse=rmse,
            max_yank=yank_result["max_yank"],
            mean_yank=yank_result["mean_yank"],
            cov=cov,
            method="simple-variance-plateau"
        )
    
    def _calculate_rmse(self, data: List[TorqueDataPoint], ref_line: Optional[List[float]] = None) -> float:
        """Calculate RMSE between observed and predicted values."""
        n = len(data)
        if n == 0:
            return 0.0
        
        if ref_line is None:
            target = [mean([p.torque for p in data])] * n
        else:
            target = ref_line
        
        sum_sq = sum((data[i].torque - target[i]) ** 2 for i in range(n))
        return math.sqrt(sum_sq / n)
    
    def _calculate_yank(self, data: List[TorqueDataPoint]) -> Dict[str, float]:
        """Calculate yank (dτ/dt) statistics."""
        yanks = []
        for i in range(len(data) - 1):
            dt = data[i + 1].time - data[i].time
            if dt <= 0:
                continue
            dy = data[i + 1].torque - data[i].torque
            yanks.append(dy / dt)
        
        if not yanks:
            return {"yanks": [], "max_yank": 0.0, "mean_yank": 0.0}
        
        max_yank = max(abs(y) for y in yanks)
        mean_yank = sum(yanks) / len(yanks)
        
        return {"yanks": yanks, "max_yank": max_yank, "mean_yank": mean_yank}
    
    def _calculate_cov(self, data: List[TorqueDataPoint]) -> float:
        """Calculate coefficient of variation."""
        torques = [p.torque for p in data]
        mu = mean(torques)
        if mu == 0:
            return float('inf')
        sigma_sq = variance(torques)
        sigma = math.sqrt(sigma_sq)
        return sigma / mu
    
    def _regress_and_trim_ci(self, segment: List[TorqueDataPoint], confidence: float) -> CIRegressionResult:
        """Perform regression and trim points within confidence intervals."""
        n = len(segment)
        if n < 5:
            raise ValueError("Need at least 5 points")
        
        x = [p.time for p in segment]
        y = [p.torque for p in segment]
        
        reg_result = linreg(x, y)
        slope = reg_result["slope"]
        intercept = reg_result["intercept"]
        stderr = reg_result["stderr"]
        r_squared = reg_result["rSquared"]
        
        z = self._z_from_confidence(confidence)
        x_bar = mean(x)
        sxx = sum((xi - x_bar) ** 2 for xi in x)
        
        upper_ci = []
        lower_ci = []
        regression_line = []
        
        for i in range(n):
            xi = x[i]
            fit = intercept + slope * xi
            pi_factor = math.sqrt(1 + 1/n + ((xi - x_bar) ** 2) / sxx) if sxx > 0 else 1.0
            delta = z * stderr * pi_factor
            
            regression_line.append(fit)
            upper_ci.append(fit + delta)
            lower_ci.append(fit - delta)
        
        # Trim points within CI - find longest continuous segment (EXACTLY matching TypeScript)
        segments = []
        in_seg = False
        seg_start = 0
        
        for i in range(n):
            pt = segment[i]
            if pt.torque >= lower_ci[i] and pt.torque <= upper_ci[i]:
                if not in_seg:
                    seg_start = i
                    in_seg = True
            else:
                if in_seg:
                    segments.append([seg_start, i - 1])
                    in_seg = False
        
        if in_seg:
            segments.append([seg_start, n - 1])
        
        if not segments:
            raise ValueError("No valid plateau segments within CI bounds")
        
        # Pick the longest segment (EXACTLY matching TypeScript)
        segments.sort(key=lambda x: (x[1] - x[0]), reverse=True)
        first_idx, last_idx = segments[0]
        
        # Extract trimmed points
        trimmed = segment[first_idx:last_idx + 1]
        
        if len(trimmed) < 10:
            raise ValueError(f"Too few points ({len(trimmed)}) within confidence bounds — try increasing confidence level.")
        
        return CIRegressionResult(
            slope=slope,
            intercept=intercept,
            regression_line=regression_line,
            r_squared=r_squared,
            stderr=stderr,
            upper_ci=upper_ci,
            lower_ci=lower_ci,
            trimmed=trimmed,
            trimmed_start_index=first_idx,
            trimmed_end_index=last_idx
        )
    
    def _z_from_confidence(self, confidence: float) -> float:
        """Calculate z-score from confidence level."""
        alpha = 1 - confidence
        p = 1 - alpha/2
        return math.sqrt(2) * erfinv(2*p - 1)

# ----------------------------
# Legacy helper functions (for backward compatibility)
# ----------------------------

def _ms_to_samples(ms: float, sr: int) -> int:
    return max(1, int(round((ms / 1000.0) * sr)))

def _slice_by_time_window(t: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    return np.where((t >= start_s) & (t <= end_s))[0]

def _linreg_slope(t: np.ndarray, y: np.ndarray) -> float:
    """Return slope d(torque)/dt using least squares; robust if >1 sample."""
    if len(t) < 2:
        return float("nan")
    # Center time for numerical stability
    tc = t - t.mean()
    # slope = cov(t,y)/var(t)
    denom = np.sum(tc * tc)
    if denom <= 0:
        return float("nan")
    return float(np.sum(tc * (y - y.mean())) / denom)

def _first_derivative(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """dτ/dt with central differences (endpoints via forward/backward)."""
    dt = np.diff(t)
    dy = np.diff(y)
    v = np.empty_like(y)
    v[1:-1] = (y[2:] - y[:-2]) / (t[2:] - t[:-2])
    v[0] = dy[0] / dt[0]
    v[-1] = dy[-1] / dt[-1]
    return v

def _spectral_entropy(y: np.ndarray) -> float:
    """Spectral entropy of a demeaned signal (base-e)."""
    if len(y) < 8:
        return float("nan")
    x = y - y.mean()
    # Power spectrum
    fft = np.fft.rfft(x)
    psd = (fft.real**2 + fft.imag**2)
    psd_sum = psd.sum()
    if psd_sum <= 0:
        return float("nan")
    p = psd / psd_sum
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))

def _zero_crossings(x: np.ndarray) -> int:
    return int(np.sum((x[:-1] <= 0) & (x[1:] > 0)) + np.sum((x[:-1] >= 0) & (x[1:] < 0)))

# ----------------------------
# RTD features (relative to onset)
# ----------------------------

def compute_rtd_features(
    active: List[TorqueDataPoint],
    onset_time: float,
    sr: int,
    early_ms: Tuple[int, int] = (0, 75),
    late_ms: Tuple[int, int] = (100, 200),
) -> Dict[str, float]:
    t = np.array([p.time for p in active], dtype=float)
    y = np.array([p.torque for p in active], dtype=float)

    # Windows from onset time
    e_start, e_end = onset_time + early_ms[0]/1000.0, onset_time + early_ms[1]/1000.0
    l_start, l_end = onset_time + late_ms[0]/1000.0,  onset_time + late_ms[1]/1000.0

    ei = _slice_by_time_window(t, e_start, e_end)
    li = _slice_by_time_window(t, l_start, l_end)

    rtd_early = _linreg_slope(t[ei], y[ei]) if len(ei) >= 2 else float("nan")
    rtd_late  = _linreg_slope(t[li], y[li]) if len(li) >= 2 else float("nan")

    return {
        "RTD_early_0_75ms": float(rtd_early),   # Nm/s (or Nm/kg/s if normalized)
        "RTD_late_100_200ms": float(rtd_late),
        "RTD_early_window_pts": int(len(ei)),
        "RTD_late_window_pts": int(len(li)),
    }

# ----------------------------
# Updated plateau detection using new method
# ----------------------------

def detect_plateau_window_new(
    active: List[TorqueDataPoint],
    sr: int,
    confidence_level: float = 0.99,
    min_duration: float = 0.5  # seconds
) -> Dict[str, int | float]:
    """
    New plateau detection using the TypeScript method.
    """
    detector = PlateauDetector(active, sr, min_duration)
    try:
        result = detector.detect_plateau(confidence_level)
        if result is None:
            # Fallback to original method if new method fails
            print("New plateau detection found no plateau, falling back to original method")
            return detect_plateau_window(active, sr)
        
        # Convert to original format for compatibility
        return {
            "start_idx": result.start_index,
            "end_idx": result.end_index,
            "mean": mean([p.torque for p in result.plateau]),
            "std": math.sqrt(variance([p.torque for p in result.plateau])),
            "cov": result.cov,
            "peak_idx": max(range(len(active)), key=lambda i: active[i].torque),
            "peak_time": active[max(range(len(active)), key=lambda i: active[i].torque)].time,
            "duration": result.duration,
            "gradient": result.gradient,
            "rmse": result.rmse,
            "r_squared": result.r_squared,
            "max_yank": result.max_yank,
            "mean_yank": result.mean_yank
        }
    except Exception as e:
        print(f"New plateau detection failed: {e}, falling back to original method")
        return detect_plateau_window(active, sr)

def detect_plateau_window(
    active: List[TorqueDataPoint],
    sr: int,
    window_ms: int = 500,                 # candidate plateau window length
    step_ms: int = 25,                    # slide step
    search_start_pct_of_peak_time: float = 0.2,  # start search after 20% of (onset->peak) by default
    cov_cap: Optional[float] = None       # optional CoV filter (e.g., 0.10 = 10%); if None, pick min CoV
) -> Dict[str, int | float]:
    """
    Original sliding window plateau detection (kept for backward compatibility).
    """
    if len(active) < 5:
        return {"start_idx": 0, "end_idx": len(active)-1}

    t = np.array([p.time for p in active], dtype=float)
    y = np.array([p.torque for p in active], dtype=float)

    # find peak time within active
    peak_idx = int(np.argmax(y))
    peak_time = t[peak_idx]

    # search region (after early rise)
    t0 = t[0]
    # estimate an onset->peak duration
    onset_peak_dur = max(1e-6, peak_time - t0)
    search_start_time = t0 + search_start_pct_of_peak_time * onset_peak_dur

    # Convert to samples
    win = _ms_to_samples(window_ms, sr)
    step = _ms_to_samples(step_ms, sr)
    if win < 3:
        win = 3

    # candidate indices
    candidates = []
    start_i = int(np.searchsorted(t, search_start_time, side="left"))
    for s in range(start_i, len(y) - win + 1, step):
        e = s + win
        seg = y[s:e]
        m = float(np.mean(seg))
        sd = float(np.std(seg, ddof=1)) if len(seg) > 1 else 0.0
        cov = (sd / m) if m != 0 else float("inf")
        candidates.append((s, e, m, sd, cov))

    if not candidates:
        return {"start_idx": 0, "end_idx": len(active)-1}

    # choose by minimal CoV; if cov_cap provided, only consider segments below it; tie-break by higher mean
    if cov_cap is not None:
        filtered = [c for c in candidates if c[4] <= cov_cap]
        pool = filtered if filtered else candidates
    else:
        pool = candidates

    pool.sort(key=lambda x: (x[4], -x[2]))  # min CoV, then max mean
    s, e, m, sd, cov = pool[0]

    return {
        "start_idx": int(s),
        "end_idx": int(e - 1),
        "mean": float(m),
        "std": float(sd),
        "cov": float(cov),
        "peak_idx": int(peak_idx),
        "peak_time": float(peak_time),
    }

def compute_plateau_features(
    active: List[TorqueDataPoint],
    sr: int,
    confidence_level: float = 0.99,
    use_new_method: bool = True,
    window_ms: int = 500,
    step_ms: int = 25,
    cov_cap: Optional[float] = None
) -> Dict[str, float | int]:
    """
    Detect plateau and report features. 
    Uses new TypeScript-based method by default, with fallback to original method.
    """
    if not active:
        return {}

    t = np.array([p.time for p in active], dtype=float)
    y = np.array([p.torque for p in active], dtype=float)

    # Use new method by default
    if use_new_method:
        pl = detect_plateau_window_new(active, sr, confidence_level)
    else:
        pl = detect_plateau_window(active, sr, window_ms, step_ms, cov_cap=cov_cap)
    
    s, e = int(pl["start_idx"]), int(pl["end_idx"])
    s = max(0, min(s, len(y)-1))
    e = max(s, min(e, len(y)-1))

    tp = t[s:e+1]
    yp = y[s:e+1]

    # Calculate features
    if use_new_method and "gradient" in pl:
        # Use values from new detection method
        slope = pl["gradient"]
        yank_mean_abs = abs(pl.get("mean_yank", 0.0))
        yank_peak_abs = pl.get("max_yank", 0.0)
        rms_resid = pl.get("rmse", 0.0)
    else:
        # Calculate traditional features
        slope = _linreg_slope(tp, yp)  # Nm/s
        # dτ/dt stats ("yank")
        d = _first_derivative(yp, tp)
        yank_mean_abs = float(np.mean(np.abs(d)))
        yank_peak_abs = float(np.max(np.abs(d)))
        # steadiness vs trend (RMS of residuals from best linear fit)
        a = slope
        b = float(yp.mean() - a * tp.mean())
        resid = yp - (a * tp + b)
        rms_resid = float(np.sqrt(np.mean(resid**2)))
    
    # Common calculations
    spec_ent = _spectral_entropy(yp - (slope * tp + (yp.mean() - slope * tp.mean())))
    d = _first_derivative(yp, tp) if not (use_new_method and "gradient" in pl) else None
    zc_d = _zero_crossings(d) if d is not None else 0

    return {
        "plateau_start_idx": s,
        "plateau_end_idx": e,
        "plateau_duration_s": float(tp[-1] - tp[0]) if len(tp) > 1 else 0.0,
        "plateau_mean": float(yp.mean()),
        "plateau_std": float(yp.std(ddof=1)) if len(yp) > 1 else 0.0,
        "plateau_cov": float((yp.std(ddof=1) / (yp.mean() if yp.mean()!=0 else np.nan))),
        "plateau_start_torque": float(yp[0]),
        "plateau_end_torque": float(yp[-1]),
        "plateau_range": float(yp.max() - yp.min()),
        "plateau_gradient": float(slope),          # Nm/s
        "plateau_yank_mean_abs": yank_mean_abs,    # mean |dτ/dt|
        "plateau_yank_peak_abs": yank_peak_abs,    # max |dτ/dt|
        "plateau_rms_residual": rms_resid,         # steadiness vs linear trend
        "plateau_spectral_entropy": float(spec_ent),
        "plateau_dtorque_zero_crossings": int(zc_d),
        "plateau_peak_idx_within_active": int(pl.get("peak_idx", np.argmax(y))),
        "plateau_peak_time_within_active": float(pl.get("peak_time", t[np.argmax(y)])),
        # Additional features from new method
        "plateau_r_squared": float(pl.get("r_squared", 0.0)),
        "plateau_method": "new-variance-ci" if use_new_method else "sliding-window",
    }

# ----------------------------
# Master feature extractor
# ----------------------------

def extract_features_for_active_segment(
    active: List[TorqueDataPoint],
    onset_time: float,
    sr: int,
    plateau_window_ms: int = 500,
    plateau_step_ms: int = 25,
    plateau_cov_cap: Optional[float] = None
) -> Dict[str, float | int]:
    """
    Given an 'active' segment (onset→end) and the onset_time, compute:
      - RTD_early (0–75 ms) and RTD_late (100–200 ms)
      - Plateau features (steadiness & dynamics)
    Return dict of features.
    """
    rtd = compute_rtd_features(active, onset_time, sr, (0, 75), (100, 200))
    pl = compute_plateau_features(
        active,
        sr,
        confidence_level=0.99,
        use_new_method=True,
        window_ms=plateau_window_ms,
        step_ms=plateau_step_ms,
        cov_cap=plateau_cov_cap,
    )
    return {**rtd, **pl}

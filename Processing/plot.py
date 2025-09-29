import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from features import *

from preprocessing import *

def plot_active_window_with_features(
    raw: list[TorqueDataPoint],
    cleaned: list[TorqueDataPoint],
    active: list[TorqueDataPoint],
    window_info: dict
):
    """
    Plots raw signal, cleaned signal, and highlights the detected active window,
    plus RTD periods and plateau detection using features.py functions.
    Also draws regression lines for RTD early, late, and plateau periods.
    window_info: dict returned by detect_active_window_indices(...)
    """

    # extract arrays
    t_raw = np.array([p.time for p in raw])
    y_raw = np.array([p.torque for p in raw])

    t_clean = np.array([p.time for p in cleaned])
    y_clean = np.array([p.torque for p in cleaned])

    t_active = np.array([p.time for p in active])
    y_active = np.array([p.torque for p in active])

    # Extract features from active segment
    onset_time = window_info["startTime"]
    sampling_rate = 1000  # from your main code
    
    # Get plateau features
    plateau_features = compute_plateau_features(active, sampling_rate)
    
    # Get RTD features
    rtd_features = compute_rtd_features(active, onset_time, sampling_rate)
    
    # make plot
    plt.figure(figsize=(14, 8))
    plt.plot(t_raw, y_raw, color="gray", alpha=0.5, label="Raw")
    plt.plot(t_clean, y_clean, color="blue", label="Cleaned")
    plt.plot(t_active, y_active, color="red", linewidth=2.5, label="Active window")

    # mark onset/end
    plt.axvline(window_info["startTime"], color="green", linestyle="--", label="Onset")
    plt.axvline(window_info["endTime"], color="purple", linestyle="--", label="End")

    # thresholds for reference
    plt.axhline(window_info["startThresholdUsed"], color="green", linestyle=":", alpha=0.6)
    plt.axhline(window_info["endThresholdUsed"], color="purple", linestyle=":", alpha=0.6)

    # RTD Early period (0-75ms from onset)
    rtd_early_start = onset_time
    rtd_early_end = onset_time + 0.075  # 75ms
    plt.axvspan(rtd_early_start, rtd_early_end, alpha=0.3, color="orange", label="RTD Early (0-75ms)")
    
    # RTD Late period (100-200ms from onset)
    rtd_late_start = onset_time + 0.100  # 100ms
    rtd_late_end = onset_time + 0.200    # 200ms
    plt.axvspan(rtd_late_start, rtd_late_end, alpha=0.3, color="yellow", label="RTD Late (100-200ms)")
    
    # Draw RTD regression lines
    def draw_rtd_regression(start_time, end_time, slope, color, label):
        # Find data points within the time window
        mask = (t_active >= start_time) & (t_active <= end_time)
        if np.any(mask):
            t_segment = t_active[mask]
            y_segment = y_active[mask]
            if len(t_segment) >= 2 and not np.isnan(slope):
                # Calculate intercept from the mean point
                t_mean = np.mean(t_segment)
                y_mean = np.mean(y_segment)
                intercept = y_mean - slope * t_mean
                
                # Draw regression line
                y_reg = slope * t_segment + intercept
                plt.plot(t_segment, y_reg, color=color, linestyle='-', linewidth=2.5, alpha=0.8, label=label)
    
    # RTD Early regression line
    rtd_early_slope = rtd_features.get('RTD_early_0_75ms', float('nan'))
    draw_rtd_regression(rtd_early_start, rtd_early_end, rtd_early_slope, "darkorange", "RTD Early Fit")
    
    # RTD Late regression line  
    rtd_late_slope = rtd_features.get('RTD_late_100_200ms', float('nan'))
    draw_rtd_regression(rtd_late_start, rtd_late_end, rtd_late_slope, "gold", "RTD Late Fit")
    
    # Plateau window and regression
    if 'plateau_start_idx' in plateau_features and 'plateau_end_idx' in plateau_features:
        plateau_start_idx = plateau_features['plateau_start_idx']
        plateau_end_idx = plateau_features['plateau_end_idx']
        
        if plateau_start_idx < len(t_active) and plateau_end_idx < len(t_active):
            plateau_start_time = t_active[plateau_start_idx]
            plateau_end_time = t_active[plateau_end_idx]
            
            # Highlight plateau region
            plt.axvspan(plateau_start_time, plateau_end_time, alpha=0.2, color="cyan", label="Plateau Window")
            
            # Plot plateau segment with different styling
            plateau_t = t_active[plateau_start_idx:plateau_end_idx+1]
            plateau_y = y_active[plateau_start_idx:plateau_end_idx+1]
            plt.plot(plateau_t, plateau_y, color="cyan", linewidth=3, alpha=0.8)
            
            # Get confidence bounds from the plateau detector
            try:
                # Run plateau detector to get confidence bounds
                detector = PlateauDetector(active, sampling_rate, 0.5)  # min 0.5s duration
                plateau_result = detector.detect_plateau(0.95)  # 95% confidence
                
                if plateau_result and plateau_result.upper_ci and plateau_result.lower_ci:
                    # The CI bounds correspond to the original detected segment before CI trimming
                    # We need to find the original segment that was used for regression
                    original_plateau_data = plateau_result.plateau  # This is the trimmed data
                    
                    # Get the time range for the confidence interval visualization
                    # This should match the time range of the original regression
                    ci_start_time = plateau_result.start_time
                    ci_end_time = plateau_result.end_time
                    
                    # Create time array for CI bounds (we'll reconstruct this from the regression data)
                    # The CI bounds array length matches the original segment before trimming
                    ci_segment_length = len(plateau_result.upper_ci)
                    
                    # Find the corresponding indices in the active array for the full CI segment
                    full_ci_start_idx = None
                    for i, point in enumerate(active):
                        if abs(point.time - ci_start_time) < 0.001:  # Small tolerance for float comparison
                            # Work backwards to find the start of the original segment
                            full_ci_start_idx = max(0, i - (ci_segment_length - len(original_plateau_data)) // 2)
                            break
                    
                    if full_ci_start_idx is not None:
                        ci_segment = active[full_ci_start_idx:full_ci_start_idx + ci_segment_length]
                        ci_t = np.array([p.time for p in ci_segment])
                        
                        if len(ci_t) == len(plateau_result.upper_ci):
                            # Plot confidence bounds
                            plt.fill_between(ci_t, plateau_result.lower_ci, plateau_result.upper_ci,
                                           alpha=0.15, color="lightblue", label="95% Confidence Bounds")
                            
                            # Plot the bounds as lines for better visibility
                            plt.plot(ci_t, plateau_result.upper_ci, color="lightsteelblue", 
                                   linestyle="--", linewidth=1.5, alpha=0.8, label="Upper CI")
                            plt.plot(ci_t, plateau_result.lower_ci, color="lightsteelblue",
                                   linestyle="--", linewidth=1.5, alpha=0.8, label="Lower CI")
                            
                            # Show which points were trimmed (outside CI) with red X markers
                            trimmed_points = []
                            for i, point in enumerate(ci_segment):
                                if (point.torque < plateau_result.lower_ci[i] or 
                                    point.torque > plateau_result.upper_ci[i]):
                                    trimmed_points.append(point)
                                    plt.plot(point.time, point.torque, 'rx', markersize=8, alpha=0.7)
                            
                            if trimmed_points:
                                # Add label for trimmed points (only once)
                                plt.plot([], [], 'rx', markersize=8, alpha=0.7, label="Trimmed Points")
                
            except Exception as e:
                print(f"Could not get confidence bounds: {e}")
            
            # Draw plateau regression line
            plateau_slope = plateau_features.get('plateau_gradient', 0.0)
            if not np.isnan(plateau_slope) and len(plateau_t) >= 2:
                # Calculate intercept from the mean point
                t_mean = np.mean(plateau_t)
                y_mean = np.mean(plateau_y)
                intercept = y_mean - plateau_slope * t_mean
                
                # Draw plateau regression line
                y_plateau_reg = plateau_slope * plateau_t + intercept
                plt.plot(plateau_t, y_plateau_reg, color="darkturquoise", linestyle='-', linewidth=3, alpha=0.9, label="Plateau Fit")

    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.title("Torque–Time Curve with Active Window, RTD Periods, Plateau Detection & Regression Lines")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Print feature values for reference
    print("\n=== RTD Features ===")
    print(f"RTD Early (0-75ms): {rtd_features.get('RTD_early_0_75ms', 'N/A'):.2f} Nm/s")
    print(f"RTD Late (100-200ms): {rtd_features.get('RTD_late_100_200ms', 'N/A'):.2f} Nm/s")
    print(f"Early window points: {rtd_features.get('RTD_early_window_pts', 'N/A')}")
    print(f"Late window points: {rtd_features.get('RTD_late_window_pts', 'N/A')}")
    
    print("\n=== Plateau Features ===")
    print(f"Plateau mean torque: {plateau_features.get('plateau_mean', 'N/A'):.2f} Nm")
    print(f"Plateau CoV: {plateau_features.get('plateau_cov', 'N/A'):.4f}")
    print(f"Plateau duration: {plateau_features.get('plateau_duration_s', 'N/A'):.3f} s")
    print(f"Plateau gradient: {plateau_features.get('plateau_gradient', 'N/A'):.3f} Nm/s")
    print(f"Plateau steadiness (RMS residual): {plateau_features.get('plateau_rms_residual', 'N/A'):.3f} Nm")
    print(f"Plateau R²: {plateau_features.get('plateau_r_squared', 'N/A'):.4f}")
    print(f"Detection method: {plateau_features.get('plateau_method', 'N/A')}")
    
    plt.show()

# Keep the original function for backward compatibility
def plot_active_window(
    raw: list[TorqueDataPoint],
    cleaned: list[TorqueDataPoint],
    active: list[TorqueDataPoint],
    window_info: dict
):
    """
    Original plotting function - calls the enhanced version.
    """
    plot_active_window_with_features(raw, cleaned, active, window_info)

df = pd.read_csv("../Data/Raw/6_Right_ACLR_0_mvic1.csv")

# extract arrays
raw_torque_array = df.iloc[:, 0].to_numpy()
raw = [TorqueDataPoint(time=i/1000.0, torque=float(val)) for i, val in enumerate(raw_torque_array)]

# 2) clean with your same chain
cleaned = process_data_with_cleaning(raw, sampling_rate_hz=100, cutoff_hz=20, order=4, ma_window_s=0.15, hampel_window_s=0.4)

# 3) detect & slice the active window (onset + end, symmetric logic)
active = slice_active_window_using_detectors(
    cleaned,
    sampling_rate_hz=1000,
    start_threshold_nm=3.0,            # or omit and use % of peak via start_threshold_pct_of_peak
    end_threshold_nm=3.0,    # 25% of peak after the peak
    min_rise_window_ms=20,
    min_fall_window_ms=20,
    walk_back_max_ms=300,
    walk_fwd_max_ms=300,
    local_min_neighborhood=3
)

window_info = detect_active_window_indices(cleaned, sampling_rate_hz=1000)

# Use the enhanced plotting function that includes RTD and plateau features
plot_active_window_with_features(raw, cleaned, active, window_info)
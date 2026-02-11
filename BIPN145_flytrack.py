#!/usr/bin/env python3
"""
BIPN 145 Fly Tracker

Tracks a fruit fly against a light background in a video, calculates
its position and velocity over time, and produces path/velocity plots.

Original MATLAB code by Jeff Stafford, modified by A. Juavinett for BIPN 145.
Python port for standalone use.

Usage:
    python BIPN145_flytrack.py                  # interactive mode (file picker + prompts)
    python BIPN145_flytrack.py video1.avi       # command-line mode with defaults
    python BIPN145_flytrack.py video1.avi video2.avi --diameter 4 --frame-rate 30
"""

# --- Dependency check: auto-install if missing ---
import subprocess
import sys


def check_install(package, import_name=None):
    try:
        __import__(import_name or package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


check_install("numpy")
check_install("opencv-python", "cv2")
check_install("matplotlib")

import argparse
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def fly_finder(roi_image, half_search, threshold, flip=True):
    """
    Find a fly (dark region) in a grayscale image.
    Returns (x, y) position or (NaN, NaN) if not found.
    """
    if flip:
        val = np.nanmin(roi_image)
    else:
        val = np.nanmax(roi_image)

    ys, xs = np.where(roi_image == val)
    xpos = np.mean(xs)
    ypos = np.mean(ys)

    h, w = roi_image.shape
    left = max(int(round(xpos) - half_search), 0)
    right = min(int(round(xpos) + half_search), w - 1)
    top = max(int(round(ypos) - half_search), 0)
    bottom = min(int(round(ypos) + half_search), h - 1)

    search_area = roi_image[top:bottom + 1, left:right + 1].astype(np.float64)

    if flip:
        search_area = 255.0 - search_area

    total = np.sum(search_area)

    if total >= threshold:
        x_indices = np.arange(search_area.shape[1])
        y_indices = np.arange(search_area.shape[0])
        x = np.sum(search_area @ x_indices) / total + left
        y = np.sum(search_area.T @ y_indices) / total + top
        return x, y
    else:
        return np.nan, np.nan


def dist_filter(array, tele_dist_threshold, num_avg=5):
    """
    Teleport filter: removes spurious points where fly position
    jumps far from the mean of surrounding frames.
    """
    filtered = array.copy()
    tele_count = 0

    for i in range(num_avg, len(filtered) - num_avg):
        point = filtered[i, 1:3]
        if np.any(np.isnan(point)):
            continue

        last_set = filtered[i - num_avg:i, 1:3]
        last_set = last_set[~np.isnan(last_set[:, 0])]
        if len(last_set) == 0:
            continue
        last_mean = np.mean(last_set, axis=0)

        next_set = filtered[i + 1:i + 1 + num_avg, 1:3]
        next_set = next_set[~np.isnan(next_set[:, 0])]
        if len(next_set) == 0:
            continue
        next_mean = np.mean(next_set, axis=0)

        if (np.linalg.norm(point - last_mean) > tele_dist_threshold or
                np.linalg.norm(point - next_mean) > tele_dist_threshold):
            filtered[i, 1:3] = np.nan
            tele_count += 1

    # More stringent check at start and end
    for idx in list(range(0, min(5, len(filtered) - 1))) + \
               list(range(max(0, len(filtered) - 6), len(filtered) - 1)):
        if np.any(np.isnan(filtered[idx, 1:3])) or np.any(np.isnan(filtered[idx + 1, 1:3])):
            continue
        if np.linalg.norm(filtered[idx, 1:3] - filtered[idx + 1, 1:3]) > tele_dist_threshold / 2:
            filtered[idx, 1:3] = np.nan
            tele_count += 1

    print(f"  {tele_count} points removed by the teleportation filter.")
    return filtered


def interpolate_pos(array, inter_dist_threshold):
    """
    Linearly interpolate fly position between NaN gaps,
    as long as the gap endpoints are within inter_dist_threshold.
    """
    result = array.copy()
    interp_count = 0

    col_pairs = [(1, 2)]
    for cx, cy in col_pairs:
        i = 0
        while i < len(result):
            if np.isnan(result[i, cx]) and i > 0:
                last_idx = i - 1
                last_point = result[last_idx, cx:cy + 1]
                remaining = result[i:, cx]
                non_nan = np.where(~np.isnan(remaining))[0]
                if len(non_nan) == 0:
                    break
                next_idx = non_nan[0] + i
                next_point = result[next_idx, cx:cy + 1]
                gap = next_idx - i

                if np.linalg.norm(last_point - next_point) <= inter_dist_threshold:
                    for j in range(1, gap + 1):
                        frac = j / (gap + 1)
                        result[last_idx + j, cx:cy + 1] = last_point + (next_point - last_point) * frac
                    interp_count += gap

                i = next_idx
            elif np.isnan(result[i, cx]) and i == 0:
                non_nan = np.where(~np.isnan(result[:, cx]))[0]
                if len(non_nan) == 0:
                    break
                i = non_nan[0]
            else:
                i += 1

    print(f"  {interp_count} points recovered through interpolation.")
    return result


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------

def select_roi(video_path):
    """
    Display the first frame and let the user draw an ROI with cv2.selectROI.
    Returns (x, y, w, h).
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read first frame of {video_path}")

    print("Draw a rectangle around the dish, then press ENTER or SPACE to confirm.")
    print("Press C to cancel and redraw.")
    roi = cv2.selectROI("Select ROI — press ENTER to confirm", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    if roi[2] == 0 or roi[3] == 0:
        raise ValueError("No ROI selected. Please try again.")

    return roi


def process_video(video_path, roi, diameter, frame_rate, search_size, per_pixel_threshold):
    """
    Process a single fly video and return the corrected position array.
    Returns Nx3 array [time_s, x_cm, y_cm].
    """
    height = diameter
    width = diameter
    roi_x, roi_y, roi_w, roi_h = roi

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    nfrm = total_frames - 1

    print(f"\nProcessing: {os.path.basename(video_path)}")
    print(f"  Video FPS: {fps}, Total frames: {total_frames}")
    print(f"  Using frame_rate parameter: {frame_rate} for time conversion")

    # --- Create background from 100 random frames ---
    print("  Calculating background...")
    bg_number = min(100, nfrm)
    bg_indices = sorted(np.random.choice(nfrm, bg_number, replace=False))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, sample = cap.read()
    gray_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    bg_array = np.zeros((*gray_sample.shape, bg_number), dtype=np.uint8)

    for idx, frame_num in enumerate(bg_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            bg_array[:, :, idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    background = np.mean(bg_array, axis=2).astype(np.uint8)

    # --- Process each frame ---
    print("  Tracking fly positions...")
    threshold = (search_size ** 2) * per_pixel_threshold
    half_search = round(search_size / 2)

    pos_array = np.zeros((nfrm, 3))

    for nofr in range(nfrm):
        cap.set(cv2.CAP_PROP_POS_FRAMES, nofr)
        ret, frame = cap.read()
        if not ret:
            pos_array[nofr] = [nofr, np.nan, np.nan]
            continue

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # Background subtraction using GIMP division formula
        frame_div = np.clip(
            (256.0 * frame_gray) / (background.astype(np.float64) + 1), 0, 255
        ).astype(np.uint8)

        # Crop to ROI
        frame_crop = frame_div[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # Find fly
        fx, fy = fly_finder(frame_crop, half_search, threshold, flip=True)
        pos_array[nofr] = [nofr, fx, fy]

        # Progress update every 10%
        if (nofr + 1) % max(1, nfrm // 10) == 0:
            pct = (nofr + 1) / nfrm * 100
            print(f"    {pct:.0f}% complete ({nofr + 1}/{nfrm} frames)")

    cap.release()

    # --- Convert to real coordinates ---
    xscale = width / roi_w
    yscale = height / roi_h

    corrected_array = np.column_stack([
        pos_array[:, 0] / frame_rate,
        pos_array[:, 1] * xscale,
        pos_array[:, 2] * yscale,
    ])

    skipped = np.sum(np.isnan(corrected_array[:, 1]))
    print(f"  {skipped} points skipped out of {nfrm}.")

    # Apply teleport filter and interpolation
    corrected_array = dist_filter(corrected_array, 2)
    corrected_array = interpolate_pos(corrected_array, 2)

    # Manual fix for 15fps videos (matching MATLAB behavior)
    if frame_rate == 15:
        corrected_array[:, 0] = corrected_array[:, 0] / 4

    return corrected_array


# ---------------------------------------------------------------------------
# Velocity calculation
# ---------------------------------------------------------------------------

def calculate_velocity(corrected, frame_rate, bin_size):
    """
    Calculate binned velocity in mm/s from a corrected position array.
    Returns (time_axis, velocity) arrays.
    """
    total_time = len(corrected) / frame_rate
    total_bins = int(np.floor(total_time / bin_size))

    if corrected[1, 0] > 0:
        data_rate = int(round(1.0 / corrected[1, 0]) * bin_size)
    else:
        data_rate = int(frame_rate * bin_size)

    if data_rate < 1:
        raise ValueError("bin_size is smaller than the minimum data rate.")

    velocity = np.zeros(total_bins)
    for row in range(0, len(corrected) - data_rate, data_rate):
        bin_idx = row // data_rate
        if bin_idx >= total_bins:
            break
        p1 = corrected[row, 1:3]
        p2 = corrected[row + data_rate, 1:3]
        if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
            velocity[bin_idx] = np.nan
        else:
            velocity[bin_idx] = 10.0 * np.linalg.norm(p1 - p2)

    velocity = velocity / bin_size
    time_axis = np.arange(total_bins) * bin_size
    return time_axis, velocity


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_path(corrected, video_name, diameter):
    """Plot the fly path color-coded by time."""
    x = corrected[:, 1]
    y = corrected[:, 2]
    t = corrected[:, 0]
    valid = ~np.isnan(x) & ~np.isnan(y)

    fig, ax = plt.subplots(figsize=(6, 6))
    points = np.array([x[valid], y[valid]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap="viridis", linewidth=2)
    lc.set_array(t[valid][:-1])
    ax.add_collection(lc)

    ax.set_xlim(0, diameter)
    ax.set_ylim(diameter, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("X-coordinate (cm)", fontsize=11)
    ax.set_ylabel("Y-coordinate (cm)", fontsize=11)
    ax.set_title(f"Fly Path — {video_name}")

    cbar = fig.colorbar(lc, ax=ax, orientation="horizontal", pad=0.1)
    cbar.set_label("Time (s)")

    plt.tight_layout()
    plt.show()


def plot_velocity(time_axis, velocity, video_name, bin_size):
    """Plot velocity over time for a single video."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_axis, velocity, linewidth=1.5)
    ax.set_xlim(0, time_axis[-1] + bin_size if len(time_axis) > 0 else 1)
    max_vel = np.nanmax(velocity)
    ax.set_ylim(0, max_vel * 1.5 if max_vel > 0 else 1)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Velocity (mm/s)", fontsize=11)
    ax.set_title(f"Fly Velocity — {video_name}")
    plt.tight_layout()
    plt.show()


def plot_all_velocities(all_velocity, bin_size):
    """Plot all fly velocities on one figure."""
    if len(all_velocity) < 2:
        return

    max_len = max(len(v) for v in all_velocity)
    time_axis = np.arange(max_len) * bin_size

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, vel in enumerate(all_velocity):
        padded = np.full(max_len, np.nan)
        padded[:len(vel)] = vel
        ax.plot(time_axis, padded, linewidth=2, label=f"Fly {i + 1}")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Velocity (mm/s)", fontsize=11)
    ax.set_title("All Fly Velocities")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prompt_param(name, default, cast=float):
    """Prompt the user for a parameter, returning default if they press Enter."""
    raw = input(f"  {name} [{default}]: ").strip()
    if raw == "":
        return cast(default)
    return cast(raw)


def interactive_mode():
    """
    Fully interactive mode: file picker + prompted parameters.
    Returns (video_files, diameter, frame_rate, bin_size, search_size, threshold).
    """
    print("\n=== BIPN 145 Fly Tracker ===\n")

    # --- File picker using tkinter ---
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()  # hide the root window
        root.attributes("-topmost", True)  # bring dialog to front
        print("Select your video file(s) in the file dialog...")
        video_files = list(filedialog.askopenfilenames(
            title="Select video(s) to analyze",
            filetypes=[
                ("Video files", "*.avi *.mp4 *.mov *.mkv *.mj2 *.mpg *.m4v"),
                ("All files", "*.*"),
            ],
        ))
        root.destroy()
    except Exception:
        # Fallback if tkinter is unavailable (rare, but possible on some Linux)
        print("Could not open file dialog. Enter video path(s) manually.")
        paths = input("Video file path(s), separated by commas: ").strip()
        video_files = [p.strip() for p in paths.split(",") if p.strip()]

    if not video_files:
        print("No files selected. Exiting.")
        sys.exit(0)

    print(f"\n{len(video_files)} file(s) selected:")
    for vf in video_files:
        print(f"  • {os.path.basename(vf)}")

    # --- Prompt for parameters (Enter accepts default) ---
    print("\nSet parameters (press Enter to accept default):\n")
    diameter = prompt_param("Dish diameter in cm", 4, float)
    frame_rate = prompt_param("Frame rate (fps)", 30, int)
    bin_size = prompt_param("Velocity bin size (s)", 1, float)
    search_size = prompt_param("Search size (px)", 20, int)
    threshold = prompt_param("Per-pixel threshold", 1.5, float)

    return video_files, diameter, frame_rate, bin_size, search_size, threshold


def main():
    # If no arguments provided, run in interactive mode
    if len(sys.argv) == 1:
        video_files, diameter, frame_rate, bin_size, search_size, threshold = interactive_mode()
    else:
        parser = argparse.ArgumentParser(
            description="BIPN 145 Fly Tracker — track fly position and velocity from video."
        )
        parser.add_argument("videos", nargs="+", help="Video file(s) to analyze")
        parser.add_argument("--diameter", type=float, default=4,
                            help="Diameter of the dish in cm (default: 4)")
        parser.add_argument("--frame-rate", type=int, default=30,
                            help="Frame rate in fps (default: 30)")
        parser.add_argument("--bin-size", type=float, default=1,
                            help="Velocity bin size in seconds (default: 1)")
        parser.add_argument("--search-size", type=int, default=20,
                            help="Search area size in pixels (default: 20)")
        parser.add_argument("--threshold", type=float, default=1.5,
                            help="Per-pixel intensity threshold (default: 1.5)")
        args = parser.parse_args()
        video_files = args.videos
        diameter = args.diameter
        frame_rate = args.frame_rate
        bin_size = args.bin_size
        search_size = args.search_size
        threshold = args.threshold

    # Validate video files exist
    for vf in video_files:
        if not os.path.isfile(vf):
            print(f"Error: file not found: {vf}")
            sys.exit(1)

    print(f"\n{len(video_files)} video(s) selected for analysis.")
    print(f"  Diameter: {diameter} cm, Frame rate: {frame_rate} fps")
    print(f"  Bin size: {bin_size} s, Search size: {search_size} px")

    # Select ROI from the first video
    roi = select_roi(video_files[0])
    print(f"ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")

    # Process all videos
    all_corrected = []
    all_velocity = []

    for vf in video_files:
        corrected = process_video(vf, roi, diameter, frame_rate,
                                  search_size, threshold)
        all_corrected.append(corrected)

        # Save tracking CSV
        base = os.path.splitext(vf)[0]
        csv_name = f"{base}_tracking.csv"
        np.savetxt(csv_name, corrected, delimiter=",",
                   header="Time_s,X_cm,Y_cm", comments="")
        print(f"  Saved {csv_name}")

        # Calculate velocity
        time_axis, velocity = calculate_velocity(corrected, frame_rate, bin_size)
        all_velocity.append(velocity)

        # Warn about absurd velocities
        if np.nanmax(velocity) > 30:
            print(f"  WARNING: Absurdly high velocities detected in {os.path.basename(vf)}.")
            print("    Consider changing the ROI or re-recording the video.")

        mean_vel = np.nanmean(velocity)
        std_vel = np.nanstd(velocity)
        print(f"\n  --- {os.path.basename(vf)} ---")
        print(f"  Mean velocity: {mean_vel:.2f} mm/s")
        print(f"  Std deviation: {std_vel:.2f} mm/s")

        # Save velocity CSV
        vel_csv = f"{base}_velocity.csv"
        vel_data = np.column_stack([time_axis, velocity])
        np.savetxt(vel_csv, vel_data, delimiter=",",
                   header="Time_s,Velocity_mm_per_s", comments="")
        print(f"  Saved {vel_csv}")

        # Plot
        plot_path(corrected, os.path.basename(vf), diameter)
        plot_velocity(time_axis, velocity, os.path.basename(vf), bin_size)

    # Summary across videos
    num_files = len(all_velocity)
    if num_files > 1:
        per_video_means = [np.nanmean(v) for v in all_velocity]
        mean_across = np.mean(per_video_means)
        sd_across = np.std(per_video_means)
        print(f"\n=== Summary Across {num_files} Videos ===")
        print(f"Mean velocity across videos: {mean_across:.2f} mm/s")
        print(f"SD of mean velocity across videos: {sd_across:.2f} mm/s")
        plot_all_velocities(all_velocity, bin_size)
    else:
        mean_vel = np.nanmean(all_velocity[0])
        sd_vel = np.nanstd(all_velocity[0])
        print(f"\n=== Summary (1 Video) ===")
        print(f"Mean velocity: {mean_vel:.2f} mm/s")
        print(f"SD of velocity: {sd_vel:.2f} mm/s")

    print("\nDone!")


if __name__ == "__main__":
    main()

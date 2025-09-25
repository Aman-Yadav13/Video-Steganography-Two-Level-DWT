import cv2
import numpy as np
import os
import time
from tqdm import tqdm

# Import the functions we've already built
from video_processor import split_video_to_frames
from metrics import calculate_metrics
from analysis import plot_psnr_per_frame, plot_original_vs_stego_frame, plot_y_channel_histogram

def analyze_videos():
    """
    Analyzes an existing stego video against its original to generate
    performance metrics and plots without re-running the embedding process.
    """
    # --- 1. Configuration ---
    # Point to the two videos you want to compare
    ORIGINAL_VIDEO_PATH = '../data/input_video.mp4'
    STEGO_VIDEO_PATH = '../output/stego_video.mp4'
    
    # Ensure the output directory exists for the plots
    os.makedirs('../output', exist_ok=True)

    print("🚀 Starting analysis of existing videos...")
    start_time = time.time()

    # --- 2. Load Frames from Both Videos ---
    print("Loading original video frames...")
    original_frames = split_video_to_frames(ORIGINAL_VIDEO_PATH)
    
    print("Loading stego video frames...")
    stego_frames = split_video_to_frames(STEGO_VIDEO_PATH)

    if not original_frames or not stego_frames:
        print("❌ Error: Could not load one or both videos.")
        return

    if len(original_frames) != len(stego_frames):
        print("⚠️ Warning: Videos have a different number of frames. Analysis will be done on the shorter length.")
        min_frames = min(len(original_frames), len(stego_frames))
        original_frames = original_frames[:min_frames]
        stego_frames = stego_frames[:min_frames]

    # --- 3. Calculate Metrics Frame-by-Frame ---
    psnr_scores = []
    ssim_scores = []

    print("\nCalculating metrics for each frame...")
    for original_frame, stego_frame in tqdm(zip(original_frames, stego_frames), total=len(original_frames)):
        psnr_val, ssim_val = calculate_metrics(original_frame, stego_frame)
        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)

    # --- 4. Display Final Metrics ---
    end_time = time.time()
    runtime = end_time - start_time
    
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)

    print("\n--- 📊 Analysis Completed ---")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Total Analysis Runtime: {runtime:.2f} seconds")

    # --- 5. Generate and Save Plots ---
    print("\n--- 📈 Generating Performance Plots ---")
    
    # Plot 1: PSNR per frame
    plot_psnr_per_frame(psnr_scores, len(original_frames), output_path='../output/psnr_per_frame.png')

    # For comparison plots, let's pick the middle frame
    frame_idx = len(original_frames) // 2
    original_comp_frame = original_frames[frame_idx]
    stego_comp_frame = stego_frames[frame_idx]
    
    # Plot 2: Original vs. Stego frame
    plot_original_vs_stego_frame(original_comp_frame, stego_comp_frame, frame_idx + 1, output_path='../output/frame_comparison.png')

    # Plot 3: Histogram comparison
    original_y = cv2.cvtColor(original_comp_frame, cv2.COLOR_BGR2YUV)[:, :, 0]
    stego_y = cv2.cvtColor(stego_comp_frame, cv2.COLOR_BGR2YUV)[:, :, 0]
    plot_y_channel_histogram(original_y, stego_y, frame_idx + 1, output_path='../output/y_channel_histogram.png')
    
    print("\n✅ All plots have been saved to the 'output' folder.")


if __name__ == "__main__":
    analyze_videos()
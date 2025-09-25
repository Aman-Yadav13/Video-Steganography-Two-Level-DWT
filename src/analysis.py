import matplotlib.pyplot as plt
import numpy as np
import cv2
import seaborn as sns

sns.set_theme(style="whitegrid")

def plot_psnr_per_frame(psnr_scores, num_frames, output_path='../output/psnr_per_frame.png'):
    """Generates and saves a PSNR per frame graph."""
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_frames + 1), psnr_scores, marker='o', linestyle='-', markersize=4, color='skyblue')
    plt.title('PSNR per Frame During Steganography', fontsize=16)
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(min(psnr_scores) - 2, max(psnr_scores) + 2) # Adjust y-axis for better visualization
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"PSNR per frame graph saved to {output_path}")
    plt.close()

def plot_original_vs_stego_frame(original_frame, stego_frame, frame_number, output_path='../output/frame_comparison.png'):
    """Generates and saves a side-by-side comparison of original and stego frames."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Convert OpenCV BGR to Matplotlib RGB
    original_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    stego_rgb = cv2.cvtColor(stego_frame, cv2.COLOR_BGR2RGB)

    axes[0].imshow(original_rgb)
    axes[0].set_title(f'Original Frame {frame_number}', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(stego_rgb)
    axes[1].set_title(f'Stego Frame {frame_number}', fontsize=14)
    axes[1].axis('off')

    plt.suptitle('Original vs. Stego Frame Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.savefig(output_path)
    print(f"Frame comparison saved to {output_path}")
    plt.close()

def plot_y_channel_histogram(original_y_channel, stego_y_channel, frame_number, output_path='../output/y_channel_histogram.png'):
    """Generates and saves histograms of the Y channel for original and stego frames."""
    plt.figure(figsize=(10, 6))
    
    sns.histplot(original_y_channel.flatten(), color='blue', alpha=0.6, label='Original Y Channel', bins=50, kde=True)
    sns.histplot(stego_y_channel.flatten(), color='red', alpha=0.6, label='Stego Y Channel', bins=50, kde=True)
    
    plt.title(f'Y Channel Histogram Comparison (Frame {frame_number})', fontsize=16)
    plt.xlabel('Pixel Intensity', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Y channel histogram saved to {output_path}")
    plt.close()

# Example usage (you would integrate this into your main.py after processing)
if __name__ == "__main__":
    # Dummy data for demonstration
    num_frames = 100
    avg_psnr = 51.5
    psnr_scores_dummy = np.random.normal(avg_psnr, 1.5, num_frames) # Simulate some variation
    
    # Create dummy frames (replace with actual frames from your processing)
    dummy_original_frame = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
    dummy_stego_frame = dummy_original_frame.copy()
    dummy_stego_frame[10:20, 10:20, 0] = (dummy_stego_frame[10:20, 10:20, 0] + 50) % 256 # Introduce a slight change

    dummy_original_y = cv2.cvtColor(dummy_original_frame, cv2.COLOR_BGR2YUV)[:,:,0]
    dummy_stego_y = cv2.cvtColor(dummy_stego_frame, cv2.COLOR_BGR2YUV)[:,:,0]

    # Generate plots
    plot_psnr_per_frame(psnr_scores_dummy, num_frames)
    plot_original_vs_stego_frame(dummy_original_frame, dummy_stego_frame, 50) # Assuming frame 50 for comparison
    plot_y_channel_histogram(dummy_original_y, dummy_stego_y, 50)
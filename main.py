import cv2
import numpy as np
import time
from src.video_processor import split_video_to_frames, mux_frames_to_video
from src.chaotic_maps import get_seed_from_password, logistic_map, henon_map
from src.encryption import encrypt_payload
from src.steganography import embed_dwt
from src.metrics import calculate_metrics
from src.analysis import plot_psnr_per_frame, plot_original_vs_stego_frame, plot_y_channel_histogram

def main():
    """
    Main function to run the entire video steganography process, including
    payload encryption, embedding, and performance analysis.
    """
    # --- 1. Configuration ---
    INPUT_VIDEO_PATH = './data/input_video.mp4'
    PAYLOAD_FILE_PATH = './data/payload.txt'
    OUTPUT_VIDEO_PATH = './output/stego_video.mp4'
    PASSWORD = "mysecretpassword"

    start_time = time.time()

    # --- 2. Load Payload ---
    with open(PAYLOAD_FILE_PATH, 'r', encoding='utf-8') as f:
        payload_string = f.read()
    payload_bytes = payload_string.encode('utf-8')
    payload_bits = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))
    print(f"✅ Payload loaded: {len(payload_bits)} bits")

    # --- 3. Generate Chaotic Keys from Password ---
    seed = get_seed_from_password(PASSWORD)
    key_size = len(payload_bits) * 2  # Generate extra for padding
    logistic_key = logistic_map(seed, key_size)
    henon_key = henon_map(seed, 20000)  # For embedding positions
    print("✅ Chaotic keys generated.")

    # --- 4. Encrypt Payload ---
    encrypted_bits = encrypt_payload(payload_bits, logistic_key)
    print(f"✅ Payload encrypted. New size: {len(encrypted_bits)} bits")

    # --- 5. Process Video Frames ---
    original_frames = split_video_to_frames(INPUT_VIDEO_PATH)
    if not original_frames:
        print("❌ Error: Could not read frames from the input video.")
        return

    stego_frames = []
    psnr_scores = []
    ssim_scores = []
    
    total_bits_to_embed = len(encrypted_bits)
    bits_embedded = 0

    # Store a specific frame for comparison plots (e.g., the middle frame)
    frame_for_comparison_idx = len(original_frames) // 2
    original_frame_comparison = None
    stego_frame_comparison = None
    original_y_comparison = None
    stego_y_comparison = None

    print("\n🚀 Starting embedding process...")
    for i, frame in enumerate(original_frames):
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv_frame)

        stego_y = y # Default to original if no embedding happens in this frame
        if bits_embedded < total_bits_to_embed:
            # Simple logic: embed a chunk of bits in each frame
            chunk_size = 6000  # bits per frame, can be made adaptive
            bits_for_this_frame = encrypted_bits[bits_embedded : bits_embedded + chunk_size]
            
            if len(bits_for_this_frame) > 0:
                stego_y = embed_dwt(y, bits_for_this_frame, henon_key)
                bits_embedded += len(bits_for_this_frame)
                
                # --- FIX 1: Ensure stego_y matches original y dimensions ---
                # This prevents errors from slight size changes during Inverse DWT.
                if stego_y.shape != y.shape:
                    stego_y = cv2.resize(stego_y, (y.shape[1], y.shape[0]))
        
        # Reconstruct the stego frame
        stego_yuv_frame = cv2.merge([stego_y.astype(np.uint8), u, v]) # Added .astype for safety
        stego_frame_bgr = cv2.cvtColor(stego_yuv_frame, cv2.COLOR_YUV2BGR)
        stego_frames.append(stego_frame_bgr)


        # Store frames and Y-channels for plotting analysis
        if i == frame_for_comparison_idx:
            original_frame_comparison = frame
            stego_frame_comparison = stego_frame_bgr
            original_y_comparison = y
            stego_y_comparison = stego_y.astype(np.uint8) # Ensure correct dtype for plotting

        # Calculate metrics for this frame
        psnr_val, ssim_val = calculate_metrics(frame, stego_frame_bgr)
        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)
        
        print(f"Processed Frame {i+1}/{len(original_frames)} | PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")

    # --- 6. Mux Frames into Output Video ---
    mux_frames_to_video(stego_frames, OUTPUT_VIDEO_PATH)
    print(f"\n✅ Stego video saved to {OUTPUT_VIDEO_PATH}")

    end_time = time.time()
    runtime = end_time - start_time

    # --- 7. Final Metrics ---
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    payload_bpp = (total_bits_to_embed / len(original_frames)) / (original_frames[0].shape[0] * original_frames[0].shape[1])
    
    print("\n--- 📊 Process Completed ---")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Payload (bpp): {payload_bpp:.4f}")
    print(f"Total Runtime: {runtime:.2f} seconds")
    # BER would be 0 as we haven't implemented extraction and comparison yet.

    # --- 8. Generate Analysis Plots ---
    print("\n--- 📈 Generating Performance Plots ---")
    plot_psnr_per_frame(psnr_scores, len(original_frames), output_path='./output/psnr_per_frame.png')
    
    if original_frame_comparison is not None:
        plot_original_vs_stego_frame(
            original_frame_comparison, stego_frame_comparison,
            frame_for_comparison_idx + 1,
            output_path='./output/frame_comparison.png'
        )
        plot_y_channel_histogram(
            original_y_comparison, stego_y_comparison,
            frame_for_comparison_idx + 1,
            output_path='./output/y_channel_histogram.png'
        )
    else:
        print("⚠️ Could not generate frame-specific plots as no frame was captured for comparison.")

if __name__ == "__main__":
    main()
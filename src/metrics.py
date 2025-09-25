import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# In src/metrics.py

def calculate_metrics(original_frame, stego_frame):
    """Calculates PSNR and SSIM between two frames."""
    if original_frame.shape != stego_frame.shape:
        h, w, _ = original_frame.shape
        stego_frame = cv2.resize(stego_frame, (w, h))

    psnr_val = psnr(original_frame, stego_frame, data_range=255)
    
    # --- FIX 2: Replace deprecated 'multichannel' with 'channel_axis=2' ---
    # This explicitly tells scikit-image how to handle the color channels (axis 2).
    ssim_val = ssim(original_frame, stego_frame, channel_axis=2, data_range=255)
    
    return psnr_val, ssim_val

def calculate_ber(original_bits, extracted_bits):
    """Calculates Bit Error Rate."""
    if len(original_bits) != len(extracted_bits):
        return 1.0 # Max error if lengths differ
    error_count = np.sum(original_bits != extracted_bits)
    return error_count / len(original_bits)
import numpy as np
import pywt

def embed_dwt(y_channel, bits_to_embed, henon_sequence):
    """Embeds bits into the Y channel using 2-Level DWT and LSB."""
    # 2-Level DWT
    coeffs = pywt.dwt2(y_channel, 'haar')
    LL, (LH, HL, HH) = coeffs
    
    coeffs2 = pywt.dwt2(LL, 'haar')
    LL2, (LH2, HL2, HH2) = coeffs2
    
    # Combine sub-bands for embedding
    sub_bands = [LH2, HL2, HH2, LH, HL, HH]
    
    total_capacity = sum(band.size for band in sub_bands)
    if len(bits_to_embed) > total_capacity:
        raise ValueError("Payload is too large for the video frame.")
        
    bit_index = 0
    
    # Generate unique embedding positions
    positions = set()
    for band_idx, band in enumerate(sub_bands):
        rows, cols = band.shape
        # Scale Henon map to band dimensions
        x_coords = (henon_sequence[:, 0] * (rows - 1)).astype(int)
        y_coords = (henon_sequence[:, 1] * (cols - 1)).astype(int)
        
        for r, c in zip(x_coords, y_coords):
            if bit_index >= len(bits_to_embed):
                break
            if (band_idx, r, c) not in positions:
                # Modify LSB
                coeff_val = int(band[r, c])
                band[r, c] = float((coeff_val & ~1) | bits_to_embed[bit_index])
                positions.add((band_idx, r, c))
                bit_index += 1
        if bit_index >= len(bits_to_embed):
            break

    # Inverse DWT
    LL = pywt.idwt2((LL2, (LH2, HL2, HH2)), 'haar')
    stego_y = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
    
    return stego_y.astype(np.uint8)
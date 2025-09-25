import numpy as np
from src.chaotic_maps import logistic_map, arnold_cat_map

def encrypt_payload(payload_bits, key_sequence):
    """Encrypts payload bits using XOR with a chaotic key sequence."""
    key_bits = (key_sequence > 0.5).astype(int)
    encrypted_bits = payload_bits ^ key_bits[:len(payload_bits)]
    
    # Apply Arnold Cat Map for diffusion
    n = int(np.ceil(np.sqrt(len(encrypted_bits))))
    padded_bits = np.pad(encrypted_bits, (0, n*n - len(encrypted_bits)), 'constant')
    bit_matrix = padded_bits.reshape((n, n))
    
    scrambled_matrix = arnold_cat_map(bit_matrix)
    
    return scrambled_matrix.flatten()

def decrypt_payload(scrambled_bits, key_sequence, original_len):
    """Decrypts the payload."""
    # Inverse Arnold Cat Map (requires running it multiple times until identity)
    n = int(np.sqrt(len(scrambled_bits)))
    bit_matrix = scrambled_bits.reshape((n, n))
    
    # This is a simplified inverse; a true inverse requires finding the period.
    # For demonstration, we assume a single iteration forward for encryption.
    # We need to implement the inverse transformation.
    # A simple way for a report is to run it forward until the original is recovered.
    # Let's use a simpler inverse for now.
    inverted_matrix = np.zeros_like(bit_matrix)
    for r in range(n):
        for c in range(n):
            orig_r = (r - c) % n
            orig_c = (-orig_r + c) % n
            inverted_matrix[orig_r, orig_c] = bit_matrix[r, c]

    decrypted_scrambled_bits = inverted_matrix.flatten()[:original_len]
    
    key_bits = (key_sequence > 0.5).astype(int)
    decrypted_bits = decrypted_scrambled_bits ^ key_bits[:original_len]
    
    return decrypted_bits
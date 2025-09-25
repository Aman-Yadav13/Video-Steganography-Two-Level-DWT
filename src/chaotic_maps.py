import numpy as np

def get_seed_from_password(password):
    """Generates a numerical seed from a string password."""
    seed = 0
    for char in password:
        seed += ord(char)
    return seed

def logistic_map(seed, size):
    """Generates a chaotic sequence using the Logistic Map."""
    x = seed / 256.0  # Normalize seed to (0, 1)
    r = 3.99
    sequence = []
    for _ in range(size):
        x = r * x * (1 - x)
        sequence.append(x)
    return np.array(sequence)

# In src/chaotic_maps.py

# In src/chaotic_maps.py

def henon_map(seed, size):
    """
    Generates a NORMALIZED and STABLE chaotic 2D sequence using the Henon Map.
    """
    x = seed / 256.0
    y = seed / 256.0
    a = 1.4
    b = 0.3
    
    x_sequence = np.zeros(size)
    y_sequence = np.zeros(size)

    for i in range(size):
        x_next = 1 - a * x * x + y
        y_next = b * x
        
        # --- ROBUST FIX: Keep values bounded within the loop ---
        # The modulo operator (%) ensures the values stay within a finite
        # range (-1.0 to 1.0), preventing them from diverging to infinity.
        x = x_next % 1.0
        y = y_next % 1.0
        
        x_sequence[i] = x
        y_sequence[i] = y

    # Now that the sequence is guaranteed to be stable, normalization will work correctly.
    x_min, x_max = np.min(x_sequence), np.max(x_sequence)
    y_min, y_max = np.min(y_sequence), np.max(y_sequence)
    
    if x_max == x_min:
        norm_x = np.zeros(size)
    else:
        norm_x = (x_sequence - x_min) / (x_max - x_min)

    if y_max == y_min:
        norm_y = np.zeros(size)
    else:
        norm_y = (y_sequence - y_min) / (y_max - y_min)
    
    return np.column_stack((norm_x, norm_y))

def arnold_cat_map(matrix):
    """Applies one iteration of Arnold's Cat Map scrambling."""
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("Arnold Cat Map requires a square matrix.")
    
    scrambled_matrix = np.zeros_like(matrix)
    for r in range(rows):
        for c in range(cols):
            new_r = (2 * r + c) % rows
            new_c = (r + c) % cols
            scrambled_matrix[new_r, new_c] = matrix[r, c]
    return scrambled_matrix
import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        # Hebbian learning: sum of outer products
        for p in patterns:
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)  # Remove self-connections

    def recall(self, pattern, steps=10):
        # Iteratively update until convergence or max steps
        for _ in range(steps):
            updated_pattern = pattern.copy()
            for i in range(self.size):
                raw_value = np.dot(self.weights[i], pattern)
                updated_pattern[i] = 1 if raw_value >= 0 else -1
            if np.array_equal(updated_pattern, pattern):  # Converged
                break
            pattern = updated_pattern
        return pattern

    def recognize_pattern(self, recalled_pattern, patterns):
        # Compare recalled pattern with each stored pattern and return closest match
        similarities = [np.sum(recalled_pattern == p) for p in patterns]
        recognized_index = np.argmax(similarities)
        return recognized_index  # Index of the most similar stored pattern

def binarize(pattern):
    flat_pattern = pattern.flatten()
    return np.array([1 if x == 1 else -1 for x in flat_pattern])

def add_noise(pattern, noise_level=0.1):
    noisy_pattern = pattern.copy()
    n_noisy = int(noise_level * len(pattern))
    flip_indices = np.random.choice(len(pattern), size=n_noisy, replace=False)
    noisy_pattern[flip_indices] *= -1
    return noisy_pattern

def print_pattern(pattern, rows=10, cols=10, label=""):
    print(f"\n{label}")
    for i in range(0, len(pattern), cols):
        print(''.join(['#' if x == 1 else '.' for x in pattern[i:i+cols]]))

# Define digit patterns (only digit_0 and digit_6 as examples; you should add all digits 0-9)
digit_0 = np.array([
    [0,1,1,1,1,1,1,1,1,0],
    [1,1,1,1,1,1,1,1,1,1],
    [1,1,1,0,0,0,0,1,1,1],
    [1,1,1,0,0,0,0,1,1,1],
    [1,1,1,0,0,0,0,1,1,1],
    [1,1,1,0,0,0,0,1,1,1],
    [1,1,1,0,0,0,0,1,1,1],
    [1,1,1,0,0,0,0,1,1,1],
    [1,1,1,1,1,1,1,1,1,1],
    [0,1,1,1,1,1,1,1,1,0]
])

digit_6 = np.array([
    [0,0,1,1,1,1,1,1,0,0],
    [0,1,1,1,1,1,1,1,1,1],
    [1,1,1,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,0],
    [1,1,1,1,1,1,1,1,1,1],
    [1,1,1,0,0,0,0,1,1,1],
    [1,1,1,0,0,0,0,1,1,1],
    [1,1,1,0,0,0,0,1,1,1],
    [0,1,1,1,1,1,1,1,1,1],
    [0,0,1,1,1,1,1,1,0,0]
])

# Convert each pattern to a 1D array with -1 and 1 values
patterns = [binarize(digit_0), binarize(digit_6)]
labels = ["Digit 0", "Digit 6"]

# Initialize the network and train with the digit patterns
hopfield_net = HopfieldNetwork(size=100)
hopfield_net.train(patterns)

# Add noise to a specific digit pattern for testing, e.g., digit 6
test_digit_index = 0  # Testing "Digit 6"
noisy_pattern = add_noise(patterns[test_digit_index], noise_level=0.2)
recalled_pattern = hopfield_net.recall(noisy_pattern)

# Recognize which digit was recalled
recognized_index = hopfield_net.recognize_pattern(recalled_pattern, patterns)
recognized_label = labels[recognized_index]

# Print noisy, recalled pattern, and recognized label
print_pattern(noisy_pattern, label="Noisy input pattern for Digit 6")
print_pattern(recalled_pattern, label=f"Recalled pattern (recognized as {recognized_label})")

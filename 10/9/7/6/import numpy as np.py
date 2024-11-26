import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))  # Initialize weights

    def train(self, patterns):
        # Hebbian learning: sum of outer products
        for p in patterns:
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)  # No self-connections

    def recall(self, pattern, steps=10):
        # Iteratively update until convergence or until steps are reached
        for _ in range(steps):
            for i in range(self.size):
                raw_value = np.dot(self.weights[i], pattern)
                pattern[i] = 1 if raw_value >= 0 else -1
        return pattern

# Define the digit patterns as 10x10 matrices, just showing 0 and 1 as an example
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

digit_1 = np.array([ [0,0,0,0,1,1,1,0,0,0], [0,0,1,1,1,1,1,0,0,0], [0,1,1,0,1,1,1,0,0,0], [0,0,0,0,1,1,1,0,0,0], [0,0,0,0,1,1,1,0,0,0], [0,0,0,0,1,1,1,0,0,0], [0,0,0,0,1,1,1,0,0,0], [0,0,0,0,1,1,1,0,0,0], [0,0,0,0,1,1,1,0,0,0], [1,1,1,1,1,1,1,1,1,1] ]) 

digit_2 = np.array([ [0,1,1,1,1,1,1,1,1,0], [1,1,1,1,1,1,1,1,1,1], [1,1,1,0,0,0,0,1,1,1], [0,0,0,0,0,0,1,1,1,1], [0,0,0,0,0,1,1,1,1,0], [0,0,0,0,1,1,1,1,0,0], [0,0,0,1,1,1,0,0,0,0], [0,0,1,1,1,0,0,0,0,0], [1,1,1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1,1,1] ]) 

digit_3 = np.array([ [0,1,1,1,1,1,1,1,1,0], [1,1,1,1,1,1,1,1,1,1], [1,1,1,0,0,0,0,1,1,1], [0,0,0,0,0,0,1,1,1,1], [0,0,0,0,1,1,1,1,0,0], [0,0,0,0,0,0,1,1,1,1], [1,1,1,0,0,0,0,1,1,1], [1,1,1,0,0,0,0,1,1,1], [1,1,1,1,1,1,1,1,1,1], [0,1,1,1,1,1,1,1,1,0] ]) 

digit_4 = np.array([ [0,0,0,0,1,1,1,1,0,0], [0,0,0,1,1,1,1,1,0,0], [0,0,1,1,1,1,1,1,0,0], [0,1,1,0,1,1,1,1,0,0], [1,1,0,0,1,1,1,1,0,0], [1,1,1,1,1,1,1,1,1,1], [0,0,0,0,0,1,1,1,0,0], [0,0,0,0,0,1,1,1,0,0], [0,0,0,0,0,1,1,1,0,0], [0,0,0,0,0,1,1,1,0,0] ]) 

digit_5 = np.array([ [1,1,1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1,1,1], [1,1,1,0,0,0,0,0,0,0], [1,1,1,1,1,1,1,1,1,0], [1,1,1,1,1,1,1,1,1,1], [0,0,0,0,0,0,0,1,1,1], [0,0,0,0,0,0,0,1,1,1], [1,1,1,0,0,0,0,1,1,1], [1,1,1,1,1,1,1,1,1,1], [0,1,1,1,1,1,1,1,1,0] ]) 

digit_6 = np.array([ [0,0,1,1,1,1,1,1,0,0], [0,1,1,1,1,1,1,1,1,1], [1,1,1,0,0,0,0,0,0,0], [1,1,1,1,1,1,1,1,1,0], [1,1,1,1,1,1,1,1,1,1], [1,1,1,0,0,0,0,1,1,1], [1,1,1,0,0,0,0,1,1,1], [1,1,1,0,0,0,0,1,1,1], [0,1,1,1,1,1,1,1,1,1], [0,0,1,1,1,1,1,1,0,0] ]) 

digit_7 = np.array([ [1,1,1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,1,1,1], [0,0,0,0,0,0,1,1,1,1], [0,0,0,0,0,0,1,1,1,0], [0,0,0,0,0,1,1,1,0,0], [0,0,0,0,0,1,1,1,0,0], [0,0,0,0,1,1,1,0,0,0], [0,0,0,0,1,1,1,0,0,0], [0,0,0,1,1,1,0,0,0,0], [0,0,0,1,1,1,0,0,0,0] ]) 

digit_8 = np.array([ [0,1,1,1,1,1,1,1,1,0], [1,1,1,1,1,1,1,1,1,1], [1,1,1,0,0,0,0,1,1,1], [1,1,1,0,0,0,0,1,1,1], [0,1,1,1,1,1,1,1,1,0], [1,1,1,0,0,0,0,1,1,1], [1,1,1,0,0,0,0,1,1,1], [1,1,1,0,0,0,0,1,1,1], [1,1,1,1,1,1,1,1,1,1], [0,1,1,1,1,1,1,1,1,0] ]) 


digit_9 = np.array([ [0,1,1,1,1,1,1,1,1,0], [1,1,1,1,1,1,1,1,1,1], [1,1,1,0,0,0,0,1,1,1], [1,1,1,0,0,0,0,1,1,1], [1,1,1,1,1,1,1,1,1,1], [0,1,1,1,1,1,1,1,1,1], [0,0,0,0,0,0,0,1,1,1], [1,1,1,0,0,0,0,1,1,1], [1,1,1,1,1,1,1,1,1,1], [0,1,1,1,1,1,1,1,1,0] ])

# Convert each pattern to a 1D array with -1 and 1 values
def binarize(pattern):
    flat_pattern = pattern.flatten()
    return np.array([1 if x == 1 else -1 for x in flat_pattern])

# Binarize all digit patterns
patterns = [binarize(digit_0), binarize(digit_1), binarize(digit_2), binarize(digit_3),
            binarize(digit_4), binarize(digit_5), binarize(digit_6), binarize(digit_7),
            binarize(digit_8), binarize(digit_9)]

# Initialize the network and train with the digit patterns
hopfield_net = HopfieldNetwork(size=100)
hopfield_net.train(patterns)

def add_noise(pattern, noise_level=0.1):
    noisy_pattern = pattern.copy()
    n_noisy = int(noise_level * len(pattern))  # Number of pixels to flip
    flip_indices = np.random.choice(len(pattern), size=n_noisy, replace=False)
    noisy_pattern[flip_indices] *= -1  # Flip bits (1 to -1 or -1 to 1)
    return noisy_pattern

# Function to print the pattern with labelings
def print_pattern(pattern, rows=10, cols=10, label=""):
    print(f"\n{label}")
    for i in range(0, len(pattern), cols):
        print(''.join(['#' if x == 1 else '.' for x in pattern[i:i+cols]]))


test_digit_index = 8
noisy_pattern = add_noise(patterns[test_digit_index], noise_level=0.2)
recalled_pattern = hopfield_net.recall(noisy_pattern)

# Print input and output patterns
print_pattern(noisy_pattern, label="Noisy input pattern")
print_pattern(recalled_pattern, label="Recalled pattern")
import numpy as np

# Load the file
data = np.load('classes.npy')

# Print the array content, shape, and data type
print(data)
print("Shape:", data.shape)
print("Data type:", data.dtype)
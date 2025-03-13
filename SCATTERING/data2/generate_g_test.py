import numpy as np

# Number of rows and columns
rows = 100
cols = 12

# Generate random numbers
data = np.random.rand(rows, cols)

# Write to file
with open('c:/Users/jocar/Desktop/LaboratorioBiofisica/SCATTERING/data2/g_test.txt', 'w') as f:
    for row in data:
        f.write(' '.join(map(str, row)) + '\n')
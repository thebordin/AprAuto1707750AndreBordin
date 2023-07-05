from PIL import Image
import numpy as np
import sys

# 1. Setup
input = sys.argv[1]
file = sys.argv[2]
output = sys.argv[3]
img = Image.open(input+file)

# 2. Convert image to NumPy array
arr = np.asarray(img)
print(f'Convertendo {file}: Array Shape: {arr.shape}')
# (771, 771, 3)
# 3. Convert 3D array to 2D list of lists
lst = []
for row in arr:
    tmp = []
    for col in row:
        tmp.append(str(col))
    lst.append(tmp)
# 4. Save list of lists to CSV
#print(lst)
with open(output+file+'.csv', 'w') as f:
    for row in lst:
        f.write(','.join(row) + '\n')
# In[]
import numpy as np
from scipy.io import loadmat

# Load the .mat file
mat = loadmat("/work/jiaqi/Network_compressibility/graphs_Protein_samples_undirected.mat")

# Inspect keys
print(mat.keys())

G_yeast = mat['G_yeast']
print(G_yeast.shape, G_yeast.dtype)
# print(G_yeast)

G = G_yeast[0][0]
print(G, type(G), G.shape)

np.savetxt('G_yeast.txt', G, fmt='%d')

# mymat = loadmat('../matrix.mat')
# print(mymat.keys())
# mymatrix = mymat['my_matrix']
# print(mymatrix.shape, mymatrix.dtype)
# print(mymatrix)
# %%
import numpy as np
from scipy.io import loadmat

mat_s_5 = loadmat("../S_7.mat")
s_5 = mat_s_5['S']
my_s_5 = np.loadtxt('../S_upper_bound_heuristic_7.txt')
s_5 = np.squeeze(s_5)
print(s_5.shape, s_5.dtype) 
print(my_s_5.shape, my_s_5.dtype)
# compare s_5 with my_s_5
print(np.allclose(s_5, my_s_5))
diff = s_5 - my_s_5
print(diff.max())
# print(diff)

# draw a line plot for diff
import matplotlib.pyplot as plt

abs_diff = np.abs(diff)
plt.plot(abs_diff)
plt.title("Difference between S and My S")
plt.xlabel("Index")
plt.ylabel("Difference")
plt.show()

relative_diff = abs_diff / np.abs(s_5)
# replace nan with 0
relative_diff = np.nan_to_num(relative_diff)
print(np.max(relative_diff))
plt.plot(relative_diff)
plt.title("Relative Difference between S and My S")
plt.xlabel("Index")
plt.ylabel("Relative Difference")
plt.show()
# %%
import numpy as np

data = np.load('../logs/logs_interactomes.max_cc.rw1000/heu_5_no_time/rate_distortion_394_0/rates_upper.npy')
print(data.shape)
# print(data)
# data is a 1D array, check if it is monitonically decreasing
count = 0
for i in range(1, len(data)):
    if data[i] > data[i-1]:
        print(f"Not monotonically decreasing at index {i}: {data[i-1]} - {data[i]} = {data[i-1] - data[i]}")
        count += 1
        # break
print(f"Total violations found: {count}")
# %%
import numpy as np

data = np.load('../data/compressibility/graphs_Animal_samples_directed_G_wetlands_d_0.npz')['array']
print(data.shape)
# save data as .mat file
from scipy.io import savemat
savemat('wetlands_0.mat', {'adj': data})
# %%
from scipy.io import loadmat
import numpy as np

results = loadmat('394_0_results_heu7.mat')
S = results['S'].squeeze()
# reverse S
S = S[::-1]
data = np.load('../logs/logs_interactomes.max_cc.rw1000/heu_7_no_time/rate_distortion_394_0/rates_upper.npy')
print(S.shape, data.shape)
print(np.allclose(S, data))
diff = np.abs(S - data)
print(diff.max())
count = 0
for i in range(1, len(S)):
    if S[i] > S[i-1]:
        print(f"Not monotonically decreasing at index {i}: {S[i-1]} - {S[i]} = {S[i-1] - S[i]}")
        count += 1
        # break
print(f"Total violations found: {count}")
# plot the two curves
import matplotlib.pyplot as plt

plt.plot(S, label='S')
plt.plot(data, label='Data')
plt.legend()
plt.show()

# %%
S[0], data[0]
# %%

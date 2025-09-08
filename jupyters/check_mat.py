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

# In[]
from scipy.io import loadmat

mat = loadmat('/work/jiaqi/Network_compressibility/graphs_Citation_samples_undirected.mat')
mat.keys()

# %%
import numpy as np
np_array = mat['G_HepPh'][0][0].toarray()
np_array
# %%

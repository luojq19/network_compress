# In[]
from scipy.io import loadmat
import numpy as np
import os
from tqdm.auto import tqdm

mat_files = []
for root, dirs, files in os.walk('/work/jiaqi/Network_compressibility'):
    for file in files:
        if file.endswith('.mat') and file != 'graphs_info.mat':
            mat_files.append(os.path.join(root, file))
print(f'mat_files: {len(mat_files)}')

save_dir = '../data/compressibility'
os.makedirs(save_dir, exist_ok=True)
keys_ignored = ['__header__', '__version__', '__globals__']
n_graphs = 0
for mat_file in tqdm(mat_files):
    base_name = os.path.basename(mat_file).split('.')[0]
    mat = loadmat(mat_file)
    keys = mat.keys()
    for k in keys:
        if k in keys_ignored:
            continue
        # print(f'Processing {base_name}, key: {k}')
        data = mat[k][0]
        for i in range(data.shape[0]):
            if type(data[i]) == np.ndarray:
                array = data[i]
            else:
                array = data[i].toarray()
            # np.save(os.path.join(save_dir, f'{base_name}_{k}_{i}.npy'), array)
            np.savez_compressed(os.path.join(save_dir, f'{base_name}_{k}_{i}.npz'), array=array)
# %%
import numpy as np
m = np.load('../data/compressibility/graphs_Animal_samples_directed_G_wetlands_d_0.npz')
m.files
# %%

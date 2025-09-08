# In[]
import os
import shutil
from tqdm.auto import tqdm

def remove_time_from_name(s):
    # Removes the date and time pattern: YYYY_MM_DD__HH_MM_SS
    import re
    return re.sub(r'\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}_', '', s)

root_dir = '../logs/logs_compressibility/heu_7'
dir_list = os.listdir(root_dir)
print(len(dir_list))
new_root_dir = '../logs/logs_compressibility/heu_7_no_time'
os.makedirs(new_root_dir, exist_ok=True)

for item in tqdm(dir_list):
    old_path = os.path.join(root_dir, item)
    new_name = remove_time_from_name(item)
    new_path = os.path.join(new_root_dir, new_name)
    shutil.copytree(old_path, new_path)

# %%

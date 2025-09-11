# In[]
import os
import shutil
from tqdm.auto import tqdm
import re

def remove_time_from_name(s):
    # Removes the date and time pattern: YYYY_MM_DD__HH_MM_SS
    return re.sub(r'\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}_', '', s)

def remove_redundant_underscores(s):
    # Replace multiple underscores with a single underscore
    return re.sub(r'_+', '_', s)

root_dir = '../logs/logs_interactomes.max_cc.rw2000/heu_7'
dir_list = os.listdir(root_dir)
print(len(dir_list))
new_root_dir = '../logs/logs_interactomes.max_cc.rw2000/heu_7_no_time'
os.makedirs(new_root_dir, exist_ok=True)

for item in tqdm(dir_list):
    old_path = os.path.join(root_dir, item)
    # new_name = remove_time_from_name(item)
    new_name = remove_redundant_underscores(item)
    new_path = os.path.join(new_root_dir, new_name)
    shutil.copytree(old_path, new_path)

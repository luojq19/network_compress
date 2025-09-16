import os
import pandas as pd
from tqdm.auto import tqdm
import time
import random
import subprocess

df_species = pd.read_csv('../data/treeoflife.species.tsv', sep='\t')
species_id_list = df_species['Species_ID'].tolist()
save_dir = '../data/STRING/ppi'
os.makedirs(save_dir, exist_ok=True)

MAX_RETRIES = 1
failed_species = []

for species_id in tqdm(species_id_list, dynamic_ncols=True):
    url = f'https://stringdb-downloads.org/download/stream/protein.physical.links.full.v12.0/{species_id}.protein.physical.links.full.v12.0.tsv.gz'
    output_path = os.path.join(save_dir, f'{species_id}.protein.physical.links.full.v12.0.txt.gz')
    if os.path.exists(output_path):
        continue

    success = False
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = subprocess.run(
                ["wget", "-q", "-O", output_path, url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                success = True
                break
            else:
                # 如果是空文件，删掉，避免后面以为下载过
                if os.path.exists(output_path) and os.path.getsize(output_path) == 0:
                    os.remove(output_path)
                print(f"Attempt {attempt} failed for {species_id}")
        except Exception as e:
            print(f"Error downloading {species_id} (attempt {attempt}): {e}")

        # 随机 sleep，避免并发时服务器压力过大
        time.sleep(random.uniform(1, 3))

    if not success:
        print(f"❌ Failed to download {species_id} after {MAX_RETRIES} attempts")
        failed_species.append(species_id)
    else:
        time.sleep(random.uniform(0.1, 1))  # 正常成功也加个短暂停顿
print("Failed species:", len(failed_species))
with open('../data/STRING/failed_species.txt', 'w') as f:
    for species_id in failed_species:
        f.write(f"{species_id}\n")
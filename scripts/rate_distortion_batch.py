import sys
sys.path.append('.')
import os
from joblib import Parallel, delayed
import random
from tqdm import tqdm
import time
from utils import commons
import shutil
import argparse

def worker(graph_path, heu, log_dir_base, script):
    graph_name = os.path.basename(graph_path).replace('.npz', '')
    if os.path.exists(os.path.join(log_dir_base, f'heu_{heu}', f'rate_distortion__{graph_name}', 'rate_distortion.png')):
        print(f'Skipping {graph_name} with heu {heu} as it is already done.')
        return
    command = f'''
        python {script} \\
        --graph {graph_path} \\
        --heu {heu} \\
        --logdir {os.path.join(log_dir_base, f'heu_{heu}')} \\
        --tag {graph_name} \\
        --no_timestamp
    '''
    # print(command)
    os.system(command)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/compressibility', help='Directory containing graph files in .npz format.')
    parser.add_argument('--log_dir_base', type=str, default='logs/logs_compressibility_debug', help='Base directory for logs.')
    parser.add_argument('--script', type=str, default='scripts/rate_distortion_debug.py', help='Path to the rate distortion script.')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of parallel workers.')
    args = parser.parse_args()
    return args

def main():
    start_overall = time.time()
    args = get_args()
    data_dir = args.data_dir
    log_dir_base = args.log_dir_base
    os.makedirs(log_dir_base, exist_ok=True)
    script = args.script
    shutil.copyfile(script, os.path.join(log_dir_base, os.path.basename(script)))

    graphs = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.npz'):
                graphs.append(os.path.join(root, file))
    random.shuffle(graphs)
    num_workers = args.num_workers
    Parallel(n_jobs=num_workers, verbose=10)(delayed(worker)(graph, heu, log_dir_base, script) for graph in tqdm(graphs, desc='Processing graphs') for heu in [5, 7])
    end_overall = time.time()
    print(f'Time elapsed: {commons.sec2hr_min_sec(end_overall - start_overall)}')

if __name__ == "__main__":
    main()
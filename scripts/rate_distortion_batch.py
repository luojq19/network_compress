import sys
sys.path.append('.')
import os
from joblib import Parallel, delayed
import random
from tqdm import tqdm
import time
from utils import commons

def worker(graph_path, heu):
    graph_name = os.path.basename(graph_path).replace('.npz', '')
    if os.path.exists(f'logs/logs_interactomes.max_cc.rw2000/heu_{heu}/rate_distortion_{graph_name}/rates_upper.npy'):
        print(f'Skipping {graph_name} with heu {heu} as it is already done.')
        return
    command = f'''
        python scripts/rate_distortion.py \\
        --graph {graph_path} \\
        --heu {heu} \\
        --logdir logs/logs_interactomes.max_cc.rw2000/heu_{heu} \\
        --tag {graph_name} \\
        --no_timestamp
    '''
    # print(command)
    os.system(command)

def main():
    start_overall = time.time()
    data_dir = 'data/treeoflife.interactomes.max_cc.rw2000_adj'
    graphs = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.npz'):
                graphs.append(os.path.join(root, file))
    random.shuffle(graphs)
    num_workers = 40
    Parallel(n_jobs=num_workers, verbose=10)(delayed(worker)(graph, heu) for graph in tqdm(graphs) for heu in [5, 7])
    end_overall = time.time()
    print(f'Time elapsed: {commons.sec2hr_min_sec(end_overall - start_overall)}')

if __name__ == "__main__":
    main()
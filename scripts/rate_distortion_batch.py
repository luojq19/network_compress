import sys
sys.path.append('.')
import os
from joblib import Parallel, delayed
import random
from tqdm import tqdm

def worker(graph_path, heu):
    graph_name = os.path.basename(graph_path).replace('.npz', '')
    command = f'''
        python scripts/rate_distortion.py \\
        --graph {graph_path} \\
        --heu {heu} \\
        --logdir logs/logs_compressibility/heu_{heu} \\
        --tag {graph_name}
    '''
    # print(command)
    os.system(command)

def main():
    data_dir = 'data/compressibility'
    graphs = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.npz'):
                graphs.append(os.path.join(root, file))
    random.shuffle(graphs)
    num_workers = 10
    Parallel(n_jobs=num_workers, verbose=10)(delayed(worker)(graph, heu) for graph in tqdm(graphs) for heu in [5, 7])

if __name__ == "__main__":
    main()
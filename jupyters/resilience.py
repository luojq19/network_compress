import sys
sys.path.append('.')
import numpy as np
import os
import random
import argparse
import networkx as nx
from utils import commons
import time
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

def connected_component_sizes(adj: np.ndarray):
    G = nx.from_numpy_array(adj)
    sizes = [len(c) for c in nx.connected_components(G)]
    assert sum(sizes) == adj.shape[0], f"Sum of component sizes must equal number of nodes: {sum(sizes)} != {adj.shape[0]}"
    return sizes

def compute_shannon_diversity(sizes):
    if len(sizes) == 1:
        return 0.0
    sizes = np.array(sizes)
    total = sizes.sum()
    if total == len(sizes):
        return 1.0
    probs = sizes / total
    shannon_div = -np.sum(probs * np.log(probs + 1e-10)) / np.log(total)
    return shannon_div.item()

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--graph', type=str, default='data/treeoflife.interactomes.max_cc_adj/394.npz')
    parser.add_argument('--frac', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--logdir', type=str, default='logs/logs/')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--no_timestamp', action='store_true', help='If set, do not append timestamp to logdir')
    args = parser.parse_args()
    return args

def main():
    start_overall = time.time()
    args = get_args()
    commons.seed_all(args.seed)

    log_dir = commons.get_new_log_dir(root=args.logdir, prefix='resilience', tag=args.tag, timestamp=not args.no_timestamp)
    logger = commons.get_logger('resilience', log_dir=log_dir)
    logger.info(f'Args: {args}')

    data = np.load(args.graph)
    key = list(data.keys())[0]
    adj = data[key]
    logger.info(f'Nodes: {adj.shape[0]}, Edges: {adj.sum() / 2}')
    div_list = [0.0]
    available_nodes = set(list(range(adj.shape[0])))
    num_nodes_to_remove = args.frac * adj.shape[0]
    num_nodes_already_removed = 0
    for i in tqdm(range(1, 100), dynamic_ncols=True, desc='Simulating node removals'):
        nodes_to_remove = random.sample(list(available_nodes), int(num_nodes_to_remove * i - num_nodes_already_removed))
        num_nodes_already_removed += len(nodes_to_remove)
        available_nodes -= set(nodes_to_remove)
        # set the rows and columns of the removed nodes to zero
        adj[nodes_to_remove, :] = 0
        adj[:, nodes_to_remove] = 0
        sizes = connected_component_sizes(adj)
        div = compute_shannon_diversity(sizes)
        div_list.append(div)
    div_list.append(1.0)

    np.save(os.path.join(log_dir, 'shannon_diversity.npy'), np.array(div_list))

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.plot(div_list)
    ax.set_xlabel('Network failure rate, f')
    ax.set_ylabel('Shannon Diversity')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'shannon_diversity.png'))

    end_overall = time.time()
    logger.info(f'Time elapsed: {end_overall - start_overall:.2f} seconds')

if __name__ == '__main__':
    main()
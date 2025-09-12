import sys
sys.path.append('.')
import numpy as np
import os
import time
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import pandas as pd

def worker(graph, seed):
    command = f'''
    python jupyters/resilience.py \\
    --graph {graph} \\
    --logdir logs/resilience_{seed}/ \\
    --seed {seed} \\
    --no_timestamp \\
    --tag {os.path.basename(graph).split(".")[0]}
    '''
    os.system(command)

def main():
    seed = 0
    data_dir = 'data/treeoflife.interactomes.max_cc_adj'
    adj_file_list = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    print(f"Found {len(adj_file_list)} adjacency matrix files.")
    df_evolution = pd.read_csv('data/treeoflife.species.evolution.tsv', sep='\t')
    df_pubmed = pd.read_csv('data/treeoflife.species.pubmed.count.tsv', sep='\t', comment='#')
    df_pubmed_ge1000 = df_pubmed[df_pubmed['Publication_count'] >= 1000]
    species_pubmed_ge1000 = df_pubmed_ge1000['Species_ID'].tolist()
    df_evolution = df_evolution[df_evolution['Species_ID'].isin(species_pubmed_ge1000)]
    species_pubmed_ge1000_evolution = df_evolution['Species_ID'].astype(str).tolist()
    print(f"Species with >=1000 PubMed publications and evolution data: {len(species_pubmed_ge1000_evolution)}")
    adj_file_list = [f for f in adj_file_list if f.split('.')[0] in species_pubmed_ge1000_evolution]
    print(f"Filtered to {len(adj_file_list)} adjacency matrix files with evolution data and >=1000 PubMed publications.")

    num_workers = 32
    Parallel(n_jobs=num_workers)(delayed(worker)(
        os.path.join(data_dir, adj_file), seed) for adj_file in tqdm(adj_file_list, desc='Processing graphs'))

if __name__ == '__main__':
    main()
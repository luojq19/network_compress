import sys
sys.path.append('.')
import numpy as np
import scipy
from scipy.sparse.linalg import eigs
from scipy.special import comb
import os, argparse, time, random, logging, json
from tqdm import tqdm
from utils import commons
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from joblib import Parallel, delayed, parallel_backend
# suppress RuntimeWarning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def visualize_network(G, title="Network Graph"):
    # G is a numpy adjacency matrix
    graph = nx.from_numpy_array(G)
    plt.figure(figsize=(6, 6))
    nx.draw(graph, node_color='blue', edge_color='gray', node_size=20, alpha=0.7)
    plt.title(title)
    plt.savefig('./graph.png', dpi=200)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def worker(k_list, I, J, transition_prob_old, stationary_state_old, log_transition_prob_old, rate_old):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ans_list = []
    for k in k_list:
        i, j = I[k], J[k]
        n_old = transition_prob_old.shape[0]
        all_inds = np.arange(n_old)
        inds_not_ij = np.delete(all_inds, [i, j])
        p_ss_merged_prob = stationary_state_old[i] + stationary_state_old[j]
        p_ss_temp = np.append(stationary_state_old[inds_not_ij], p_ss_merged_prob)
        p_ss_unmerged = stationary_state_old[inds_not_ij]
        P_to_ij = transition_prob_old[np.ix_(inds_not_ij, [i, j])]
        numerator_1 = np.sum(p_ss_unmerged[:, np.newaxis] * P_to_ij, axis=1)
        denominator_1 = p_ss_temp[:-1]
        P_temp_1 = numerator_1 / denominator_1
        numerator_2 = stationary_state_old[i] * transition_prob_old[i, inds_not_ij] + stationary_state_old[j] * transition_prob_old[j, inds_not_ij]
        P_temp_2 = numerator_2 / p_ss_merged_prob
        submatrix = transition_prob_old[np.ix_([i, j], [i, j])]
        p_ss_sub = stationary_state_old[[i, j]]
        numerator_3 = np.sum(p_ss_sub[:, np.newaxis] * submatrix)
        P_temp_3 = numerator_3 / p_ss_merged_prob

        logP_temp_1 = np.log2(P_temp_1)
        # set inf to 0
        logP_temp_1[np.isinf(logP_temp_1)] = 0
        logP_temp_2 = np.log2(P_temp_2)
        logP_temp_2[np.isinf(logP_temp_2)] = 0
        logP_temp_3 = np.log2(P_temp_3)
        logP_temp_3 = 0 if np.isinf(logP_temp_3) else logP_temp_3
        # assert there's no inf or nan
        assert not np.any(np.isnan(logP_temp_1)) and not np.any(np.isinf(logP_temp_1))
        assert not np.any(np.isnan(logP_temp_2)) and not np.any(np.isinf(logP_temp_2))
        assert not np.any(np.isnan(logP_temp_3)) and not np.any(np.isinf(logP_temp_3))
        
        i_new_part = (
            -np.sum(p_ss_temp[:-1] * P_temp_1 * logP_temp_1)
            - p_ss_temp[-1] * np.sum(P_temp_2 * logP_temp_2)
            - p_ss_temp[-1] * P_temp_3 * logP_temp_3
        )
        i_old_part = (
            + np.sum(stationary_state_old * transition_prob_old[:, i] * log_transition_prob_old[:, i])
            + np.sum(stationary_state_old * transition_prob_old[:, j] * log_transition_prob_old[:, j])
            + stationary_state_old[i] * np.sum(transition_prob_old[i, :] * log_transition_prob_old[i, :])
            + stationary_state_old[j] * np.sum(transition_prob_old[j, :] * log_transition_prob_old[j, :])
            - stationary_state_old[i] * (transition_prob_old[i, i] * log_transition_prob_old[i, i] + transition_prob_old[i, j] * log_transition_prob_old[i, j])
            - stationary_state_old[j] * (transition_prob_old[j, j] * log_transition_prob_old[j, j] + transition_prob_old[j, i] * log_transition_prob_old[j, i])
        )
        
        dS = i_new_part + i_old_part
        S_temp = rate_old + dS
        ans_list.append(S_temp)
    return ans_list

def rate_distortion(G, heuristic, num_pairs, logger, n_jobs=4):
    N = G.shape[0]
    E = G.sum() / 2
    logger.debug(f'N: {N}, E: {E}')

    rates_upper, rates_lower = [], []
    clusters, Gs = [], []

    transition_prob_old = G / G.sum(axis=1, keepdims=True)

    eigenvalues, eigenvectors = eigs(transition_prob_old.T, k=1, which='LR')
    stationary_state = eigenvectors[:, 0].real
    stationary_state /= stationary_state.sum()
    # logger.debug(f'eigenvalues: {eigenvalues}')
    # logger.debug(f'stationary_state:\n{stationary_state}')
    stationary_state_old = stationary_state.copy()

    log_transition_prob_old = np.log2(transition_prob_old)
    # logger.debug(f'log_transition_prob_old:\n{log_transition_prob_old}')
    log_transition_prob_old[np.isinf(log_transition_prob_old)] = 0
    # logger.debug(f'log_transition_prob_old (after inf removal):\n{log_transition_prob_old}')
    rate_old = -np.sum(stationary_state_old * np.sum(transition_prob_old * log_transition_prob_old, axis=1))
    logger.debug(f'rate_old: {rate_old}')
    P_joint = transition_prob_old * stationary_state_old.reshape(-1, 1)
    # logger.debug(f'transition_prob_old:\n{transition_prob_old[:5, :5]}')
    # logger.debug(f'stationary_state_old:\n{stationary_state_old[:5]}')
    # logger.debug(f'P_joint:\n{P_joint[:5, :5]}')
    P_low = transition_prob_old.copy()

    rates_upper.append(rate_old.item())
    rates_lower.append(rate_old.item())
    initial_cluster = [[i] for i in range(N)]
    clusters.append(initial_cluster)
    Gs.append(G)

    for n in tqdm(range(N - 1, 1, -1), dynamic_ncols=True):
        if heuristic == 1:
            # Try combining all pairs:
            pairs = list(combinations(range(n + 1), 2))
            
            # DEBUG
            # pairs = pairs[:num_pairs]

            I = [i for i, j in pairs]
            J = [j for i, j in pairs]
        elif heuristic == 2:
            # Randomly sample pairs
            pairs = list(combinations(range(n + 1), 2))
            indices = random.sample(range(len(pairs)), min(num_pairs, len(pairs)))
            I = [pairs[i][0] for i in indices]
            J = [pairs[i][1] for i in indices]
        elif heuristic == 3:
            # Try combining all pairs connected by an edge
            upper = np.triu(transition_prob_old + transition_prob_old.T, 1)
            I, J = np.where(upper > 0)
        elif heuristic == 4:
            # Pick num_pairs node pairs at random that are connected by an edge
            upper = np.triu(transition_prob_old + transition_prob_old.T, 1)
            I, J = np.where(upper > 0)
            indices = random.sample(range(len(I)), min(num_pairs, len(I)))
            I = I[indices]
            J = J[indices]
        elif heuristic == 5:
            # Pick num_pairs node pairs with largest joint transition probabilities
            P_joint_symm = np.triu(P_joint + P_joint.T, 1)
            flat = P_joint_symm.flatten()
            num_candidates = min(num_pairs, np.sum(flat > 0))
            linear_indices = np.argsort(flat)[-num_candidates:]
            I, J = np.unravel_index(linear_indices, P_joint_symm.shape)
        elif heuristic == 6:
            diag_p_joint = np.diag(P_joint)
            score_matrix = P_joint + P_joint.T
            score_matrix += np.tile(diag_p_joint, (P_joint.shape[0], 1))
            score_matrix += np.tile(diag_p_joint, (P_joint.shape[0], 1)).T
            P_joint_symm = np.triu(score_matrix, k=1)
            flat = P_joint_symm.flatten()
            num_candidates = min(num_pairs, np.sum(flat > 0))
            linear_indices = np.argsort(flat)[-num_candidates:]
            I, J = np.unravel_index(linear_indices, P_joint_symm.shape)
        elif heuristic == 7:
            score_matrix = stationary_state_old[:, np.newaxis] + stationary_state_old
            stationary_state_temp = np.triu(score_matrix, k=1)
            flat = stationary_state_temp.flatten()
            num_candidates = min(num_pairs, comb(n + 1, 2, exact=True))
            # print(f'flat: {flat}')
            # print(f'num_candidates: {num_candidates}')
            linear_indices = np.argsort(flat)[-num_candidates:]
            I, J = np.unravel_index(linear_indices, stationary_state_temp.shape)
        else:
            raise NotImplementedError(f'Unknown heuristic: {heuristic}')

        assert len(I) == len(J), f'Inconsistent pair lengths: {len(I)} != {len(J)}'
        num_selected_pairs = len(I)
        # logger.info(f'num_selected_pairs: {num_selected_pairs}')
        # selected_entropies = []
        batch_size = 50
        num_batches = int(np.ceil(num_selected_pairs / batch_size))
        batches_k = [list(range(i * batch_size, min((i + 1) * batch_size, num_selected_pairs))) for i in range(num_batches)]
        selected_entropies_list = Parallel(n_jobs=n_jobs)(delayed(worker)(k_list, I, J, transition_prob_old, stationary_state_old, log_transition_prob_old, rate_old) for k_list in batches_k)
        selected_entropies = [item for sublist in selected_entropies_list for item in sublist]

        min_entropy = np.min(selected_entropies)
        rate_old = min_entropy
        # logger.info(f'Minimum entropy: {min_entropy}')
        # input()
        min_indices = np.where(selected_entropies == min_entropy)[0]
        # min_idx = np.random.choice(min_indices)
        min_idx = min_indices[0]  # choose the first one for reproducibility

        rate_new = selected_entropies[min_idx]
        rates_upper.append(rate_new.item())

        i_new, j_new = I[min_idx], J[min_idx]
        # logger.info(f'[i_new, j_new]=[{i_new}, {j_new}]')
        all_indices = np.arange(n + 1)
        indices_not_ij = np.delete(all_indices, [i_new, j_new])
        stationary_state_old = np.append(stationary_state_old[indices_not_ij], stationary_state_old[i_new] + stationary_state_old[j_new])
        # logger.info(f'n={n}')
        # logger.info(f'stationary_state_old:\n{stationary_state_old}')
        # Top-left block (unmerged to unmerged)
        P_joint_TL = P_joint[np.ix_(indices_not_ij, indices_not_ij)]
        # Right block (unmerged to new merged)
        P_joint_R = np.sum(P_joint[np.ix_(indices_not_ij, [i_new, j_new])], axis=1, keepdims=True)
        # Bottom block (new merged to unmerged)
        P_joint_B = np.sum(P_joint[np.ix_([i_new, j_new], indices_not_ij)], axis=0, keepdims=True)
        # Bottom-right corner (new merged to new merged)
        P_joint_BR = np.sum(P_joint[np.ix_([i_new, j_new], [i_new, j_new])])
        # Combine the blocks to form the new joint probability matrix
        P_joint = np.block([
            [P_joint_TL, P_joint_R],
            [P_joint_B, P_joint_BR]
        ])
        # logger.info(f'n={n}')
        # logger.info(f'P_joint:\n{P_joint[:10, :10]}')
        transition_prob_old = P_joint / stationary_state_old.reshape(-1, 1)
        log_transition_prob_old = np.log2(transition_prob_old)
        log_transition_prob_old[np.isinf(log_transition_prob_old)] = 0
        # logger.info(f'log_transition_prob_old:\n{log_transition_prob_old[:10, :10]}')
        # record clusters and graph
        clusters_old = clusters[-1]
        unmerged_clusters = [clusters_old[idx] for idx in indices_not_ij]
        merged_cluster = clusters_old[i_new] + clusters_old[j_new]
        clusters_new = unmerged_clusters + [merged_cluster]
        clusters.append(clusters_new)
        
        # Gs.append(P_joint*2*E)

        # Compute lower bound on mutual information
        unmerged_cols = P_low[:, indices_not_ij]
        merged_col = np.sum(P_low[:, [i_new, j_new]], axis=1, keepdims=True)
        P_low = np.hstack([unmerged_cols, merged_col])
        logP_low = np.log2(P_low)
        logP_low[np.isinf(logP_low)] = 0
        rate_low = -np.sum(stationary_state * np.sum(P_low * logP_low, axis=1))
        rates_lower.append(rate_low.item())
        # logger.debug(f'rate_low: {rate_low}')
        # logger.debug(f'i_new: {i_new}, j_new: {j_new}')
        # logger.debug(f'P_low:\n{P_low[:5, :5]}')
        # input()
    rates_upper.append(0.0)
    rates_lower.append(0.0)
    clusters.append([list(range(N))])
    # Gs = [gs.tolist() for gs in Gs]
    # Gs.append([[0]])
    return rates_upper, rates_lower, clusters, Gs

def load_graph(network_file):
    if network_file.endswith('.txt'):
        return np.loadtxt(network_file)
    elif network_file.endswith('.npz'):
        data = np.load(network_file)
        key = data.files[0]
        return data[key]
    elif network_file.endswith('.mat'):
        return scipy.io.loadmat(network_file)['G'].squeeze()
    else:
        raise ValueError("Unsupported file format.")

def plot_rate_distortion(rates_upper, rates_lower, save_path):
    plt.figure(figsize=(6, 4), dpi=200)
    plt.plot(rates_upper, label='rates_upper', color='blue')
    plt.plot(rates_lower, label='rates_lower', color='orange')
    plt.xlabel('Scale')
    plt.ylabel('Rate')
    plt.legend()
    plt.savefig(save_path, dpi=200)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, default='karate.mat')
    parser.add_argument('--heu', type=int, default=7, help='Heuristic value')
    parser.add_argument('--num_pairs', type=int, default=100, help='Number of pairs to consider')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--logdir', type=str, default='logs/logs')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--n_jobs', type=int, default=4, help='Number of parallel jobs')

    return parser.parse_args()

def main():
    start_overall = time.time()
    args = get_args()
    commons.seed_all(args.seed)

    log_dir = commons.get_new_log_dir(root=args.logdir, prefix='rate_distortion_parallel', tag=args.tag)
    logger = commons.get_logger('rate_distortion_parallel', log_dir=log_dir)
    logger.info(f'Args: {args}')

    G = load_graph(args.graph)
    logger.info(f'Loaded graph: {G.shape}')

    rates_upper, rates_lower, clusters, Gs = rate_distortion(G=G, 
                                                             heuristic=args.heu, 
                                                             num_pairs=args.num_pairs, 
                                                             logger=logger,
                                                             n_jobs=args.n_jobs)

    plot_rate_distortion(rates_upper, rates_lower, save_path=os.path.join(log_dir, 'rate_distortion.png'))

    # save the four list into json files
    # with open(os.path.join(log_dir, 'rates_upper.json'), 'w') as f:
    #     json.dump(rates_upper, f)
    # with open(os.path.join(log_dir, 'rates_lower.json'), 'w') as f:
    #     json.dump(rates_lower, f)
    # with open(os.path.join(log_dir, 'clusters.json'), 'w') as f:
    #     json.dump(clusters, f)
    # with open(os.path.join(log_dir, 'Gs.json'), 'w') as f:
    #     json.dump(Gs, f)
    # save the four list into npz file
    np.save(os.path.join(log_dir, 'rates_upper.npy'), rates_upper)
    np.save(os.path.join(log_dir, 'rates_lower.npy'), rates_lower)
    np.savez_compressed(os.path.join(log_dir, 'clusters.npz'), clusters=np.array(clusters, dtype=object))
    # np.savez_compressed(os.path.join(log_dir, 'Gs.npz'), Gs=np.array(Gs, dtype=object))

    end_overall = time.time()
    logger.info(f'Time elapsed: {end_overall - start_overall:.2f} seconds')


if __name__ == "__main__":
    main()
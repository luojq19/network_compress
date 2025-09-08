import sys
sys.path.append('.')
from itertools import combinations
from typing import List, Tuple, Optional, Any, Dict
import numpy as np

def _row_normalize(G: np.ndarray) -> np.ndarray:
    """
    Row-normalize a nonnegative matrix to make a (possibly substochastic) transition matrix.
    Rows that sum to 0 are kept as all-zeros (consistent with MATLAB's division behavior
    followed by log handling that zeroes infs).
    """
    G = np.asarray(G, dtype=float)
    rs = G.sum(axis=1, keepdims=True)
    # Avoid division-by-zero: keep zero-rows as zeros.
    P = np.divide(G, rs, out=np.zeros_like(G, dtype=float), where=(rs != 0))
    return P


def _power_iteration_left(P: np.ndarray,
                          max_iter: int = 10000,
                          tol: float = 1e-12) -> np.ndarray:
    """
    Compute the stationary distribution (left eigenvector for eigenvalue 1)
    using simple power iteration: start with uniform p, repeatedly p <- p P.
    This is robust and avoids complex eigenvectors from dense eigensolvers.

    Returns a 1D vector p_ss with sum = 1. If the chain is not ergodic,
    this still converges for many practical cases to a stationary distribution
    of the communicating class reached from the initial distribution.
    """
    n = P.shape[0]
    p = np.full(n, 1.0 / n, dtype=float)
    for _ in range(max_iter):
        p_next = p @ P
        s = p_next.sum()
        if s > 0:
            p_next = p_next / s
        # Convergence check (L1 distance)
        if np.linalg.norm(p_next - p, ord=1) < tol:
            p = p_next
            break
        p = p_next
    # Numerical clean-up: clip tiny negatives from rounding
    p = np.clip(p, 0.0, None)
    s = p.sum()
    return p / s if s > 0 else np.full(n, 1.0 / n, dtype=float)


def _safe_log2(x: np.ndarray) -> np.ndarray:
    """Return log2(x) with log2(0) treated as 0 (consistent with MATLAB code that sets -inf -> 0)."""
    out = np.zeros_like(x, dtype=float)
    mask = x > 0
    out[mask] = np.log2(x[mask])
    return out


def _topk_indices_from_flat(arr2d: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (I, J) for the top-k entries by value from a 2D array, reading only where entries > 0.
    Ties are resolved by argpartition's behavior; exact order is not important for selection.
    """
    flat = arr2d.ravel()
    pos_mask = flat > 0
    if not np.any(pos_mask):
        return np.array([], dtype=int), np.array([], dtype=int)
    flat_pos = flat[pos_mask]
    # k may exceed count; cap it
    k = min(k, flat_pos.size)
    # indices among the positive subset
    idx_pos = np.argpartition(-flat_pos, kth=k-1)[:k]
    # map back to original flat indices
    idx_all_pos = np.flatnonzero(pos_mask)[idx_pos]
    I, J = np.unravel_index(idx_all_pos, arr2d.shape)
    return I, J


def _pair_indices_connected(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return upper-triangular (i<j) indices for pairs connected by an edge in the symmetrized matrix.
    Equivalent to MATLAB: [I,J] = find(triu(P_old + P_old', 1))
    """
    M = np.triu(P + P.T, k=1)
    I, J = np.nonzero(M > 0)
    return I, J


def _all_pairs(n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    All i<j pairs for current 'n_nodes' items, zero-based (Python).
    MATLAB uses 1..(n+1); here we use 0..n inclusive which has size n+1.
    """
    pairs = np.fromiter((i for c in combinations(range(n_nodes), 2) for i in c),
                        dtype=int)
    return pairs[0::2], pairs[1::2]


def rate_distortion(
    G: np.ndarray,
    heuristic: int,
    num_pairs: int,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray, List[List[List[int]]], List[np.ndarray]]:
    """
    Python rewrite of the MATLAB rate_distortion.m

    Parameters
    ----------
    G : (N,N) array_like
        Adjacency matrix of a (possibly weighted, directed) network.
    heuristic : int
        Determines how candidate node (cluster) pairs are chosen each iteration.
        1: all pairs
        2: random 'num_pairs' pairs from all pairs
        3: all pairs connected by an edge (in P + P^T)
        4: random 'num_pairs' connected pairs
        5: top 'num_pairs' by joint transition (P_joint + P_joint^T) on upper triangle
        6: top 'num_pairs' by (P_joint + P_joint^T + diag terms) on upper triangle
        7: top 'num_pairs' by combined stationary mass (p_i + p_j)
        8: iteratively merge a fixed node into the last (I=0, J=last)
    num_pairs : int
        How many candidate pairs to consider for heuristics that subsample.
    rng : np.random.Generator, optional
        Optional numpy RNG for reproducibility. If None, uses default.

    Returns
    -------
    S : (N,) ndarray
        Upper bound on mutual information after clustering into n clusters (S[n-1]).
    S_low : (N,) ndarray
        Lower bound on mutual information.
    clusters : list of length N
        clusters[n-1] is a list of current clusters when there are n clusters,
        each inner cluster is a list of 1-based node indices (to match MATLAB).
    Gs : list of length N
        Gs[n-1] is the joint transition probability matrix (scaled back to adjacency) for n clusters.
    """
    if rng is None:
        rng = np.random.default_rng()

    G = np.asarray(G, dtype=float)
    N = G.shape[0]
    assert G.ndim == 2 and G.shape[0] == G.shape[1], "G must be square"

    # Total edge weight E = sum(G)/2 (matches MATLAB, works for directed too as a scaling)
    E = G.sum() / 2.0

    # Storage
    S = np.zeros(N, dtype=float)       # Upper bound on entropy rate after clustering
    S_low = np.zeros(N, dtype=float)   # Lower bound on entropy rate
    clusters: List[List[List[int]]] = [None] * N
    Gs: List[np.ndarray] = [None] * N

    # Transition probability matrix
    P_old = _row_normalize(G)

    # Stationary distribution (left eigenvector of P with eigenvalue 1)
    # MATLAB used eigs(P_old'). We use robust power iteration on the left.
    p_ss = _power_iteration_left(P_old)
    p_ss_old = p_ss.copy()  # will be updated as clusters merge
    p_ss_init = p_ss.copy()  # keep original for lower bound (mirrors MATLAB's use of p_ss)

    # Initial entropy S_old = -sum_i p_ss(i) * sum_j P(i,j) log2 P(i,j)
    logP_old = _safe_log2(P_old)
    S_old = -np.sum(p_ss_old * np.sum(P_old * logP_old, axis=1))

    # Joint transition matrix for current partition (P_joint = diag(p_ss) * P)
    P_joint = (p_ss_old[:, None]) * P_old

    # Lower bound initialization
    P_low = P_old.copy()

    # Record initial state at n = N clusters (index N-1)
    S[-1] = S_old
    S_low[-1] = S_old
    # MATLAB clusters are 1-based node ids
    clusters[-1] = [[i] for i in range(1, N + 1)]
    Gs[-1] = G.copy()

    # Main loop: n goes from N-1 down to 2 (inclusive) in MATLAB; Python uses 0-based indexing
    # At the start of an iteration with target 'n' clusters, we currently have 'n+1' clusters.
    for n in range(N - 1, 1, -1):
        # Number of current clusters before merging this step
        cur_k = n + 1  # matches MATLAB's (n+1)

        # --- Choose candidate pairs (i, j) among the current cur_k clusters (0..cur_k-1) ---
        if heuristic == 1:
            # All pairs
            I, J = _all_pairs(cur_k)

        elif heuristic == 2:
            # Random subset of all pairs
            I_all, J_all = _all_pairs(cur_k)
            total = I_all.size
            k = min(num_pairs, total)
            if total > 0:
                sel = rng.choice(total, size=k, replace=False)
                I, J = I_all[sel], J_all[sel]
            else:
                I = J = np.array([], dtype=int)

        elif heuristic == 3:
            # All connected pairs (by edge in P_old + P_old^T)
            I, J = _pair_indices_connected(P_old)

        elif heuristic == 4:
            # Random subset of connected pairs
            I_all, J_all = _pair_indices_connected(P_old)
            total = I_all.size
            k = min(num_pairs, total)
            if total > 0:
                sel = rng.choice(total, size=k, replace=False)
                I, J = I_all[sel], J_all[sel]
            else:
                I = J = np.array([], dtype=int)

        elif heuristic == 5:
            # Top 'num_pairs' by upper-triangular (P_joint + P_joint^T)
            P_joint_symm = np.triu(P_joint + P_joint.T, k=1)
            I, J = _topk_indices_from_flat(P_joint_symm, k=min(num_pairs, np.count_nonzero(P_joint_symm > 0)))

        elif heuristic == 6:
            # Top 'num_pairs' by upper-triangular (P_joint + P_joint^T + diag terms expanded)
            diag_Pj = np.diag(P_joint)
            A = P_joint + P_joint.T + np.tile(diag_Pj, (cur_k, 1)) + np.tile(diag_Pj[:, None], (1, cur_k))
            P_joint_symm = np.triu(A, k=1)
            I, J = _topk_indices_from_flat(P_joint_symm, k=min(num_pairs, np.count_nonzero(P_joint_symm > 0)))

        elif heuristic == 7:
            # Top 'num_pairs' by combined stationary mass p_i + p_j
            T = np.triu(np.tile(p_ss_old, (cur_k, 1)) + np.tile(p_ss_old[:, None], (1, cur_k)), k=1)
            I, J = _topk_indices_from_flat(T, k=min(num_pairs, (cur_k * (cur_k - 1)) // 2))

        elif heuristic == 8:
            # Iteratively add random nodes to one large cluster (MATLAB set I=1; J=n+1)
            # Use fixed I=0, J=last to mirror that behavior deterministically
            if cur_k >= 2:
                I = np.array([0], dtype=int)
                J = np.array([cur_k - 1], dtype=int)
            else:
                I = J = np.array([], dtype=int)
        else:
            raise ValueError('Variable "heuristic" is not properly defined (must be 1..8).')

        num_pairs_temp = I.size
        if num_pairs_temp == 0:
            # No candidate pairs found; cannot merge further. We keep values and break.
            # (This is a safety fallback; ideally the heuristics always provide at least one pair.)
            break

        # --- Evaluate S_temp (upper bound on MI) for each candidate pair ---
        S_all = np.zeros(num_pairs_temp, dtype=float)

        for t in range(num_pairs_temp):
            i = I[t]
            j = J[t]
            if i > j:
                i, j = j, i  # ensure i < j (as in upper triangle)

            # Indices excluding i and j, in order [0..i-1, i+1..j-1, j+1..cur_k-1]
            inds_not_ij = np.concatenate([np.arange(0, i),
                                          np.arange(i + 1, j),
                                          np.arange(j + 1, cur_k)], axis=0)

            # New stationary distribution after merging clusters i and j
            p_ss_temp = np.concatenate([p_ss_old[inds_not_ij], [p_ss_old[i] + p_ss_old[j]]])

            # New transition probabilities:
            # P_temp_1: for each remaining cluster r (not i/j), probability to go into {i,j} (aggregated and renormalized)
            #   numerator: sum over dest in {i,j} of p_r * P(r, dest)
            #   denominator: p_ss_temp[r] = p_r (unchanged for those)
            # resulting shape: (cur_k-2+1) = (n-1) entries (excluding the merged cluster row)
            num_1 = np.sum((p_ss_old[inds_not_ij, None]) * P_old[inds_not_ij][:, [i, j]], axis=1)
            den_1 = p_ss_temp[:-1]  # p_r for r not merged
            P_temp_1 = np.divide(num_1, den_1, out=np.zeros_like(num_1), where=(den_1 > 0))

            # P_temp_2: from merged cluster to each remaining cluster (renormalized)
            #   numerator: sum over src in {i,j} of p_src * P(src, r)
            #   denominator: p_ss_temp[-1] = p_i + p_j
            num_2 = np.sum((p_ss_old[[i, j], None]) * P_old[[i, j]][:, inds_not_ij], axis=0)
            den_2 = p_ss_temp[-1]
            P_temp_2 = (num_2 / den_2) if den_2 > 0 else np.zeros_like(num_2)

            # P_temp_3: self-transition of merged cluster (renormalized)
            #   numerator: sum over src in {i,j}, dest in {i,j} of p_src * P(src, dest)
            num_3 = np.sum((p_ss_old[[i, j], None]) * P_old[[i, j]][:, [i, j]])
            den_3 = p_ss_temp[-1]
            P_temp_3 = (num_3 / den_3) if den_3 > 0 else 0.0

            # Logs with 0 -> 0 convention
            logP_temp_1 = _safe_log2(P_temp_1)
            logP_temp_2 = _safe_log2(P_temp_2)
            logP_temp_3 = _safe_log2(np.array(P_temp_3, ndmin=1))[0]

            # Compute change in the upper bound on mutual information (dS), matching MATLAB's formula.
            # Break the long expression into readable chunks.
            term_new = -np.sum(p_ss_temp[:-1] * P_temp_1 * logP_temp_1) \
                       - p_ss_temp[-1] * np.sum(P_temp_2 * logP_temp_2) \
                       - p_ss_temp[-1] * P_temp_3 * logP_temp_3

            # Old contributions to be removed/adjusted
            col_i = P_old[:, i]
            col_j = P_old[:, j]
            row_i = P_old[i, :]
            row_j = P_old[j, :]

            log_col_i = logP_old[:, i]
            log_col_j = logP_old[:, j]
            log_row_i = logP_old[i, :]
            log_row_j = logP_old[j, :]

            term_old_cols = np.sum(p_ss_old * col_i * log_col_i) + np.sum(p_ss_old * col_j * log_col_j)
            term_old_rows = p_ss_old[i] * np.sum(row_i * log_row_i) + p_ss_old[j] * np.sum(row_j * log_row_j)
            term_old_self = p_ss_old[i] * (P_old[i, i] * log_row_i[i] + P_old[i, j] * log_row_i[j]) \
                            + p_ss_old[j] * (P_old[j, j] * log_row_j[j] + P_old[j, i] * log_row_j[i])

            dS = term_new + term_old_cols + term_old_rows - term_old_self
            S_all[t] = S_old + dS

        # Pick the minimum-entropy pair; break ties randomly (mirror MATLAB datasample)
        min_val = S_all.min()
        min_inds = np.flatnonzero(S_all == min_val)
        choice = rng.choice(min_inds)
        S_old = S_all[choice]
        S[n - 1] = S_old  # store at index for 'n' clusters

        # --- Update the partition with the chosen merge (i_new, j_new) ---
        i_new, j_new = I[choice], J[choice]
        if i_new > j_new:
            i_new, j_new = j_new, i_new

        inds_not_ij = np.concatenate([np.arange(0, i_new),
                                      np.arange(i_new + 1, j_new),
                                      np.arange(j_new + 1, cur_k)], axis=0)

        # New stationary distribution after merge
        p_ss_new = np.concatenate([p_ss_old[inds_not_ij], [p_ss_old[i_new] + p_ss_old[j_new]]])

        # Update P_joint by merging rows/cols i_new and j_new
        # Start from current joint = diag(p_ss_old) * P_old
        P_joint = (p_ss_old[:, None]) * P_old

        # Build merged joint matrix:
        #   upper-left: keep rows/cols not in {i,j}
        #   last column: sum of columns i and j (for rows not in {i,j})
        #   last row: sum of rows i and j (for cols not in {i,j})
        #   bottom-right: sum over {i,j}x{i,j}
        UL = P_joint[np.ix_(inds_not_ij, inds_not_ij)]
        last_col = np.sum(P_joint[np.ix_(inds_not_ij, [i_new, j_new])], axis=1, keepdims=True)
        last_row = np.sum(P_joint[np.ix_([i_new, j_new], inds_not_ij)], axis=0, keepdims=True)
        bottom_right = np.array([[np.sum(P_joint[np.ix_([i_new, j_new], [i_new, j_new])])]])

        P_joint = np.block([
            [UL,            last_col],
            [last_row,      bottom_right]
        ])

        # New transition matrix after merge
        with np.errstate(invalid='ignore', divide='ignore'):
            P_old = np.divide(P_joint, p_ss_new[:, None], out=np.zeros_like(P_joint), where=(p_ss_new[:, None] > 0))
        p_ss_old = p_ss_new

        # Update logs
        logP_old = _safe_log2(P_old)

        # Record clusters and "graph" Gs (convert joint back to adjacency scale by *2E like MATLAB)
        # clusters[n-1] is built from clusters[n] by merging i_new and j_new
        prev_clusters = clusters[n]  # clusters at (n+1) clusters
        merged_cluster = prev_clusters[i_new] + prev_clusters[j_new]
        # keep order: [0..i_new-1], [i_new+1..j_new-1], [j_new+1..end], then merged at the end
        new_list = [prev_clusters[k] for k in inds_not_ij] + [merged_cluster]
        clusters[n - 1] = new_list
        Gs[n - 1] = P_joint * (2.0 * E)

        # Lower bound update: P_low merges ONLY THE COLUMNS like in MATLAB code
        # P_low = [P_low(:, others), P_low(:,i) + P_low(:,j)]
        merged_col = (P_low[:, i_new] + P_low[:, j_new])[:, None]
        P_low = np.concatenate([P_low[:, inds_not_ij], merged_col], axis=1)

        # Recompute lower bound S_low(n) = -sum_i p_ss_init(i) * sum_j P_low(i,j) log2 P_low(i,j)
        logP_low = _safe_log2(P_low)
        S_low[n - 1] = -np.sum(p_ss_init * np.sum(P_low * logP_low, axis=1))

    return S, S_low, clusters, Gs

def read_edge_list_to_adjmatrix(filename):
    edges = []
    nodes_set = set()

    # 逐行读取文件
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():  # 跳过空行
                u, v = line.strip().split()
                edges.append((u, v))
                nodes_set.update([u, v])

    # 给每个节点分配一个整数索引
    nodes = sorted(list(nodes_set))  # 排序，确保结果稳定
    node_index = {node: idx for idx, node in enumerate(nodes)}

    # 初始化邻接矩阵
    N = len(nodes)
    adj_matrix = np.zeros((N, N), dtype=int)

    # 填充邻接矩阵
    for u, v in edges:
        i, j = node_index[u], node_index[v]
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1  # 无向图 → 对称

    return adj_matrix, nodes

# ----------------------------
# Example usage (remove or adapt in your codebase)
# ----------------------------
if __name__ == "__main__":
    # Small demo graph (undirected-like)
    # G_demo = np.array([
    #     [0, 1, 0, 1],
    #     [1, 0, 0, 0],
    #     [0, 0, 0, 1],
    #     [1, 0, 1, 0]
    # ])
    # G_demo, _ = read_edge_list_to_adjmatrix('data/treeoflife.interactomes/394.txt')
    # # G_demo is a np array, I want to write it to a txt file
    # from scipy.io import savemat
    # savemat("matrix.mat", {"my_matrix": G_demo})
    from scipy.io import loadmat

    # Load the .mat file
    mat = loadmat("/work/jiaqi/Network_compressibility/graphs_Protein_samples_undirected.mat")

    # Inspect keys
    print(mat.keys())

    G_yeast = mat['G_yeast']
    print(G_yeast.shape, G_yeast.dtype)
    # print(G_yeast)

    # G_demo = G_yeast[0][0]
    # print(G_demo, type(G_demo), G_demo.shape)
    import networkx as nx

    G_demo = nx.to_numpy_array(nx.karate_club_graph())
    import time
    heuristic = 1
    start_time = time.time()
    rates_upper, rates_lower, clusters, Gs = rate_distortion(G_demo, heuristic=heuristic, num_pairs=100, rng=np.random.default_rng(42))
    end_time = time.time()
    print("S (upper bound):\n", rates_upper)
    print("S_low (lower bound):\n", rates_lower)

    import scipy
    # load karate_results.mat
    results = scipy.io.loadmat('karate_results.mat')
    print(f'Loaded results: {results.keys()}')
    S = results['S'].squeeze()
    S_low = results['S_low'].squeeze()
    # reverse S
    # S = S[::-1]
    # S_low = S_low[::-1]
    print(f'{S.shape}, {S_low.shape}')
    # check if results match, print passed if match and failed if not
    tol = 1e-5
    upper_match = np.allclose(rates_upper, S, atol=tol)
    lower_match = np.allclose(rates_lower, S_low, atol=tol)
    if upper_match:
        print("Passed: rates_upper matches S.")
    else:
        print("Failed: rates_upper does not match S.")
        diff = rates_upper - S
        print(f'Difference (upper):\n{diff}')
    if lower_match:
        print("Passed: rates_lower matches S_low.")
    else:
        print("Failed: rates_lower does not match S_low.")
        diff = rates_lower - S_low
        print(f'Difference (lower):\n{diff}')


    # import matplotlib.pyplot as plt

    from utils import commons
    dirr = commons.get_new_log_dir()
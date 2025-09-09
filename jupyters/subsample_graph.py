# In[]
import os
import numpy as np
import networkx as nx
from tqdm.auto import tqdm
import numpy as np

def random_walk_subgraph(adj: np.ndarray, n: int, seed: int | None = None, start: int | None = None):
    """
    Perform a (weighted) random walk on a graph given by its adjacency matrix.
    Start at a (random or specified) node, keep walking until n *distinct* nodes
    have been reached, and return the induced adjacency matrix and the list of
    visited node indices in order of first visit.

    Parameters
    ----------
    adj : np.ndarray
        Square (N x N) adjacency matrix. Nonnegative entries are treated as edge weights.
    n : int
        Number of distinct nodes to collect.
    seed : int | None
        Optional seed for reproducibility.
    start : int | None
        Optional start node index. If None, a non-isolated node is chosen at random.

    Returns
    -------
    sub_adj : np.ndarray
        (n x n) adjacency matrix induced by the reached nodes, ordered by first visit.
    nodes : list[int]
        Indices of the n nodes, in the order they were first reached.

    Raises
    ------
    ValueError
        If adj is not square, n < 1, n > size of the connected component reachable
        from the start node (treating the graph as undirected for reachability),
        or if no valid start node exists (all nodes isolated).
    """
    # --- basic checks ---
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be a square 2D numpy array.")
    if n < 1:
        raise ValueError("n must be >= 1.")
    N = adj.shape[0]
    if n > N:
        raise ValueError("n cannot exceed the number of nodes in the graph.")

    if np.any(adj < 0):
        raise ValueError("adjacency matrix must be nonnegative (edge weights).")

    rng = np.random.default_rng(seed)

    # nodes with at least one outgoing edge (including self-loop)
    outdeg = adj.sum(axis=1)
    non_isolated = np.flatnonzero(outdeg > 0)

    if non_isolated.size == 0:
        raise ValueError("All nodes are isolated; random walk cannot proceed.")

    # choose start if not provided
    if start is None:
        start = int(rng.choice(non_isolated))
    else:
        if not (0 <= start < N):
            raise ValueError("start index out of bounds.")
        if outdeg[start] == 0:
            raise ValueError("Chosen start node is isolated (no outgoing edges).")

    # --- pre-check: can we possibly reach n distinct nodes from start? ---
    # Treat edges as undirected to compute the connected component for reachability.
    # (This prevents getting stuck trying to exceed the component's size.)
    G = (adj > 0)
    undirected = G | G.T

    # BFS/DFS to find reachable set from start
    stack = [start]
    seen = set([start])
    while stack:
        u = stack.pop()
        nbrs = np.flatnonzero(undirected[u])
        for v in nbrs:
            if v not in seen:
                seen.add(int(v))
                stack.append(int(v))
    comp_size = len(seen)
    if n > comp_size:
        raise ValueError(
            f"Requested n={n} exceeds the size of the start node's reachable component ({comp_size})."
        )

    # --- random walk collecting first-time visits ---
    visited = [start]
    visited_set = {start}
    current = start

    # A loose cap to avoid pathological loops; practically, you'll hit n quickly.
    max_steps = max(10_000, 500 * n)
    steps = 0

    while len(visited) < n:
        steps += 1
        if steps > max_steps:
            raise RuntimeError(
                "Random walk step cap exceeded before reaching n distinct nodes. "
                "This is unlikely; consider increasing the cap or verifying the graph."
            )

        neighbors = np.flatnonzero(adj[current] > 0)
        if neighbors.size == 0:
            # Shouldn't happen due to earlier checks (non-isolated start & undirected reachability),
            # but guard anyway.
            raise RuntimeError(f"Walk got stuck at node {current} with no outgoing edges.")

        weights = adj[current, neighbors].astype(float)
        p = weights / weights.sum()

        nxt = int(rng.choice(neighbors, p=p))

        if nxt not in visited_set:
            visited.append(nxt)
            visited_set.add(nxt)

        current = nxt

    sub_adj = adj[np.ix_(visited, visited)]
    return sub_adj, visited


data_dir = '../data/treeoflife.interactomes.max_cc_adj'
node_dir = '../data/treeoflife.interactomes.max_cc_nodes'
save_data_dir = '../data/treeoflife.interactomes.max_cc.rw1000_adj'
save_node_dir = '../data/treeoflife.interactomes.max_cc.rw1000_nodes'
os.makedirs(save_data_dir, exist_ok=True)
os.makedirs(save_node_dir, exist_ok=True)
interactome_list = []
for file in os.listdir(data_dir):
    if file.endswith('.npz'):
        interactome_list.append(file.split('.')[0])
print(f'Found {len(interactome_list)} interactomes.')
print(interactome_list[:5])
threshold = 1000
repeats = 50
# for interactome in tqdm(interactome_list):
#     # print(f'Processing {interactome}...')
#     data = np.load(os.path.join(data_dir, f'{interactome}.npz'))
#     adj = data['adj']
#     nodes = np.loadtxt(os.path.join(node_dir, f'{interactome}_nodes.txt'), dtype=str)
#     if len(adj) <= threshold:
#         np.savez_compressed(os.path.join(save_data_dir, f'{interactome}_0.npz'), adj=adj)
#         np.savetxt(os.path.join(save_node_dir, f'{interactome}_0.txt'), nodes, fmt='%s')
#     else:
#         for i in range(50):
#             try:
#                 submatrix, subnode_indices = random_walk_subgraph(adj, threshold, seed=i)
#                 subnodes = nodes[subnode_indices]
#                 np.savez_compressed(os.path.join(save_data_dir, f'{interactome}_{i}.npz'), adj=submatrix)
#                 np.savetxt(os.path.join(save_node_dir, f'{interactome}_{i}_nodes.txt'), subnodes, fmt='%s')
#             except Exception as e:
#                 print(f"Error processing {interactome}_{i}: {e}")
interactome = '322710'
i = 18
data = np.load(os.path.join(data_dir, f'{interactome}.npz'))
adj = data['adj']
nodes = np.loadtxt(os.path.join(node_dir, f'{interactome}_nodes.txt'), dtype=str)
submatrix, subnode_indices = random_walk_subgraph(adj, threshold, seed=i * 10)
subnodes = nodes[subnode_indices]
print(os.path.exists(os.path.join(save_data_dir, f'{interactome}_{i}.npz')))
np.savez_compressed(os.path.join(save_data_dir, f'{interactome}_{i}.npz'), adj=submatrix)
np.savetxt(os.path.join(save_node_dir, f'{interactome}_{i}_nodes.txt'), subnodes, fmt='%s')
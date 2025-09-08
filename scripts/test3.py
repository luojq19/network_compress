import numpy as np  # 数值计算基础包
from itertools import combinations  # 用于生成组合（对应 MATLAB 的 nchoosek）
from numpy.random import default_rng  # 随机数生成器（对应 MATLAB 的 randsample/datasample）

try:
    from scipy.sparse.linalg import eigs as sparse_eigs  # 近似 MATLAB 的 eigs
    _has_scipy = True
except Exception:
    _has_scipy = False  # 没有 scipy 时，退化为 numpy 的 eig


def rate_distortion(G, heuristic, num_pairs, random_state=None):
    """
    这是对 MATLAB 版 rate_distortion 的逐行忠实 Python 翻译。
    参数与返回值保持一致语义（维度/含义），下方每行尽量与原始 MATLAB 行对齐并注释。

    输入:
      G: 形状 NxN 的邻接矩阵（可以是加权、有向）
      heuristic: 整数，决定在每次迭代中选择哪些成对聚合的候选
      num_pairs: 每次迭代考虑的候选对数（在某些 heuristic 下使用）
      random_state: 可选，随机种子，便于复现

    返回:
      S: 长度为 N 的 numpy 数组，上界互信息（文中称“熵率”）随簇数变化的曲线
      S_low: 长度为 N 的 numpy 数组，下界互信息（熵率下界）
      clusters: 长度为 N 的 Python 列表，第 n 项是 n 个簇各自包含的原始节点列表
      Gs: 长度为 N 的 Python 列表，第 n 项为 n 簇的“联合转移概率矩阵”再乘以 2E（与 MATLAB 同步）
    """
    rng = default_rng(random_state)  # 随机数生成器，等价于 MATLAB 的 randsample/datasample 可复现

    # % Size of network:
    N = G.shape[0]  # 网络规模 N（对应 MATLAB: N = size(G,1);）
    E = np.sum(G) / 2.0  # E = sum(G(:))/2; 若无向图这相当于边数，加权图是权重和的一半

    # % Variables to save:
    S = np.zeros(N)       # 上界熵率（互信息上界），对应 S = zeros(1, N);
    S_low = np.zeros(N)   # 下界熵率（互信息下界），对应 S_low = zeros(1, N);
    clusters = [None]*N   # clusters{n}，这里使用 Python 列表来模拟元胞数组
    Gs = [None]*N         # Gs{n}，记录每一步的（联合转移）图

    # % Transition probability matrix:
    # P_old = G./repmat(sum(G,2),1,N);
    row_sums = G.sum(axis=1, keepdims=True)                    # 按行求和（出度）
    P_old = np.divide(G, row_sums, where=row_sums!=0)  # 行归一化得到转移概率矩阵

    # % Compute steady-state distribution:
    # [p_ss, D] = eigs(P_old'); % Works for all networks
    # [~, ind] = max(diag(D));
    # p_ss = p_ss(:,ind)/sum(p_ss(:,ind));
    # p_ss_old = p_ss;
    #
    # Python 实现说明：
    #   我们需要 P_old' 的特征分解来拿到特征值最“靠近 1”的特征向量（稳态分布）。
    #   优先用 scipy.sparse.linalg.eigs(P_old.T, k=1)，否则退化到 numpy.linalg.eig 再选取“实部最大/最接近 1”的特征值。
    if _has_scipy:
        # 使用 scipy 的 eigs 求解最大的模特征值对应的特征向量（与 MATLAB 的 eigs 类似）
        vals, vecs = sparse_eigs(P_old.T, k=1)  # 返回一个特征值和对应的右特征向量
        p_ss = np.real(vecs[:, 0])              # 取实部作为稳态分布的“方向”
    else:
        # 退化：用 numpy.linalg.eig，取实部最大的特征值对应的特征向量
        vals, vecs = np.linalg.eig(P_old.T)
        # 选择“实部”最大的特征值（与 MATLAB 代码中 max(diag(D)) 的精神近似）
        idx = np.argmax(vals.real)
        p_ss = np.real(vecs[:, idx])

    # 归一化，使其为概率分布；并将其调整为非负（与马尔可夫稳态向量同号即可）
    # 对于可能出现的全负或复数方向，我们取实部并按 L1 归一化
    if p_ss.sum() == 0:
        # 退路：若数值奇异，回退为均匀分布（避免除零）
        p_ss = np.ones(N) / N
    else:
        # 如果向量总和为负，整体乘以 -1，使总和为正（仅改变方向，不改变特征）
        if p_ss.sum() < 0:
            p_ss = -p_ss
        p_ss = p_ss / p_ss.sum()

    p_ss_old = p_ss.copy()  # 保存为“旧”的稳态分布（对应 MATLAB: p_ss_old = p_ss;）

    # % Caluclate initial entropy:
    # logP_old = log2(P_old);
    logP_old = np.zeros_like(P_old)  # 先分配，便于逐元素处理
    with np.errstate(divide='ignore'):
        logP_old = np.log2(P_old, where=(P_old > 0), out=logP_old)  # 对 >0 的位置取 log2
    logP_old[~np.isfinite(logP_old)] = 0  # 将 -inf/inf/nan 置零（对应 MATLAB：isinf置0）

    # S_old = -sum(p_ss_old.*sum(P_old.*logP_old,2));
    S_old = -np.sum(p_ss_old * np.sum(P_old * logP_old, axis=1))  # 初始的熵率（互信息上界）

    # P_joint = P_old.*repmat(p_ss_old, 1, N);
    P_joint = P_old * p_ss_old[:, None]  # 对应逐行乘以稳态概率，得到联合概率 P(i,j)=p(i)P(i->j)

    # P_low = P_old;
    P_low = P_old.copy()  # 下界部分的工作矩阵

    # % Record initial values:
    # S(end) = S_old;  MATLAB 的 end 是末尾下标；Python 直接用 [-1]
    S[-1] = S_old
    S_low[-1] = S_old
    # clusters{end} = num2cell(1:N);
    # Python 里保留节点编号为 1..N（忠实于 MATLAB 的 1-based 存储）
    clusters[-1] = [[k] for k in range(1, N + 1)]
    # Gs{end} = G;
    Gs[-1] = G.copy()

    # % Loop over the number of clusterings:
    # for n = (N-1):-1:2
    for n in range(N - 1, 1, -1):  # 递减循环 n = N-1, N-2, ..., 2

        # % 不同的候选节点对 I,J 的生成（对应 heuristic 设定）:
        if heuristic == 1:
            # % Try combining all pairs:
            # pairs = nchoosek(1:(n+1),2);
            # I = pairs(:,1); J = pairs(:,2);
            nodes = list(range(1, n + 2))  # MATLAB 1:(n+1)
            pairs = np.array(list(combinations(nodes, 2)), dtype=int)
            I = pairs[:, 0]
            J = pairs[:, 1]

        elif heuristic == 2:
            # % Pick num_pairs node pairs at random:
            nodes = list(range(1, n + 2))
            all_pairs = np.array(list(combinations(nodes, 2)), dtype=int)
            k = min(num_pairs, len(all_pairs))  # 取不超过全部对数
            inds = rng.choice(len(all_pairs), size=k, replace=False)  # 无放回随机抽样
            chosen = all_pairs[inds]
            I, J = chosen[:, 0], chosen[:, 1]

        elif heuristic == 3:
            # % Try combining all pairs connected by an edge:
            # [I, J] = find(triu(P_old + P_old', 1));
            M = np.triu(P_old + P_old.T, k=1)  # 上三角（不含对角）
            I, J = np.where(M > 0)            # 注意：这里 I,J 是 0-based 索引
            # 将 0-based 索引转换为 1-based（与 MATLAB 对齐）
            I = (I + 1).astype(int)
            J = (J + 1).astype(int)

        elif heuristic == 4:
            # % Pick num_pairs node pairs at random that are connected by an edge:
            M = np.triu(P_old + P_old.T, k=1)
            I0, J0 = np.where(M > 0)
            m = len(I0)
            if m == 0:
                I = np.array([], dtype=int)
                J = np.array([], dtype=int)
            else:
                k = min(num_pairs, m)
                pick = rng.choice(m, size=k, replace=False)
                I = (I0[pick] + 1).astype(int)  # 转为 1-based
                J = (J0[pick] + 1).astype(int)

        elif heuristic == 5:
            # % Pick num_pairs node pairs with largest joint transition probabilities:
            # P_joint_symm = triu(P_joint + P_joint', 1);
            # [~, inds] = maxk(P_joint_symm(:), min([num_pairs, sum(P_joint_symm(:) > 0)]));
            # [I, J] = ind2sub([n+1, n+1], inds);
            M = np.triu(P_joint + P_joint.T, k=1)
            flat = M.ravel()
            positive_mask = flat > 0
            num_pos = int(np.sum(positive_mask))
            k = min(num_pairs, num_pos)
            if k == 0:
                I = np.array([], dtype=int)
                J = np.array([], dtype=int)
            else:
                # 取前 k 大的索引（等价 maxk）
                # argpartition 得到未完全排序的前 k 大，再精排
                pos_idx = np.where(positive_mask)[0]
                pos_vals = flat[pos_idx]
                order = np.argpartition(-pos_vals, kth=k-1)[:k]
                top_idx_unsorted = pos_idx[order]
                # 为了尽量贴近“k 大”的排序，此处再按数值降序排序（与 MATLAB maxk 顺序一致）
                top_vals = flat[top_idx_unsorted]
                sorted_order = np.argsort(-top_vals)
                top_idx = top_idx_unsorted[sorted_order]
                # 将线性索引还原到矩阵索引
                I0, J0 = np.unravel_index(top_idx, M.shape, order='C')
                I = (I0 + 1).astype(int)  # 转为 1-based
                J = (J0 + 1).astype(int)

        elif heuristic == 6:
            # % Pick num_pairs node pairs with largest joint transition probabilities
            # % plus self-transition probabilities:
            # P_joint_symm = triu(P_joint + P_joint' + repmat(diag(P_joint), 1, n+1) +...
            #     repmat(diag(P_joint)', n+1, 1), 1);
            d = np.diag(P_joint)  # diag(P_joint)
            M = np.triu(P_joint + P_joint.T + d[:, None] + d[None, :], k=1)
            flat = M.ravel()
            positive_mask = flat > 0
            num_pos = int(np.sum(positive_mask))
            k = min(num_pairs, num_pos)
            if k == 0:
                I = np.array([], dtype=int)
                J = np.array([], dtype=int)
            else:
                pos_idx = np.where(positive_mask)[0]
                pos_vals = flat[pos_idx]
                order = np.argpartition(-pos_vals, kth=k-1)[:k]
                top_idx_unsorted = pos_idx[order]
                top_vals = flat[top_idx_unsorted]
                sorted_order = np.argsort(-top_vals)
                top_idx = top_idx_unsorted[sorted_order]
                I0, J0 = np.unravel_index(top_idx, M.shape, order='C')
                I = (I0 + 1).astype(int)
                J = (J0 + 1).astype(int)

        elif heuristic == 7:
            # % Pick num_pairs node pairs with largest combined stationary probabilities:
            # P_ss_temp = triu(repmat(p_ss_old, 1, n+1) + repmat(p_ss_old', n+1, 1), 1);
            # [~, inds] = maxk(P_ss_temp(:), min([num_pairs, nchoosek(n+1,2)]));
            # [I, J] = ind2sub([n+1, n+1], inds);
            p = p_ss_old
            M = np.triu(p[:, None] + p[None, :], k=1)  # 上三角
            flat = M.ravel()
            # 最大的候选对数不超过 C(n+1,2)
            max_pairs = (n + 1) * n // 2
            k = min(num_pairs, max_pairs)
            if k == 0:
                I = np.array([], dtype=int)
                J = np.array([], dtype=int)
            else:
                order = np.argpartition(-flat, kth=k-1)[:k]
                top_vals = flat[order]
                sorted_order = np.argsort(-top_vals)
                top_idx = order[sorted_order]
                I0, J0 = np.unravel_index(top_idx, M.shape, order='C')
                I = (I0 + 1).astype(int)
                J = (J0 + 1).astype(int)

        elif heuristic == 8:
            # % Iteratively add random nodes to one large cluster:
            # I = 1; J = n+1;
            I = np.array([1], dtype=int)
            J = np.array([n + 1], dtype=int)

        else:
            raise ValueError('Variable "setting" is not properly defined.')

        # % Number of pairs:
        # num_pairs_temp = length(I);
        num_pairs_temp = len(I)

        # % Keep track of all entropies:
        # S_all = zeros(1,num_pairs_temp);
        S_all = np.zeros(num_pairs_temp)

        # % Loop over the pairs of nodes:
        # for ind = 1:num_pairs_temp
        for ind in range(num_pairs_temp):
            # i = I(ind); j = J(ind);
            i = int(I[ind])   # 注意：这里的 i,j 是 1-based（与 MATLAB 对齐）
            j = int(J[ind])

            # inds_not_ij = [1:(i-1),(i+1):(j-1),(j+1):(n+1)];
            # 这里也保持 1-based 逻辑，然后稍后映射到 Python 的 0-based 切片
            left1 = list(range(1, i))
            mid = list(range(i + 1, j))
            right1 = list(range(j + 1, n + 2))
            inds_not_ij_1b = left1 + mid + right1  # 仍然 1-based

            # % Compute new stationary distribution:
            # p_ss_temp = [p_ss_old(inds_not_ij); p_ss_old(i) + p_ss_old(j)];
            p_ss_temp = np.concatenate([
                p_ss_old[np.array(inds_not_ij_1b) - 1],  # 转成 0-based 取值
                [p_ss_old[i - 1] + p_ss_old[j - 1]]
            ])

            # % Compute new transition probabilities:
            # P_temp_1 = sum(repmat(p_ss_old(inds_not_ij), 1, 2).*P_old(inds_not_ij,[i j]), 2);
            # P_temp_1 = P_temp_1./p_ss_temp(1:(end-1));
            rows = np.array(inds_not_ij_1b) - 1  # 0-based 行索引
            cols_ij = np.array([i - 1, j - 1])   # 0-based 列索引 [i j]
            # 逐元素乘以对应行的 p_ss_old（广播），再对列求和（axis=1）
            P_temp_1 = np.sum(p_ss_old[rows, None] * P_old[np.ix_(rows, cols_ij)], axis=1)
            P_temp_1 = np.divide(P_temp_1, p_ss_temp[:-1], out=np.zeros_like(P_temp_1), where=p_ss_temp[:-1] != 0)

            # P_temp_2 = sum(repmat(p_ss_old([i j]), 1, n-1).*P_old([i j], inds_not_ij), 1);
            # P_temp_2 = P_temp_2/p_ss_temp(end);
            cols_not = np.array(inds_not_ij_1b) - 1  # 0-based 列
            P_temp_2 = np.sum((p_ss_old[[i - 1, j - 1], None] * P_old[np.ix_([i - 1, j - 1], cols_not)]), axis=0)
            P_temp_2 = P_temp_2 / (p_ss_temp[-1] if p_ss_temp[-1] != 0 else 1.0)

            # P_temp_3 = sum(sum(repmat(p_ss_old([i j]), 1, 2).*P_old([i j], [i j])));
            # P_temp_3 = P_temp_3/p_ss_temp(end);
            P_temp_3 = np.sum((p_ss_old[[i - 1, j - 1], None] * P_old[np.ix_([i - 1, j - 1], [i - 1, j - 1])]))
            P_temp_3 = P_temp_3 / (p_ss_temp[-1] if p_ss_temp[-1] != 0 else 1.0)

            # logP_temp_* 三者的 log2，并将 inf 位置置 0
            with np.errstate(divide='ignore'):
                logP_temp_1 = np.log2(P_temp_1, where=(P_temp_1 > 0), out=np.zeros_like(P_temp_1))
                logP_temp_2 = np.log2(P_temp_2, where=(P_temp_2 > 0), out=np.zeros_like(P_temp_2))
                logP_temp_3 = np.log2(P_temp_3) if P_temp_3 > 0 else 0.0

            # % Compute change in upper bound on mutual information:
            # dS = -sum(p_ss_temp(1:(end-1)).*P_temp_1.*logP_temp_1) - p_ss_temp(end)*sum(P_temp_2.*logP_temp_2) -...
            #     p_ss_temp(end)*P_temp_3*logP_temp_3 +...
            #     sum(p_ss_old.*P_old(:,i).*logP_old(:,i)) + sum(p_ss_old.*P_old(:,j).*logP_old(:,j)) +...
            #     p_ss_old(i)*sum(P_old(i,:).*logP_old(i,:)) + p_ss_old(j)*sum(P_old(j,:).*logP_old(j,:)) -...
            #     p_ss_old(i)*(P_old(i,i)*logP_old(i,i) + P_old(i,j)*logP_old(i,j)) -...
            #     p_ss_old(j)*(P_old(j,j)*logP_old(j,j) + P_old(j,i)*logP_old(j,i));
            term1 = -np.sum(p_ss_temp[:-1] * P_temp_1 * logP_temp_1)
            term2 = -p_ss_temp[-1] * np.sum(P_temp_2 * logP_temp_2)
            term3 = -p_ss_temp[-1] * P_temp_3 * (logP_temp_3 if np.isfinite(logP_temp_3) else 0.0)
            term4 = np.sum(p_ss_old * P_old[:, i - 1] * logP_old[:, i - 1])
            term5 = np.sum(p_ss_old * P_old[:, j - 1] * logP_old[:, j - 1])
            term6 = p_ss_old[i - 1] * np.sum(P_old[i - 1, :] * logP_old[i - 1, :])
            term7 = p_ss_old[j - 1] * np.sum(P_old[j - 1, :] * logP_old[j - 1, :])
            term8 = -p_ss_old[i - 1] * (P_old[i - 1, i - 1] * logP_old[i - 1, i - 1] + P_old[i - 1, j - 1] * logP_old[i - 1, j - 1])
            term9 = -p_ss_old[j - 1] * (P_old[j - 1, j - 1] * logP_old[j - 1, j - 1] + P_old[j - 1, i - 1] * logP_old[j - 1, i - 1])
            dS = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9

            # S_temp = S_old + dS;
            S_temp = S_old + dS

            # % Keep track of all entropies:
            # S_all(ind) = S_temp;
            S_all[ind] = S_temp

        # % Find minimum entropy:
        # min_inds = find(S_all == min(S_all));
        min_val = np.min(S_all) if len(S_all) > 0 else S_old  # 空时保护
        min_inds = np.where(S_all == min_val)[0]

        # % 在并列最小者中随机挑一个（与 MATLAB datasample 保持一致）:
        if len(min_inds) == 0:
            min_ind = 0
        else:
            min_ind = int(rng.choice(min_inds, size=1)[0])

        # % Save mutual information:
        # S_old = S_all(min_ind);
        S_old = S_all[min_ind] if len(S_all) > 0 else S_old
        # S(n) = S_old;
        S[n - 1] = S_old  # 注意 Python 0-based：第 n 项放到索引 n-1

        # % Compute old transition probabilities:
        # i_new = I(min_ind); j_new = J(min_ind);
        if len(I) == 0:
            # 没有候选对的极端情况，直接跳过（与 MATLAB 一致性之外的健壮性保护）
            continue
        i_new = int(I[min_ind])
        j_new = int(J[min_ind])

        # inds_not_ij = [1:(i_new-1),(i_new+1):(j_new-1),(j_new+1):(n+1)];
        left1 = list(range(1, i_new))
        mid = list(range(i_new + 1, j_new))
        right1 = list(range(j_new + 1, n + 2))
        inds_not_ij_1b = left1 + mid + right1

        # p_ss_new = [p_ss_old(inds_not_ij); p_ss_old(i_new) + p_ss_old(j_new)];
        p_ss_new = np.concatenate([
            p_ss_old[np.array(inds_not_ij_1b) - 1],
            [p_ss_old[i_new - 1] + p_ss_old[j_new - 1]]
        ])

        # P_joint = repmat(p_ss_old, 1, n+1).*P_old;
        # P_joint = [P_joint(inds_not_ij, inds_not_ij), sum(P_joint(inds_not_ij, [i_new j_new]),2);...
        #     sum(P_joint([i_new j_new], inds_not_ij),1), sum(sum(P_joint([i_new j_new], [i_new j_new])))];
        P_joint = P_old * p_ss_old[:, None]
        rows = np.array(inds_not_ij_1b) - 1
        cols = np.array(inds_not_ij_1b) - 1
        block_11 = P_joint[np.ix_(rows, cols)]  # 左上角块
        block_12 = np.sum(P_joint[np.ix_(rows, [i_new - 1, j_new - 1])], axis=1, keepdims=True)  # 右上列向量
        block_21 = np.sum(P_joint[np.ix_([i_new - 1, j_new - 1], cols)], axis=0, keepdims=True)  # 左下行向量
        block_22 = np.sum(P_joint[np.ix_([i_new - 1, j_new - 1], [i_new - 1, j_new - 1])])      # 右下标量
        # 拼成新的 (n x n) 的 P_joint
        P_joint = np.block([
            [block_11,            block_12],
            [block_21, np.array([[block_22]])]
        ])

        # P_old = P_joint./repmat(p_ss_new, 1, n);
        P_old = np.divide(P_joint, p_ss_new[:, None], out=np.zeros_like(P_joint), where=(p_ss_new[:, None] != 0))
        p_ss_old = p_ss_new.copy()

        # logP_old = log2(P_old);
        # logP_old(isinf(logP_old)) = 0;
        logP_old = np.zeros_like(P_old)
        with np.errstate(divide='ignore'):
            logP_old = np.log2(P_old, where=(P_old > 0), out=logP_old)
        logP_old[~np.isfinite(logP_old)] = 0

        # % Record clusters and graph:
        # clusters{n} = [clusters{n+1}([1:(i_new-1),(i_new+1):(j_new-1),(j_new+1):(n+1)]),...
        #     [clusters{n+1}{i_new}, clusters{n+1}{j_new}]];
        prev_clusters = clusters[n]  # 注意：clusters[-1] 是初始，clusters[n] 对应 MATLAB 的 {n+1}
        # 取出除 i_new 和 j_new 外的簇（保持顺序）
        keep_1b = inds_not_ij_1b  # 仍为 1-based
        kept = [prev_clusters[k - 1] for k in keep_1b]
        merged = prev_clusters[i_new - 1] + prev_clusters[j_new - 1]  # 将两个簇的节点列表拼接
        clusters[n - 1] = kept + [merged]  # 更新第 n 项（对应 MATLAB 的 clusters{n}）

        # Gs{n} = P_joint*2*E;
        Gs[n - 1] = P_joint * (2.0 * E)

        # % Compute lower bound on mutual information:
        # P_low = [P_low(:, [1:(i_new-1),(i_new+1):(j_new-1),(j_new+1):(n+1)]),...
        #     P_low(:,i_new) + P_low(:,j_new)];
        # 这里对列进行“合并”：将 i_new、j_new 两列相加作为新簇列，并删除原列
        cols_keep_0b = np.array(inds_not_ij_1b) - 1
        new_col = (P_low[:, i_new - 1] + P_low[:, j_new - 1])[:, None]
        P_low = np.concatenate([P_low[:, cols_keep_0b], new_col], axis=1)

        # logP_low = log2(P_low);
        # logP_low(isinf(logP_low)) = 0;
        logP_low = np.zeros_like(P_low)
        with np.errstate(divide='ignore'):
            logP_low = np.log2(P_low, where=(P_low > 0), out=logP_low)
        logP_low[~np.isfinite(logP_low)] = 0

        # S_low(n) = -sum(p_ss.*sum(P_low.*logP_low,2));
        # 注意：原 MATLAB 代码这里使用的是 p_ss（初始稳态分布），而不是 p_ss_old/p_ss_new。
        # 为了“忠实”转换，我们也保持一致（即使用最初的 p_ss）。
        S_low[n - 1] = -np.sum(p_ss * np.sum(P_low * logP_low, axis=1))

    # 循环结束
    return S, S_low, clusters, Gs

if __name__ == "__main__":
    # Small demo graph (undirected-like)
    G_demo = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ])

    S, S_low, clusters, Gs = rate_distortion(G_demo, heuristic=1, num_pairs=100, random_state=42)
    print("S (upper bound):", S)
    print("S_low (lower bound):", S_low)
    print("Clusters by n:", clusters)
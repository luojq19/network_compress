import numpy as np
import scipy.sparse.linalg as sp_linalg
import itertools
import warnings

def rate_distortion(G, heuristic, num_pairs):
    """
    计算给定图G在压缩时的率失真曲线。

    Args:
        G (np.ndarray): 一个NxN的邻接矩阵，可以是有向、有权的。
        heuristic (int): 一个整数，用于决定选择哪种启发式策略来筛选要合并的聚类对。
        num_pairs (int): 一个整数，决定在每次迭代中要评估多少对候选聚类。

    Returns:
        tuple: 包含以下元素的元组:
            S (np.ndarray): 一个1xN的数组，S[n]是在聚类为n+1个簇时信息率的上界。
            S_low (np.ndarray): 一个1xN的数组，S_low[n]是信息率的下界。
            clusters (list): 一个列表，clusters[n]记录了在有n+1个簇时，每个簇包含的原始节点。
            Gs (list): 一个列表，Gs[n]是n+1个簇时的粗粒化邻接矩阵。
    """
    # -------------------- 1. 初始化阶段 --------------------
    # 获取网络大小 (节点数量)
    N = G.shape[0]
    # 计算网络中的总边数 (主要用于无向图，以便将联合概率矩阵转换回邻接矩阵)
    E = np.sum(G) / 2

    # 初始化用于保存结果的变量
    S = np.zeros(N)          # 率失真曲线的上界 (S_upper)
    S_low = np.zeros(N)      # 率失真曲线的下界
    clusters = [None] * N    # 存储每个聚类数n时的聚类结果
    Gs = [None] * N          # 存储每个聚类数n时的粗粒化邻接矩阵

    # 计算初始的转移概率矩阵 P_old
    row_sums = np.sum(G, axis=1, keepdims=True)
    # 避免除以零的错误 (对于没有出度的节点)
    P_old = np.divide(G, row_sums, where=row_sums!=0)

    # 计算稳态分布 p_ss (适用于所有有向/无向网络)
    # SciPy的eigs默认寻找最大模的特征值，这正是我们需要的(特征值为1)
    # k=1表示我们只需要一个特征向量
    eigenvalues, eigenvectors = sp_linalg.eigs(P_old.T, k=1, which='LM')
    p_ss = np.real(eigenvectors[:, 0]) / np.sum(np.real(eigenvectors[:, 0]))
    p_ss_old = p_ss.copy() # p_ss_old 将在循环中被迭代更新

    # 计算初始熵 (n=N 时的信息率)
    # 使用 where=(P_old > 0) 来避免 log2(0) 的警告
    with warnings.catch_warnings(): # 忽略计算中可能出现的无效值警告
        warnings.simplefilter("ignore")
        logP_old = np.log2(P_old, out=np.zeros_like(P_old), where=(P_old > 0))
    
    # S_old = -sum(pi_i * sum_j(P_ij * log2(P_ij)))
    S_old = -np.sum(p_ss_old * np.sum(P_old * logP_old, axis=1))
    
    # 初始化联合转移概率矩阵和用于计算下界的矩阵
    P_joint = P_old * p_ss_old[:, np.newaxis]
    P_low = P_old.copy()

    # 记录初始值 (n=N 的情况, 在Python中索引为 N-1)
    S[N - 1] = S_old
    S_low[N - 1] = S_old
    # 初始时，每个节点自成一个聚类 (节点索引从0到N-1)
    clusters[N - 1] = [[i] for i in range(N)]
    Gs[N - 1] = G.copy()

    # -------------------- 2. 主循环：聚合聚类 --------------------
    # 从 n=N-1 个聚类开始，每次循环减少一个聚类，直到剩下2个
    # 在Python中，这意味着循环结束后，聚类数量为2 (索引为1)
    # MATLAB的 for n=(N-1):-1:2 对应Python的 range(N-1, 1, -1)
    for n_clusters_after_merge in range(N - 1, 1, -1):
        n_clusters_before_merge = n_clusters_after_merge + 1

        # --- 3. 启发式策略：筛选候选合并对 ---
        cluster_indices = range(n_clusters_before_merge)
        
        if heuristic == 1: # 暴力法，尝试所有可能的聚类对
            pairs = np.array(list(itertools.combinations(cluster_indices, 2)))
            I, J = pairs[:, 0], pairs[:, 1]
        
        elif heuristic == 2: # 随机选择num_pairs对聚类
            all_pairs = np.array(list(itertools.combinations(cluster_indices, 2)))
            num_to_sample = min(num_pairs, len(all_pairs))
            indices = np.random.choice(len(all_pairs), num_to_sample, replace=False)
            pairs = all_pairs[indices]
            I, J = pairs[:, 0], pairs[:, 1]

        elif heuristic == 3: # 尝试所有有连接的聚类对
            # 对称化以考虑双向连接，并取上三角避免重复
            adj_symm = np.triu(P_old + P_old.T, k=1)
            I, J = np.where(adj_symm > 0)

        elif heuristic == 4: # 从有连接的聚类对中随机选择num_pairs对
            adj_symm = np.triu(P_old + P_old.T, k=1)
            i_coords, j_coords = np.where(adj_symm > 0)
            num_to_sample = min(num_pairs, len(i_coords))
            indices = np.random.choice(len(i_coords), num_to_sample, replace=False)
            I, J = i_coords[indices], j_coords[indices]

        elif heuristic == 5: # 选择联合转移概率最大的num_pairs对
            P_joint_symm = np.triu(P_joint + P_joint.T, k=1)
            num_to_select = min(num_pairs, np.sum(P_joint_symm > 0))
            if num_to_select > 0:
                flat_indices = np.argsort(P_joint_symm.ravel())[-num_to_select:]
                I, J = np.unravel_index(flat_indices, P_joint_symm.shape)
            else:
                I, J = np.array([]), np.array([])
        
        elif heuristic == 6: # 策略5的变体，加上了自转移的概率
            diag_sum = np.diag(P_joint)[:, np.newaxis] + np.diag(P_joint)[np.newaxis, :]
            P_joint_symm = np.triu(P_joint + P_joint.T + diag_sum, k=1)
            num_to_select = min(num_pairs, np.sum(P_joint_symm > 0))
            if num_to_select > 0:
                flat_indices = np.argsort(P_joint_symm.ravel())[-num_to_select:]
                I, J = np.unravel_index(flat_indices, P_joint_symm.shape)
            else:
                 I, J = np.array([]), np.array([])

        elif heuristic == 7: # 选择组合稳态概率最大的num_pairs对
            ss_sum_matrix = np.triu(p_ss_old[:, np.newaxis] + p_ss_old[np.newaxis, :], k=1)
            num_to_select = min(num_pairs, len(list(itertools.combinations(cluster_indices, 2))))
            if num_to_select > 0:
                flat_indices = np.argsort(ss_sum_matrix.ravel())[-num_to_select:]
                I, J = np.unravel_index(flat_indices, ss_sum_matrix.shape)
            else:
                I, J = np.array([]), np.array([])

        elif heuristic == 8: # 迭代地将节点添加到一个大聚类中
            I, J = np.array([0]), np.array([n_clusters_after_merge])

        else:
            raise ValueError("变量 'heuristic' 没有被正确定义。")

        # 实际的候选对数量
        num_pairs_temp = len(I)
        # 如果没有候选对（例如在一个完全断开的图中），则跳过这一轮
        if num_pairs_temp == 0:
            # 简单地合并最后两个聚类作为默认行为
            I, J = np.array([n_clusters_before_merge - 2]), np.array([n_clusters_before_merge - 1])
            num_pairs_temp = 1
        
        S_all = np.zeros(num_pairs_temp) # 存储所有候选合并方案产生的新熵值

        # --- 4. 评估候选对：计算信息率变化量 ---
        for ind in range(num_pairs_temp):
            i, j = I[ind], J[ind]
            
            # 获取除了i和j之外的其他聚类的索引
            all_inds = np.arange(n_clusters_before_merge)
            inds_not_ij = np.setdiff1d(all_inds, [i, j])

            # 模拟计算合并后的新稳态分布
            p_ss_i, p_ss_j = p_ss_old[i], p_ss_old[j]
            p_ss_temp = np.concatenate((p_ss_old[inds_not_ij], [p_ss_i + p_ss_j]))
            
            # (以下代码块是dS公式的数值实现，用于计算熵变)
            # 为了避免数值问题，当稳态概率为0时，直接将变化量设为0
            if p_ss_temp[-1] == 0 or np.any(p_ss_temp[:-1] == 0):
                dS = 0
            else:
                P_temp_1 = np.sum(p_ss_old[inds_not_ij, np.newaxis] * P_old[np.ix_(inds_not_ij, [i, j])], axis=1) / p_ss_temp[:-1]
                P_temp_2 = np.sum(p_ss_old[[i, j], np.newaxis] * P_old[np.ix_([i, j], inds_not_ij)], axis=0) / p_ss_temp[-1]
                P_temp_3 = np.sum(p_ss_old[[i, j], np.newaxis] * P_old[np.ix_([i, j], [i, j])]) / p_ss_temp[-1]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    logP_temp_1 = np.log2(P_temp_1, out=np.zeros_like(P_temp_1), where=(P_temp_1 > 0))
                    logP_temp_2 = np.log2(P_temp_2, out=np.zeros_like(P_temp_2), where=(P_temp_2 > 0))
                    logP_temp_3 = np.log2(P_temp_3, out=np.zeros_like(P_temp_3), where=(P_temp_3 > 0))

                # 计算信息率的变化量 dS (对应补充材料的Eq. S14)
                new_entropy_terms = -np.sum(p_ss_temp[:-1] * P_temp_1 * logP_temp_1) \
                                    -p_ss_temp[-1] * np.sum(P_temp_2 * logP_temp_2) \
                                    -p_ss_temp[-1] * P_temp_3 * logP_temp_3
                
                old_entropy_terms = np.sum(p_ss_old * P_old[:, i] * logP_old[:, i]) \
                                    + np.sum(p_ss_old * P_old[:, j] * logP_old[:, j]) \
                                    + p_ss_old[i] * np.sum(P_old[i, :] * logP_old[i, :]) \
                                    + p_ss_old[j] * np.sum(P_old[j, :] * logP_old[j, :]) \
                                    - p_ss_old[i] * (P_old[i, i] * logP_old[i, i] + P_old[i, j] * logP_old[i, j]) \
                                    - p_ss_old[j] * (P_old[j, j] * logP_old[j, j] + P_old[j, i] * logP_old[j, i])
                
                dS = new_entropy_terms + old_entropy_terms
            
            S_all[ind] = S_old + dS
        
        # --- 5. 决策与更新 ---
        # 找到导致信息率最小的合并方案
        min_val = np.min(S_all)
        min_inds = np.where(S_all == min_val)[0]
        min_ind_in_candidates = np.random.choice(min_inds) # 随机选择一个以打破平局

        # 更新当前的最优信息率
        S_old = S_all[min_ind_in_candidates]
        # 将本次迭代的最优信息率记录到输出数组S中 (Python索引为n-1)
        S[n_clusters_after_merge - 1] = S_old

        # 获取被选中的最佳合并对
        i_new, j_new = I[min_ind_in_candidates], J[min_ind_in_candidates]
        
        # 正式执行合并，更新网络状态
        all_inds = np.arange(n_clusters_before_merge)
        inds_not_ij = np.setdiff1d(all_inds, [i_new, j_new])

        # 更新稳态分布
        p_ss_new = np.concatenate((p_ss_old[inds_not_ij], [p_ss_old[i_new] + p_ss_old[j_new]]))

        # 更新联合转移概率矩阵 P_joint
        P_joint = p_ss_old[:, np.newaxis] * P_old
        
        # 构造新的联合转移概率矩阵
        # 块1: 未受影响的聚类之间的连接
        block11 = P_joint[np.ix_(inds_not_ij, inds_not_ij)]
        # 块2: 其他聚类到新合并聚类的总概率流 (新矩阵的最后一列)
        block12 = np.sum(P_joint[np.ix_(inds_not_ij, [i_new, j_new])], axis=1, keepdims=True)
        # 块3: 新合并聚类到其他聚类的总概率流 (新矩阵的最后一行)
        block21 = np.sum(P_joint[np.ix_([i_new, j_new], inds_not_ij)], axis=0, keepdims=True)
        # 块4: 新合并聚类内部的总概率流
        block22 = np.sum(P_joint[np.ix_([i_new, j_new], [i_new, j_new])]).reshape(1,1)

        P_joint = np.block([[block11, block12], [block21, block22]])
        
        # 更新转移概率矩阵 P_old 和稳态分布 p_ss_old，为下一次迭代做准备
        P_old = np.divide(P_joint, p_ss_new[:, np.newaxis], out=np.zeros_like(P_joint), where=p_ss_new[:, np.newaxis]!=0)
        p_ss_old = p_ss_new

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logP_old = np.log2(P_old, out=np.zeros_like(P_old), where=(P_old > 0))

        # 记录本次迭代后的聚类结果和粗粒化网络
        old_clusters = clusters[n_clusters_before_merge - 1]
        new_clusters_list = [old_clusters[i] for i in inds_not_ij]
        merged_cluster = old_clusters[i_new] + old_clusters[j_new]
        new_clusters_list.append(merged_cluster)
        clusters[n_clusters_after_merge - 1] = new_clusters_list
        Gs[n_clusters_after_merge - 1] = P_joint * 2 * E
        
        # --- 6. 计算信息率下界 ---
        P_low_new_col = P_low[:, i_new] + P_low[:, j_new]
        P_low = np.hstack((P_low[:, inds_not_ij], P_low_new_col[:, np.newaxis]))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logP_low = np.log2(P_low, out=np.zeros_like(P_low), where=(P_low > 0))
        
        # 注意：下界计算使用原始的稳态分布p_ss
        S_low[n_clusters_after_merge - 1] = -np.sum(p_ss * np.sum(P_low * logP_low, axis=1))

    return S, S_low, clusters, Gs


if __name__ == '__main__':
    # --- 使用示例 ---
    # 创建一个简单的网络 (例如，一个4节点的环形图)
    G_example = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ])

    print("正在计算一个4节点环形图的率失真曲线...")
    # 使用启发式策略1 (所有对)
    S_upper, S_lower, final_clusters, final_Gs = rate_distortion(G_example, 1, 100)

    print("\n计算完成！")
    print("S (upper bound):", S_upper)
    print("S_low (lower bound):", S_lower)
    print("Clusters by n:", final_clusters)
    # print("率失真曲线上界 (S_upper):")
    # # Python索引从0开始，S_upper[i] 对应 i+1 个聚类
    # for i, val in enumerate(S_upper):
    #     if val != 0: # 只打印计算出的值
    #         print(f"  {i+1} 个聚类时: {val:.4f} bits")
    
    # print("\n最终的聚类结果 (2个聚类时):")
    # # clusters[1] 对应 2 个聚类
    # print(f"  {final_clusters[1]}")
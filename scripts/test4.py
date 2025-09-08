import numpy as np
import scipy.sparse.linalg as spla
import itertools
import random

def rate_distortion(G, heuristic, num_pairs):
    """
    该函数用于计算在压缩给定图G时的率失真曲线。

    输入:
    G: 一个N*N的邻接矩阵，代表一个可能带权重、有向的网络。
    heuristic: 一个整数，决定了如何选择要尝试合并的节点对。
    num_pairs: 一个整数，决定了在每次迭代中要尝试合并多少对节点。

    输出:
    S: 一个长度为N的向量，S[n-1]是在聚类成n个簇后，原始网络上的随机游走与聚类后序列之间的互信息（上界）。
       我们假设聚类后的随机游走是马尔可夫的，因此这是一个上界。
    S_low: 互信息的下界。
    clusters: 一个列表，clusters[n-1]列出了n个簇中每个簇包含的原始节点。
    Gs: 一个列表，Gs[n-1]是聚类成n个簇时的联合转移概率矩阵（乘以总边权重的两倍）。
    """
    # --- 初始化 ---
    
    # 获取网络的大小 (节点数)
    N = G.shape[0]
    # 计算图中总边权重的一半 (对于无向图，这等于边的数量)
    E = np.sum(G) / 2
    
    # 初始化用于存储结果的变量
    # S(n)是聚类为n个簇时的互信息上界
    S = np.zeros(N) 
    # S_low(n)是互信息的下界
    S_low = np.zeros(N)
    # clusters[n-1]列出了n个簇中每个簇包含的节点
    clusters = [None] * N
    # Gs[n-1]是n个簇的联合转移概率矩阵
    Gs = [None] * N

    # --- 计算初始状态 (N个簇) ---

    # 计算转移概率矩阵 P_old = G / G的行和
    row_sums = np.sum(G, axis=1, keepdims=True)
    # 为避免除以零，如果某一行和为0，则该行所有概率设为0
    P_old = np.divide(G, row_sums, out=np.zeros_like(G, dtype=float), where=row_sums != 0)

    # 计算稳态分布 p_ss (stationary distribution)
    # 对于所有类型的网络（有向/无向），都可以通过计算特征向量来找到稳态分布
    # 我们需要寻找转移矩阵转置的特征值为1所对应的特征向量
    # 使用scipy.sparse.linalg.eigs，并设置sigma=1来寻找最接近1的特征值
    eigenvalues, eigenvectors = spla.eigs(P_old.T, k=1, sigma=1)
    # eigs的计算结果可能是复数，我们只取实部
    p_ss = np.real(eigenvectors[:, 0])
    # 将特征向量归一化，使其所有元素之和为1，得到概率分布
    p_ss /= np.sum(p_ss)
    p_ss_old = p_ss.copy()

    # 计算初始熵率 (Entropy Rate)
    # 使用np.log2计算对数，并通过where参数处理P_old中为0的元素，使其结果为0而非-inf
    logP_old = np.log2(P_old, where=P_old > 0, out=np.zeros_like(P_old, dtype=float))
    # 根据公式计算熵率 H = - sum_i p_ss(i) * sum_j P(i,j) * log2(P(i,j))
    S_old = -np.sum(p_ss_old * np.sum(P_old * logP_old, axis=1))

    # 计算联合概率矩阵 P_joint(i, j) = p_ss(i) * P(i, j)
    # p_ss_old[:, np.newaxis] 将一维行向量转换为列向量以进行广播
    P_joint = P_old * p_ss_old[:, np.newaxis]
    # 初始化用于计算下界的转移概率矩阵
    P_low = P_old.copy()

    # 记录初始值 (当有N个簇时，即原始图)
    # Python使用0-based索引，所以N个簇的结果存在索引N-1处
    S[N-1] = S_old
    S_low[N-1] = S_old
    # 初始时，每个节点自成一簇 (节点编号我们保持和MATLAB一致，从1到N)
    clusters[N-1] = [[i + 1] for i in range(N)]
    Gs[N-1] = G.copy()

    # --- 主循环：从 N-1 个簇迭代到 2 个簇 ---
    # range(N - 1, 1, -1) 生成序列 N-1, N-2, ..., 2
    for n in range(N - 1, 1, -1):
        
        # 当前迭代的簇数量是 n+1
        current_num_clusters = n + 1
        
        # --- 启发式策略：选择要尝试合并的簇对 ---
        if heuristic == 1:
            # 策略1：尝试合并所有可能的簇对
            # itertools.combinations生成所有组合
            pairs = list(itertools.combinations(range(current_num_clusters), 2))
            # 将对分解为两个列表 I 和 J
            I = [p[0] for p in pairs]
            J = [p[1] for p in pairs]

        elif heuristic == 2:
            # 策略2：随机选择 num_pairs 对簇进行尝试
            all_pairs = list(itertools.combinations(range(current_num_clusters), 2))
            # 确定要抽样的数量，不能超过总对数
            num_to_sample = min(num_pairs, len(all_pairs))
            # 从所有可能的对中随机抽样
            pairs = random.sample(all_pairs, num_to_sample)
            I = [p[0] for p in pairs]
            J = [p[1] for p in pairs]

        elif heuristic == 3:
            # 策略3：尝试合并所有由边连接的簇对
            # 创建一个对称矩阵来表示连接关系 (如果P(i,j)>0或P(j,i)>0则认为有连接)
            adj_matrix = (P_old + P_old.T) > 0
            # np.triu(..., k=1)取矩阵的上三角部分（不含对角线），.nonzero()找到非零元素的索引
            I_tuple, J_tuple = np.triu(adj_matrix, k=1).nonzero()
            I = list(I_tuple)
            J = list(J_tuple)

        elif heuristic == 4:
            # 策略4：随机选择 num_pairs 对由边连接的簇进行尝试
            adj_matrix = (P_old + P_old.T) > 0
            I_all, J_all = np.triu(adj_matrix, k=1).nonzero()
            all_indices = list(range(len(I_all)))
            num_to_sample = min(num_pairs, len(all_indices))
            # 随机选择索引
            sampled_indices = random.sample(all_indices, num_to_sample)
            I = [I_all[i] for i in sampled_indices]
            J = [J_all[i] for i in sampled_indices]

        elif heuristic == 5:
            # 策略5：选择联合转移概率 (P_joint) 最大的 num_pairs 对簇
            P_joint_symm = np.triu(P_joint + P_joint.T, k=1)
            # 确定要选择的对数
            num_to_select = min(num_pairs, np.sum(P_joint_symm > 0))
            # np.argsort找到排序后元素的索引，取最后num_to_select个即为最大值
            flat_indices = np.argsort(P_joint_symm.flatten())[-num_to_select:]
            # np.unravel_index将一维索引转换回二维
            I_tuple, J_tuple = np.unravel_index(flat_indices, P_joint_symm.shape)
            I = list(I_tuple)
            J = list(J_tuple)

        elif heuristic == 6:
            # 策略6：选择 "联合转移概率 + 自转移概率" 最大的 num_pairs 对簇
            diag_P_joint = np.diag(P_joint)
            # 构建一个矩阵，其(i,j)元素为 diag(i)+diag(j)
            diag_sum_matrix = np.tile(diag_P_joint, (current_num_clusters, 1)) + \
                              np.tile(diag_P_joint, (current_num_clusters, 1)).T
            P_joint_symm = np.triu(P_joint + P_joint.T + diag_sum_matrix, k=1)
            num_to_select = min(num_pairs, np.sum(P_joint_symm > 0))
            flat_indices = np.argsort(P_joint_symm.flatten())[-num_to_select:]
            I_tuple, J_tuple = np.unravel_index(flat_indices, P_joint_symm.shape)
            I = list(I_tuple)
            J = list(J_tuple)

        elif heuristic == 7:
            # 策略7：选择稳态概率之和 (p_ss) 最大的 num_pairs 对簇
            # p_ss_old[:, np.newaxis] + p_ss_old 通过广播创建和矩阵
            p_ss_matrix_sum = p_ss_old[:, np.newaxis] + p_ss_old
            P_ss_temp = np.triu(p_ss_matrix_sum, k=1)
            num_combinations = len(list(itertools.combinations(range(current_num_clusters), 2)))
            num_to_select = min(num_pairs, num_combinations)
            flat_indices = np.argsort(P_ss_temp.flatten())[-num_to_select:]
            I_tuple, J_tuple = np.unravel_index(flat_indices, P_ss_temp.shape)
            I = list(I_tuple)
            J = list(J_tuple)

        elif heuristic == 8:
            # 策略8：迭代地将节点添加到一个大簇中
            # MATLAB代码中I=1, J=n+1。在0-based索引中对应第0个和第n个簇
            I = [0]
            J = [n]

        else:
            # 如果提供了无效的启发式策略编号，则报错
            raise ValueError("Variable 'heuristic' is not properly defined.")
            
        # 获取本次迭代实际尝试的对数
        num_pairs_temp = len(I)
        
        # 如果根据策略没有找到可以合并的对（例如在一个无连接的图中），则提前终止
        if num_pairs_temp == 0:
            # 将剩余的S和S_low值设置为上一个有效值
            S[:n] = S[n]
            S_low[:n] = S_low[n]
            break

        # 创建一个数组来跟踪所有尝试合并产生的熵
        S_all = np.zeros(num_pairs_temp)

        # --- 内循环：遍历所有选定的簇对，计算合并后的熵 ---
        for ind in range(num_pairs_temp):
            
            i = I[ind]
            j = J[ind]
            
            # 确保 i < j，方便后续处理
            if i > j:
                i, j = j, i
            
            # 获取除了i和j之外的所有簇的索引
            inds_not_ij = [k for k in range(current_num_clusters) if k != i and k != j]
            
            # 计算合并后的新稳态分布
            p_ss_temp = np.concatenate((p_ss_old[inds_not_ij], [p_ss_old[i] + p_ss_old[j]]))
            
            # 计算合并后的新转移概率（根据信息瓶颈方法的公式）
            # P_temp_1: 从其他未合并簇 -> 到新合并簇 的转移概率
            # np.ix_用于方便地进行高级索引，选择特定的行和列
            P_temp_1_num = np.sum(p_ss_old[inds_not_ij, np.newaxis] * P_old[np.ix_(inds_not_ij, [i, j])], axis=1)
            P_temp_1 = np.divide(P_temp_1_num, p_ss_temp[:-1], out=np.zeros_like(P_temp_1_num), where=p_ss_temp[:-1]!=0)
            
            # P_temp_2: 从新合并簇 -> 到其他未合并簇 的转移概率
            P_temp_2_num = np.sum(p_ss_old[[i, j], np.newaxis] * P_old[np.ix_([i, j], inds_not_ij)], axis=0)
            P_temp_2 = np.divide(P_temp_2_num, p_ss_temp[-1], out=np.zeros_like(P_temp_2_num), where=p_ss_temp[-1]!=0)
            
            # P_temp_3: 从新合并簇 -> 到自身 的转移概率
            P_temp_3_num = np.sum(p_ss_old[[i, j], np.newaxis] * P_old[np.ix_([i, j], [i, j])])
            P_temp_3 = P_temp_3_num / p_ss_temp[-1] if p_ss_temp[-1] != 0 else 0
            
            # 计算这些概率的对数，并处理概率为0的情况
            logP_temp_1 = np.log2(P_temp_1, where=P_temp_1 > 0, out=np.zeros_like(P_temp_1, dtype=float))
            logP_temp_2 = np.log2(P_temp_2, where=P_temp_2 > 0, out=np.zeros_like(P_temp_2, dtype=float))
            logP_temp_3 = np.log2(P_temp_3, where=P_temp_3 > 0) if P_temp_3 > 0 else 0

            # 计算互信息上界的变化量 dS，这个复杂的公式是信息损失的直接计算
            # 这个公式是直接从MATLAB代码翻译过来的，它代表了合并操作导致的熵率变化
            term1 = -np.sum(p_ss_temp[:-1] * P_temp_1 * logP_temp_1)
            term2 = -p_ss_temp[-1] * np.sum(P_temp_2 * logP_temp_2)
            term3 = -p_ss_temp[-1] * P_temp_3 * logP_temp_3
            term4 = np.sum(p_ss_old * P_old[:, i] * logP_old[:, i])
            term5 = np.sum(p_ss_old * P_old[:, j] * logP_old[:, j])
            term6 = p_ss_old[i] * np.sum(P_old[i, :] * logP_old[i, :])
            term7 = p_ss_old[j] * np.sum(P_old[j, :] * logP_old[j, :])
            term8 = -p_ss_old[i] * (P_old[i, i] * logP_old[i, i] + P_old[i, j] * logP_old[i, j])
            term9 = -p_ss_old[j] * (P_old[j, j] * logP_old[j, j] + P_old[j, i] * logP_old[j, i])
            dS = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
            
            # 计算合并后的总熵
            S_temp = S_old + dS
            
            # 记录这次尝试合并得到的熵
            S_all[ind] = S_temp
            
        # --- 更新状态：选择最优合并并更新变量 ---

        # 找到导致熵最小（信息损失最小）的合并操作
        min_val = np.min(S_all)
        # 可能有多个合并操作得到相同的最小熵，从中随机选择一个
        min_inds = np.where(S_all == min_val)[0]
        min_ind = random.choice(min_inds)
        
        # 将最优合并后的熵作为当前的新熵
        S_old = S_all[min_ind]
        # 保存互信息（上界）到结果数组中 (索引为n-1对应n个簇)
        S[n-1] = S_old
        
        # 获取最优合并对的索引
        i_new = I[min_ind]
        j_new = J[min_ind]
        
        # 再次确保 i_new < j_new
        if i_new > j_new:
            i_new, j_new = j_new, i_new
        
        # 获取除了最优合并对之外的簇的索引
        inds_not_ij = [k for k in range(current_num_clusters) if k != i_new and k != j_new]

        # 更新稳态分布
        p_ss_new = np.concatenate((p_ss_old[inds_not_ij], [p_ss_old[i_new] + p_ss_old[j_new]]))
        
        # 更新联合概率矩阵 P_joint (矩阵收缩操作)
        new_size = n
        P_joint_new = np.zeros((new_size, new_size))
        # 填充左上角 (未合并簇之间的转移)
        P_joint_new[:-1, :-1] = P_joint[np.ix_(inds_not_ij, inds_not_ij)]
        # 填充最后一列 (未合并簇 -> 新合并簇)
        P_joint_new[:-1, -1] = np.sum(P_joint[np.ix_(inds_not_ij, [i_new, j_new])], axis=1)
        # 填充最后一行 (新合并簇 -> 未合并簇)
        P_joint_new[-1, :-1] = np.sum(P_joint[np.ix_([i_new, j_new], inds_not_ij)], axis=0)
        # 填充右下角 (新合并簇 -> 自身)
        P_joint_new[-1, -1] = np.sum(P_joint[np.ix_([i_new, j_new], [i_new, j_new])])
        P_joint = P_joint_new
        
        # 更新转移概率矩阵 P_old 和稳态分布 p_ss_old 为下一次迭代做准备
        P_old = np.divide(P_joint, p_ss_new[:, np.newaxis], out=np.zeros_like(P_joint), where=p_ss_new[:, np.newaxis]!=0)
        p_ss_old = p_ss_new
        
        # 更新 log(P_old)
        logP_old = np.log2(P_old, where=P_old > 0, out=np.zeros_like(P_old, dtype=float))

        # 记录簇的构成和收缩后的图
        prev_clusters = clusters[n] # 获取n+1个簇时的列表
        # 创建新的簇列表，包含未合并的簇
        new_clusters_list = [prev_clusters[k] for k in inds_not_ij]
        # 添加由i_new和j_new合并而成的新簇
        new_clusters_list.append(prev_clusters[i_new] + prev_clusters[j_new])
        clusters[n-1] = new_clusters_list # 保存到n个簇的位置
        
        # 根据更新后的联合概率矩阵计算新的邻接矩阵并保存
        Gs[n-1] = P_joint * 2 * E
        
        # --- 计算互信息的下界 ---
        # 合并P_low矩阵的对应列（i_new 和 j_new）
        P_low_other_cols = P_low[:, inds_not_ij]
        P_low_merged_col = P_low[:, i_new] + P_low[:, j_new]
        # 使用np.hstack水平堆叠数组，形成新的P_low
        P_low = np.hstack((P_low_other_cols, P_low_merged_col[:, np.newaxis]))
        
        # 计算 log(P_low)
        logP_low = np.log2(P_low, where=P_low > 0, out=np.zeros_like(P_low, dtype=float))
        # 计算下界熵并记录 (注意这里的p_ss是原始的稳态分布，没有被迭代更新)
        S_low[n-1] = -np.sum(p_ss * np.sum(P_low * logP_low, axis=1))

    # 函数返回所有计算得到的结果
    return S, S_low, clusters, Gs

if __name__ == '__main__':
    # --- 使用示例 ---
    # 创建一个示例邻接矩阵G (4个节点)
    G_test = np.array([
        [0, 1, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0]
    ])

    # 调用函数
    # 使用启发式策略1 (尝试所有对)，num_pairs在这里不重要
    for heuristic in range(1, 9):
        print(f"Heuristic {heuristic}:")
        S_py, S_low_py, clusters_py, Gs_py = rate_distortion(G_test, heuristic=heuristic, num_pairs=100)
        print("S (upper bound):", S_py)
        print("S_low (lower bound):", S_low_py)
    # S_py, S_low_py, clusters_py, Gs_py = rate_distortion(G_test, heuristic=3, num_pairs=100)

    # print("S (upper bound):", S_py)
    # print("S_low (lower bound):", S_low_py)
    # print("Clusters by n:", clusters_py)
    # 打印结果
    # print("\nS (互信息上界):")
    # # S[i] 对应 i+1 个簇时的值。S[0]和S[1]可能为0，因为循环到n=2为止。
    # print(S_py)
    
    # print("\nS_low (互信息下界):")
    # print(S_low_py)
    
    # print("\n当聚类为2个簇时的构成:")
    # # 索引为1的地方存储的是2个簇的结果
    # print(clusters_py[1])
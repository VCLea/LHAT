import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def cosine_dist(x1: torch.Tensor, x2: torch.Tensor, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t() + eps)

def hyperedge_concat(*H_list):
    H = None
    for h in H_list:
        if h is not None and h != []:
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H

def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    H = torch.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = torch.argsort(-dis_vec)
        avg_dis = torch.mean(dis_vec)
        if not torch.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx
        index = torch.tensor(nearest_idx).long()
        for node_idx in index[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = torch.exp(-dis_vec[node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H

def construct_H_with_KNN(X, K_neigs=[10], split_diff_scale=False, is_probH=True, m_prob=1):
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])
    if type(K_neigs) == int:
        K_neigs = [K_neigs]
    dis_mat = cosine_dist(X, X)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H

def load_feature_construct_H(fts, m_prob=1, K_neigs=[4], is_probH=True, split_diff_scale=False):
    print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
    H = None
    tmp = construct_H_with_KNN(fts, K_neigs=K_neigs,
                               split_diff_scale=split_diff_scale,
                               is_probH=is_probH, m_prob=m_prob)
    H = hyperedge_concat(H, tmp)
    return H

def generate_G_from_H(H, variable_weight=False):
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G

def _generate_G_from_H(H, variable_weight=False):
    if isinstance(H, torch.Tensor):
        H = H.numpy()
    H = np.array(H)
    n_edge = H.shape[1]
    W = np.ones(n_edge)
    DV = np.sum(H * W, axis=1)
    DE = np.sum(H, axis=0)
    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T
    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return torch.tensor(G, dtype=torch.float32)

def gen_trte_inc_mat(data, k_neigs):
    G_train_list = []
    for i in range(len(data)):
        G_train_list.append(generate_G_from_H(load_feature_construct_H(data[i], K_neigs=k_neigs)))
    return G_train_list


# ================== 可学习超图模块（支持可配置投影维度） ==================
class LearnableHypergraph(nn.Module):
    """
    基于稀疏注意力的可学习超图构建模块。
    输入：节点特征矩阵 X (n_nodes, in_features)
    输出：超图拉普拉斯矩阵 G (n_nodes, n_nodes)
    """
    def __init__(self, in_features, proj_dim=None, k=20, num_heads=4, dropout=0.2):
        super(LearnableHypergraph, self).__init__()
        self.k = k
        self.num_heads = num_heads
        self.dropout = dropout
        # 投影维度：若未指定，则等于输入维度
        self.proj_dim = proj_dim if proj_dim is not None else in_features

        # 线性变换：将输入特征投影到 proj_dim 维度
        self.W = nn.Linear(in_features, self.proj_dim, bias=False)
        # 多头注意力参数 a: (num_heads, 2*proj_dim)
        self.a = nn.Parameter(torch.zeros(1, num_heads, 2 * self.proj_dim))
        nn.init.xavier_uniform_(self.a)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, X):
        n = X.size(0)
        k = min(self.k, n)

        # 1. 基于余弦相似度预选邻居（不参与梯度）
        with torch.no_grad():
            X_norm = F.normalize(X, p=2, dim=1)
            cos_sim = torch.mm(X_norm, X_norm.t())
            _, topk_idx = torch.topk(cos_sim, k, dim=1)   # [n, k]

        # 2. 线性变换（投影到 proj_dim 维）
        H = self.W(X)                                      # [n, proj_dim]

        # 3. 提取邻居特征
        neighbor_feat = H[topk_idx]                        # [n, k, proj_dim]

        # 4. 计算注意力分数（只对邻居）
        Q = H.unsqueeze(1).expand(-1, k, -1)               # [n, k, proj_dim]
        concat = torch.cat([Q, neighbor_feat], dim=-1)     # [n, k, 2*proj_dim]

        # 多头注意力
        concat = concat.unsqueeze(2).expand(-1, -1, self.num_heads, -1)  # [n, k, num_heads, 2*proj_dim]
        e = torch.einsum('ijkh,kh->ijk', concat, self.a.squeeze(0))       # [n, k, num_heads]
        e = self.leaky_relu(e)                              # [n, k, num_heads]

        attn = F.softmax(e, dim=1)                           # [n, k, num_heads]
        attn = attn.mean(dim=-1)                             # [n, k]  平均多头

        # 5. 构建稀疏关联矩阵 H_attn
        row_idx = topk_idx.reshape(-1)                        # [n*k]
        col_idx = torch.arange(n, device=X.device).repeat_interleave(k)  # [n*k]
        values = attn.reshape(-1)                             # [n*k]
        H_attn = torch.sparse_coo_tensor(
            torch.stack([row_idx, col_idx]), values, (n, n)
        ).to_dense()

        # 6. 计算拉普拉斯矩阵
        DV = H_attn.sum(dim=1)                                # [n]
        DE = H_attn.sum(dim=0)                                # [n]

        eps = 1e-8
        DV_inv_sqrt = torch.where(DV > eps, DV.pow(-0.5), torch.zeros_like(DV))
        DE_inv = torch.where(DE > eps, DE.pow(-1), torch.zeros_like(DE))

        DV_inv_sqrt_mat = torch.diag(DV_inv_sqrt)             # [n, n]
        DE_inv_mat = torch.diag(DE_inv)                       # [n, n]

        G = DV_inv_sqrt_mat @ H_attn @ DE_inv_mat @ H_attn.t() @ DV_inv_sqrt_mat
        return G
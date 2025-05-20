#!/usr/bin/env python
# coding=utf-8
from models import hypergraph_utils
from models.hgnn_models import HGNN
from torch import nn
#import torch.nn.functional as F
import torch
from tqdm import tqdm


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    """
    Construct hypergraph incidence matrix from hypergraph node distance matrix using PyTorch.
    :param dis_mat: Node distance matrix (tensor)
    :param k_neig: K nearest neighbor
    :param is_probH: Prob Vertex-Edge matrix or binary
    :param m_prob: Probability parameter
    :return: N_object X N_hyperedge tensor
    """
    n_obj = dis_mat.size(0)
    n_edge = n_obj
    H = torch.zeros(n_obj, n_edge, device=dis_mat.device)

    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0  # Set self-distance to zero
        dis_vec = dis_mat[center_idx]

        # Get the indices sorted by distance (ascending)
        nearest_idx = torch.argsort(dis_vec)

        # Calculate the average distance
        avg_dis = torch.mean(dis_vec)

        # Ensure center_idx is included in the nearest indices if not already present
        if not torch.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = torch.exp(-dis_vec[node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0

    return H

def construct_similarity_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    """
    Construct similarity matrix from distance matrix using PyTorch.
    :param dis_mat: Node distance matrix (torch tensor)
    :param k_neig: K nearest neighbor
    :param is_probH: Probability for similarity (if True) or binary (if False)
    :param m_prob: Probability parameter
    :return: N_object X N_object similarity matrix
    """
    n_obj = dis_mat.size(0)  # 获取对象数量
    similarity_matrix = torch.zeros((n_obj, n_obj), device=dis_mat.device)  # 初始化相似度矩阵

    for center_idx in range(n_obj):
        dis_vec = dis_mat[center_idx]
        avg_dis = torch.mean(dis_vec)  # 计算平均距离

        # 计算相似度
        if is_probH:
            similarity_vec = torch.exp(-dis_vec ** 2 / (m_prob * avg_dis) ** 2)
        else:
            similarity_vec = torch.where(dis_vec < 1e-5, torch.tensor(1.0, device=dis_mat.device), torch.tensor(0.0, device=dis_mat.device))

        # 找到 K 最近邻的索引
        nearest_idx = torch.argsort(dis_vec)[:k_neig]
        similarity_matrix[center_idx, nearest_idx] = similarity_vec[nearest_idx]

    return similarity_matrix

def compute_scores_fast(Z, i, topmin_min=0, topmin_max=0.3): # from MuSc
    # speed fast but space large
    # compute anomaly scores
    device = Z.device
    image_num, patch_num, c = Z.shape
    print(Z.shape)

    patch2image = torch.tensor([]).to(device)
    Z_ref = torch.cat((Z[:i], Z[i+1:]), dim=0)
    print(Z_ref.shape)
    print(Z[i:i+1].reshape(-1, c).shape)
    print(Z_ref.reshape(-1, c).shape)
    patch2image = torch.cdist(Z[i:i+1].reshape(-1, c), Z_ref.reshape(-1, c)).reshape(patch_num, image_num-1, patch_num)
    print('simi shape')
    print(patch2image.shape)
    if False:
        # interval average
        k_max = topmin_max
        k_min = topmin_min
        if k_max < 1:
            k_max = int(patch2image.shape[1]*k_max)
        if k_min < 1:
            k_min = int(patch2image.shape[1]*k_min)
        if k_max < k_min:
            k_max, k_min = k_min, k_max
        patch2image = torch.min(patch2image, -1)[0]
        vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
        vals, _ = torch.topk(vals.float(), k_max-k_min, largest=True, sorted=True)
        patch2image = vals.clone()
        return torch.mean(patch2image, dim=1)
    return torch.mean(patch2image, dim = -1)

def cross_domain_pseudo_feature(image_feature, bank, topmin_min = 0.0, topmin_max = 0.3):
    # for single image_feature
    print('image_feature.shape')
    print(image_feature.shape)
    print(bank.shape)

    device = image_feature.device
    Z = torch.concat([image_feature.squeeze(0), bank.squeeze(0)], dim = 0).unsqueeze(1)
    anomaly_scores_matrix = torch.tensor([]).double().to(device)
    pseudo_features = torch.tensor([]).double().to(device)
    for i in tqdm(range(image_feature.shape[1])):
        # each image
        anomaly_scores_i = compute_scores_fast(Z, i, topmin_min, topmin_max)#.unsqueeze(0)
        print(anomaly_scores_i.shape)
        anomaly_scores_matrix = torch.cat((anomaly_scores_matrix, anomaly_scores_i.double()), dim=0)    # (N, B)
        
        _, max_indexes = torch.min(anomaly_scores_i, dim=-1)
        print('max_indexes')
        print(max_indexes)

        index = (len(image_feature) + max_indexes).tolist()
        print(index)
        pseudo_features = torch.cat((pseudo_features, Z[index]), dim = 0)
    return pseudo_features


def cross_domain_pseudo_feature_only_foreign(image_feature, bank, topmin_min = 0.0, topmin_max = 0.3):
    anomaly_scores_matrix = torch.tensor([]).double().to(device)
    for i in tqdm(range(image_feature.shape[0])):
        anomaly_scores_i = compute_scores_fast_foreign(image_feature, bank, i, topmin_min, topmin_max).unsqueeze(0)
        anomaly_scores_matrix = torch.cat((anomaly_scores_matrix, anomaly_scores_i.double()), dim=0)    # (N, B)
    return 


class HyperDomainFusionModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
     
        self.device = "cuda:0"
        self.HGNN = HGNN(in_ch=input_dim,
            n_class=output_dim,
            n_hid=1024, # 64
            dropout=0.2).to(self.device).float()#.half()

        self.knn_neibs = 10



    def forward(self, image_feature, bank, adj = None, adj_type = 'dist'):

        #pseudo_features = cross_domain_pseudo_feature(image_feature, bank.reshape(-1, image_feature.shape[-2], image_feature.shape[-1]))
        #pseudo_features = cross_domain_pseudo_feature(image_feature, bank)

        nodes_feature = torch.concat([image_feature.reshape(-1, 768), bank.reshape(-1, 768)], dim = 0)

        #adj = torch.cdist(image_feature, bank, p=2)

        #print('construct adj:...')
        
        if adj_type == 'simi':
            #print(adj.shape)
            #adj = construct_H_with_KNN_from_distance(adj, 5)
            adj = construct_similarity_from_distance(nodes_feature, 5)
            #adj = torch.cosine_similarity(nodes_feature.unsqueeze(1), nodes_feature.unsqueeze(0), dim=-1)
            print(adj.shape)
        elif adj_type == 'dist':
            adj = torch.cdist(nodes_feature, nodes_feature, p=2)
            adj = 1/(1+adj)
            feature_number = image_feature.shape[1]
            score_map = adj[:feature_number, :feature_number]
            values, index_top2 = torch.topk(score_map, 2)
            index = index_top2[:, 1]
            if False:
                print(image_feature.shape)
                print(bank.shape)
                print(feature_number)
                print(adj.shape)
                print('adj.shape')
                print(score_map.shape)
                print('top2')
                print(index_top2[:, 1])
                print('index:')
                print(index.shape)
            #assert index.max() <= fea
            #index = index + feature_number
            #print(index)
            pseudo_features = nodes_feature[index.tolist()]

        #adj = hypergraph_utils.construct_H_with_KNN(nodes_feature.cpu().detach(), K_neigs=[self.knn_neibs],
        #                                split_diff_scale=False,
        #                                is_probH=True, m_prob=1)
        #adj = torch.from_numpy(adj).half().cuda()

        #adj = adj.float()
        #nodes_feature = nodes_feature.half()

        # adj need to be dense

        #output = self.HGNN(nodes_feature.to(self.device), adj)
        output = self.HGNN(nodes_feature.to(self.device)[:feature_number], adj[:feature_number, :feature_number])

        #print(output.shape)
        #print(output[:image_feature.shape[1]].shape)
        output = output[:feature_number]

        #print('pseudo_features.shape')
        #print(pseudo_features.shape)

        #_, max_index1 = torch.min(adj, dim=1)
        #print(max_index[10000:])
        #max_indexes = []
        #for index in range(len(adj)):
        #    _, max_index = torch.min(adj[index], dim=0)
        #    max_indexes.append(max_index)
        #max_indexes = torch.tensor(max_indexes).cuda()
        #print(torch.tensor(max_indexes)[10000:].cuda())
        #print(max_index1 == max_indexes)
        #print(max_index0 == max_indexes)
        #print(max_index1.float().mean())
        #print(max_index0.float().mean())
        #print(torch.tensor(max_indexes).float().mean())

        return output, pseudo_features



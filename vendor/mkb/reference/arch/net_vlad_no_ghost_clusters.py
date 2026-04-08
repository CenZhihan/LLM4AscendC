# Pybind: 7 tensors (x, clusters, clusters2, bn_w, bn_b, bn_m, bn_var) — NetVLAD, no ghost clusters.
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, clusters, clusters2, bn_w, bn_b, bn_m, bn_var):
        # x: [B, N, D], clusters: [D, K], clusters2: [1, D, K]
        b, n, d = x.shape
        k = clusters.size(1)
        x_flat = x.reshape(-1, d)
        assignment = x_flat @ clusters
        assignment = F.batch_norm(
            assignment.unsqueeze(-1),
            bn_w,
            bn_b,
            bn_m,
            bn_var,
            training=False,
            eps=1e-5,
        ).squeeze(-1)
        assignment = F.softmax(assignment, dim=1)
        assignment = assignment.view(b, n, k)
        a_sum = assignment.sum(dim=1, keepdim=True)
        a = a_sum * clusters2
        assignment = assignment.transpose(1, 2)
        vlad = assignment @ x
        vlad = vlad.transpose(1, 2)
        vlad = vlad - a
        vlad = F.normalize(vlad, p=2, dim=1)
        vlad = vlad.reshape(b, k * d)
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad

B, N, D, K = 2048, 100, 512, 32

def get_inputs():
    x = torch.rand(B, N, D)
    clusters = torch.rand(D, K)
    clusters2 = torch.rand(1, D, K)
    bn_w = torch.rand(K)
    bn_b = torch.rand(K)
    bn_m = torch.rand(K)
    bn_var = torch.rand(K).abs() + 1e-5
    return [x, clusters, clusters2, bn_w, bn_b, bn_m, bn_var]

def get_init_inputs():
    return []

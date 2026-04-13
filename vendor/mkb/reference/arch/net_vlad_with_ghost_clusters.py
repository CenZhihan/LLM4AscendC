# Copyright 2018 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code modified from here
https://github.com/albanie/collaborative-experts/blob/master/model/net_vlad.py
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th


class Model(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(Model, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        # The `clusters` weights are the `(w,b)` in the paper
        self.clusters = nn.Parameter(init_sc * th.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        # The `clusters2` weights are the visual words `c_k` in the paper
        self.clusters2 = nn.Parameter(init_sc * th.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size

    def forward(self, x, clusters, clusters2, bn_weight, bn_bias, bn_mean, bn_var):
        """Aggregates feature maps into a fixed size representation.  In the following
        notation, B = batch_size, N = num_features, K = num_clusters, D = feature_size.

        Args:
            x (th.Tensor): B x N x D
            clusters: cluster assignments weights
            clusters2: cluster centers
            bn_weight, bn_bias, bn_mean, bn_var: batch norm parameters

        Returns:
            (th.Tensor): B x DK
        """
        max_sample = x.size()[1]
        x = x.view(-1, self.feature_size)  # B x N x D -> BN x D

        if x.device != clusters.device:
            msg = f"x.device {x.device} != cluster.device {clusters.device}"
            raise ValueError(msg)

        assignment = th.matmul(x, clusters)  # (BN x D) x (D x (K+G)) -> BN x (K+G)

        # Apply batch norm with provided parameters
        if bn_mean is not None and bn_var is not None:
            bn_mean_exp = bn_mean.unsqueeze(0)
            bn_var_exp = bn_var.unsqueeze(0)
            bn_weight_exp = bn_weight.unsqueeze(0) if bn_weight is not None else None
            bn_bias_exp = bn_bias.unsqueeze(0) if bn_bias is not None else None
            assignment = (assignment - bn_mean_exp) / (bn_var_exp.sqrt() + 1e-5)
            if bn_weight_exp is not None:
                assignment = assignment * bn_weight_exp
            if bn_bias_exp is not None:
                assignment = assignment + bn_bias_exp

        assignment = F.softmax(assignment, dim=1)  # BN x (K+G) -> BN x (K+G)
        # remove ghost assigments
        assignment = assignment[:, :self.cluster_size]
        assignment = assignment.view(-1, max_sample, self.cluster_size)  # -> B x N x K
        a_sum = th.sum(assignment, dim=1, keepdim=True)  # B x N x K -> B x 1 x K
        a = a_sum * clusters2

        assignment = assignment.transpose(1, 2)  # B x N x K -> B x K x N

        x = x.view(-1, max_sample, self.feature_size)  # BN x D -> B x N x D
        vlad = th.matmul(assignment, x)  # (B x K x N) x (B x N x D) -> B x K x D
        vlad = vlad.transpose(1, 2)  # -> B x D x K
        vlad = vlad - a

        # L2 intra norm
        vlad = F.normalize(vlad)

        # flattening + L2 norm
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)  # -> B x DK
        vlad = F.normalize(vlad)
        return vlad  # B x DK

batch_size = 2048
num_features = 100
num_clusters = 32
feature_size = 512
ghost_clusters = 16  # K+G = 48 (matches custom op TORCH_CHECK)

def get_inputs():
    x = torch.rand(batch_size, num_features, feature_size)
    clusters = torch.rand(feature_size, num_clusters + ghost_clusters)
    clusters2 = torch.rand(1, feature_size, num_clusters)
    bn_weight = torch.rand(num_clusters + ghost_clusters)
    bn_bias = torch.rand(num_clusters + ghost_clusters)
    bn_mean = torch.rand(num_clusters + ghost_clusters)
    bn_var = torch.rand(num_clusters + ghost_clusters)
    return [x, clusters, clusters2, bn_weight, bn_bias, bn_mean, bn_var]

def get_init_inputs():
    return [num_clusters, feature_size, ghost_clusters]

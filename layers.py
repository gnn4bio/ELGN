# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from turtle import forward
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
from pgl.utils.helper import generate_segment_id_from_index
from pgl.utils import op
import pgl.math as math
from utils import generate_segment_id
import time
import math as math1

class DenseLayer(nn.Layer):
    def __init__(self, in_dim, out_dim, activation=F.relu, bias=True):
        super(DenseLayer, self).__init__()
        self.activation = activation
        if not bias:
            self.fc = nn.Linear(in_dim, out_dim, bias_attr=False)
        else:
            self.fc = nn.Linear(in_dim, out_dim)
    
    def forward(self, input_feat):
        return self.activation(self.fc(input_feat))

class BondLayer(nn.Layer):
    """Implementation of Bond Layer.
    """
    def __init__(self, atom_dim, bond_dim, activation=F.relu):
        super(BondLayer, self).__init__()
        in_dim = atom_dim  
        self.fc_agg = DenseLayer(in_dim, bond_dim, activation=activation, bias=True)
        in_channels = atom_dim
        dropout_rate = 0.2
        self.dropout = nn.Dropout(p=dropout_rate)

    def send_func(self,src_feat, dst_feat, edge_feat):
        return src_feat

    def recv_func(self,msg):
        return msg.reduce_sum(msg["h"])

    def forward(self, g, atom_feat): 
        h_node = self.dropout(F.normalize(atom_feat))

        msg = g.send(self.send_func, src_feat={'h': h_node})
        bond_feat = g.recv(self.recv_func, msg)

        bond_feat = self.fc_agg(bond_feat)
        return bond_feat

class AtomicTPoolLayer(nn.Layer):
    """Implementation of Atomic type Pooling Layer.
    """
    def __init__(self, bond_dim, hidden_dim):
        super(AtomicTPoolLayer, self).__init__()
        self.bond_dim = bond_dim
        self.num_type = 37
        self.fc_2 = nn.Linear(hidden_dim, 1, bias_attr=False)
        self.softmax = nn.Softmax(axis=1)
    
    def forward(self, bond_types_batch, type_count_batch, bond_feat):
        """
        Input example:
            bond_types_batch: [0,0,2,0,1,2] + [0,0,2,0,1,2] + [2]
            type_count_batch: [[3, 3, 0], [1, 1, 0], [2, 2, 1]] # [num_type, batch_size]
        """
        inter_mat_list =[]
        for type_i in range(self.num_type):
            type_i_index = paddle.masked_select(paddle.arange(len(bond_feat)), bond_types_batch==type_i)
            if paddle.sum(type_count_batch[type_i]) == 0:
                inter_mat_list.append(paddle.to_tensor(np.array([0.]*len(type_count_batch[type_i])), dtype='float32'))
                continue
            bond_feat_type_i = paddle.gather(bond_feat, type_i_index)
            graph_bond_index = op.get_index_from_counts(type_count_batch[type_i])
            # graph_bond_id = generate_segment_id_from_index(graph_bond_index)
            graph_bond_id = generate_segment_id(graph_bond_index)
            graph_feat_type_i = math.segment_pool(bond_feat_type_i, graph_bond_id, pool_type='sum')
            mat_flat_type_i = self.fc_2(graph_feat_type_i).squeeze(1)

            # print(graph_bond_id)
            # print(graph_bond_id.shape, graph_feat_type_i.shape, mat_flat_type_i.shape)
            my_pad = nn.Pad1D(padding=[0, len(type_count_batch[type_i])-len(mat_flat_type_i)], value=-1e9)
            mat_flat_type_i = my_pad(mat_flat_type_i)
            inter_mat_list.append(mat_flat_type_i)

        inter_mat_batch = paddle.stack(inter_mat_list, axis=1) # [batch_size, num_type]
        inter_mat_mask = paddle.ones_like(inter_mat_batch) * -1e9
        inter_mat_batch = paddle.where(type_count_batch.transpose([1, 0])>0, inter_mat_batch, inter_mat_mask)
        inter_mat_batch = self.softmax(inter_mat_batch)
        return inter_mat_batch


class AtomBondPoolLayer(nn.Layer):
    """Implementation of Atom-Bond pooling Layer.
    """
    def __init__(self, atom_dim, hidden_dim_list):
        super(AtomBondPoolLayer, self).__init__()
        self.pool = pgl.nn.GraphPool(pool_type='sum')
        self.pool1 = pgl.nn.GraphPool(pool_type='sum')
        self.mlp = nn.LayerList()
        for hidden_dim in hidden_dim_list:
            self.mlp.append(DenseLayer(atom_dim, hidden_dim, activation=F.relu))
            atom_dim = hidden_dim
        self.output_layer = nn.Linear(atom_dim, 1)
    
    def forward(self, g, line_g, atom_feat, line_feat):
        graph_feat = self.pool(g, atom_feat)
        line_feat = self.pool1(line_g, line_feat)
        graph_feat = paddle.concat([graph_feat, line_feat], axis=1)
    
        for layer in self.mlp:
            graph_feat = layer(graph_feat)
        output = self.output_layer(graph_feat)
        return output


class AtomLayer(nn.Layer):
    """
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, dim_tmp= 0,nodes_att_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False):
        super(AtomLayer, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 1


        self.edge_mlp = nn.Sequential(
            nn.Linear(dim_tmp, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        weight_attr = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.XavierNormal())
        layer = paddle.nn.Linear(hidden_nf, 1, weight_attr=weight_attr,bias_attr=False)


        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            # self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)


        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

        self.node_dec = nn.Sequential(nn.Linear(hidden_nf, hidden_nf),
                                      act_fn,
                                      nn.Linear(hidden_nf, hidden_nf),
                                      nn.ELU())
  
        self.fc = nn.Linear(hidden_nf, hidden_nf)
        self.dropout = nn.Dropout(p=0.2)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = paddle.concat([source, target, radial], axis=1)
        else:
            out = paddle.concat([source, target, radial, edge_attr], axis=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def unsorted_segment_sum(self, data, segment_ids, num_segments):
        """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
        result_shape = (num_segments, data.shape[1])
        result = paddle.Tensor(np.zeros(result_shape))
        # result = data.new_full(result_shape, 0)  # Init empty result tensor.
        # segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.shape[1])
        segment_ids = paddle.Tensor(np.array(segment_ids))
        segment_ids = segment_ids.unsqueeze(-1)
        segment_ids = paddle.expand(segment_ids, shape=[-1, data.shape[1]])
        result = paddle.cast(result, 'float32')
        result = self.sc(result, segment_ids, data)
        return result

    def unsorted_segment_mean(self, data, segment_ids, num_segments):
        result_shape = (num_segments, data.shape[1])
        segment_ids = paddle.Tensor(np.array(segment_ids))
        segment_ids = segment_ids.unsqueeze(-1)
        segment_ids = paddle.expand(segment_ids,shape=[-1, data.shape[1]])

        result = paddle.Tensor(np.zeros(result_shape))
        count = paddle.Tensor(np.zeros(result_shape))
        # result = data.new_full(result_shape, 0)  # Init empty result tensor.
        # count = data.new_full(result_shape, 0)
        result = paddle.cast(result, 'float32')
        count = paddle.cast(count, 'float32')
        result = self.sc(result, segment_ids, data)
        count = self.sc(count, segment_ids, paddle.ones_like(data))
        return result / count.clip(min=1)

    def sc(self,x, seg, data):
        x = x
        updates = data
        index = seg
        i, j = index.shape
        grid_x, grid_y = paddle.meshgrid(paddle.arange(i), paddle.arange(j))
        index = paddle.stack([index.flatten(), grid_y.flatten()], axis=1)
        updates_index = paddle.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
        updates = paddle.gather_nd(updates, index=updates_index)
        res = paddle.scatter_nd_add(x, index, updates)
        return res

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = self.unsorted_segment_sum(edge_attr, row, num_segments=x.shape[0])
        if node_attr is not None:
            agg = paddle.concat([x, agg, node_attr], axis=1)
        else:
            agg = paddle.concat([x, agg], axis=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = paddle.clip(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train
        agg = self.unsorted_segment_mean(trans, row, num_segments=coord.shape[0])
        coord += agg*self.coords_weight
        return coord


    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = paddle.sum((coord_diff)**2, 1).unsqueeze(1)
        if self.norm_diff:
            norm = paddle.sqrt(radial) + 1
            coord_diff = coord_diff/(norm)

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
       
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        h = self.dropout1(h)
        node_attr = self.dropout(node_attr)
    

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        edge_feat = self.dropout2(edge_feat)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        h = self.node_dec(h)
        return h, coord, edge_feat


   
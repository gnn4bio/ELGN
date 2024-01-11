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
"""
Model code for Equivariant  Line Graph Networks (ELGN).
"""
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
from layers import  BondLayer, AtomicTPoolLayer, AtomBondPoolLayer, AtomLayer
import networkx as nx

class ELGN(nn.Layer):
    def __init__(self, args, **params):
        super(ELGN, self).__init__()
        dense_dims = args.dense_dims
        infeat_dim = args.infeat_dim
        hidden_dim = args.hidden_dim

        cut_dist = args.cut_dist
        activation = args.activation
   

    
        self.egcl_layer = AtomLayer(hidden_dim, hidden_dim, hidden_dim, edges_in_d=0, dim_tmp=257, nodes_att_dim=37, act_fn=nn.Silu(), recurrent=True, coords_weight=1, attention=True)
        self.egcl_layer1 = AtomLayer(hidden_dim, hidden_dim, hidden_dim, edges_in_d=0, dim_tmp=385, nodes_att_dim=37, act_fn=nn.Silu(),recurrent=True, coords_weight=1, attention=True)
        self.egcl_layer2 = AtomLayer(hidden_dim, hidden_dim, hidden_dim, edges_in_d=0, dim_tmp=385, nodes_att_dim=37,act_fn=nn.Silu(), recurrent=True, coords_weight=1, attention=True)
        self.egcl_layer3 = AtomLayer(hidden_dim, hidden_dim, hidden_dim, edges_in_d=0, dim_tmp=385, nodes_att_dim=37, act_fn=nn.Silu(), recurrent=True, coords_weight=1, attention=True)
       
        self.pipool_layer = AtomicTPoolLayer(hidden_dim, hidden_dim)
        self.output_layer = AtomBondPoolLayer(hidden_dim*2, dense_dims)
        self.new_embedding = nn.Linear(37*2, hidden_dim)
        self.embedding = nn.Linear(37, hidden_dim)
        self.bond2bond1 = BondLayer(128, 128)
        self.bond2bond2 = BondLayer(128, 128)
        self.bond2bond3 = BondLayer(128, 128)
        self.bond2bond4 = BondLayer(128, 128)
 
    def forward(self, a2a_g, line_g, coords, bond_types, type_count):
        atom_feat = a2a_g.node_feat['feat']
        atom_feat = paddle.cast(atom_feat, 'float32')
        print(a2a_g.num_edges)
        feature = line_g.node_feat['feat']
        feature = paddle.cast(feature, 'float32')
        atom_h = atom_feat
      
        x = coords
        h = self.embedding(atom_h)
        edges_index = paddle.t(a2a_g.edges)
    

        # The ELGN model on the PDBbind core set
        h, _, bond_h = self.egcl_layer(h, edges_index, x, node_attr=atom_h)
        h1 = self.bond2bond1(line_g, bond_h)
        h1 = self.bond2bond2(line_g, h1)
        h, _, bond_h = self.egcl_layer1(h, edges_index, x, node_attr=atom_h, edge_attr=h1)
        h1 = self.bond2bond3(line_g, bond_h)
        h1 = self.bond2bond4(line_g, h1)
        h, _, bond_h = self.egcl_layer2(h, edges_index, x, node_attr=atom_h, edge_attr=h1)

        # The ELGN model on the CSAR-HiQ set
        # h, _, bond_h = self.egcl_layer(h, edges_index, x, node_attr=atom_h)
        # h1 = self.bond2bond1(line_g, bond_h)
        # h1 = self.bond2bond2(line_g, h1)
        # h, _, bond_h = self.egcl_layer1(h, edges_index, x, node_attr=atom_h, edge_attr=h1)
   

        pred_inter_mat = self.pipool_layer(bond_types, type_count, bond_h)
        pred_socre = self.output_layer(a2a_g, line_g, h, bond_h)
        return  pred_inter_mat, pred_socre


        

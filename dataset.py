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
Dataset code for protein-ligand complexe interaction graph construction.
"""

import os
import numpy as np
import paddle
import pgl
import pickle
from pgl.utils.data import Dataset as BaseDataset
from pgl.utils.data import Dataloader
from scipy.spatial import distance
from scipy.sparse import coo_matrix
from utils import cos_formula, setxor
from tqdm import tqdm
import networkx as nx
import math

prot_atom_ids = [6, 7, 8, 16]
drug_atom_ids = [6, 7, 8, 9, 15, 16, 17, 35, 53]
pair_ids = [(i, j) for i in prot_atom_ids for j in drug_atom_ids]

class ComplexDataset(BaseDataset):
    def __init__(self, data_path, dataset, cut_dist, save_file=True):
        self.data_path = data_path
        self.dataset = dataset
        self.cut_dist = cut_dist
        self.save_file = save_file
    
        self.labels = []
        self.a2a_graphs = []
        self.line_graph = []
        self.coords = []
        self.inter_feats_list = []
        self.bond_types_list = []
        self.type_count_list = []

        self.load_data()
        

    def __len__(self):
        """ Return the number of graphs. """
        return len(self.labels)
    
    def __getitem__(self, idx):
        """ Return graphs and label. """
        return self.a2a_graphs[idx], self.line_graph[idx],\
               self.coords[idx],self.inter_feats_list[idx],self.bond_types_list[idx], self.type_count_list[idx], self.labels[idx]

    def has_cache(self):
        """ Check cache file."""
        self.graph_path = f'{self.data_path}/{self.dataset}_{int(self.cut_dist)}_pgl_graph.pkl'
        return os.path.exists(self.graph_path)
    
    def save(self):
        """ Save the generated graphs. """
        print('Saving processed complex data...')
        graphs = [self.a2a_graphs, self.line_graph]
        global_feat = [self.coords,self.inter_feats_list, self.bond_types_list, self.type_count_list]
        with open(self.graph_path, 'wb') as f:
            pickle.dump((graphs, global_feat, self.labels), f)

    def load(self):
        """ Load the generated graphs. """
        print('Loading processed complex data...')
        with open(self.graph_path, 'rb') as f:
            graphs, global_feat, labels = pickle.load(f)
        return graphs, global_feat, labels
    
    def build_graph(self, mol):
        num_atoms_d, coords, features, atoms, inter_feats = mol
        coord = np.vstack([coords,coords.mean(axis=0)])
        ##################################################
        # prepare distance matrix and interaction matrix #
        ##################################################
        dist_mat = distance.cdist(coords, coords, 'euclidean')
        np.fill_diagonal(dist_mat, np.inf)
        inter_feats.append(len(coords)*2)
        inter_feats = np.array([inter_feats])
        inter_feats = inter_feats / inter_feats.sum()

        ############################
        # build atom to atom graph #
        ############################
        num_atoms = len(coords)
        dist_graph_base = dist_mat.copy()
        dist_feat = dist_graph_base[dist_graph_base < self.cut_dist].reshape(-1,1)
        dist_graph_base[dist_graph_base >= self.cut_dist] = 0.
        atom_graph = coo_matrix(dist_graph_base)
        a2a_edges = list(zip(atom_graph.row, atom_graph.col))
        edges_raw = a2a_edges
        for i in range(num_atoms):
            a2a_edges.append((num_atoms,i))
            a2a_edges.append((i,num_atoms))
        num_atom = num_atoms + 1
        a2a_edges = sorted(a2a_edges)
        vnode_feat = np.mean(features, axis=0)
        vnode_flag = np.ones([1])
        vnode_feat = np.concatenate([vnode_feat, vnode_flag])
        not_vnode_flag = np.zeros([len(features),1]) 
        features = np.concatenate([features, not_vnode_flag], axis=1)
        features = np.vstack([features,vnode_feat])
        a2a_graph = pgl.Graph(a2a_edges, num_nodes=num_atom, node_feat={"feat": features})
      
        ############################
        # build line graph         #
        ############################
        G = nx.Graph()
        G.add_nodes_from([i for i in range(num_atoms)])
        G.add_edges_from(a2a_edges)
        L = nx.line_graph(G)
        a2a_edges_new = list()
        edge_index = {tuple(a2a_edges[i]): i for i in range(len(a2a_edges))}
        for i in L.edges:
            a2a_edges_new.append((edge_index[i[0]], edge_index[i[1]]))
        num_nodes_new = len(a2a_edges)
        feature = list()
        new_coord = list()
        for i in range(num_nodes_new):
            feature.append(np.hstack([features[a2a_edges[i][0]], features[a2a_edges[i][1]]]))
            new_coord.append((coord[a2a_edges[i][0]]+coord[a2a_edges[i][1]])/2)                            
        feature = np.array(feature)
        new_coord = np.array(new_coord)
        line_graph = pgl.Graph(a2a_edges_new, num_nodes=num_nodes_new, node_feat={"feat": feature})

        ######################
        # prepare bond nodes #
        ######################
        indices = []
        bond_pair_atom_types = []
        for i in a2a_edges:
            if i[0] == num_atoms or i[1] == num_atoms:
                bond_pair_atom_types += [36]
                continue
            at_i, at_j = atoms[i[0]], atoms[i[1]]
            if i[0] < num_atoms_d and i[1] >= num_atoms_d and (at_j, at_i) in pair_ids:
                bond_pair_atom_types += [pair_ids.index((at_j, at_i))]
            elif i[0] >= num_atoms_d and i[1] < num_atoms_d and (at_i, at_j) in pair_ids:
                bond_pair_atom_types += [pair_ids.index((at_i, at_j))]      
            else:
                bond_pair_atom_types += [-1]
            indices.append([i[0], i[1]])

        #########################################
        # build index for inter-molecular bonds #
        #########################################
        bond_types = bond_pair_atom_types
        type_count = [0 for _ in range(len(pair_ids)+1)]
        for type_i in bond_types:
            if type_i != -1:
                type_count[type_i] += 1
        bond_types = np.array(bond_types)
        type_count = np.array(type_count)

        graphs = a2a_graph, line_graph
        global_feat = coord, inter_feats, bond_types, type_count
        return graphs, global_feat

    def load_data(self):
        """ Generate complex
         interaction graphs. """
        if self.has_cache():
            graphs, global_feat, labels = self.load()
            self.a2a_graphs, self.line_graph = graphs
            self.coords,self.inter_feats_list, self.bond_types_list, self.type_count_list = global_feat
            self.labels = labels
        else:
            print('Processing raw protein-ligand complex data...')
            file_name = os.path.join(self.data_path, "{0}.pkl".format(self.dataset))
            with open(file_name, 'rb') as f:
                data_mols, data_Y = pickle.load(f)
            

            for mol, y in tqdm(zip(data_mols, data_Y)):
                graphs, global_feat = self.build_graph(mol)
                if graphs is None:
                    continue
                self.a2a_graphs.append(graphs[0])
                self.line_graph.append(graphs[1])

                self.coords.append(global_feat[0])
                self.inter_feats_list.append(global_feat[1])
                self.bond_types_list.append(global_feat[2])
                self.type_count_list.append(global_feat[3])
                self.labels.append(y)

            self.labels = np.array(self.labels).reshape(-1, 1)
            # self.labels = np.array(data_Y).reshape(-1, 1)
            if self.save_file:
                self.save()


def collate_fn(batch):
    a2a_gs, line_g, coords, feats, types, counts, labels  = map(list, zip(*batch))


    a2a_g = pgl.Graph.batch(a2a_gs).tensor()
    line_g = pgl.Graph.batch(line_g).tensor()
    coords = paddle.concat([paddle.to_tensor(t) for t in coords])
    feats = paddle.concat([paddle.to_tensor(f, dtype='float32') for f in feats])
    types = paddle.concat([paddle.to_tensor(t) for t in types])
    counts = paddle.stack([paddle.to_tensor(c) for c in counts], axis=1)
    labels = paddle.to_tensor(np.array(labels), dtype='float32')

    return a2a_g, line_g, coords, feats, types, counts, labels


if __name__ == "__main__":
    complex_data = ComplexDataset("./data/", "pdbbind2016_test", 5)
    loader = Dataloader(complex_data,
                        batch_size=32,
                        shuffle=False,
                        num_workers=1,
                        collate_fn=collate_fn)
    cc = 0
    for batch in loader:
        a2a_g, b2a_g, b2b_gl, feats, types, counts, labels = batch
        print(labels)
        cc += 1
        if cc == 2:
            break
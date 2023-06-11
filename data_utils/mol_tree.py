import copy
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import torch
import tqdm
from biopandas.pdb import PandasPdb

sys.path.append('.')
from data_utils.chemutils import (decode_stereo, enum_assemble, get_clique_mol,
                                  get_mol, get_smiles, set_atommap,
                                  tree_decomp)


def read_pdb(path, mol, raid=6.0):
    pdb_df = PandasPdb().read_pdb(path)

    protein_coord = pdb_df.df['ATOM'][['x_coord', 'y_coord', 'z_coord']]
    atom_type = pdb_df.df['ATOM']['atom_name']
    residue_name = pdb_df.df['ATOM']['chain_id'] + pdb_df.df['ATOM']['residue_number'].astype(str)
    residue_type = pdb_df.df['ATOM']['residue_name']
    
    ligand_coord = np.array(mol.GetConformer().GetPositions())
    protein_coord = np.array(protein_coord)
    pocket_residue = set()
    for i in range(len(protein_coord)):
        for j in range(len(ligand_coord)):
            if np.linalg.norm(protein_coord[i] - ligand_coord[j]) < raid:
                pocket_residue.add(residue_name[i])
    protein_atom_idx = [i for i,r in enumerate(residue_name) if r in pocket_residue]
    protein = {
        'atom_type': [atom_type[i] for i in protein_atom_idx],
        'residue_name': [residue_name[i] for i in protein_atom_idx],
        'residue_type': [residue_type[i] for i in protein_atom_idx],
        'coord': [protein_coord[i] for i in protein_atom_idx]
    }
    CA_idex = [i for i,atom in enumerate(protein['atom_type']) if atom == 'CA']
    protein_CA = {
        'residue_type': [protein['residue_type'][i] for i in CA_idex],
        'coord': [protein['coord'][i] for i in CA_idex],
        'ligand_name': path.split('/')[-1].split('.')[0],
        'pocket_name': path.split('/')[-2]
    }
    return protein_CA


def read_protein_mol(mol_path, protein_path, name):
    
    mol_suppl = Chem.SDMolSupplier(mol_path)
    mol = [x for x in mol_suppl if x is not None][0]
    protein_data = read_pdb(protein_path, mol)
    try:
        if name == "train":
            jt = MolTree(mol, vocab=vocab)
        else:
            jt = mol
        return jt, protein_data
    except:
        print("incompatible mol")
        return None

def get_slots(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return [(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in mol.GetAtoms()]

class Vocab(object):

    def __init__(self, smiles_list, fp_df):
        self.vocab = smiles_list
        self.vmap = {x:i for i,x in enumerate(self.vocab)}
        self.slots = [get_slots(smiles) for smiles in self.vocab]
        self.fp_df = fp_df
        self.fps = [np.array(self.fp_df.loc[smiles]) for smiles in self.vocab]
        self.mol_sizes = [Chem.MolFromSmiles(smiles).GetNumHeavyAtoms() for smiles in self.vocab]

        
    def get_index(self, smiles):
        return self.vmap[smiles]

    def get_size(self, size):
        return [i for i,x in enumerate(self.mol_sizes) if x==size]
    
    def get_array(self, array):
        return [i for i,x in enumerate(self.fps) if np.array_equal(x, array)]
    
    def compute_size_for_idx(self, idx):
        return Chem.MolFromSmiles(self.vocab[idx]).GetNumHeavyAtoms()

    def get_smiles(self, idx):
        return self.vocab[idx]
    
    def get_fp(self, smiles):
        return np.array(self.fp_df.loc[smiles])

    def get_slots(self, idx):
        return copy.deepcopy(self.slots[idx])

    def size(self):
        return len(self.vocab)

class MolTreeNode(object):

    def __init__(self, smiles, pos, clique=[], vocab=None, hbd=None):
        self.smiles = smiles
        self.mol = get_mol(self.smiles)
        self.wid = None
        self.fp = None
        if vocab:
            self.fp = vocab.fp_df.loc[smiles]#new embedding
            self.wid = vocab.get_index(smiles)

        self.clique = [x for x in clique] #copy
        self.neighbors = []
        self.pos = pos
        self.hbd = hbd
        
    def add_neighbor(self, nei_node):
        self.neighbors.append(nei_node)

    def recover(self, original_mol):
        clique = []
        clique.extend(self.clique)
        if not self.is_leaf:
            for cidx in self.clique:
                original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(self.nid)

        for nei_node in self.neighbors:
            clique.extend(nei_node.clique)
            if nei_node.is_leaf: #Leaf node, no need to mark 
                continue
            for cidx in nei_node.clique:
                #allow singleton node override the atom mapping
                if cidx not in self.clique or len(nei_node.clique) == 1:
                    atom = original_mol.GetAtomWithIdx(cidx)
                    atom.SetAtomMapNum(nei_node.nid)

        clique = list(set(clique))
        label_mol = get_clique_mol(original_mol, clique)
        self.label = Chem.MolToSmiles(Chem.MolFromSmiles(get_smiles(label_mol)))
        self.label_mol = get_mol(self.label)

        for cidx in clique:
            original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)

        return self.label

class MolTreeNode_blur(object):

    def __init__(self, fp, pos, size):
        self.fp = fp
        self.wid = None#check this to differentiate from MolTreeNode
        self.neighbors = []
        self.pos = pos
        self.size = size
        
    def add_neighbor(self, nei_node):
        self.neighbors.append(nei_node)
    

class MolTree(object):

    def __init__(self, mol, nodes=None, edge_index=None, vocab=None):
        if mol:#use for data preprocess
            self.smiles = Chem.MolToSmiles(mol)
            self.mol3D = mol
            self.mol3D = Chem.RemoveHs(self.mol3D)
            Chem.Kekulize(self.mol3D)

            cliques, edges = tree_decomp(self.mol3D)
            self.adj_matrix = np.zeros((len(cliques), len(cliques)))
            self.nodes = []
            root = 0
            for i,c in enumerate(cliques):
                Chem.AddHs(self.mol3D)
                mol3D_checkH = self.mol3D
                Chem.AddHs(mol3D_checkH)
                Hydro_start = ('O', 'N', 'S', 'P')
                node_hbd = 0
                for atom_idx in c:
                    atom = mol3D_checkH.GetAtomWithIdx(atom_idx)
                    if atom.GetSymbol() in Hydro_start:
                        node_hbd += atom.GetTotalNumHs()
                
                cmol = get_clique_mol(self.mol3D, c)
                try:
                    node_pos = np.mean([self.mol3D.GetConformer().GetAtomPosition(x) for x in c], axis=0)
                except:
                    #print('Bad Conformer, init 0 position \r')
                    node_pos = np.zeros((1,3))
                node = MolTreeNode(get_smiles(cmol), node_pos, c, vocab=vocab, hbd=node_hbd)
                self.nodes.append(node)
                if min(c) == 0:
                    root = i

            for x,y in edges:
                self.nodes[x].add_neighbor(self.nodes[y])
                self.nodes[y].add_neighbor(self.nodes[x])
                self.adj_matrix[x, y] = 1
                self.adj_matrix[y, x] = 1#met bug before
            if root > 0:
                self.nodes[0],self.nodes[root] = self.nodes[root],self.nodes[0]
                self.adj_matrix[[0, root], :] = self.adj_matrix[[root, 0], :]
                self.adj_matrix[:, [0, root]] = self.adj_matrix[:, [root, 0]]

            for i,node in enumerate(self.nodes):
                node.nid = i + 1
                if len(node.neighbors) > 1: #Leaf node mol is not marked
                    set_atommap(node.mol, node.nid)
                node.is_leaf = (len(node.neighbors) == 1)

        elif nodes is not None:#use for reconstruction
            #self.nodes = [MolTreeNode(node[0], node[1], vocab=vocab) for node in nodes]
            self.nodes = nodes
            for i in range(len(self.nodes)):
                self.nodes[i].idx = i
            self.adj_matrix = np.zeros((len(nodes), len(nodes)))
            self.decode_adj_matrix = np.zeros((len(nodes), len(nodes)))
            if edge_index is not None:
                exist_edge = set()
                for ind in range(edge_index[0].shape[0]):
                    i, j = edge_index[0][ind], edge_index[1][ind]
                    self.adj_matrix[i, j] = 1
                    self.adj_matrix[j, i] = 1
                    if (i, j) not in exist_edge:
                        self.nodes[i].add_neighbor(self.nodes[j])
                        exist_edge.add((i, j))
                    if (j, i) not in exist_edge:
                        self.nodes[j].add_neighbor(self.nodes[i])
                        exist_edge.add((j, i))
            
        else:
            raise ValueError('Invalid input for MolTreeNodes')
    def add_node(self, node, link_index=None):
        if link_index is not None:
            for i in link_index:
                self.nodes[i].add_neighbor(node)
                node.add_neighbor(self.nodes[i])
            new_adj_matrix = np.zeros((len(self.nodes) + 1, len(self.nodes) + 1))
            new_adj_matrix[:self.adj_matrix.shape[0], :self.adj_matrix.shape[1]] = self.adj_matrix
            new_decode_adj_matrix = np.zeros((len(self.nodes) + 1, len(self.nodes) + 1))
            new_decode_adj_matrix[:self.adj_matrix.shape[0], :self.adj_matrix.shape[1]] = self.decode_adj_matrix
            for i in link_index:
                new_adj_matrix[-1, i] = 1
                new_adj_matrix[i, -1] = 1
                new_decode_adj_matrix[i, -1] = 1
            self.adj_matrix = new_adj_matrix
            self.decode_adj_matrix = new_decode_adj_matrix

        self.nodes.append(node)
    
    def add_edge(self, i, j):
        self.adj_matrix[i, j] = 1
        self.adj_matrix[j, i] = 1
        self.nodes[i].add_neighbor(self.nodes[j])
        self.nodes[j].add_neighbor(self.nodes[i])
        self.decode_adj_matrix[i, j] = 1

    def size(self):
        return len(self.nodes)

    def recover(self):
        for node in self.nodes:
            node.recover(self.mol3D)

    def assemble(self):
        for node in self.nodes:
            node.assemble()

if __name__ == "__main__":
    import sys
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    vocab_dir = 'dataset/vocab.txt'
    with open(vocab_dir, 'r') as f:
        vocab = [x.strip() for x in f.readlines()]
    vocab_fp = pd.read_csv('dataset/vocab_blur_fps_updated.csv', index_col=0)
    vocab = Vocab(vocab, vocab_fp)
    data_name = sys.argv[1]
    if data_name == 'crossdock':
        crossdock_dir_sdf = 'data/crossdock_mols.sdf'
        suppl = Chem.SDMolSupplier(crossdock_dir_sdf)
        mol_list = [x for x in suppl if x is not None]
        output_dir = 'data/crossdock_blur_trees'
            #code used for save mol trees of crossdock dataset
        for i, mol in enumerate(tqdm.tqdm(mol_list)):
            tree_list = []
            try:#check vocab compat automatically
                jt = MolTree(mol, vocab=vocab)
                tree_list.append(jt)
            except:
                continue
            with open(os.path.join(output_dir, f'{i}_drug_trees.pkl'), 'wb') as f:
                    pickle.dump(tree_list, f)
        print(f'{len(tree_list)} trees saved')
    

    elif data_name == 'GEOM_drug':
        base_dir = 'data/GEOM/rdkit_folder/drugs/'
        output_dir = 'data/GEOM_drugs_trees_blur_correct_adj'
        pickled_path = [os.path.join(base_dir, d) for d in os.listdir(base_dir)]
        start_time = time.time()
        for i, p in enumerate(tqdm.tqdm(pickled_path)):
            tree_list = []
            with open(p, 'rb') as f:
                try:
                    file = pickle.load(f)
                    mols = [file['conformers'][i]['rd_mol'] for i in range(len(file['conformers']))]
                    random.shuffle(mols)
                    if len(mols) > 4:
                        mols = mols[:4]
                except:
                    continue #some pickled files are corrupted
                for mol in mols:
                    try:
                        jt = MolTree(mol, vocab=vocab)
                        tree_list.append(jt)
                    except:
                        continue#some mols are corrupted
                    
            if len(tree_list) > 1:
                with open(os.path.join(output_dir, f'{i}_drug_trees.pkl'), 'wb') as f:
                    pickle.dump(tree_list, f)
        
    elif data_name == 'crossdock_cond':
        crossdock_dir_split = "data/split_by_name.pt"
        crossdock_dir_data = "data/crossdocked_pocket10"
        split_paths = torch.load(crossdock_dir_split)
        output_trees = {"train": [], "test": []}
        num_workers = 64
        if not num_workers:
            for name in split_paths.keys():
                for line in tqdm.tqdm(split_paths[name]):
                    mol_path = os.path.join(crossdock_dir_data, line[1])
                    protein_path = os.path.join(crossdock_dir_data, line[0])
                    jt, protein_data = read_protein_mol(mol_path, protein_path, name)
                    
        else:
            #multiprocessing
            pool = mp.Pool(processes=num_workers)
            for name in split_paths.keys():
                batch_paths = []
                for line in tqdm.tqdm(split_paths[name]):
                    mol_path = os.path.join(crossdock_dir_data, line[1])
                    protein_path = os.path.join(crossdock_dir_data, line[0])
                    batch_paths.append((mol_path, protein_path, name))
                    if len(batch_paths) == num_workers:
                        results = pool.starmap(read_protein_mol, batch_paths)
                        for result in results:
                            if result is not None:
                                output_trees[name].append(result)
                        batch_paths = []
                if len(batch_paths) > 0:
                    results = pool.starmap(read_protein_mol, batch_paths)
                    for result in results:
                        if result is not None:
                            output_trees[name].append(result)
                
        for i, d in tqdm.tqdm(enumerate(output_trees["train"])):
            with open(f"data/crossdock_trees_cond_name/train_{i}", "wb") as f:
                pickle.dump(d, f)
        for i, d in tqdm.tqdm(enumerate(output_trees["test"])):
            with open(f"data/crossdock_trees_cond_name/test_{i}", "wb") as f:
                pickle.dump(d, f)


    else:
        raise ValueError('Wrong data name')

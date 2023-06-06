import copy
import pickle
import sys
from collections import deque

import numpy as np
import rmsd
import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

sys.path.append('/home/songyuxuan/code/3Dmolgen')


from data_utils.mol_tree import *


def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]  # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    H = np.matmul(np.transpose(AA),BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        #print("Reflection detected")
        Vt[2, :] *= -1
        R = np.matmul(Vt.T,U.T)

    t = -np.matmul(R, centroid_A) + centroid_B
    # err = B - np.matmul(A,R.T) - t.reshape([1, 3])
    return R, t

def flexible_transform_3D(A, B):
    rotation = rmsd.kabsch(A - rmsd.centroid(A), B - rmsd.centroid(B))
    return rotation, (rmsd.centroid(A), rmsd.centroid(B))

def move_frag(mol, clique, coord):
  conf = mol.GetConformer()
  frag = []
  for a in mol.GetAtoms():
    if a.GetIdx() in clique:
        frag.append(a)
        #print(a.GetSymbol())
  original_xyz = np.stack([np.array(mol.GetConformer().GetAtomPosition(i)) for i in clique])
  center = np.mean(original_xyz, axis=0)
  shift = coord - center
  for ind, i in enumerate(clique):
     mol.GetConformer().SetAtomPosition(i, Point3D(*(original_xyz[ind] + shift)))
  return mol

def move_leaf(mol, clique, reference_mol, attached_pos, attached_clique):
  r, t = rigid_transform_3D(attached_pos[0], attached_pos[1])
  new_xyz = np.stack([np.array(reference_mol.GetConformer().GetAtomPosition(i)) for i in clique])
  new_xyz = np.matmul(new_xyz, r.T) + t.reshape([1, 3])
  #print(new_xyz)
  for ind, i in enumerate(clique):
    if i not in attached_clique:
      mol.GetConformer().SetAtomPosition(i, Point3D(*(new_xyz[ind])))
  return mol

    
def check_visit(v_set, clique):
  overlap = []
  for num in clique:
    if num in v_set:
      overlap.append(num)
  return overlap

class bfs_node(object):
    def __init__(self, idx, links):
        self.idx = idx
        self.links = links
        self.depth = None

def forfor(a):
        return [item for sublist in a for item in sublist]

def get_bfs_order(edges, n_nodes):
    edges = list(zip(*edges))
    bfs_links = [[] for _ in range(n_nodes)]
    for i in range(n_nodes):
        for link_parent, link_son in edges:
            if link_parent == i:
                bfs_links[i].append(link_son)
            elif link_son == i:
                bfs_links[i].append(link_parent)
    bfs_nodes = [bfs_node(idx, links) for idx, links in enumerate(bfs_links)]
    queue = deque([bfs_nodes[0]])
    visited = set([bfs_nodes[0].idx])
    bfs_nodes[0].depth = 0
    order1,order2 = [],[]
    bfs_order = [0,]
    while len(queue) > 0:
        x = queue.popleft()
        for y in x.links:
            y = bfs_nodes[y]
            if y.idx not in visited:
                queue.append(y)
                visited.add(y.idx)
                y.depth = x.depth + 1
                if y.depth > len(order1):
                    order1.append([])
                    order2.append([])
                order1[y.depth-1].append( (x.idx, y.idx) )
                order2[y.depth-1].append( (y.idx, x.idx) )
                bfs_order.append(y.idx)
    return bfs_order, forfor(order1)


def RMSD_package_tree(mol3D_1, mol3D_2):
    tree1, tree2 = MolTree(mol3D_1), MolTree(mol3D_2)
    xyz1, xyz2 = np.stack([n.pos for n in tree1.nodes]), np.stack([n.pos for n in tree2.nodes])
    return rmsd.kabsch_rmsd(xyz1, xyz2, translate=True)

def RMSD_package_mol(mol3D_1, mol3D_2):
    xyz1, xyz2 = np.stack([np.array(mol3D_1.GetConformer().GetAtomPosition(i)) for i in range(mol3D_1.GetNumAtoms())]), np.stack([np.array(mol3D_2.GetConformer().GetAtomPosition(i)) for i in range(mol3D_2.GetNumAtoms())])
    return rmsd.kabsch_rmsd(xyz1, xyz2, translate=True)


def set_rmsd(mol, amap, tree):
  smiles = Chem.MolToSmiles(mol)
  m3d_rdkit = Chem.AddHs(mol)
  AllChem.EmbedMolecule(m3d_rdkit, randomSeed=1)
  #AllChem.UFFOptimizeMoleculeConfs(m3d_rdkit)
  reference_mol = m3d_rdkit
  
  m3d_rdkit = Chem.RemoveHs(m3d_rdkit)
  #m3d_tree = Chem.RemoveHs(m3d_tree)
  reference_mol = Chem.RemoveHs(reference_mol)
  rdkit_xyz = np.stack([np.array(m3d_rdkit.GetConformer().GetAtomPosition(i)) for i in range(m3d_rdkit.GetNumAtoms())])
  node_atom_map = [list(a.values()) for a in amap]
  rdkit_xyz = np.stack([np.mean(rdkit_xyz[i], axis=0) for i in node_atom_map])
  tree_xyz = np.stack([np.array(n.pos) for n in tree.nodes])
  rotation, translate = flexible_transform_3D(rdkit_xyz, tree_xyz)

  mol_xyz = np.stack([np.array(m3d_rdkit.GetConformer().GetAtomPosition(i)) for i in range(m3d_rdkit.GetNumAtoms())])
  mol_xyz = np.dot(mol_xyz - translate[0], rotation) + translate[1]
  for i in range(m3d_rdkit.GetNumAtoms()):
      m3d_rdkit.GetConformer().SetAtomPosition(i, Point3D(*(mol_xyz[i])))
  
  visited = set()
  reference_nodes = tree.nodes
  attach_order, _ = get_bfs_order(tree.adj_matrix.nonzero(), len(tree.nodes))
  reference_nodes = [reference_nodes[i] for i in attach_order]
  #reference_nodes = sorted(reference_nodes, key=lambda n: len(n.clique), reverse=False)
  for i, n in enumerate(reference_nodes):
    n.clique = amap[i]
  for n in reference_nodes[:1]:
      if len(check_visit(visited, n.clique)) == 0:
        neighbors_ref_pos = np.stack([reference_nodes[i].pos for i in n.neighbors])
        neighbors_rdkit_pos = np.stack([np.mean([reference_mol.GetConformer().GetAtomPosition(c) for c in nei.clique], axis=0) for nei in n.neighbors])
        m3d_rdkit = move_leaf(m3d_rdkit, n.clique, reference_mol, attached_pos=[neighbors_rdkit_pos, neighbors_ref_pos], attached_clique=[])
        for v in n.clique:
            visited.add(v)
  for n in reference_nodes[1:]:
    attach = check_visit(visited, n.clique)
    neighbors_ref_pos = [reference_nodes[i].pos for i in n.neighbors]
    neighbors_ref_pos = np.stack(neighbors_ref_pos + [m3d_rdkit.GetConformer().GetAtomPosition(c) for c in attach])
    neighbors_rdkit_pos = [np.mean([reference_mol.GetConformer().GetAtomPosition(c) for c in nei.clique], axis=0) for nei in n.neighbors]
    neighbors_rdkit_pos = np.stack(neighbors_rdkit_pos + [reference_mol.GetConformer().GetAtomPosition(c) for c in attach])
    m3d_rdkit = move_leaf(m3d_rdkit, n.clique, reference_mol, attached_pos=[neighbors_rdkit_pos, neighbors_ref_pos], attached_clique=attach)
    for v in n.clique:
        visited.add(v)
  m3d_rdkit_org = copy.deepcopy(m3d_rdkit)
  AllChem.UFFOptimizeMoleculeConfs(m3d_rdkit)

  return {'tree_rmsd': RMSD_package_tree(m3d_rdkit, m3d_rdkit_org),
          'mol_rmsd': RMSD_package_mol(m3d_rdkit, m3d_rdkit_org),
          'mol_uff': m3d_rdkit, 'mol_org': m3d_rdkit_org}

def base_rmsd(mol):
  mol1 = copy.deepcopy(mol)
  mol2 = copy.deepcopy(mol)
  try:
    AllChem.UFFOptimizeMoleculeConfs(mol2)#some mol will violate the constraint
  except:
    return None
  return {'tree': RMSD_package_tree(mol1, mol2),
            'mol': RMSD_package_mol(mol1, mol2)}

'''
if __name__ == '__main__':
  with open('/home/AI4Science/qiangb/data_from_brain++/sharefs/3D_jtvae/crossdock_blur_trees_test.pkl', 'rb') as f:
    data = pickle.load(f)
  
  t_r, m_r = [], []
  for tree in tqdm.tqdm(list(data.values())[:200]):
    rmsd_ = set_rmsd(tree[0])
    t_r.append(rmsd_['tree'])
    m_r.append(rmsd_['mol'])
  print(f'tree RMSD---mean: {np.mean(t_r)} ---max: {np.max(t_r)}')
  print(f'mol RMSD---mean: {np.mean(m_r)} ---max: {np.max(m_r)}')
'''

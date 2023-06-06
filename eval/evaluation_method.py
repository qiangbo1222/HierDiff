import itertools
import os
import sys

sys.path.append('/home/AI4Science/qiangb/data_from_brain++/molgen/3D_jtvae')
from collections import Counter

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import tqdm
from rdkit.Chem import QED, AllChem, Descriptors, Draw, RDConfig
from rdkit.Chem.Scaffolds import MurckoScaffold

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from data_utils.mol_tree import *
from sklearn import metrics

#from RAscore import RAscore_NN
#nn_scorer = RAscore_NN.RAScorerNN()


base_dir = '/home/AI4Science/qiangb/data_from_brain++/molgen/3D_jtvae/notebooks'
_mcf = pd.read_csv(os.path.join(base_dir, 'mcf.csv'))
_pains = pd.read_csv(os.path.join(base_dir, 'wehi_pains.csv'),
                     names=['smarts', 'names'])
_filters = [Chem.MolFromSmarts(x) for x in
            _mcf.append(_pains, sort=True)['smarts'].values]

#sanitize and remove duplicates
def rdmols_cleaner(mols):
    mols_can = [Chem.MolFromSmiles(Chem.MolToSmiles(mol), sanitize=True) for mol in mols]
    mols_unique = set([Chem.MolToSmiles(mol) for mol in mols_can if mol])
    mols_unique  = [Chem.MolFromSmiles(mol) for mol in mols_unique]
    return mols_unique


#compute molecular weight from rdmol
def cal_MW(mols):
    mols = rdmols_cleaner(mols)
    return np.array([Descriptors.ExactMolWt(mol) for mol in mols if mol])

#compute the rate that can pass Chemical filter(https://github.com/molecularsets/moses/) from 
def mol_passes_filters(mol,
                       allowed=None,
                       isomericSmiles=False):
    """
    Checks if mol
    * passes MCF and PAINS filters,
    * has only allowed atoms
    * is not charged
    """
    allowed = allowed or {'C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H'}
    if mol is None:
        return False
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() != 0 and any(
            len(x) >= 8 for x in ring_info.AtomRings()
    ):
        return False
    h_mol = Chem.AddHs(mol)
    if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
        return False
    if any(atom.GetSymbol() not in allowed for atom in mol.GetAtoms()):
        return False
    if any(h_mol.HasSubstructMatch(smarts) for smarts in _filters):
        return False
    smiles = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)
    if smiles is None or len(smiles) == 0:
        return False
    if Chem.MolFromSmiles(smiles) is None:
        return False
    return True

def filter_rate(mols):
    mols = rdmols_cleaner(mols)
    return np.sum(np.array([mol_passes_filters(mol) for mol in mols])) / len(mols)
 
#compute logP
def cal_logP(mols):
    mols = rdmols_cleaner(mols)
    return np.array([Descriptors.MolLogP(mol) for mol in mols if mol if mol])

#compute the number of rotatable bonds
def cal_numrb(mols): 
    mols = rdmols_cleaner(mols)
    return np.array([Descriptors.NumRotatableBonds(mol) for mol in mols if mol])

#scaffold diversity (The Shannon entropy) 0 means all structures share the same scaffold
def scaffold_entropy(mols):
    mols = rdmols_cleaner(mols)
    scaffolds = [Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)) for mol in mols]
    p = np.array(list(Counter(scaffolds).values())) / len(mols)
    entropy = - np.sum(np.log(p) * p)
    return entropy
    
#The max fingerprint similarity with reference set
def max_sim_fp(mols, ref_mols):
    mols = rdmols_cleaner(mols)
    mols_fp = [np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)) for mol in mols if mol]
    ref_mols = rdmols_cleaner(ref_mols)
    ref_mols_fp = [np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)) for mol in ref_mols]
    sim_mat = np.zeros([len(mols_fp), len(ref_mols_fp)])
    for i in range(len(mols_fp)):
        for j in range(len(ref_mols_fp)):
            xo = (mols_fp[i] + ref_mols_fp[j] == 2)
            xx = ((mols_fp[i] + ref_mols_fp[j]) != 0)
            sim_mat[i, j] = np.sum(xo) / np.sum(xx)
    max_sim = np.max(sim_mat, axis=0)
    max_index = np.argmax(sim_mat, axis=0)
    return max_sim, [(mols[max_index[i]], ref_mols[i]) for i in range(max_index.size)]

#mininum RMSD with the UFF optimized conformers
def compute_rmsd(mol_origin, num=32, worker=4):
    try:
        #mol_origin = samples_pool.rdmol[0]
        mol = Chem.AddHs(mol_origin)
        mol_origin = Chem.AddHs(mol_origin)
        cids = AllChem.EmbedMultipleConfs(mol,numConfs=num, numThreads=worker)
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=worker)
        origin_id = mol.AddConformer(mol_origin.GetConformer(0))
        return np.mean([AllChem.GetConformerRMS(mol, origin_id, id_, prealigned=False) for id_ in range(len(mol.GetConformers()) - 1) if id_ != origin_id])
    except:
        return 'invalid mol generated'

def group_mean_rmsd(mols):
    smis = [Chem.MolToSmiles(mol) for mol in mols]
    smis_unique = set(smis)
    unique_index = [smis.index(smi) for smi in smis_unique]
    mols = [mols[ind] for ind in unique_index]
    rmsd = [compute_rmsd(mol) for mol in mols]
    rmsd = [dis for dis in rmsd if not isinstance(dis, str)]
    return np.mean(np.array(rmsd))

def cal_SAS(mols):
    return np.array([sascorer.calculateScore(mol) for mol in mols])

def cal_QED(mols):
    return np.array([QED.qed(mol) for mol in mols])

def cal_RA(mols):
    smiles = [Chem.MolToSmiles(mol) for mol in mols]
    return np.array([nn_scorer.predict(smi) for smi in smiles])

def cal_Rings(mols):
    ri_info = []
    for mol in mols:
        ri = mol.GetRingInfo()
        ri_info.append([len(r) for r in ri.AtomRings()])
    ri_count = np.array([len(r) for r in ri_info])
    ri_sizes = np.array([np.mean(r) if np.mean(r) >=0 else 0 for r in ri_info])
    return [ri_sizes, ri_count]

def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    if len(X.shape) == 1:
        X = np.expand_dims(X, 1)
        Y = np.expand_dims(Y, 1)
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def x_atom_par(mol):
    num_atom = mol.GetNumAtoms()
    num_x = Descriptors.NumHeteroatoms(mol)
    return num_x / num_atom

def x_atom_par_mols(mols):
    return np.array([x_atom_par(mol) for mol in mols])

def node_freq(mols, vocab):
    trees = []
    unfind = 0
    for mol in mols:
        try:
            trees.append(MolTree(mol, vocab=vocab))
        except:
            unfind += 1
    if unfind !=0:
        print(f'{unfind/len(mols)} molecules not found in the vocabulary')
    freq, fp = [], []
    for tree in trees:
        for node in tree.nodes:
            freq.append(node.wid)
            fp.append(node.fp)
    fp_array = np.mean(np.stack(fp, axis=0), axis=0)
    freq_array = np.zeros(vocab.size())
    for wid in freq:
        freq_array[wid] += 1
    return freq_array, fp_array


def calculate_ro5_properties(molecule):
    # Calculate Ro5-relevant chemical properties
    molecular_weight = Descriptors.ExactMolWt(molecule)
    n_hba = Descriptors.NumHAcceptors(molecule)
    n_hbd = Descriptors.NumHDonors(molecule)
    logp = Descriptors.MolLogP(molecule)
    numrb = Descriptors.NumRotatableBonds(molecule)
    # Check if Ro5 conditions fulfilled
    conditions = [molecular_weight <= 500, n_hba <= 10, n_hbd <= 5, logp <= 5, numrb <= 10]
    ro5_fulfilled = sum(conditions)
    # Return True if no more than one out of four conditions is violated
    return ro5_fulfilled

def ro5(mols):
    return np.array([calculate_ro5_properties(mol) for mol in mols])

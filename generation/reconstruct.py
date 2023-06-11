import argparse
import math
import pickle
import random
import sys

import tqdm

sys.path.append('./generation/jtnn')
sys.path.append('.')

import rdkit
import rdkit.Chem as Chem
import torch
import torch.nn as nn
from data_utils.mol_tree import *
from jtnn_dec import can_assemble
from torch.autograd import Variable

from generation.ar_sampling import beam_tree
from jtnn import *

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--start_num', type=int, default=0)
parser.add_argument('--tree_path', type=str, default='/home/qiangb/code/hierdiff_private/test/sampled_trees_from_prop_embed_0-1000__5.pkl')
parser.add_argument('--vocab_path', type=str, default='dataset/vocab.txt')
parser.add_argument('--output_dir', type=str, default='output_results.pkl')
parser.add_argument(
    "--vocab_fp_dir",
    type=str,
    default="dataset/vocab_blur_fps_updated.csv",
)
parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--depth', type=int, default=3)
parser.add_argument('--stereo', type=int, default=0)
args = parser.parse_args()
   
vocab = [x.strip("\r\n ") for x in open(args.vocab_path)]

vocab_fp = pd.read_csv(args.vocab_fp_dir, index_col=0)
vocab = Vocab(vocab, vocab_fp)

hidden_size = int(args.hidden_size)
latent_size = int(args.latent_size)
depth = int(args.depth)
stereo = True if int(args.stereo) == 1 else False



model = JTNNVAE(vocab, hidden_size, latent_size, depth, stereo=stereo)

with open(args.tree_path, 'rb') as f:
    data = pickle.load(f)



def check_qm9_atom(smi):
    mol = Chem.MolFromSmiles(smi)
    unallowed_atoms = ['B', 'P', 'S', 'Cl', 'Br', 'I']
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in unallowed_atoms:
            return False
        
    return True

results = []
smiles = []
all_smi_counter = 0
with torch.no_grad():
    for i, tree in enumerate(tqdm.tqdm(data, desc='Reconstructing molecules from trees')):
        #tree.tree.context = tree.context
        try:
            pocket_name = tree.name
        except:
            pocket_name = 'unknown'
        tree = tree.tree
        if len(tree.nodes) < 100:
            for idx, node in enumerate(tree.nodes):
                node.is_leaf = (len(node.neighbors) == 1)
                node.idx = idx
                for nei in node.neighbors:
                    node.is_leaf = (len(node.neighbors) == 1)
                    node.idx = idx
            output = model.sample_tree(tree, vocab, args)
            
            if output != 'max9':
                all_smi_counter += 1
                if output != 'invalid':
                    output_mol, amap, smi = output
                    can_smiles = Chem.MolToSmiles(smi)
                    #if check_qm9_atom(can_smiles):
                    node_pos = [n.pos for n in tree.nodes]
                    results.append((output_mol, amap[1:], pocket_name, smi))
                    smiles.append(can_smiles)
                    print(Chem.MolToSmiles(smi))

print(f'{len(results)} molecules reconstructed.')
print(f'valid: {len(results) / all_smi_counter}')
print(f'unique: {len(set(smiles)) / len(smiles)}')
print(f'average atom num: {sum([len(mol.GetAtoms()) for mol, _, _, _ in results]) / len(results)}')
with open(args.output_dir+f'{args.start_num}__5.pkl', 'wb') as f:
    pickle.dump(results, f)


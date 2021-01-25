import os
from tqdm import tqdm
import argparse
from rdkit import Chem
from rdkit.Chem import Draw
import deepchem as dc
"""
m = Chem.MolFromMolFile('data/input.mol')
Chem.MolToSmiles(m,isomericSmiles=False)
consider adding diversity to training data if its an issue
https://www.rdkit.org/docs/GettingStartedInPython.html
featurizer = dc.feat.graph_features.ConvMolFeaturizer()
mol_object = featurizer.featurize(mols=molecules)
"""

state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
torch.save(state, args.save_name)
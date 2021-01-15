import os
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

def main(args):
    count = 0
    with open(args.input_file,'r') as f:
        with open(os.path.join(args.output_folder,'labels.smi'),'w') as w:
            for l in f:
                l = l.strip()
                m = Chem.MolFromSmiles(l)
                if m != None:
                    Draw.MolToFile(m,os.path.join(args.output_folder, '{}.png'.format(count)))
                    w.write("{}\t{}\n".format(l,count))
                    count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='data/validation.smi', help='source of input smiles data')
    parser.add_argument('--output_folder', type=str, default='data/validation_images')
    parser.add_argument('--smiles', action='store_true', help='diversify images produces')
    parser.add_argument('--diversify', action='store_true', help='diversify images produces')
    args = parser.parse_args()
    main(args)
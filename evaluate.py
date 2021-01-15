import os
import argparse
from rdkit import Chem
from rdkit.Chem import Draw
import deepchem as dc

def main(args):
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=str, help='a text file of candiate smiles strings.')
    parser.add_argument('--reference', type=str, help='a text file of reference smiles strings.')
    parser.add_argument('--strict', action='store_true', help='Binary labels on correct string')
    args = parser.parse_args()
    main(args)
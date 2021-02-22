import os
from tqdm import tqdm
import argparse
from rdkit import Chem

def main(args):
    count = 0
    with open(args.input_file,'r') as f:
        with open(args.output_file,'w') as w:
            for i, l in enumerate(tqdm(f)):
                l = l.strip()
                try:
                    canonical = Chem.CanonSmiles(l)
                    w.write("{}\n".format(canonical))
                except:
                    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate SMILES to canonical smiles')
    parser.add_argument('--input_file', type=str, default='data/validation.smi', help='source of input smiles data')
    parser.add_argument('--output_file', type=str, default='data/validation_canonical.smi', help='cleaned cannonical')
    args = parser.parse_args()
    main(args)
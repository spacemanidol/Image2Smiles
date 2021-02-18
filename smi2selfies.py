import os
from tqdm import tqdm
import argparse
import selfies as sf

def main(args):
    with open(args.input_file,'r') as f:
        with open(args.output_file,'w') as w:
            for l in f:
                l = l.strip()
                w.write('{}\n'.format(sf.encoder(l)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate smiles to selfies')
    parser.add_argument('--input_file', type=str, default='data/training.smi', help='source of input smiles files')
    parser.add_argument('--output_file', type=str, default='data/training.selfies', help='selfies file')
    args = parser.parse_args()
    main(args)



import os
from tqdm import tqdm
import argparse
from rdkit import Chem

def main(args):
    files = os.listdir(args.input_dir)
    with open(args.output_file,'w') as w:
        for a_file in files:
            try:
                m = Chem.MolFromMolFile(os.path.join(args.input_dir, a_file))
                w.write('{}\t{}TIF\n'.format(Chem.MolToSmiles(m),a_file[:-3]))
            except:
                pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate mol files to SMI')
    parser.add_argument('--input_dir', type=str, default='data/uspto-validation-updated/molfiles/', help='source of input mol files')
    parser.add_argument('--output_file', type=str, default='data/uspto-validation-updated/images/labels.smi', help='smiles processed')
    args = parser.parse_args()
    main(args)


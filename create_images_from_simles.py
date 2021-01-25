import os
from tqdm import tqdm
import argparse
from rdkit import Chem
from rdkit.Chem import Draw

def main(args):
    count = 0
    with open(args.input_file,'r') as f:
        with open(os.path.join(args.output_folder,'labels.smi'),'w') as w:
            for i, l in enumerate(tqdm(f)):
                l = l.strip()
                try:
                    m = Chem.MolFromSmiles(l)
                    if m != None:
                        Draw.MolToFile(m,os.path.join(args.output_folder, '{}.png'.format(count)))
                        w.write("{}\t{}\n".format(l,count))
                        count += 1
                except:
                    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Images and associated captions labels for use in downstream training.')
    parser.add_argument('--input_file', type=str, default='data/validation.smi', help='source of input smiles data')
    parser.add_argument('--output_folder', type=str, default='data/validation_images')
    parser.add_argument('--smiles', action='store_true', help='diversify images produces')
    parser.add_argument('--diversify', action='store_true', help='diversify images produces. Nothing implemented yet')
    args = parser.parse_args()
    main(args)
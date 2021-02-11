import os
from tqdm import tqdm
import argparse
from rdkit import Chem
from rdkit.Chem import Draw

def main(args):
    count = args.count_start
    with open(args.input_file,'r') as f:
        with open(os.path.join(args.output_folder,'labels.smi'),'w') as w:
            for i, l in enumerate(tqdm(f)):
                l = l.strip()
                try:
                    m = Chem.MolFromSmiles(l)
                    if m != None:
                        Draw.MolToFile(m,os.path.join(args.output_folder, '{}.png'.format(count)), size=(args.img_size,args.img_size))
                        w.write("{}\t{}.png\n".format(l,count))
                        count += 1
                except:
                    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Images and associated captions labels for use in downstream training.')
    parser.add_argument('--input_file', type=str, default='data/validation.smi', help='source of input smiles data')
    parser.add_argument('--output_folder', type=str, default='data/validation_images')
    parser.add_argument('--img_size', type=int, default=256, help='size of image to output')
    parser.add_argument('--diversify', action='store_true', help='diversify images produces. Nothing implemented yet')
    parser.add_argument('--count_start', type=int, default=0, help='start range for image numbering')
    args = parser.parse_args()
    main(args)

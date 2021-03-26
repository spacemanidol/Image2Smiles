import chemschematicresolver as csr
import osra_rgroup
import os
from tqdm import tqdm
import argparse

def main(args):
    with open(args.images_to_predict,'r') as f:
        with open(args.output,'w') as w:
            for i, l in enumerate(tqdm(f)):
                path = os.path.join(args.directory_path,l.strip())
                result = csr.extract_image(path)
                w.write("{}\t{}\n".format(top, l.strip()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction via CSR')
    parser.add_argument('--images_to_predict', default=None, type=str, help='a file indicating what images to predict. One png name per line')
    parser.add_argument('--directory_path', type=str, help='directory of images to predict')
    parser.add_argument('--output', type=str, default='output.txt', help='file name to produce model predictions for each image.')
    args = parser.parse_args()
    main(args)    

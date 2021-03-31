import osra_rgroup
import os
import argparse

def main(args):
    with open(args.images_to_predict,'r') as f:
        with open(args.output,'w') as w:
            for l in f:
                try:
                    path = os.path.join(args.directory_path,l.strip().split('\t')[1])
                    result = osra_rgroup.read_diagram(input_file=path, superatom_file='superatom.txt', spelling_file='spelling.txt')
                    w.write("{}\t{}\n".format(result.strip(), l.strip().split('\t')[1]))
                except:
                    pass
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction via CSR')
    parser.add_argument('--images_to_predict', default='../outputs/USPTO-reference.txt', type=str, help='a file indicating what images to predict. One png name per line')
    parser.add_argument('--directory_path', default='../data/uspto_images/', type=str, help='directory of images to predict')
    parser.add_argument('--output', type=str, default='../outputs/USPTO-pyosra-predictions.txt', help='file name to produce model predictions for each image.')
    args = parser.parse_args()
    main(args)    

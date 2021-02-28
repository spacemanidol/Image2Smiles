import chemschematicresolver as csr
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
    with torch.no_grad():
        with open(args.images_to_predict,'r') as f:
            with open(args.output,'w') as w:
                for i, l in enumerate(tqdm(f)):
                    path = os.path.join(args.directory_path,l.strip())
                    top = predict_captions(args, encoder, decoder, tokenizer,  path, transform, device)
                    if len(top) > 0:
                        top = top[0]
                    w.write("{}\t{}\n".format(top, l.strip()))
    print("Done Predicting")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction via CSR')
    parser.add_argument('--images_to_predict', default=None, type=str, help='a file indicating what images to predict. One png name per line')
    parser.add_argument('--directory_path', type=str, help='directory of images to predict')
    parser.add_argument('--beam_size', type=int, default=20, help='Beam size for candidate generation')
    parser.add_argument('--branch_rounds', type=int, default=5, help='Branch round for expanding beam')
    parser.add_argument('--branch_factor', type=int, default=5, help='How much to branch every beam by')
    parser.add_argument('--branches_to_expand', type=int, default=5, help='How many top branches to expand')
    parser.add_argument('--img_size', type=int, default=256, help='Image')
    parser.add_argument('--test_string', type=str, default='CC(C)CCNc1cnnc(NCCc2ccc(S(N)(=O)=O)cc2)n1', help='a SMILES string to test tokenizer with')
    parser.add_argument('--output', type=str, default='output.txt', help='file name to produce model predictions for each image.')
    parser.add_argument('--tokenizer', default='tokenizers/tokenizer_vocab_2000.json', type=str, help='tokenizer name in the folder tokenizers/')
    parser.add_argument('--encoder_type', default='RESNET101', type=str, help='Type of encoder architecture', choices=['RESNET101'])
    parser.add_argument('--decoder_type', default='LSTM+Attention', type=str, help='Type of decoder architecture', choices=['LSTM+Attention'])
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--cuda_device', default='cuda:2', type=str, help='cuda device to use. aka gpu')
    parser.add_argument('--model_path', default='models/vocab200checkpoint_0', type=str, help='model path')
    args = parser.parse_args()
    main(args)    



result = csr.extract_image('<path/to/image/file>')
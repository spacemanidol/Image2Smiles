import os
import argparse
import numpy as np
import json
from tqdm import tqdm
from scipy.misc import imread, imresize
from nltk.translate.bleu_score import corpus_bleu
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from rdkit import Chem

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

from encoders import Resnet101Encoder
from decoders import DecoderWithAttention

def encoder_img(args, encoder, transform, path, device):
    img = imread(path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (args.img_size, args.img_size))
    img = img.transpose(2, 0, 1)
    assert img.shape == (3, args.img_size, args.img_size)
    assert np.max(img) <= 255
    img  = torch.FloatTensor(img/255.)
    img = transform(img)
    img = torch.stack([img]).to(device)
    return encoder(img).to(device)

def get_closest(args, candidates, encoder_out, encoder, path, device, transform):
    idx = 0
    if len(candidates) > 1:
        similarities = []
        for smi in top:
            cur_sim = 0
            try:
                m = Chem.MolFromSmiles(l)
                if m != None:
                    Draw.MolToFile(m,'tmp.png'), size=(args.img_size,args.img_size))
                    cand_out = encoder_img(args, encoder, 'tmp.png', device)
                    similarities.append(math.log(torch.sum(torch.abs(encoder_out, cand_out))))
        max_sim = np.max(similarities)  
        idx = similarities.index(max_sim)
    return candidates[idx]

def load_selfies_vocab(input_file):
    idx2selfies, selfies2idx = {}, {}
    idx = 0
    with open(input_file) as f:
        for l in f:
            l = l.strip()
            idx2selfies[idx] = l
            selfies2idx[l] = idx
            idx += 1
    return idx2selfies, selfies2idx

def main(args):
    # Load Tokenizer
    print("Loading Tokenizer: {}.".format(args.tokenizer))
    tokenizer = Tokenizer.from_file(args.tokenizer)
    print("Testing with SMILES String: {}".format(args.test_string))
    encoding = tokenizer.encode(args.test_string)
    print("Encoded string: {}".format(encoding.tokens))
    decoded = tokenizer.decode(encoding.ids)
    print("Decoded string: {}".format(decoded))
    print("Tokenizer Loaded.")
    
    # Load model
    try:
        print("Loading models: {}".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        decoder = checkpoint['decoder']
        encoder = checkpoint['encoder']
    except: 
        print("Models couldn't be loaded. Aborting")
        exit(0)
    
    # Deal With CUDA
    if args.cuda:
        device = args.cuda_device
        cudnn.benchmark = True
    else:
        device = 'cpu'
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    decoder.eval()
    encoder.eval()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([normalize])
    use_selfies = args.use_selfies
    idx2selfies, selfies2idx = load_selfies_vocab(args.selfies_vocab)
    print("Predicting images")
    with torch.no_grad():
        with open(args.images_to_predict,'r') as f:
            with open(args.output,'w') as w:
                for i, l in enumerate(tqdm(f)):
                    path = os.path.join(args.directory_path,l.strip())
                    encoder_out = encoder_img(args, encoder, transform, path, device)
                    candidates = decoder.predict(encoder_out, tokenizer, args.beam_size, args.branch_rounds, args.branch_factor, args.branches_to_expand, device, args.use_selfies, idx2selfies)
                    candidate = 'NONE'
                    if len(candidates) > 0:
                        candidate = get_closest(args, candidates, encoder_out, encoder, path, device, transform)
                    w.write("{}\t{}\n".format(candidate, l.strip()))
    print("Done Predicting")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Smiles Given an input image')
    parser.add_argument('--selfies_vocab', type=str, default = 'data/selfies.vocab', help='vocab file for selfies encoding')
    parser.add_argument('--use_selfies', action='store_true', help='Use selfies')
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

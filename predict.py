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

def predict_captions(args, encoder, decoder, tokenizer, path, transform,device):
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
    encoder_out = encoder(img).to(device)
    top = decoder.predict(encoder_out, tokenizer, args.beam_size,device)
    return top


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
    print("Predicting images")
    with torch.no_grad():
        with open(args.images_to_predict,'r') as f:
            with open(args.output,'w') as w:
                for i, l in enumerate(tqdm(f)):
                    path = os.path.join(args.directory_path,l.strip())
                    top = predict_captions(args, encoder, decoder, tokenizer,  path, transform, device)
                    w.write("{}\t{}\n".format(top, l.strip()))
    print("Done Predicting")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Smiles Given an input image')
    parser.add_argument('--images_to_predict', default=None, type=str, help='a file indicating what images to predict. One png name per line')
    parser.add_argument('--directory_path', type=str, help='directory of images to predict')
    parser.add_argument('--beam_size', type=int, default=20, help='Beam size for candidate generation')
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

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img = imread(path)
if len(img.shape) == 2:
    img = img[:, :, np.newaxis]
    img = np.concatenate([img, img, img], axis=2)


img = imresize(img, (img_size, img_size))
img = img.transpose(2, 0, 1)
assert img.shape == (3, img_size, img_size)
assert np.max(img) <= 255
img  = torch.FloatTensor(img/255.)
img = transform(img)
img = torch.stack([img])
k = 5
encoder_out = encoder(img)

enc_image_size = encoder_out.size(1)
encoder_dim = encoder_out.size(3)
# Flatten encoding
encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
num_pixels = encoder_out.size(1)
# We'll treat the problem as having a batch size of k
encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
# Tensor to store top k previous words at each step; now they're just <start>
k_prev_words = torch.LongTensor([[1]] * k).to(device) # (k, 1)
# Tensor to store top k sequences; now they're just <start>
seqs = k_prev_words  # (k, 1)
# Tensor to store top k sequences' scores; now they're just 0
top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
# Tensor to store top k sequences' alphas; now they're just 1s
seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)
# Lists to store completed sequences, their alphas and scores
complete_seqs = list()
complete_seqs_scores = list()
# Start decoding
step = 1
h, c = decoder.init_hidden_state(encoder_out)
# s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
while True:
    embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
    awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
    alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)
    gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
    awe = gate * awe
    h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
    scores = decoder.fc(h)  # (s, vocab_size)
    scores = F.log_softmax(scores, dim=1)
    # Add
    scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
    # For the first step, all k points will have the same scores (since same k previous words, h, c)
    if step == 1:
        top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
    else:
        # Unroll and find top scores, and their unrolled indices
        top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)
    # Convert unrolled indices to actual indices of scores
    prev_word_inds = top_k_words / vocab_size  # (s)
    next_word_inds = top_k_words % vocab_size  # (s)
    # Add new words to sequences, alphas
    seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
    # Which sequences are incomplete (didn't reach <end>)?
    incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != -1]
    complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
    # Set aside complete sequences
    if len(complete_inds) > 0:
        complete_seqs.extend(seqs[complete_inds].tolist())
        complete_seqs_scores.extend(top_k_scores[complete_inds])
    k -= len(complete_inds)  # reduce beam length accordingly
    # Proceed with incomplete sequences
    if k == 0:
        break
    seqs = seqs[incomplete_inds]
    h = h[incomplete_inds]
    c = c[incomplete_inds]
    encoder_out = encoder_out[incomplete_inds]
    top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
    k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
    # Break if things have been going on too long
    if step > 150:
        break
    step += 1




    #i = complete_seqs_scores.index(max(complete_seqs_scores))
    #seq = complete_seqs[i]
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
    vocab_size = tokenizer.get_vocab_size()
    
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
        device = 'cuda:0'
        cudnn.benchmark = True
    else:
        device = 'cpu'
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    decoder.eval()
    encoder.eval()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    # Custom dataloaders
    print("Loading image")

def load_normalize_img(path, normalize, img_size=256):

img = imread(path)
if len(img.shape) == 2:
    img = img[:, :, np.newaxis]
    img = np.concatenate([img, img, img], axis=2)


img = imresize(img, (img_size, img_size))
img = img.transpose(2, 0, 1)
assert img.shape == (3, img_size, img_size)
assert np.max(img) <= 255
img  = torch.FloatTensor(img/255.)
img = transform(img)
img = torch.stack([img])
encoder_out = encoder([img])
print("Image loaded")


with torch.no_grad():
    
    imgs = []
    if args.predict_whole_directory:
        #load whole Directory of image
    else:
        pass
    print("Images loaded. There are {} images to predict".format(len(imgs)))

    # Predict Captions
    with open(args.output_dir,'r') as w:
        for img in imgs:
            predictions = caption_image(encoder, decoder, img, idx2word, beam_size)
            smiles = []
            for prediction in predictions:
                smiles.append(tokenizer.decode(encoding.ids))
            smiles2condfidence = generateImages
            # Create Images from Predicted SMILES and choose one which matches the original image most. 
    with torch.no_grad():
        for imgs, caps, caplens in tqdm(loader):
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            # Forward pass
            imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
            scores_copy = scores.clone().to(device)
            targets = caps_sorted[:, 1:] #remove <start> and <end> tokens
            targets_copy = targets.clone().to(device)
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).to(device) #remove padding tokens
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).to(device)
            # Calculate loss
            loss = criterion(scores.data, targets.data).to(device)
            # Add doubly stochastic attention regularization
            loss += args.alphac * ((1. - alphas.sum(dim=1).to(device)) ** 2).mean()
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp = list()
            for j, p in enumerate(preds):
                temp.append(preds[j][:decode_lengths[j]])  # remove pads
            candidates.extend(temp)
            refs = targets_copy.tolist()
            temp = list()
            for j, p in enumerate(refs):
                temp.append(refs[j][:decode_lengths[j]])
            references.extend(temp)
        print("Done Validation Moving to tokenize and BLEU")
        temp = list()
        for ref in references:
            tmp = tokenizer.decode(ref)
            tmp = tokenizer.encode(tmp)
            pad_index = len(tmp.tokens) - 1
            if '<pad>' in tmp.tokens:
                pad_index = tmp.tokens.index('<pad>')
            temp.append(' '.join(tmp.tokens[:pad_index]))
        references = temp
        print("Done translating references")
        temp = list()
        for ref in candidates:
            tmp = tokenizer.decode(ref)
            tmp = tokenizer.encode(tmp)
            pad_index = len(tmp.tokens) - 1
            if '<pad>' in tmp.tokens:
                pad_index = tmp.tokens.index('<pad>')
            temp.append(' '.join(tmp.tokens[:pad_index]))
        candidates = temp
        print("Done translating candidates")

        assert len(references) == len(candidates)
        bleu4 = corpus_bleu(references, candidates)
        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))
        return bleu4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Smiles Given an input image')
    parser.add_argument('--image_path', type=str, help='path to image to be predicted')
    parser.add_argument('--directory_path', type=str, help='directory of images to predict')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for candidate generation')
    parser.add_argument('--encoder_type', type=str, default='Resnet101Encoder', help='Architecture used for the encoder')
    parser.add_argument('--test_string', type=str, default='CC(C)CCNc1cnnc(NCCc2ccc(S(N)(=O)=O)cc2)n1', help='a SMILES string to test tokenizer with')
    parser.add_argument('--beam_size', default=5, type=int, help='beam size for prediction creation')
    parser.add_argument('--output_dir', type=str, default='output.txt', help='file name to produce model predictions for each image.')
    parser.add_argument('--max_length', type=int, default=150, help='Max length of tokenized smiles')
    parser.add_argument('--tokenizer', default='tokenizers/tokenizer_vocab_2000.json', type=str, help='tokenizer name in the folder tokenizers/')
    parser.add_argument('--test_string', type=str, default='CC(C)CCNc1cnnc(NCCc2ccc(S(N)(=O)=O)cc2)n1', help='a SMILES string to test tokenizer with')
    parser.add_argument('--num_workers', default=8, type=int, help='Workers for data loading')
    parser.add_argument('--batch_size', default=32, type=int, help='Size of sampled batch')
    parser.add_argument('--encoder_type', default='RESNET101', type=str, help='Type of encoder architecture', choices=['RESNET101'])
    parser.add_argument('--decoder_type', default='LSTM+Attention', type=str, help='Type of decoder architecture', choices=['LSTM+Attention'])
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--model_path', default='models/', type=str, help='model path')
    args = parser.parse_args()
    main(args)    
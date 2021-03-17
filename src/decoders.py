from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils

import copy
import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from rdkit import Chem
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import selfies as sf

class Attention(nn.Module):
    """
    Attention Network.
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
    
        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()
    
        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size) #.to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels) #.to(device)
        
        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),(h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    def predict(self, encoder_out, tokenizer, beam_size, branch_rounds, branch_factor, branches_to_expand, device, use_selfies, idx2selfies, selfies2idx):
        """
        Caption prediction
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param tokenizer: a BPE tokenizer 
        :param beam_size: caption lengths, a tensor of dimension (batch_size, 1)
        :param branch rounds: how many times do we branch top beams
        :param branch_factor: how many branches will each beam fork have
        :param branches_to_expand: how many beams to branch
        :param device: 
        :return: a list of possible captions for each image. 
        """
        start_token_id = tokenizer.encode('<start>').ids[0]
        end_token_id = tokenizer.encode('<end>').ids[0]
        pad_token_id = tokenizer.get_vocab_size()
        vocab_size = tokenizer.get_vocab_size()+ 1
        if use_selfies:
            start_token_id = selfies2idx['[start]']
            end_token_id = selfies2idx['[end]']
            pad_token_id = selfies2idx['[pad]']
            vocab_size = len(selfies2idx)
        k =  beam_size
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)# Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim).to(device)  # (k, num_pixels, encoder_dim)
        k_prev_words = torch.LongTensor([[start_token_id]] * k).to(device) # (k, 1)
        seqs = k_prev_words  # (k, 1)
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
        complete_seqs = list()
        complete_seqs_scores = list()
        step = 1
        h, c = self.init_hidden_state(encoder_out)
        while True:
            embeddings = self.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            awe, _ = self.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (s, encoder_dim)
            h, c = self.decode_step(torch.cat([embeddings, gate * awe], dim=1), (h, c))  # (s, decoder_dim)
            scores = self.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)   # For the first step, all k points will have the same scores (since same k previous words, h, c)                
            else:
                #print("{}\t{}\t{}\t{}".format(k,step, branches_to_expand, top_k_scores))
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # Unroll and find top scores, and their unrolled indices
                if branch_rounds > 0:
                    tmp = branches_to_expand
                    if branches_to_expand > len(top_k_scores):
                        tmp = len(top_k_scores) 
                    _, expand_index = torch.topk(top_k_scores, tmp, dim=0) 
            next_word_inds = top_k_words % vocab_size  #  Convert unrolled indices to actual indices of scores
            if step > 2 and branch_rounds > 0:
                branch_rounds -= 1 
                for i in range(0, len(expand_index)):
                    next_beam_scores, next_beam_words = scores[expand_index[i]].topk(branch_factor+1)
                    in_play = [ind for ind, next_word in enumerate(next_beam_words) if next_word != next_word_inds[expand_index[i]]]
                    next_beam_scores = next_beam_scores[in_play]
                    next_beam_words = next_beam_words[in_play]
                    seq_to_add = torch.stack([seqs[expand_index[i]]])
                    h_to_add = torch.stack([h[expand_index[i]]])
                    c_to_add = torch.stack([c[expand_index[i]]])
                    encoder_to_add = torch.stack([encoder_out[expand_index[i]]])
                    score_to_add = torch.stack([top_k_scores[expand_index[i]]])
                    for j in range(0, branch_factor):
                        k+= 1
                        seqs = torch.cat([seqs, seq_to_add])
                        next_word_inds = torch.cat([next_word_inds, torch.stack([next_beam_words[j]])] )
                        h = torch.cat([h, h_to_add])
                        c = torch.cat([c, c_to_add])
                        encoder_out = torch.cat([encoder_out, encoder_to_add])
                        top_k_scores = torch.cat([top_k_scores, score_to_add])
            seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != end_token_id] # Which sequences are incomplete (didn't reach <end>)?
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds)) # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[incomplete_inds]
            c = c[incomplete_inds]
            encoder_out = encoder_out[incomplete_inds]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            if step > 150:
                break
            step += 1
        mol_index = np.argsort(complete_seqs_scores)
        real_molecules = []
        if use_selfies == False:
            for i in mol_index:
                try:
                    smi = tokenizer.decode(complete_seqs[i][1:-1]) #remove start and end token
                    mol = Chem.MolFromSmiles(smi)
                    can_smi = Chem.MolToSmiles(mol, True) 
                    if len(can_smi) > 1:
                        real_molecules.append(can_smi)
                except:
                    pass
        else:
            for i in mol_index:
                tmp = complete_seqs[i][1:-1]
                cur_smiles = ''.join([idx2selfies[j] for j in tmp])
                can_smi = sf.decoder(cur_smiles)
                try:
                    can_smi = sf.decoder(cur_smiles)
                    if len(can_smi) > 1:
                        real_molecules.append(can_smi)
                except:
                    pass
        return real_molecules

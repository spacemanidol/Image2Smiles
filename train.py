
#Logging Of Model
#import wandb
#wandb.init(project="game-theorectic-pruning")
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu

import os
import argparse
from tqdm import tqdm


from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

from encoders import Resnet101Encoder
from decoders import DecoderWithAttention
from utils import CaptionDataset, clip_gradient, adjust_learning_rate, AverageMeter

def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """
    Saves model checkpoint.
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)

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
    
    # Load Checkpoint if exists
    start_epoch = 0
    if args.load:
        try:
            print("Loading models: {}".format(args.model_path))
            checkpoint = torch.load(args.model_model)
            start_epoch = checkpoint['epoch'] + 1
            best_bleu = checkpoint['bleu']
            decoder = checkpoint['decoder']
            decoder_optimizer = checkpoint['decoder_optimizer']
            encoder = checkpoint['encoder']
            encoder_optimizer = checkpoint['encoder_optimizer']
            encoder.fine_tune(args.fine_tune_encoder)
            if encoder_optimizer is None:
                encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),lr=args.encoder_lr)
            print("Models Loaded")
        except:
            print("Models couldn't be loaded. Aborting")
            exit(0)
    else:
        # Load encoder
        if args.encoder_type == 'RESNET101':
            encoder = Resnet101Encoder()
        else:
            print("No other encoders implemented yet.")
            exit(0)
        encoder.fine_tune(args.fine_tune_encoder)
        encoder_optimizer = None
        if args.fine_tune_encoder:
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=args.encoder_lr)
        # Load Decoder
        if args.decoder_type == 'LSTM+Attention':
            decoder = DecoderWithAttention(attention_dim=args.attention_dim,embed_dim=args.embedding_dim, decoder_dim=args.decoder_dim, encoder_dim=args.encoder_dim, vocab_size=vocab_size, dropout=args.dropout)
        else:
            print("No other decoders implemented yet.")
            exit(0)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=args.decoder_lr)
        print("Models Loaded")
    
    # Deal With CUDA
    if args.cuda:
        device = 'cuda'
        cudnn.benchmark = True
        encoder = torch.nn.DataParallel(encoder)
        decoder = torch.nn.DataParallel(encoder)
        
    else:
        device = 'cpu'
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    print("Loading Datasets")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    train_dataset = CaptionDataset(args.data_dir, 'validation', transform=transforms.Compose([normalize]))
    val_dataset = CaptionDataset(args.data_dir, 'validation', transform=transforms.Compose([normalize]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print("Datasets loaded")


    # Train and validate
    print("Moving on to training")
    epochs_since_improvement = 0
    for epoch in range(start_epoch, args.epochs):
        if epochs_since_improvement == 20:
            print("No improvement in 20 epochs. Ending training")
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)
        decoder.train()  # train mode (dropout and batchnorm is used)
        encoder.train()
        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter()  # loss (per word decoded)
        top5accs = AverageMeter()  # top5 accuracy
        start = time.time()
        # Batches
        for i, (imgs, caps, caplens) in enumerate(train_loader):
            data_time.update(time.time() - start)
            # Move to GPU, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            exit(0)
            # Forward prop.
            imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Back prop.
            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            if grad_clip is not None:
                clip_gradient(decoder_optimizer, grad_clip)
                if encoder_optimizer is not None:
                    clip_gradient(encoder_optimizer, grad_clip)

            # Update weights
            decoder_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

            # Keep track of metrics
            top5 = accuracy(scores, targets, 5)
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                            batch_time=batch_time,
                                                                            data_time=data_time, loss=losses,
                                                                            top5=top5accs))
            exit(0)

        # One epoch's validation
        decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))
        recent_bleu4 = bleu4

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer, recent_bleu4, is_best)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train molecule captioning model')   
    parser.add_argument('--max_length', type=int, default=150, help='Max length of tokenized smiles')
    parser.add_argument('--tokenizer', default='tokenizers/tokenizer_vocab_2000.json', type=str, help='tokenizer name in the folder tokenizers/')
    parser.add_argument('--test_string', type=str, default='CC(C)CCNc1cnnc(NCCc2ccc(S(N)(=O)=O)cc2)n1', help='a SMILES string to test tokenizer with')
    parser.add_argument('--data_dir', default='data/', type=str, help='directory of data to be processed. Expect a labels.smi file and associated images')
    parser.add_argument('--epochs', default=100, type=int, help='Train epochs')
    parser.add_argument('--num_workers', default=4, type=int, help='Workers for data loading')
    parser.add_argument('--batch_size', default=16, type=int, help='Size of sampled batch')
    parser.add_argument('--dropout', default=0.5, type=float, help='Rate of dropout')
    parser.add_argument('--embedding_dim', default=512, type=int, help='embedding dimension')
    parser.add_argument('--encoder_type', default='RESNET101', type=str, help='Type of encoder architecture', choices=['RESNET101'])
    parser.add_argument('--decoder_type', default='LSTM+Attention', type=str, help='Type of decoder architecture', choices=['LSTM+Attention'])
    parser.add_argument('--decoder_dim', default=2048, type=int, help='dimension of decoder')
    parser.add_argument('--encoder_dim', default=2048, type=int, help='dimension of encoder')
    parser.add_argument('--attention_dim', default=2048, type=int, help='attention dim')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--model_path', default='models/', type=str, help='model path')
    parser.add_argument('--load', action='store_true', help='load existing model')
    parser.add_argument('--encoder_lr', default=1e-4, type=float, help='encoder learning rate if fine tuning')
    parser.add_argument('--decoder_lr', default=4e-4, type=float, help='decoder learning rate')
    parser.add_argument('--gradient_clip', default=5.0, type=float, help='clip gradients at an abosulte value of')
    parser.add_argument('--prune', action='store_true', help='prune network')
    parser.add_argument('--fine_tune_encoder', action='store_true', help='fine tune encoder')
    args = parser.parse_args()
    main(args)
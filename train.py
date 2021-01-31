
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


from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

from encoders import Resnet101Encoder
from decoders import DecoderWithAttention
from utils import CaptionDataset, clip_gradient, adjust_learning_rate, AverageMeter, accuracy

def save_checkpoint(model_path, epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, bleu4, is_best):
    print("Saving model")
    state = {'epoch': epoch,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = model_path + 'checkpoint_' + str(epoch)
    torch.save(state, filename)
    if is_best:
        torch.save(state, filename+ '_BEST')

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
            checkpoint = torch.load(args.model_path)
            start_epoch = checkpoint['epoch'] + 1
            best_bleu = checkpoint['bleu-4']
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
        device = 'cuda:0'
        cudnn.benchmark = True
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        #encoder = torch.nn.DataParallel(encoder)
        #decoder = torch.nn.DataParallel(decoder)
    else:
        device = 'cpu'
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Custom dataloaders
    print("Loading Datasets")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    train_dataset = CaptionDataset(args.data_dir, 'training', transform=transforms.Compose([normalize]))
    print("There are {} training data examples".format(len(train_dataset)))
    val_dataset = CaptionDataset(args.data_dir, 'validation', transform=transforms.Compose([normalize]))
    print("There are {} validation data examples".format(len(val_dataset)))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print("Datasets loaded")
    
    epochs_since_improvement = 0
    best_bleu4 = 0
    print("Validating Model")
    best_bleu4 = validate(encoder, decoder, val_loader, decoder_optimizer, encoder_optimizer, device,  criterion, tokenizer)
    print("BLEU Score: {}".format(best_bleu4))
    # Train and validate
    print("Moving on to training")
    
    for epoch in range(start_epoch, args.epochs):
        print("Starting epoch {}".format(epoch))
        if epochs_since_improvement == 20:
            print("No improvement in 20 epochs. Ending training")
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)
        train(encoder, decoder, train_loader, decoder_optimizer, encoder_optimizer, device, criterion)
        cur_bleu4 = validate(encoder, decoder, val_loader, decoder_optimizer, encoder_optimizer, device,  criterion, tokenizer)
        is_best= False
        epochs_since_improvement += 1
        if cur_bleu4 > best_bleu4:
            is_best = True
            best_bleu4 = cur_bleu4
            epochs_since_improvement = 0 
        save_checkpoint(args.model_path, epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, cur_bleu4, is_best)

def train(encoder, decoder, loader, decoder_optimizer, encoder_optimizer, device, criterion):
    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy
    i = 0
    for data in tqdm(loader):
        imgs = data[0]
        caps = data[1]
        caplens = data[2]
        # Forward pass
        imgs = encoder(imgs.to(device)).to(device)
        caps = caps.to(imgs.device)
        caplens = caplens.to(imgs.device)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
        #print(decode_lengths)
        targets = caps_sorted[:, 1:] #remove <start> and <end> tokens
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).to(device) #remove padding tokens
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).to(device)
        # Calculate loss
        loss = criterion(scores.data, targets.data).to(imgs.device)
        # Add doubly stochastic attention regularization
        loss += args.alphac * ((1. - alphas.sum(dim=1).to(device)) ** 2).mean()
        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        if args.gradient_clip is not None:
            clip_gradient(decoder_optimizer, args.gradient_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, args.gradient_clip)
        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()
        # Keep track of metrics
        top5 = accuracy(scores.data, targets.data, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        # Print status
        if i % args.print_freq == 0:
            print('Loss {loss.val:.4f} ({loss.avg:.4f})\t''Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(loss=losses,top5=top5accs))
        i += 1
                
def validate(encoder, decoder, loader, decoder_optimizer, encoder_optimizer, device, criterion, tokenizer):
    decoder.eval()
    encoder.eval() 
    losses = AverageMeter()
    top5accs = AverageMeter()
    references, candidates = list(), list()
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
    parser = argparse.ArgumentParser(description='Train molecule captioning model')   
    parser.add_argument('--max_length', type=int, default=150, help='Max length of tokenized smiles')
    parser.add_argument('--tokenizer', default='tokenizers/tokenizer_vocab_2000.json', type=str, help='tokenizer name in the folder tokenizers/')
    parser.add_argument('--test_string', type=str, default='CC(C)CCNc1cnnc(NCCc2ccc(S(N)(=O)=O)cc2)n1', help='a SMILES string to test tokenizer with')
    parser.add_argument('--data_dir', default='data/', type=str, help='directory of data to be processed. Expect a labels.smi file and associated images')
    parser.add_argument('--epochs', default=10, type=int, help='Train epochs')
    parser.add_argument('--num_workers', default=8, type=int, help='Workers for data loading')
    parser.add_argument('--batch_size', default=32, type=int, help='Size of sampled batch')
    parser.add_argument('--dropout', default=0.5, type=float, help='Rate of dropout')
    parser.add_argument('--embedding_dim', default=512, type=int, help='embedding dimension')
    parser.add_argument('--encoder_type', default='RESNET101', type=str, help='Type of encoder architecture', choices=['RESNET101'])
    parser.add_argument('--decoder_type', default='LSTM+Attention', type=str, help='Type of decoder architecture', choices=['LSTM+Attention'])
    parser.add_argument('--decoder_dim', default=512, type=int, help='dimension of decoder')
    parser.add_argument('--encoder_dim', default=2048, type=int, help='dimension of encoder')
    parser.add_argument('--attention_dim', default=512, type=int, help='attention dim')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--model_path', default='models/', type=str, help='model path')
    parser.add_argument('--load', action='store_true', help='load existing model')
    parser.add_argument('--encoder_lr', default=1e-4, type=float, help='encoder learning rate if fine tuning')
    parser.add_argument('--decoder_lr', default=4e-4, type=float, help='decoder learning rate')
    parser.add_argument('--gradient_clip', default=5.0, type=float, help='clip gradients at an abosulte value of')
    parser.add_argument('--alphac', default=1, type=int, help="regularization param")
    parser.add_argument('--prune', action='store_true', help='prune network')
    parser.add_argument('--fine_tune_encoder', action='store_true', help='fine tune encoder')
    parser.add_argument('--print_freq', default=5000, type=int, help="print loss and top5 acc every n batches")
    args = parser.parse_args()
    main(args)
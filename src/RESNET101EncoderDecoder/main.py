import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision as tv

from utils import NestedTensor, nested_tensor_from_tensor_list, set_seed, load_selfies, save_model, create_caption_and_mask, under_max
from dataset import MoleculeCaption
from caption import Caption

from PIL import Image
import argparse
import numpy as np
import math
import time
import sys
import os
import tqdm

import wandb
import selfies as sf

def train(args, model, criterion, data_loader,optimizer, device, epoch, max_norm, scheduler, data_loader_eval):
    model.train()
    criterion.train()
    epoch_loss = 0.0
    total = len(data_loader)
    i = 0
    with tqdm.tqdm(total=total) as pbar:
        for images, masks, caps, cap_masks in data_loader:
            samples = NestedTensor(images, masks).to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)
            outputs = model(samples, caps[:, :-1], cap_masks[:, :-1])
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])
            loss_value = loss.item()
            if i % args.log_interval == 0:
                wandb.log({"loss":loss_value})
            epoch_loss += loss_value
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            pbar.update(1)
            if i % args.scheduler_updates == 0:
                scheduler.step()
            if i % args.eval_interval == 0 and args.do_eval:
                eval_loss = evaluate(model, criterion, data_loader_eval, device)
                print("Eval loss after {} batches is {}".format(i, eval_loss))
            i += 1
    return epoch_loss / total, scheduler
    
@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()
    validation_loss = 0.0
    total = len(data_loader)
    with tqdm.tqdm(total=total) as pbar:
        for images, masks, caps, cap_masks in data_loader:
            samples = NestedTensor(images, masks).to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)
            outputs = model(samples, caps[:, :-1], cap_masks[:, :-1])
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])
            validation_loss += loss.item()
            pbar.update(1)
    wandb.log({"eval_loss":validation_loss / total})
    return validation_loss / total

@torch.no_grad()
def predict_image(args, model, device, idx2selfies, selfies2idx):
    model.eval()
    val_transform = tv.transforms.Compose([
    tv.transforms.Lambda(under_max),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    start_token_id = selfies2idx['[start]']
    end_token_id = selfies2idx['[end]']
    pad_token_id = selfies2idx['[pad]']
    exclude = set()
    exclude.add(start_token_id)
    exclude.add(end_token_id)
    exclude.add(pad_token_id)
    vocab_size = len(selfies2idx)
    image = Image.open(args.image_path)
    image = val_transform(image)
    image = image.unsqueeze(0).to(device)
    caption, cap_mask = create_caption_and_mask(start_token_id, args.max_length-1)
    cap_mask = cap_mask.to(device)
    caption = caption.to(device)
    for i in range(args.max_length - 2):
        predictions = model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)
        if predicted_id[0] == end_token_id: # aka 190
            break
        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False
    cap_len = 0
    for i in cap_mask[0]:
        if i == torch.ones(1).type(torch.bool)[0]:
            break
        else:
            cap_len += 1
    caption = caption[0][:cap_len].tolist()
    selfies_caption = ''.join([idx2selfies[i] for i in caption if i not in exclude ])
    print("{}\t{}".format(sf.decoder(selfies_caption), args.image_path))

@torch.no_grad()
def predict_images(args, image_paths, model, device, idx2selfies, selfies2idx):
    model.eval()
    val_transform = tv.transforms.Compose([
    tv.transforms.Lambda(under_max),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    start_token_id = selfies2idx['[start]']
    end_token_id = selfies2idx['[end]']
    pad_token_id = selfies2idx['[pad]']
    exclude = set()
    exclude.add(start_token_id)
    exclude.add(end_token_id)
    exclude.add(pad_token_id)
    vocab_size = len(selfies2idx)
    with open(args.output_file, 'w') as w:
        for image_path in image_paths:
            image = Image.open(os.path.join(args.image_dir,image_path))
            image = val_transform(image)
            image = image.unsqueeze(0).to(device)
            caption, cap_mask = create_caption_and_mask(start_token_id, args.max_length-1)
            cap_mask = cap_mask.to(device)
            caption = caption.to(device)
            for i in range(args.max_length - 2):
                predictions = model(image, caption, cap_mask)
                predictions = predictions[:, i, :]
                predicted_id = torch.argmax(predictions, axis=-1)
                if predicted_id[0] == end_token_id: # aka 190
                    break
                caption[:, i+1] = predicted_id[0]
                cap_mask[:, i+1] = False
            cap_len = 0
            for i in cap_mask[0]:
                if i == torch.ones(1).type(torch.bool)[0]:
                    break
                else:
                    cap_len += 1
            caption = caption[0][:cap_len].tolist()
            selfies_caption = ''.join([idx2selfies[i] for i in caption if i not in exclude ])
            w.write("{}\t{}\n".format(sf.decoder(selfies_caption), image_path))

def load_image(filename):
    image_paths = []
    with open(filename,'r') as f:
        for l in f:
            l = l.strip().split('\t')[1] #remove if not known file
            image_paths.append(l)
    return image_paths

def main(args):
    print("Loading selfies")
    idx2selfies, selfies2idx = load_selfies(args.selfies_vocab)
    print("Selfies loaded.\nVocab size {}".format(len(idx2selfies)))
    print("Loading Model")
    if args.cuda:
        device = "cuda"
    else:
        device = "cpu"    
    device = torch.device(device)
    start_epoch =  1
    model = Caption(device)
    if args.cuda:
        pass #model = nn.DataParallel(model)
    param_dicts = [{"params": [p for n, p in model.named_parameters()],"lr": args.lr,},]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size)
    if args.load_model:
        checkpoint = torch.load(args.model_path)
        model = checkpoint
        #model = model.load_state_dict(checkpoint['model'])
        #optimizer = optimizer.load_state_dict(checkpoint['optimizer'])
        #scheduler = scheduler.load_state_dict(checkpoint['scheduler'])
        #start_epoch = checkpoint['epoch'] + 1
    
    print("Model has {} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    set_seed(args.seed)

    if args.do_train:
        print("Loading training data")
        wandb.init(project="MoleculeCaptioning")
        trainingData =  MoleculeCaption(args.train_data_dir, args.max_length)   #MoleculeCaption(args.training_data_dir, args.max_length)
        sampler = torch.utils.data.RandomSampler(trainingData)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler, args.batch_size, drop_last=True)
        data_loader_train = DataLoader(trainingData, batch_sampler=batch_sampler_train, num_workers=args.num_workers)
        print("Training data loaded successfully it has {} samples".format(len(trainingData)))
        
    if args.do_eval:
        print("Loading eval data")
        evalData = MoleculeCaption(args.eval_data_dir, args.max_length, args.eval_data_size)   
        sampler = torch.utils.data.SequentialSampler(evalData)
        data_loader_eval = DataLoader(evalData, args.batch_size, sampler=sampler, drop_last=False, num_workers=args.num_workers)
        print("Eval data loaded successfully it has {} samples".format(len(evalData)))

    if args.do_train:
        for epoch in range(start_epoch, args.epochs):
            if args.do_eval:
                eval_loss = evaluate(model, criterion, data_loader_eval, device)
                print("Eval Loss at epoch {}:{}".format(epoch, eval_loss))
            save_model(model, optimizer, scheduler, epoch, args)
            print("Starting Training for epoch :{}".format(epoch))
            train_loss = train(args, model, criterion, data_loader_train,optimizer, device, epoch, args.clip_max_norm, scheduler, data_loader_eval)
            print("Epoch Loss:{}".format(train_loss))
            save_model(model, optimizer, scheduler, epoch, args)

    if args.do_eval:
        eval_loss = evaluate(model, criterion, data_loader_eval, device)
        save_model(model, optimizer, scheduler, 0, args)
        print("Eval Loss:{}".format(eval_loss))

    if args.do_predict:
        if args.predict_list != 'None':
            image_paths = load_image(args.predict_list)
            print("Predicting smiles for {} images".format(len(image_paths)))
            predict_images(args,image_paths, model, device, idx2selfies, selfies2idx)
        else:
            predict_image(args,model, device, idx2selfies, selfies2idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Molecule Captioning via RESNET + ENCODER DECODER')  
    parser.add_argument('--num_workers', default=8, type=int, help='Workers for data loading')
    parser.add_argument('--output_file', default='predictions.tsv', type=str, help='file name for predictions')
    parser.add_argument('--predict_list', default='None', type=str, help='Path of a file for images to predict')
    parser.add_argument('--train_data_dir', default='data/training_images', type=str, help='Folder where training images are located')
    parser.add_argument('--eval_data_dir', default='data/validation_images', type=str, help='Folder where validation images are located')
    parser.add_argument('--seed', default=42, type=int, help='seed value')
    parser.add_argument('--image_dir', default='data/uspto_images/', type=str, help='Location of images to evaluate')
    parser.add_argument('--batch_size', default=32, type=int, help='Size of sampled batch')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--max_length', type=int, default=150, help='Max length of tokenized smiles')
    parser.add_argument('--epochs', default=10, type=int, help='Train epochs')
    parser.add_argument('--image_path', default='molecule.png', type=str, help='Predict SMI of molecule expected SMI is Cc1nc(CN(C)c2ncc(C(=O)[O-])s2)n[nH]1')
    parser.add_argument('--do_train', action='store_true', help='Train model')
    parser.add_argument('--do_eval', action='store_true', help='Eval model')
    parser.add_argument('--do_predict', action='store_true', help='Predict')
    parser.add_argument('--load_model', action='store_true', help='Load model')
    parser.add_argument('--model_path', default='model', type=str, help='model path')
    parser.add_argument('--eval_data_size', type=int, default=4096 ,help='How much of eval to run')
    parser.add_argument('--selfies_vocab', default='data/selfies.vocab', type=str, help='vocab mapping for selfies')
    parser.add_argument('--lr', default=1e-5, type=float, help='decoder learning rate')
    parser.add_argument('--lr_step_size', default=30, type=int, help='Step size for lr decay')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Learning rate weight decay')
    parser.add_argument('--dropout', default=0.1, type=float, help='Rate of dropout')
    parser.add_argument('--clip_max_norm', default=0.1)
    parser.add_argument('--log_interval', default=50, type=int, help='log batch loss every n batches')
    parser.add_argument('--scheduler_updates', default=2000, type=int, help='Update scheduler after how many batches')
    parser.add_argument('--eval_interval', default=5000, type=int, help='Evaluate model on validation every n batches')
    args = parser.parse_args()
    main(args)

    

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import NestedTensor, nested_tensor_from_tensor_list, set_seed, load_selfies, save_model, create_caption_and_mask
from dataset import MoleculeCaption

from PIL import Image
import argparse
import numpy as np
import math
import time
import sys
import os
import tqdm

import wandb

def train(model, criterion, data_loader,optimizer, device, epoch, max_norm):
    model.train()
    criterion.train()
    epoch_loss = 0.0
    total = len(data_loader)
    with tqdm.tqdm(total=total) as pbar:
        for images, masks, caps, cap_masks in data_loader:
            samples = utils.NestedTensor(images, masks).to(device)
            captions = caps.to(device)
            caption_masks = cap_masks.to(device)
            outputs = model(samples, caps[:, :-1], cap_masks[:, :-1])
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])
            loss_value = loss.item()
            epoch_loss += loss_value
            if not math.isfinite(loss_value):
                print(f'Loss is {loss_value}, stopping training')
                sys.exit(1)
            optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            pbar.update(1)
    return epoch_loss / total
    
@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()
    validation_loss = 0.0
    total = len(data_loader)
    with tqdm.tqdm(total=total) as pbar:
        for images, masks, caps, cap_masks in data_loader:
            samples = utils.NestedTensor(images, masks).to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)
            outputs = model(samples, caps[:, :-1], cap_masks[:, :-1])
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])
            validation_loss += loss.item()
            pbar.update(1)
    return validation_loss / total

@torch.no_grad()
def predict(model, image_path, device):
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    start_token_id = selfies2idx['[start]']
    end_token_id = selfies2idx['[end]']
    pad_token_id = selfies2idx['[pad]']
    vocab_size = len(selfies2idx)
    start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
    end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
    image = Image.open(image_path)
    image = coco.val_transform(image)
    image = image.unsqueeze(0)
    caption, cap_mask = create_caption_and_mask(start_token, config.max_position_embeddings)
    output = evaluate()
    result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    print(result.capitalize())
    for i in range(config.max_position_embeddings - 1):
        predictions = model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)
        if predicted_id[0] == 102:
            return caption
        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False
    return caption

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
    set_seed(args.seed)

    if args.do_train:
        print("Loading training data")
        trainingData = MoleculeCaption(args.training_data_dir, args.max_length)
        sampler = torch.utils.data.RandomSampler(trainingData)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler, args.batch_size, drop_last=True)
        data_loader_train = DataLoader(trainingData, batch_sampler=batch_sampler_train, num_workers=args.num_workers)
        print("Training data loaded successfully it has {} samples".format(len(trainingData)))
    if args.do_eval:
        print("Loading eval data")
        evalData = MoleculeCaption(args.eval_data_dir, args.max_length)   
        sampler = torch.utils.data.SequentialSampler(evalData)
        data_loader_eval = DataLoader(evalData, args.batch_size, sampler=sampler, drop_last=False, num_workers=args.num_workers)
        print("Eval data loaded successfully it has {} samples".format(len(evalData)))
    
    for image, mask, caption, caption_mask, caption_lengh in data_loader_eval:
        print(image)
        print(mask)
        print(caption)
        print(caption_mask)
        print(caption_lengh)
        break
    
    """

    model, criterion = caption.build_model(config)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

    param_dicts = [
        {"params": [p for n, p in model.named_parameters(
        ) if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": config.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)

    if os.path.exists(config.checkpoint):
        print("Loading Checkpoint...")
        checkpoint = torch.load(config.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.start_epoch = checkpoint['epoch'] + 1

    print("Start Training..")
    for epoch in range(config.start_epoch, config.epochs):
        print(f"Epoch: {epoch}")
        epoch_loss = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, config.clip_max_norm)
        lr_scheduler.step()
        print(f"Training Loss: {epoch_loss}")


        validation_loss = evaluate(model, criterion, data_loader_val, device)
        print(f"Validation Loss: {validation_loss}")

        print()
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Molecule Captioning via RESNET + ENCODER DECODER')  
    parser.add_argument('--num_workers', default=8, type=int, help='Workers for data loading')
    parser.add_argument('--training_data_dir', default='data/training_images', type=str, help='Folder where training images are located')
    parser.add_argument('--eval_data_dir', default='data/validation_images', type=str, help='Folder where validation images are located')
    parser.add_argument('--seed', default=42, type=int, help='seed value')
    parser.add_argument('--batch_size', default=32, type=int, help='Size of sampled batch')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--max_length', type=int, default=150, help='Max length of tokenized smiles')
    parser.add_argument('--epochs', default=1, type=int, help='Train epochs')
    parser.add_argument('--do_train', action='store_true', help='Train model')
    parser.add_argument('--do_eval', action='store_true', help='Eval model')
    parser.add_argument('--do_predict', action='store_true', help='Predict')
    parser.add_argument('--selfies_vocab', default='data/selfies.vocab', type=str, help='vocab mapping for selfies')
    parser.add_argument('--encoder_lr', default=1e-5, type=float, help='encoder learning rate')
    parser.add_argument('--decoder_lr', default=4e-4, type=float, help='decoder learning rate')
    parser.add_argument('--lr_step_size', default=30, type=int, help='Step size for lr decay')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Learning rate weight decay')
    parser.add_argument('--dropout', default=0.1, type=float, help='Rate of dropout')
    parser.add_argument('--clip_max_norm', default=0.1)
    args = parser.parse_args()
    main(args)
"""
        # Transformer
        self.hidden_dim = 256
        self.pad_token_id = 0
        self.max_position_embeddings = 128
        self.layer_norm_eps = 1e-12
        self.dropout = 0.1
        self.vocab_size = 30522

        self.enc_layers = 6
        self.dec_layers = 6
        self.dim_feedforward = 2048
        self.nheads = 8
        self.pre_norm = True

        # Dataset
        self.dir = '../coco'
        self.limit = -1
        """
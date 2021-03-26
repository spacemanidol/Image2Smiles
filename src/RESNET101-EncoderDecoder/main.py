import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import NestedTensor, nested_tensor_from_tensor_list, set_seed, load_selfies, save_model
from encoder import Encoder
from decoder import Decoder

from PIL import Image
import argparse
import numpy as np
import math
import time
import sys
import os
import tqdm

def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)
    caption_template[:, 0] = start_token
    mask_template[:, 0] = False
    return caption_template, mask_template

def train_one_epoch(model, criterion, data_loader,optimizer, device, epoch, max_norm):
    model.train()
    criterion.train()
    epoch_loss = 0.0
    total = len(data_loader)
    with tqdm.tqdm(total=total) as pbar:
        for images, masks, caps, cap_masks in data_loader:
            samples = utils.NestedTensor(images, masks).to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)
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
def predict(model, image, device):
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    config = Config()

    start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
    end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

    image = Image.open(image_path)
    image = coco.val_transform(image)
    image = image.unsqueeze(0)
    caption, cap_mask = create_caption_and_mask(start_token, config.max_position_embeddings)
    output = evaluate()
    result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    #result = tokenizer.decode(output[0], skip_special_tokens=True)
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

class Caption(nn.Module):
    def __init__(self, backbone, transformer, hidden_dim, vocab_size):
        super().__init__()
        self.backbone = backbone
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.transformer = transformer
        self.mlp = MLP(hidden_dim, 512, vocab_size, 3)

    def forward(self, samples, target, target_mask):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()

        assert mask is not None

        hs = self.transformer(self.input_proj(src), mask,
                              pos[-1], target, target_mask)
        out = self.mlp(hs.permute(1, 0, 2))
        return out


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

    backbone = build_backbone(config)
    transformer = build_transformer(config)
    model = Caption(backbone, transformer, config.hidden_dim, config.vocab_size)
    criterion = torch.nn.CrossEntropyLoss()  
def main(args):
    print("Loading selfies")
    idx2selfies, selfies2idx = load_selfies(args.selfies_vocab)
    print("Selfies loaded.\n Vocab size {}".format(idx2selfies))

    print("Loading Model")
    if args.cuda:
        device = "cuda"
    else:
        device = "cpu"    
    device = torch.device(args.device)
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
        sampler = torch.utils.data.SequentialSampler(validationData)
        data_loader_eval = DataLoader(evalData, args.batch_size, sampler=sampler, drop_last=False, num_workers=args.num_workers)
    
    for image, mask, caption, caption_mask, caption_lengh in data_loader_train:
        print(image)
        print(mask)
        print(caption)
        print(caption_mask)
        print(caption_length)
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

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
        }, config.checkpoint)

        validation_loss = evaluate(model, criterion, data_loader_val, device)
        print(f"Validation Loss: {validation_loss}")

        print()
"""
if __name__ == "__main__":
    if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Molecule Captioning via RESNET + ENCODER DECODER')  
    parser.add_argument('--num_workers', default=8, type=int, help='Workers for data loading')
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
    parser.add_argument('--epochs', default=1, type=int, help='Train epochs')
    parser.add_argument('--lr_step_size', default=30, type=int, help='Step size for lr decay')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Learning rate weight decay')



    parser.add_argument('--max_length', type=int, default=150, help='Max length of tokenized smiles')
    parser.add_argument('--data_dir', default='data/', type=str, help='directory of data to be processed. Expect a labels.smi file and associated images')
    parser.add_argument('--epochs', default=10, type=int, help='Train epochs')
    
    parser.add_argument('--dropout', default=0.5, type=float, help='Rate of dropout')
    parser.add_argument('--embedding_dim', default=512, type=int, help='embedding dimension')
    parser.add_argument('--encoder_type', default='RESNET101', type=str, help='Type of encoder architecture', choices=['RESNET101'])
    parser.add_argument('--decoder_type', default='LSTM+Attention', type=str, help='Type of decoder architecture', choices=['LSTM+Attention'])
    parser.add_argument('--decoder_dim', default=512, type=int, help='dimension of decoder')
    parser.add_argument('--encoder_dim', default=2048, type=int, help='dimension of encoder')
    parser.add_argument('--attention_dim', default=512, type=int, help='attention dim')
    parser.add_argument('--model_path', default='models/', type=str, help='model path')
    parser.add_argument('--load', action='store_true', help='load existing model')
    parser.add_argument('--encoder_lr', default=4e-4, type=float, help='encoder learning rate if fine tuning')
    parser.add_argument('--decoder_lr', default=4e-4, type=float, help='decoder learning rate')
    parser.add_argument('--lr_update_freq', default=1000, type=int, help='How often to decrease lr')
    parser.add_argument('--decay_rate', default=.9, type=float, help='how much to update LR by')
    parser.add_argument('--gradient_clip', default=5.0, type=float, help='clip gradients at an abosulte value of')
    parser.add_argument('--alphac', default=1, type=int, help="regularization param")
    parser.add_argument('--prune', action='store_true', help='prune network')
    parser.add_argument('--fine_tune_encoder', action='store_true', help='fine tune encoder')
    parser.add_argument('--print_freq', default=5000, type=int, help="print loss and top5 acc every n batches")
    parser.add_argument('--seed', default=42, type=int, help='Set random seed')
    parser.add_argument('--cuda_device', default='cuda:0', type=str, help='cuda device to use. aka gpu')
    parser.add_argument('--checkpoint_freq', default=1000, type=int, help='how often to checkpoint model')
    parser.add_argument('--do_eval', action='store_true', help ='validate the model')
    parser.add_argument('--do_train', action='store_true', help='train the model')
    args = parser.parse_args()
    main(args)

    
    

        # Backbone
        self.backbone = 'resnet101'
        self.position_embedding = 'sine'
        self.dilation = True
        
        # Basic
        self.device = 'cuda'
        self.seed = 42
        self.batch_size = 32
        self.num_workers = 8
        self.checkpoint = './checkpoint.pth'
        self.clip_max_norm = 0.1

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
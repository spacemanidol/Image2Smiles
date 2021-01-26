import os
import argparse
from tqdm import tqdm

from encoders import Resnet101Encoder

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

def main(args):
    # Load Tokenizer
    print("Loading Tokenizer: {}.".format(args.tokenizer))
    tokenizer = Tokenizer.from_file(os.path.join('tokenizers',args.tokenizer))
    print("Testing with SMILES String: {}".format(args.test_string))
    encoding = tokenizer.encode(args.test_string)
    print("Encoded string: {}".format(encoding.tokens))
    decoded = tokenizer.decode(encoding.ids)
    print("Decoded string: {}".format(decoded))
    print("Tokenizer Loaded.")

    # Load Model File
    print("Loading models: {}".format(args.model))
    checkpoint = torch.load(args.model)
    # Load encoder
    if args.encoder_type == 'Resnet101Encoder':
        encoder = Resnet101Encoder()
        encoder.load_state_dict(checkpoint['encoder'])
    else:
        print("No other encoders implemented yet.")
        exit(0)

    # Load Decoder
    
    # Deal With CUDA

    if args.cuda:
        device = 'cuda'
        cudnn.benchmark = True
        encoder = torch.nn.DataParallel(encoder)
        
    else:
        device = 'cpu'


    decoder = decoder.to(device)
    encoder = encoder.to(device)
    decoder.eval()
    encoder.eval()

    print("Models Loaded.")

    # Load Image 
    print("Loading Images")
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
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Smiles Given an input image')
    parser.add_argument('--image', type=str, help='path to image')
    parser.add_argument('--predict_whole_directory', action='store_true', help='image path is a directory and predict all molecules.')
    parser.add_argument('--encoder_type', type=str, default='Resnet101Encoder', help='Architecture used for the encoder')
    parser.add_argument('--tokenizer', type=str, help='tokenizer name in the folder tokenizers/')
    parser.add_argument('--model', help='model path')
    parser.add_argument('--test_string', type=str, default='CC(C)CCNc1cnnc(NCCc2ccc(S(N)(=O)=O)cc2)n1', help='a SMILES string to test tokenizer with')
    parser.add_argument('--beam_size', default=5, type=int, help='beam size for prediction creation')
    parser.add_argument('--output_dir', type=str, default='output.txt', help='file name to produce model predictions for each image.')
    parser.add_argument('--cuda', action='store_true', help='Use cuda')
    args = parser.parse_args()
    main(args)    
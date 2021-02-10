import os
import pickle
import numpy as np
import h5py
import json
import torch
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import argparse
from tqdm import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors


def create_input_files(args, dataset_path, config_output_name, output_name, output_path, img_size, dataset_size=4000000):
    '''
    Creates input files for using with models.
    :param dataset_path: the path to the data to be processed
    '''
    data = pickle.load(open(os.path.join(dataset_path, config_output_name),'rb'))
    captions = []
    caption_lengths = []
    if args.process_img:
        print("Processing images")
        processed_image_path = os.path.join(output_path, output_name + '.hdf5')
        if os.path.exists(processed_image_path):
            os.remove(processed_image_path)
        with h5py.File(processed_image_path, 'a') as h:
            images = h.create_dataset('images', (len(data['images']), 3, img_size, img_size), dtype='uint8')
            print('\nReading images and captions, storing to file...\n')
            for i, cur_data in enumerate(tqdm(data['images'])):
                if i > dataset_size:
                    break
                path = os.path.join(cur_data['filepath'], cur_data['filename'])
                img = imread(path)
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (img_size, img_size))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, img_size, img_size)
                assert np.max(img) <= 255
                # Save image to HDF5 file
                images[i] = img
                captions.append(cur_data['sentences'][0]['ids'])
                caption_lengths.append(cur_data['sentences'][0]['length'])
    else:
        print("Just processing captions")
        for i, cur_data in enumerate(data['images']):
            captions.append(cur_data['sentences'][0]['ids'])
            caption_lengths.append(cur_data['sentences'][0]['length'])


    # Save encoded captions and their lengths to JSON files
    with open(os.path.join(output_path, args.output_prefix + 'captions_' + output_name + '.json'), 'w') as j:
        json.dump(captions, j)

    with open(os.path.join(output_path, args.output_prefix + 'captions_length_' + output_name + '.json'), 'w') as j:
        json.dump(caption_lengths, j)

def create_tokenized_smiles_json(tokenizer, data_dir, split, config_output_name, max_length, label_filename):
    data = {"images" : []}
    start_token_id = tokenizer.encode('<start>').ids[0]
    end_token_id = tokenizer.encode('<end>').ids[0]
    pad_token_id = tokenizer.get_vocab_size()
    print("The <start> token id is:{}\nThe <end> token id is:{}\nThe <pad> token id is {}".format(start_token_id, end_token_id, pad_token_id)) 
    with open(os.path.join(data_dir, label_filename), "r") as f:
        for i, l in enumerate(tqdm(f)):
            try:
                smiles, idx = l.strip().split("\t")
                encoding = tokenizer.encode(smiles)
                encodingids = encoding.ids
                encodingids = [start_token_id] + encodingids[:-1] #add <start> token and shorten to 150
                cap_len = max_length
                for j in range(0, len(encodingids)):
                    if encodingids[j] == pad_token_id:
                        cap_len = j+1
                        encodingids[j] = end_token_id
                        break
                current_sample = {"filepath": data_dir, "filename": "{}".format(idx), "imgid": 0, "split": split, "sentences" : [{"tokens": encoding.tokens, "raw": smiles, "ids": encodingids , "length": cap_len}] } # note if image augmentation ever happens need to introduce a sentence id token. see mscoco json for example
                data["images"].append(current_sample)
            except:
                pass
    pickle.dump(data, open(os.path.join(data_dir, config_output_name),'wb'))
    del data
    
def main(args):
    # Load Tokenizer
    print('Loading Tokenizer: {}.'.format(args.tokenizer))
    tokenizer = Tokenizer.from_file(args.tokenizer)
    print('Testing with SMILES String: {}'.format(args.test_string))
    encoding = tokenizer.encode(args.test_string)
    print('Encoded string: {}'.format(encoding.tokens))
    decoded = tokenizer.decode(encoding.ids)
    print('Decoded string: {}'.format(decoded))
    print('Tokenizer Loaded.')

    # Create tokenized captions
    print("Creating JSON")
    create_tokenized_smiles_json(tokenizer, args.data_dir, args.data_split, args.config_output_name, args.max_length, args.label_filename)
    print("JSON created")

    # Save Images and processed Captions
    print("Processing and Saving Images")
    create_input_files(args, args.data_dir, args.config_output_name, args.image_output_filename, args.output_path, args.img_size)
    print("Done processing dataset")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess training data')
    parser.add_argument('--max_length', type=int, default=150, help='Max length of tokenized smiles')
    parser.add_argument('--img_size', type=int, default=256, help='Size of image X and Y dimensions')
    parser.add_argument('--label_filename', type=str, default='labels.smi', help='name of labels file in case dataset is enourmous')
    parser.add_argument('--tokenizer', default='tokenizers/tokenizer_vocab_2000.json', type=str, help='tokenizer name in the folder tokenizers/')
    parser.add_argument('--test_string', type=str, default='CC(C)CCNc1cnnc(NCCc2ccc(S(N)(=O)=O)cc2)n1', help='a SMILES string to test tokenizer with')
    parser.add_argument('--data_dir', default='data/validation_images', type=str, help='directory of data to be processed. Expect a labels.smi file and associated images')
    parser.add_argument('--data_split', default='validation', type=str, help='name of the portion of data being processed. Typical names are training, validation, and evaluation.')
    parser.add_argument('--config_output_name', default='dataset_img2smi.pkl', type=str, help='name of json file to store processable metadata')
    parser.add_argument('--image_output_filename', default='validation', type=str, help='prefix for output image, caption, and caption length files.')
    parser.add_argument('--output_path', default='data/', type=str, help='output folder path.')
    parser.add_argument('--output_prefix', default='vocab2000', type=str, help='name for output captions files based ')
    parser.add_argument('--process_img', action='store_true', default=False, help='create image files')
    args = parser.parse_args()
    main(args)

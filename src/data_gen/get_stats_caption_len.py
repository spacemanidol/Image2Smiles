import os
import numpy as np
from tqdm import tqdm
import argparse
import selfies as sf
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

def get_tokenizers(directory):
    tokenizers = {}
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            tokenizers[filename] = Tokenizer.from_file(os.path.join(directory, filename))
        else:
            continue
    return tokenizers

def get_smiles_length(input_file, tokenizer):
    molecule_length = {}
    end_token_id = tokenizer.encode('<end>').ids[0]
    with open(input_file,'r') as f:
        for l in f:
            l = l.strip()
            encoding = tokenizer.encode(args.test_string)
            cur_molecule_length = 0
            for i in range(0,len(encoding.ids)):
                if encoding.ids[i] == end_token_id:
                    break
                cur_molecule_length += 1
            cur_molecule_length = len(list(sf.split_selfies(l)))

def get_selfies_length(input_file, remove_null):
    molecule_length = []
    with open(input_file,'r') as f:
        for l in f:
            l = l.strip()
            cur_molecule_length = len(list(sf.split_selfies(l)))
            if remove_null == True:
                if cur_molecule_length > 0:
                    molecule_length.append(cur_molecule_length)
            else:
                molecule_length.append(cur_molecule_length)
    print("######################################")
    print("The tokenized captions lengths using selfies for {}.\n The statistics are as follows:Average:{}, Min:{}, Max:{}, Median{}, 5% percentile {}, 95% percentile {}, 50% percentile {}.".format(
        input_file, 
        np.average(molecule_length), 
        np.min(molecule_length), 
        np.max(molecule_length), 
        np.median(molecule_length), 
        np.percentile(molecule_length, 0.05), 
        np.percentile(molecule_length, 0.95), 
        np.percentile(molecule_length, 0.5)))
    print("######################################")

  
def main(args):
    # Load tokenizers
    tokenizers = get_tokenizers(args.tokenizer_directory)
    for tokenizer in tokenizers:
        print("Evaluation on tokenizer {}".format(tokenizer))
        get_tokenized_smiles(args.smiles_input_file, tokenizer)

    # Do Selfies
    get_selfies_length(args.selfies_input_file, 1)
    get_selfies_length(args.selfies_input_file, 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate captions statistics based on tokenization methods')
    parser.add_argument('--selfies_input_file', type=str, default='data/training.selfies', help='source of input selfies files')
    parser.add_argument('--smiles_input_file', type=str, default='data/training.smi', help='source of input smiles files')
    parser.add_argument('--tokenizer_director', type=str, default='tokenizers', help='directory of tokenizers')
    args = parser.parse_args()
    main(args)



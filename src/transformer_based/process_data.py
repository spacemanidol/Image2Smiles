import argparse
import selfies as sf
import json

from utils import load_selfies

def create_annotations(input_file, output_file, idx2selfies, selfies2idx, max_length):
    with open(input_file, 'r) as f:
        for l in f:
            l = l.strip().split('\t')
            if len(l) > 1:
                smiles = l[0]
                img = l[1]
            try:
                selfies = [i + ']' for i in sf.encoder(l[0]).split(']')[:-1]]
                selfies = [i + ']' for i in s]
            except:
                pass

def main(args):
    idx2selfies, selfies2idx = load_selfies(args.selfies_vocab)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create dataset for captioning system')
    parser.add_argument('--input_file', type=str, default='data/', help='source of input smiles data')
    parser.add_argument('--output_file)
    parser.add_argument('--pad_len', type=int, default=150, help='how much to pad tokenized input')
    parser.add_argument('--do_train', action='store_true', help='Train a tokenizer' )
    parser.add_argument('--do_test', action='store_true', help='Test the tokenizer found in tokenizer dir file')
    parser.add_argument('--test_string', type=str, default='CC(C)CCNc1cnnc(NCCc2ccc(S(N)(=O)=O)cc2)n1', help='a SMILES string to test tokenizer with')
    parser.add_argument('--tokenizer_name', type=str, default='tokenizer_vocab_2000.json')
    parser.add_argument('--vocab_size', type=int, default=2000, help='Size of the vocab to rain')
    parser.add_argument('--min_frequency', type=int, default=2, help='min fequency of word in corpus')
    args = parser.parse_args()
    main(args)

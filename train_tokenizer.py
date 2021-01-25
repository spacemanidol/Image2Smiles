import os
import argparse
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

def get_smi_files(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".smi"):
            files.append(os.path.join(directory, filename))
        else:
            continue
    return files

def main(args):
    if args.do_train:
        # Initialize a tokenizer
        files = get_smi_files(args.training_files)
        print("Training BPE tokenizer using the following files:{}".format(files))
        tokenizer = Tokenizer(models.BPE())
        tokenizer.normalizer = Sequence([NFKC()])
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        # Train the tokenizer
        trainer = trainers.BpeTrainer(show_progress=True, vocab_size=args.vocab_size, min_frequency=args.min_frequency)
        tokenizer.train(files, trainer=trainer)
        tokenizer.save(os.path.join('tokenizers',args.tokenizer_name), pretty=True)
        print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))

    if args.do_test:
        # Test the tokenizer
        tokenizer = Tokenizer.from_file(os.path.join('tokenizers',args.tokenizer_name))
        print("Testing with SMILES String: {}".format(args.test_string))
        encoding = tokenizer.encode(args.test_string)
        print("Encoded string: {}".format(encoding.tokens))
        print(encoding.ids)
        decoded = tokenizer.decode(encoding.ids)
        print("Decoded string: {}".format(decoded))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a smiles tokenizer given candidate files and target configs.')
    parser.add_argument('--training_files', type=str, default='data/', help='source of input smiles data')
    parser.add_argument('--do_train', action='store_true', help='Train a tokenizer' )
    parser.add_argument('--do_test', action='store_true', help='Test the tokenizer found in tokenizer dir file')
    parser.add_argument('--test_string', type=str, default='CC(C)CCNc1cnnc(NCCc2ccc(S(N)(=O)=O)cc2)n1', help='a SMILES string to test tokenizer with')
    parser.add_argument('--tokenizer_name', type=str, default='tokenizer_vocab_2000.json')
    parser.add_argument('--vocab_size', type=int, default=2000, help='Size of the vocab to rain')
    parser.add_argument('--min_frequency', type=int, default=2, help='min fequency of word in corpus')
    args = parser.parse_args()
    main(args)

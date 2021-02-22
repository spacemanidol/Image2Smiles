import os
import import argparse
import selfies as sf

dataset = ['[C][O][C]', '[F][C][F]', '[O][=O]', '[C][C][O][C][C]']
alphabet = sf.get_alphabet_from_selfies(dataset)

def load_selfies(directory):
    selfies = []
    for filename in os.listdir(directory):
        if filename.endswith(".selfies"):
            print("Found file {}. Loading samples".format(filename))
            with open(os.path.join(directory, filename), 'r') as f:
                for l in f:
                    l = l.strip()
                    selfies.append(l)
        else:
            continue
    print("All files in directory loaded.\n There are {} molecules.".format(len(selfies)))
    return selfies

def main(args):
    # Load Tokenizer
    print("Loading selfied in directory: {}.".format(args.source_directory))
    selfies = load_selfies(args.directory)
    print("Extracting alphabet from smiles samples")
    print("The longest sample in dataset is {}".format(max(sf.len_selfies(s) for s in selfies)))
    alphabet = sf.get_alphabet_from_selfies(selfies)
    alphabet.add('[start]')  # '[start]'
    alphabet.add('[end]')
    alphabet.add('[pad]')
    alphabet.add('[unk]')
    alphabet = list(alphabet)
    symbol_to_idx = {s: i for i, s in enumerate(alphabet)}
    idx_to_symbol = {i: s for i, s in enumerate(alphabet)}
    with open(args.output_file, 'w') as w:
        for i in range(0, len(idx_to_symbol)):
            w.write("{}\n".format(idx_to_symbol[i]))
    print("Alphabet written")

"""
print(sf.selfies_to_encoding(dimethyl_ether,
                             vocab_stoi=symbol_to_idx,
                             pad_to_len=pad_to_len,
                             enc_type='label'))
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create an alphabet from a set of selfies files.')
    parser.add_argument('--directory', type=str, help='directory containing selfies files.')
    parser.add_argument('--output_file', type=str, help='name of output file where vocabulary will be written.')
    args = parser.parse_args()
    main(args)    

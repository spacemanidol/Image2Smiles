import os
import random
import math
import argparse
import Levenshtein
import numpy as np
from tqdm import tqdm
from scipy.misc import imread, imresize
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, AllChem, rdmolops, Draw
from nltk.translate.bleu_score import sentence_bleu
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
import torch
import torchvision.transforms as transforms

"""
Evaluation Metrics
1. Molecule Found(Binary)
2. Valid Smiles(Binary)
3. Edit Distance(Levenshtein distance)
4. Edit Distance(BLEU1)
5. Molecule Distance via MACCS
6. Molecule Distance via Path
7. Molecule distance via morgan fingerprint
8. Image similarity
"""

def convert_smi_to_mol(smis):
    mols = []
    for smi in smis:
        try:
            mol = Chem.MolFromSmiles(smi) # Generate mole
            can_smi = Chem.MolToSmiles(mol, True) #ensure we generate canonical smile
            mols.append(Chem.MolFromSmiles(can_smi)) #final molecule for evaluation
        except:
            pass
    print("Done Converting SMI to mol")
    return mols

def levenshtein_eval(references):
    """
    Similarity via Levenshtein Distance
    """
    print("Calculating Levenshtein Distance")
    scores = []
    for reference in references:
        cur_scores = []
        for candidate in references:
            if candidate != reference:
                cur_scores.append(Levenshtein.distance(reference, candidate))
        scores.append(np.mean(cur_scores))
    return round(np.mean(scores),4)

def bleu_eval(args, references):
    """
    BLEU-1 via tokenized smile
    """
    print("Loading Tokenizer: {}.".format(args.tokenizer))
    tokenizer = Tokenizer.from_file(args.tokenizer)
    scores = []
    for smi in references:
        cur_scores = []
        for smi2 in references:
            if smi2 != smi:
                reference = tokenizer.encode(smi)
                candidate = tokenizer.encode(smi2)
                cur_scores.append(sentence_bleu(reference.tokens, candidate.tokens,weights=(1.0, 0, 0, 0)))
        scores.append(np.mean(cur_scores))
    return round(np.mean(scores),4)
  
def morgan_fingerprint_evaluation(references):
    """
    Circular based fingerprints
    https://doi.org/10.1021/ci100050t
    """
    scores = []
    for reference in references:
        cur_scores = []
        for candidate in references:
            if reference != candidate:
                candidate_morgan = AllChem.GetMorganFingerprintAsBitVect(candidate, 2, nBits=1024)
                reference_morgan = AllChem.GetMorganFingerprintAsBitVect(reference, 2, nBits=1024)
                cur_scores.append(round(DataStructs.TanimotoSimilarity(reference_morgan,candidate_morgan), 4))
        scores.append(np.mean(cur_scores))
    return round(np.mean(scores),4)

def rd_fingerprint_evaluation(references):
    """
    Enumerate linear Fragement
    """
    scores = []
    for reference in references:
        cur_scores = []
        for candidate in references:
            if reference != candidate:
                candidate_rdkfingerprint = rdmolops.RDKFingerprint(candidate, fpSize=2048, minPath=1, maxPath=7)
                reference_rdkfingerprint = rdmolops.RDKFingerprint(references, fpSize=2048, minPath=1, maxPath=7)
                cur_scores.append(round(DataStructs.TanimotoSimilarity(reference_rdkfingerprint,candidate_rdkfingerprint), 4))
        scores.append(np.mean(cur_scores))
    return round(np.mean(scores),4)

def maacs_fingerprint_evaluation(references):
    """ 
    Generate Similarity via MACCSKeys
    """
    scores = []
    for reference in references:
        cur_scores = []
        for candidate in references:
            if reference_smi != candidate_smi:
                candidate_maccs = MACCSkeys.GenMACCSKeys(candidate)
                reference_maccs = MACCSkeys.GenMACCSKeys(reference)
                cur_scores.append(round(DataStructs.TanimotoSimilarity(reference_maccs,candidate_maccs), 4))
        scores.append(np.mean(cur_scores))
    return round(np.mean(scores),4)

def load_img(args, path, transform):
    """
    Load Image and transform
    """
    img = imread(path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (args.img_size, args.img_size))
    img = img.transpose(2, 0, 1)
    assert img.shape == (3, args.img_size, args.img_size)
    assert np.max(img) <= 255
    img  = torch.FloatTensor(img/255.)
    img = transform(img)
    img = torch.stack([img])
    return img

def img_distance_eval(args, smis, transform):
    """
    Edit distance between two images of generated molecules:
    """
    scores = []
    for reference_smi in smis:
        cur_scores = []
        for candidate_smi in smis:
            if reference_smi != candidate_smi:
                m = Chem.MolFromSmiles(reference_smi)
                Draw.MolToFile(m,"tmp.png", size=(args.img_size,args.img_size))
                reference_img = load_img(args, encoder, "tmp.png", transform)
                m = Chem.MolFromSmiles(candidate_smi)
                Draw.MolToFile(m,"tmp.png", size=(args.img_size,args.img_size))
                candidate_img = load_img(args, encoder, "tmp.png", transform)
                cur_scores.append(math.log(torch.sum(torch.abs(reference_img- candidate_img))))
        scores.append(np.mean(cur_scores))
    return round(np.mean(scores),4)


def load_smi(filename):
    smi = []
    with open(filename, "r") as f:
        for l in f:
            try:
#                mol = Chem.MolFromSmiles(l.strip())
                smi.append(l.strip())
            except:
                pass
    return smi

def main(args):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([normalize])
    smi = load_smi(args.input_file)
    print("loaded smiles")
    random.shuffle(smi)
    references = smi[:args.sample_size]
    print("{} molecule sampled at random".format(args.sample_size))
    mol_references = convert_smi_to_mol(smi)
    print("SMI converted to mol")
    print("Mean Levenshtein:{}".format(levenshtein_eval(references)))
    print("Mean BLEU-1:{}".format(bleu_eval(args, references)))
    print("Mean Morgan Fingergprint:{}".format(morgan_fingerprint_evaluation(mol_references)))
    print("Mean RDkit Fingergprint:{}".format(rd_fingerprint_evaluation(mol_references)))
    print("Mean MAACS Fingergprint:{}".format(maacs_fingerprint_evaluation(mol_references)))
    print("MEAN image distance:{}".format(img_distance_eval(args, references, transform)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Base Metrics for random distribution')
    parser.add_argument('--input_file', type=str, default='data/validation.smi', help='source of input smiles files')
    parser.add_argument('--sample_size', type=int, default=5, help='amount of molecules to compare')
    parser.add_argument('--tokenizer', default='data/tokenizers/tokenizer_vocab_2000.json', type=str, help='tokenizer to use in BLEU evaluation')
    args = parser.parse_args()
    main(args)

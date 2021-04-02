import os
import math
import argparse
import Levenshtein
import numpy as np
from tqdm import tqdm
from PIL import Image
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

def load_img2smi(filename):
    img2smi = {}
    count = 0
    print("Loading file {}".format(filename))
    with open(filename,'r') as f:
        for l in f:
            l = l.strip().split('\t')
            if '.' in l[0]: #No smiles generated so just an extension
                img2smi[l[0]] = 'NONE'
                count += 1
            else:
                if l[0] == '':
                    l[0] = 'NONE'
                img2smi[l[1]] = l[0]
            if l[0] == 'NONE':
                count += 1
            else:
                try: 
                    mol = Chem.MolFromSmiles(smi) # Generate mol
                    img2smi[l[1]] = Chem.MolToSmiles(mol, True) #ensure we generate canonical smile
                except:
                    pass
    print("Done loading file {}".format(filename))
    return img2smi, count

def convert_smi_to_mol(img2smi, filename):
    mols = {}
    count = 0
    print("Converting SMI in {} to mol representation".format(filename))
    for img in img2smi:
        smi = img2smi[img]
        try:
            mol = Chem.MolFromSmiles(smi) # Generate mole
            can_smi = Chem.MolToSmiles(mol, True) #ensure we generate canonical smile
            mols[img] = Chem.MolFromSmiles(can_smi) #final molecule for evaluation
            count += 1 #Made it this far so valid smi
        except:
            pass
    print("Done Converting SMI to mol")
    return mols, count

def levenshtein_eval(references, candidates):
    """
    Similarity via Levenshtein Distance
    """
    print("Calculating Levenshtein Distance")
    distances = []
    for img in references:
        candidate_smi = ''
        if img in candidates:
            candidate_smi = candidates[img]
        distances.append(Levenshtein.distance(references[img], candidate_smi))
    print("Donte Calculating Levenshtein Distance.")
    print("Average Distance:{}".format(round(np.mean(distances),4)))
    return round(np.mean(distances),4)

def bleu_eval(args, references, candidates):
    """
    BLEU-1 via tokenized smile
    """
    print("Loading Tokenizer: {}.".format(args.tokenizer))
    tokenizer = Tokenizer.from_file(args.tokenizer)
    print("Testing with SMILES String: {}".format(args.test_string))
    encoding = tokenizer.encode(args.test_string)
    print("Encoded string: {}".format(encoding.tokens))
    decoded = tokenizer.decode(encoding.ids)
    print("Decoded string: {}".format(decoded))
    print("Tokenizer Loaded.\n Calculating BLEU-1")
    scores = []
    for img in references:
        score = 0
        if img in candidates:
            reference = tokenizer.encode(references[img])
            try:
                candidate = tokenizer.encode(candidates[img])
            except:
                pass
            score = sentence_bleu(reference.tokens, candidate.tokens,weights=(1.0, 0, 0, 0))
        scores.append(score)
    print("Done calculating BLEU-1. {} average score".format(round(np.mean(scores),4)))
    return round(np.mean(scores),4)
    
def morgan_fingerprint_evaluation(references, candidates):
    """
    Circular based fingerprints
    https://doi.org/10.1021/ci100050t
    """
    print("Calculating Similarity via Morgan based Circular Fingerprint")
    similarities = [[],[],[],[],[]] # various similarities: Tanimoto, Dice, Cosine, Sokal, McConnaughey
    for img in references:
        similarity = [0,0,0,0,0]
        if img in candidates:
            morgan_fp_candidate = AllChem.GetMorganFingerprintAsBitVect(candidates[img], 2, nBits=1024)
            morgan_fp_reference = AllChem.GetMorganFingerprintAsBitVect(references[img], 2, nBits=1024)
            similarity[0] = round(DataStructs.TanimotoSimilarity(morgan_fp_reference,morgan_fp_candidate), 4)
            similarity[1] = round(DataStructs.DiceSimilarity(morgan_fp_reference,morgan_fp_candidate), 4)
            similarity[2] = round(DataStructs.CosineSimilarity(morgan_fp_reference,morgan_fp_candidate), 4)
            similarity[3] = round(DataStructs.SokalSimilarity(morgan_fp_reference,morgan_fp_candidate), 4)
            similarity[4] = round(DataStructs.McConnaugheySimilarity(morgan_fp_reference,morgan_fp_candidate), 4)
        similarities[0].append(similarity[0])
        similarities[1].append(similarity[1])
        similarities[2].append(similarity[2])
        similarities[3].append(similarity[3])
        similarities[4].append(similarity[4])
    print("Done Calculating Similarity via  Morgan based Circular Fingerprint")
    print("##########################################")
    print("Tanimoto Similarity:{}".format(round(np.mean(similarities[0]),4)))
    print("Dice Similarity:{}".format(round(np.mean(similarities[1]),4)))
    print("Cosine Similarity:{}".format(round(np.mean(similarities[2]),4)))
    print("Sokal Similarity:{}".format(round(np.mean(similarities[3]),4)))
    print("McConnaughey Similarity:{}".format(round(np.mean(similarities[4]),4)))
    print("##########################################")
    return round(np.mean(similarities[0]),4)

def rd_fingerprint_evaluation(references, candidates):
    """
    Enumerate linear Fragement
    """
    print("Calculating Similarity via RDFIngerprint Path Similarity")
    similarities = [[],[],[],[],[]] # various similarities: Tanimoto, Dice, Cosine, Sokal, McConnaughey
    for img in references:
        similarity = [0,0,0,0,0]
        if img in candidates:
            candidate_rdkfingerprint = rdmolops.RDKFingerprint(candidates[img], fpSize=2048, minPath=1, maxPath=7)
            reference_rdkfingerprint = rdmolops.RDKFingerprint(references[img], fpSize=2048, minPath=1, maxPath=7)
            similarity[0] = round(DataStructs.TanimotoSimilarity(reference_rdkfingerprint,candidate_rdkfingerprint), 4)
            similarity[1] = round(DataStructs.DiceSimilarity(reference_rdkfingerprint,candidate_rdkfingerprint), 4)
            similarity[2] = round(DataStructs.CosineSimilarity(reference_rdkfingerprint,candidate_rdkfingerprint), 4)
            similarity[3] = round(DataStructs.SokalSimilarity(reference_rdkfingerprint,candidate_rdkfingerprint), 4)
            similarity[4] = round(DataStructs.McConnaugheySimilarity(reference_rdkfingerprint,candidate_rdkfingerprint), 4)
        similarities[0].append(similarity[0])
        similarities[1].append(similarity[1])
        similarities[2].append(similarity[2])
        similarities[3].append(similarity[3])
        similarities[4].append(similarity[4])
    print("Done Calculating Similarity via RDFIngerprint Path Similarity")
    print("##########################################")
    print("Tanimoto Similarity:{}".format(round(np.mean(similarities[0]),4)))
    print("Dice Similarity:{}".format(round(np.mean(similarities[1]),4)))
    print("Cosine Similarity:{}".format(round(np.mean(similarities[2]),4)))
    print("Sokal Similarity:{}".format(round(np.mean(similarities[3]),4)))
    print("McConnaughey Similarity:{}".format(round(np.mean(similarities[4]),4)))
    print("##########################################")
    return round(np.mean(similarities[0]),4)

def maacs_fingerprint_evaluation(references, candidates):
    """ 
    Generate Similarity via MACCSKeys
    """
    print("Calculating Similarity via MACCS Keys")
    similarities = [[],[],[],[],[]] # various similarities: Tanimoto, Dice, Cosine, Sokal, McConnaughey
    for img in references:
        similarity = [0,0,0,0,0]
        if img in candidates:
            candidate_maccs = MACCSkeys.GenMACCSKeys(candidates[img])
            reference_maccs = MACCSkeys.GenMACCSKeys(references[img])
            similarity[0] = round(DataStructs.TanimotoSimilarity(reference_maccs,candidate_maccs), 4)
            similarity[1] = round(DataStructs.DiceSimilarity(reference_maccs,candidate_maccs), 4)
            similarity[2] = round(DataStructs.CosineSimilarity(reference_maccs,candidate_maccs), 4)
            similarity[3] = round(DataStructs.SokalSimilarity(reference_maccs,candidate_maccs), 4)
            similarity[4] = round(DataStructs.McConnaugheySimilarity(reference_maccs,candidate_maccs), 4)
        similarities[0].append(similarity[0])
        similarities[1].append(similarity[1])
        similarities[2].append(similarity[2])
        similarities[3].append(similarity[3])
        similarities[4].append(similarity[4])
    print("Done Calculating Similarity via MACCS Keys")
    print("##########################################")
    print("Tanimoto Similarity:{}".format(round(np.mean(similarities[0]),4)))
    print("Dice Similarity:{}".format(round(np.mean(similarities[1]),4)))
    print("Cosine Similarity:{}".format(round(np.mean(similarities[2]),4)))
    print("Sokal Similarity:{}".format(round(np.mean(similarities[3]),4)))
    print("McConnaughey Similarity:{}".format(round(np.mean(similarities[4]),4)))
    print("##########################################")
    return round(np.mean(similarities[0]),4)

def load_img(args, path, transform):
    """
    Load Image and transform
    """
    val_transform = tv.transforms.Compose([
    tv.transforms.Lambda(under_max),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = Image.open(path)
    img = val_transform(img)
    img = img.unsqueeze(0)
    return img

def img_distance_eval(args, references, candidates, transform):
    """
    Edit distance between two images of generated molecules:
    """
    scores = []
    for img in references:
        if img in candidates:
            try:
                m = Chem.MolFromSmiles(references[img])
                Draw.MolToFile(m,"tmp.png", size=(args.img_size,args.img_size))
                reference_img = load_img(args, "tmp.png", transform)
                m = Chem.MolFromSmiles(candidates[img])
                Draw.MolToFile(m,"tmp.png", size=(args.img_size,args.img_size))
                candidate_img = load_img(args, "tmp.png", transform)
                score = math.log(torch.sum(torch.abs(reference_img- candidate_img)))
                scores.append(score)
            except:
                pass
    return round(np.mean(scores),4)
    
def get_exact_match(references, candidates):
    """
    Exact Matc between two smiles
    """
    exact_match = 0
    for img in references:
        candidate_smi = ''
        if img in candidates and references[img] == candidates[img]:
            exact_match += 1
    return exact_match

def main(args):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([normalize])
    reference_img2smi, _ = load_img2smi(args.reference_file)
    candidate_img2smi, count = load_img2smi(args.candidate_file)
    candidate_keys = candidate_img2smi.keys()
    reference_img2smi = {key: reference_img2smi[key] for key in candidate_keys if key in reference_img2smi}
    print(len(reference_img2smi))
    percent_candidate_molecules_captioned =  1 - (count/len(reference_img2smi))
    reference_mols, _ = convert_smi_to_mol(reference_img2smi, args.reference_file)
    candidate_mols, count = convert_smi_to_mol(candidate_img2smi, args.candidate_file)
    percent_candidate_valid_molecules_captioned = (count/len(reference_img2smi))
    levenshtein_distance = levenshtein_eval(reference_img2smi, candidate_img2smi)
    bleu_score = bleu_eval(args, reference_img2smi, candidate_img2smi)
    maccs_score = maacs_fingerprint_evaluation(reference_mols, candidate_mols)
    rd_score = rd_fingerprint_evaluation(reference_mols, candidate_mols)
    morgan_score = morgan_fingerprint_evaluation(reference_mols, candidate_mols)
    image_distance = img_distance_eval(args, reference_img2smi, candidate_img2smi, transform)
    exact_match = get_exact_match(reference_img2smi, candidate_img2smi)
    with open(args.output_file, 'w') as w:
        w.write("Reference File:{}\n".format(args.reference_file))
        w.write("Candidate File:{}\n".format(args.candidate_file))
        w.write("There are {} examples being evaluated".format(len(reference_img2smi)))
        w.write("Percent Captions Generated:{}\n".format(round(percent_candidate_molecules_captioned,4)))
        w.write("Percent Valid SMI Generated:{}\n".format(round(percent_candidate_valid_molecules_captioned,4)))
        w.write("Exact matches for Cannonical SMI strings:{}\n".format(exact_match))
        w.write("Levenshtein Distance:{}\n".format(levenshtein_distance))
        w.write("BLEU-1 using {} tokenizer:{}\n".format(args.tokenizer, bleu_score))
        w.write("MACCS Fingerprinting Tanimoto Similarity:{}\n".format(maccs_score))
        w.write("RD Path Fingerprinting Tanimoto Similarity:{}\n".format(rd_score))
        w.write("Morgan Fingerprint Tanimoto Similarity:{}\n".format(morgan_score))
        w.write("Image Distance:{}\n".format(image_distance))
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate smiles captioning')
    parser.add_argument('--reference_file', type=str, default='exp/references_validation.txt', help='reference img2smi file which is a tsv with SMI\tIMG.PNG')
    parser.add_argument('--candidate_file', type=str, default='exp/osra_validation.txt', help='candidate img2smi file which is a tsv with SMI\tIMG.PNG')
    parser.add_argument('--output_file', type=str, default ='exp/osra_validation_results.txt', help='Filename where eval results will be written')
    parser.add_argument('--tokenizer', default='data/tokenizers/tokenizer_vocab_20000.json', type=str, help='tokenizer to use in BLEU evaluation')
    parser.add_argument('--test_string', type=str, default='CC(C)CCNc1cnnc(NCCc2ccc(S(N)(=O)=O)cc2)n1', help='a SMILES string to test tokenizer with') 
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    args = parser.parse_args()
    main(args)

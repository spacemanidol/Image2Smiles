# Image2smiles
Given an image of a molecule create a smiles or mol represenatation. 

conda create -n molecule_extraction python=3.7
conda activate molecule_extraction
conda install -c conda-forge rdkit deepchem==2.3.0
pip install tensorflow-gpu==1.14

# Data
Data is the collection of a bunch of different SMILES molecules

[Validation Set](https://spacemanidol.blob.core.windows.net/blob/valitation.tar.gz)




# Tokenizers

python train_tokenizer.py  --training_files data/ --do_train --do_test --tokenizer_name tokenizer_vocab_2000.json --vocab_size 2000 --min_frequency 2

Note Tokenizers need to be cased. 



# Baseline
https://cactus.nci.nih.gov/osra/
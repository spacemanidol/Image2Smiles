# Image2Smiles
In order to mine the vast chemistry literature we need to create a processor which given a molecule produces a SMILES string. 
We treat this problem like an image captioning problem. Given an Image produce the associated caption. 

## High Level
Note Tokenizers need to be cased. 
## Details
### Setup
Create your enviorment by running the following commands

```bash
conda env create -f environment.yaml
```

### Data
Data is the collection of a bunch of different SMILES molecules. These smiles strings were joined, shuffled and then images were created from each of the strings. The Validation and Evaluation sets consists of 100,000 image and smiles captions. While experimenting with model tweaks we recommend a subsample of 10,000 images for validation to allow for quicker evaluation. The training corpus consists of ~20m smile images. For ease of experimentation we have produced processed files and stored them on Azure. 

[Training set]()
[Validation Set]()
[Evaluation Set]()

### Tokenizers
```bash
python src/train_tokenizer.py  --training_files data/ --do_train --do_test --tokenizer_name tokenizer_vocab_20000.json --vocab_size 20000 --min_frequency 2
python src/train_tokenizer.py  --training_files data/ --do_train --do_test --tokenizer_name tokenizer_vocab_2000.json --vocab_size 2000 --min_frequency 2
python src/train_tokenizer.py  --training_files data/ --do_train --do_test --tokenizer_name tokenizer_vocab_500.json --vocab_size 500 --min_frequency 2
python src/train_tokenizer.py  --training_files data/ --do_train --do_test --tokenizer_name tokenizer_vocab_200.json --vocab_size 200 --min_frequency 2
python src/train_tokenizer.py  --training_files data/ --do_train --do_test --tokenizer_name tokenizer_vocab_100.json --vocab_size 100 --min_frequency 2
```


# Baseline
https://cactus.nci.nih.gov/osra/

The most popular similarity measure for comparing chemical structures represented by means of fingerprints is the Tanimoto (or Jaccard) coefficient T. Two structures are usually considered similar if T > 0.85 (for Daylight fingerprints). However, it is a common misunderstanding that a similarity of T > 0.85 reflects similar bioactivities

https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3
https://docs.eyesopen.com/toolkits/python/graphsimtk/measure.html


### Train 
```bash
python train.py --max_length 150  --tokenizer tokenizers/tokenizer_vocab_100.json  --captions_prefix vocab100 --data_dir data/ --epochs 1 --num_workers 8 --batch_size 64 --dropout 0.5  --embedding_dim 512  --decoder_dim 512 --encoder_dim 2048 --encoder_lr 1e-4 --decoder_lr 4e-4 --encoder_type RESNET101 --decoder_type LSTM+Attention --model_path models/vocab100 --cuda --cuda_device cuda:0
python src/train.py --tokenizer data/tokenizers/tokenizer_vocab_20000.json  --captions_prefix vocab20000 --data_dir data --epochs 1 --num_workers 16 --batch_size 48 --model_path models/vocab20000_ --cuda --cuda_device cuda:3
python src/train.py --tokenizer data/tokenizers/tokenizer_vocab_20000.json  --captions_prefix vocab20000 --data_dir data --epochs 1 --num_workers 16 --batch_size 48 --model_path models/vocab20000_finetune_ --cuda --cuda_device cuda:3 --fine_tune

```

Models
Done
100 
200
200finetune
500
500finetune
2000
selfies
2000finetune
selfies_finetune


In Progress
100finetune


Need
20000
20000finetune


### Predict
Explore beam search and branching beam search
```bash
python predict.py --output exp/vocab200_50_branchfacto5_brachround10_branchestoexpand25 --images_to_predict exp/to_predict.txt --directory_path data/tmp/validation_images/ --beam_size 50 --tokenizer tokenizers/tokenizer_vocab_200.json --cuda --model_path models/vocab200checkpoint_62000 --cuda_device cuda:0 --branch_rounds 10 --branch_factor 5 --branches_to_expand 25
python predict.py --output exp/vocab20000_beam50_branchfacto5_brachround10_branchestoexpand25 --images_to_predict exp/to_predict.txt --directory_path data/tmp/validation_images/ --beam_size 50 --tokenizer tokenizers/tokenizer_vocab_20000.json --cuda --model_path models/vocab20000_checkpoint_0 --cuda_device cuda:0 --branch_rounds 10 --branch_factor 5 --branches_to_expand 25
```

### Results

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
```python
python src/train_tokenizer.py  --training_files data/ --do_train --do_test --tokenizer_name tokenizer_vocab_20000.json --vocab_size 20000 --min_frequency 2
python src/train_tokenizer.py  --training_files data/ --do_train --do_test --tokenizer_name tokenizer_vocab_2000.json --vocab_size 2000 --min_frequency 2
python src/train_tokenizer.py  --training_files data/ --do_train --do_test --tokenizer_name tokenizer_vocab_500.json --vocab_size 500 --min_frequency 2
python src/train_tokenizer.py  --training_files data/ --do_train --do_test --tokenizer_name tokenizer_vocab_200.json --vocab_size 200 --min_frequency 2
python src/train_tokenizer.py  --training_files data/ --do_train --do_test --tokenizer_name tokenizer_vocab_100.json --vocab_size 100 --min_frequency 2
```


### Train 
```python
python train.py --max_length 150 --tokenizer tokenizers/tokenizer_vocab_200.json --data_dir data/ --epochs 10 --num_workers 8 --batch_size 32 --dropout 0.5 --embedding_dim 512 --encoder_type RESNET101 --decoder_type LSTM+Attention --decoder_dim 512 --encoder_dim 2048 --attention_dim 512 --cuda --model_path models/basevocab200/ --encoder_lr 1e-4 --decoder_lr 4e-4 --gradient_clip 5.0 --alphac 1 --print_freq 100
```

### Baseline
https://cactus.nci.nih.gov/osra/

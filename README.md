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

### Train 
```bash
python train.py --max_length 150  --tokenizer tokenizers/tokenizer_vocab_100.json  --captions_prefix vocab100 --data_dir data/ --epochs 1 --num_workers 8 --batch_size 64 --dropout 0.5  --embedding_dim 512  --decoder_dim 512 --encoder_dim 2048 --encoder_lr 1e-4 --decoder_lr 4e-4 --encoder_type RESNET101 --decoder_type LSTM+Attention --model_path models/vocab100 --cuda --cuda_device cuda:0
python src/train.py --tokenizer data/tokenizers/tokenizer_vocab_20000.json  --captions_prefix vocab20000 --data_dir data --epochs 1 --num_workers 16 --batch_size 48 --model_path models/vocab20000_ --cuda --cuda_device cuda:3
python src/train.py --tokenizer data/tokenizers/tokenizer_vocab_20000.json  --captions_prefix vocab20000 --data_dir data --epochs 1 --num_workers 16 --batch_size 48 --model_path models/vocab20000_finetune_ --cuda --cuda_device cuda:3 --fine_tune

```

#### Models trained
selfiesbase
selfies_finetune
vocab100base
vocab100finetune
vocab200base
vocab200finetune
vocab500base
vocab500finetune
vocab2000base
vocab2000finetune
vocab20000base
vocab20000finetune
### Predict
Explore beam search and branching beam search
```bash
python src/predict.py --output selfies_finetune_10_5_10_30 --images_to_predict outputs/predictions_valid/targets.txt --directory_path data/tmp/validation_images/ --use_selfies --cuda --model_path models/done/selfiesfinetune --branch_rounds 10 --branch_factor 5 --branches_to_expand 10 --beam_size 30 --cuda_device cuda:2
```

### Evaluate
```bash
python src/evaluate.py --reference_file outputs/predictions_valid/references.txt  --candidate_file outputs/predictions_valid/vocab200_fine_tune_beam_30 --tokenizer data/tokenizers/tokenizer_vocab_200.json --output_file outputs/eval_results_valid/vocab200_fine_tune_beam30_eval_results.txt
```
### Results

#### Non Finetuned Encoder

| Model Name     | Beam Size | Beam Branching | Vocabulary Size | Images Captioned(\%) | Valid SMILES (\%) | Levenshtein Distance | BLEU-1 | MACCS Similatiry | Path Similarity | Morgan Similarity | Image Reconstruction |
|----------------|-----------|----------------|-----------------|----------------------|-------------------|----------------------|--------|------------------|-----------------|-------------------|----------------------|
| OSRA(Baseline) | 1         | 0              | N/A             | 86.5                 | 65.2              | 30.841               | 0.0533 | 0.3849           | 0.2835          | 0.286             | 9.0777               |
| SELFI-NET      | 3         | 0              | 713             | 61.8                 | 61.9              | 53.02                | 0.0289 | 0.1526           | 0.0954          | 0.0451            | 9.5457               |
| SELFI-NET      | 30        | 0              | 713             | 98.4                 | 93.4              | 47.0                 | 0.0421 | 0.2075           | 0.1044          | 0.057             | 9.4969               |
| SELFI-NET      | 3         | 5              | 713             | 99.8                 | 99.9              | 55.06                | 0.0425 | 0.2449           | 0.1304          | 0.069             | 9.5255               |
| SMI-NET        | 3         | 0              | 100             | 3.5                  | 3.6               | 42.381               | 0.0128 | 0.0073           | 0.0061          | 0.0028            | 9.7297               |
| SMI-NET        | 30        | 0              | 100             | 4.2                  | 4.3               | 42.346               | 0.0115 | 0.0052           | 0.0016          | 0.0021            | 9.3643               |
| SMI-NET        | 3         | 5              | 100             | 6.0                  | 6.0               | 42.136               | 0.0119 | 0.0072           | 0.0019          | 0.0029            | 9.5126               |
| SMI-NET        | 3         | 0              | 200             | 3.5                  | 3.5               | 42.379               | 0.0073 | 0.006            | 0.0056          | 0.0022            | 9.6775               |
| SMI-NET        | 30        | 0              | 200             | 1                    | 1                 | 42.397               | 0.006  | 0.0018           | 0.0014          | 0.0008            | 9.5883               |
| SMI-NET        | 3         | 5              | 200             | 3.1                  | 3                 | 42.269               | 0.0062 | 0.0037           | 0.001           | 0.002             | 9.4938               |
| SMI-NET        | 3         | 0              | 500             | 3.3                  | 3.3               | 42.267               | 0.0071 | 0.0068           | 0.0057          | 0.0032            | 9.6521               |
| SMI-NET        | 30        | 0              | 500             | 3.3                  | 3.3               | 42.267               | 0.0071 | 0.0068           | 0.0057          | 0.0032            | 9.6521               |
| SMI-NET        | 3         | 5              | 500             | 5.7                  | 5.7               | 42.225               | 0.0064 | 0.0064           | 0.0016          | 0.0032            | 9.5274               |
| SMI-NET        | 3         | 0              | 2000            | 10.2                 | 10.2              | 40.956               | 0.0049 | 0.0515           | 0.027           | 0.0241            | 9.4772               |
| SMI-NET        | 30        | 0              | 2000            | 10                   | 10                | 41.3                 | 0.0036 | 0.0416           | 0.0201          | 0.0196            | 9.4044               |
| SMI-NET        | 3         | 5              | 2000            | 8.8                  | 8.8               | 41.733               | 0.0029 | 0.0216           | 0.0084          | 0.0088            | 9.4106               |
| SMI-NET        | 3         | 0              | 20000           | 5.4                  | 5.4               | 42.187               | 0.0024 | 0.0092           | 0.0074          | 0.0042            | 9.6335               |
| SMI-NET        | 30        | 0              | 20000           | 4.2                  | 4.2               | 42.215               | 0.0011 | 0.0058           | 0.002           | 0.0026            | 9.5251               |
| SMI-NET        | 3         | 5              | 20000           | 7.9                  | 7.9               | 42.074               | 0.0016 | 0.0105           | 0.0024          | 0.0043            | 9.4702               |

#### Fine Tuned Encoder

| Model Name     | Beam Size | Beam Branching | Vocabulary Size | Images Captioned(\%) | Valid SMILES (\%) | Levenshtein Distance | BLEU-1 | MACCS Similatiry | Path Similarity | Morgan Similarity | Image Reconstruction |
|----------------|-----------|----------------|-----------------|----------------------|-------------------|----------------------|--------|------------------|-----------------|-------------------|----------------------|
| OSRA(Baseline) | 1         | 0              | N/A             | 86.5                 | 65.2              | 30.841               | 0.0533 | 0.3849           | 0.2835          | 0.286             | 9.0777               |
| SELFI-NET      | 3         | 0              | 713             | 99.3                 | 99.4              | 33.636               | 0.0549 | 0.418            | 0.2309          | 0.1328            | 9.5994               |
| SELFI-NET      | 30        | 0              | 713             | 100                  | 71.3              | 35.751               | 0.0501 | 0.228            | 0.1115          | 0.062             | 9.5994               |
| SELFI-NET      | 3         | 5              | 713             | 99.9                 | 97.8              | 34.072               | 0.048  | 0.308            | 0.1498          | 0.0854            | 9.4851               |
| SMI-NET        | 3         | 0              | 100             | 2.9                  | 2.9               | 38.229               | 0.024  | 0.1551           | 0.0773          | 0.073             | 9.5184               |
| SMI-NET        | 30        | 0              | 100             | 29.9                 | 29.9              | 38.482               | 0.0225 | 0.1475           | 0.0733          | 0.0693            | 9.491                |
| SMI-NET        | 3         | 5              | 100             | 13.5                 | 13.5              | 41.236               | 0.0141 | 0.0444           | 0.0175          | 0.0207            | 9.372                |
| SMI-NET        | 3         | 0              | 200             | 22.6                 | 22.6              | 39.089               | 0.0172 | 0.0172           | 0.0568          | 0.0501            | 9.4919               |
| SMI-NET        | 30        | 0              | 200             | 21.5                 | 21.5              | 39.544               | 0.0148 | 0.1015           | 0.0473          | 0.0455            | 9.4374               |
| SMI-NET        | 3         | 5              | 200             | 12.0                 | 12.0              | 41.519               | 0.0086 | 0.0349           | 0.013           | 0.0139            | 9.374                |
| SMI-NET        | 3         | 0              | 500             | 8.5                  | 8.5               | 41.423               | 0.0097 | 0.0385           | 0.0183          | 0.0162            | 9.4737               |
| SMI-NET        | 30        | 0              | 500             | 10                   | 10                | 41.53                | 0.0085 | 0.0355           | 0.014           | 0.0152            | 9.4144               |
| SMI-NET        | 3         | 5              | 500             | 12.5                 | 12.5              | 41.677               | 0.0079 | 0.0311           | 0.0084          | 0.0114            | 9.3872               |
| SMI-NET        | 3         | 0              | 2000            | 20                   | 20                | 39.955               | 0.0115 | 0.0787           | 0.0393          | 0.0285            | 9.5625               |
| SMI-NET        | 30        | 0              | 2000            | 18.2                 | 18.2              | 40.215               | 0.0094 | 0.068            | 0.0323          | 0.0255            | 9.5258               |
| SMI-NET        | 3         | 5              | 2000            | 8.8                  | 8.8               | 41.733               | 0.0029 | 0.0216           | 0.0084          | 0.0088            | 9.4106               |
| SMI-NET        | 3         | 0              | 20000           | 7.5                  | 7.5               | 41.563               | 0.0037 | 0.0317           | 0.0159          | 0.0118            | 9.5906               |
| SMI-NET        | 30        | 0              | 20000           | 11.1                 | 11.1              | 41.735               | 0.0023 | 0.0287           | 0.011           | 0.0113            | 9.4111               |
| SMI-NET        | 3         | 5              | 20000           | 18.1                 | 18.0              | 41.406               | 0.0042 | 0.0412           | 0.0121          | 0.0139            | 9.4855               |
# Image2Smiles
In order to mine the vast chemistry literature we need to create a processor which given a molecule produces a SMILES string. 
We treat this problem like an image captioning problem so given an image X produce a caption in SMILES string. This repo cointains three main portions, the data generation code, a RNN based captioning system and a transformer based captioning system. 

### Setup
Create your enviorment by running the following commands. Note the RNN based method and the transformer based method have different enviorments and each has an enviorment file in its root.

```bash
conda env create -f environment.yaml
```

### Data
Data is the collection of a bunch of different SMILES molecules. These smiles strings were joined, shuffled and then images were created from each of the strings. The Validation and Evaluation sets consists of 5,000 image and smiles captions and the training corpus consists of ~1m smile images. For ease of experimentation we have produced processed files and stored them on Azure. If you wish to generate the corpus or augment the corpus we crteated please look at the code in src/data_gen


Files for Azure
[MOLCAP](https://spacemanidol.blob.core.windows.net/blob/MOLCAP.gz) full 82 million smiles strings
[MOLCAP Images and Captions](https://spacemanidol.blob.core.windows.net/blob/data.zip) 1 million image training corpus and evaluation corpus.

### Tokenizers
Tokeniers are trained to desired size using Hugging Face's tokenizers library. Training can be done as shown below.
```bash
python src/train_tokenizer.py  --training_files data/ --do_train --do_test --tokenizer_name tokenizer_vocab_20000.json --vocab_size 20000 --min_frequency 2
python src/train_tokenizer.py  --training_files data/ --do_train --do_test --tokenizer_name tokenizer_vocab_2000.json --vocab_size 2000 --min_frequency 2
python src/train_tokenizer.py  --training_files data/ --do_train --do_test --tokenizer_name tokenizer_vocab_500.json --vocab_size 500 --min_frequency 2
python src/train_tokenizer.py  --training_files data/ --do_train --do_test --tokenizer_name tokenizer_vocab_200.json --vocab_size 200 --min_frequency 2
python src/train_tokenizer.py  --training_files data/ --do_train --do_test --tokenizer_name tokenizer_vocab_100.json --vocab_size 100 --min_frequency 2
```

### RNN Model
To train the RNN based model use the command below
```bash
python src/rnn_based/train.py --max_length 150  --tokenizer data/tokenizers/tokenizer_vocab_100.json  --captions_prefix vocab100 --data_dir data/ --epochs 1 --num_workers 8 --batch_size 64 --dropout 0.5  --embedding_dim 512  --decoder_dim 512 --encoder_dim 2048 --encoder_lr 1e-4 --decoder_lr 4e-4 --encoder_type RESNET101 --decoder_type LSTM+Attention --model_path models/vocab100 --cuda --cuda_device cuda:0

```
To predict use the command below. 
```bash
python src/rnn_based/predict.py --output selfies_finetune_10_5_10_30 --images_to_predict outputs/predictions_valid/targets.txt --directory_path data/tmp/validation_images/ --use_selfies --cuda --model_path models/done/selfiesfinetune --beam_size 1 --cuda_device cuda:2
```
If you wish to use beam search or branch use some form of this command
```bash
python src/rnn_based/predict.py --output selfies_finetune_10_5_10_30 --images_to_predict outputs/predictions_valid/targets.txt --directory_path data/tmp/validation_images/ --use_selfies --cuda --model_path models/done/selfiesfinetune --branch_rounds 10 --branch_factor 5 --branches_to_expand 10 --beam_size 30 --cuda_device cuda:2
```

### Transformer Model
Training can be done as shown below. To freeze the encoder just add --freeze_encoder
```bash
python src/transformer_based/main.py --do_train --do_eval --do_predict --lr 5e-5 --cuda --model_path model5e5_finetune
```

To run inference you can use the command below.
```bash
python src/transformer_based/main.py --do_predict --load_model --model_path models/model5e5_finetune_epoch_5.model --predict_list data/evaluation_labels.smi  --cuda --image_dir data/evaluation_images--output_file outputs/5e5finetune-predictions.txt
```
python src/transformer_based/main.py --do_predict --load_model --model_path models/model5e5_base_epoch_5.model --predict_list data/evaluation_labels.smi  --cuda --image_dir data/evaluation_images --output_file outputs/5e5base-predictions.txt

### Metrics Evaluations and Pyosra
To run inference with OSRA use the command below.
```bash
python src/run_pyosra.py
```
If you wish to generate metrics for random selection of molecules run the command below.
```bash
python src/get_eval_stats_for_random.py
```

To run evaluation you simply need a candidate and reference file as shown below. 
```bash
python src/evaluate.py --candidate_file outputs/5e5base-predictions.txt --output_file outputs/5e5base-results.txt
```
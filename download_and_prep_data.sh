#!/bin/bash
echo('Downloading data')
mkdir data
cd data
wget https://raw.githubusercontent.com/undeadpixel/reinvent-randomized/master/training_sets/chembl.training.smi
wget https://raw.githubusercontent.com/undeadpixel/reinvent-randomized/master/training_sets/chembl.validation.smi
wget https://raw.githubusercontent.com/undeadpixel/reinvent-randomized/master/training_sets/decorator_scaffolds_drd2.training.smi
wget https://raw.githubusercontent.com/undeadpixel/reinvent-randomized/master/training_sets/decorator_scaffolds_recap.training.smi
wget https://raw.githubusercontent.com/undeadpixel/reinvent-randomized/master/training_sets/decorator_scaffolds_drd2.validation.smi
wget https://raw.githubusercontent.com/undeadpixel/reinvent-randomized/master/training_sets/decorator_scaffolds_recap.validation.smi
wget https://raw.githubusercontent.com/undeadpixel/reinvent-randomized/master/training_sets/gdb13.1M.training.smi
wget https://raw.githubusercontent.com/undeadpixel/reinvent-randomized/master/training_sets/gdb13.1M.validation.smi
wget https://raw.githubusercontent.com/alexarnimueller/SMILES_generator/master/data/chembl24_10uM_20-100.csv
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pubchem.zip
wget https://cactus.nci.nih.gov/osra/uspto-validation-updated.zip
unzip pubchem.zip
unzip uspto-validation-updated.zip
rm uspto-validation-updated.zip pubchem.zip
echo('Done Downloading Data')

echo('Formating Data')
touch smiles_data.smi
for filename in pubchem/*.smi; do
    cat $filename >> smiles_data.smi
done
rm -rf pubchem/
cat chembl.training.smi >> smiles_data.smi
cat chembl.validation.smi >> smiles_data.smi
cat decorator_scaffolds_recap.training.smi >> smiles_data.smi
cat decorator_scaffolds_recap.validation.smi >> smiles_data.smi
cat decorator_scaffolds_drd2.training.smi >> smiles_data.smi
cat decorator_scaffolds_drd2.validation.smi >> smiles_data.smi
cat gdb13.1M.training.smi >> smiles_data.smi
cat gdb13.1M.validation.smi >> smiles_data.smi
cat chembl24_10uM_20-100.csv >> smiles_data.smi
uniq  smiles_data.smi > smiles_data_unique.smi
shuf smiles_data_unique.smi > shuffled_unique_smiles.smi
{
  head -n 100000 > validation.smi
  head -n 100000 > evaluation.smi
  cat > training.smi
} < shuffled_unique_smiles.smi
rm shuffled_unique_smiles.smi smiles_data.smi smiles_data_unique.smi
rm decorator_scaffolds_recap.training.smi decorator_scaffolds_recap.validation.smi decorator_scaffolds_drd2.training.smi decorator_scaffolds_drd2.validation.smi chembl.training.smi chembl.validation.smi gdb13.1M.training.smi gdb13.1M.validation.smi chembl24_10uM_20-100.csv
mkdir training_images validation_images evaluation_images
cd ..
echo('Done Formating Data')
echo('Make Canonical')
python make_canonical.py --input_file data/validation.smi --output_file data/validation_canonical.smi
python make_canonical.py --input_file data/evaluation.smi --output_file data/evaluation_canonical.smi
python make_canonical.py --input_file data/training.smi --output_file data/training_canonical.smi
uniq data/evaluation_canonical.smi > data/evaluation.smi
uniq data/validation_canonical.smi > data/validation.smi
uniq data/training_canonical.smi > data/training.smi
rm data/evaluation_canonical.smi data/validation_canonical.smi data/training_canonical.smi
echo('Done Making Cannonical')

echo('Converting Data to Images')
python create_images_from_smiles.py --input_file data/validation.smi --output_folder data/validation_images
python create_images_from_smiles.py --input_file data/evaluation.smi --output_folder data/evaluation_images
python create_images_from_smiles.py --input_file data/training.smi --output_folder data/training_images
echo('Done Converting Data to Images')

echo('Training Tokenizers')
python train_tokenizer.py  --training_files data/ --do_train --do_test --tokenizer_name tokenizer_vocab_20000.json --vocab_size 20000 --min_frequency 2
python train_tokenizer.py  --training_files data/ --do_train --do_test --tokenizer_name tokenizer_vocab_2000.json --vocab_size 2000 --min_frequency 2
python train_tokenizer.py  --training_files data/ --do_train --do_test --tokenizer_name tokenizer_vocab_500.json --vocab_size 500 --min_frequency 2
python train_tokenizer.py  --training_files data/ --do_train --do_test --tokenizer_name tokenizer_vocab_200.json --vocab_size 200 --min_frequency 2
python train_tokenizer.py  --training_files data/ --do_train --do_test --tokenizer_name tokenizer_vocab_100.json --vocab_size 100 --min_frequency 2
echo('Done Training Tokenizers')

echo('Preprocessing Data with Tokenizers for model Training')
split -n 4 data/training_images/labels.smi
mv xaa labels1.smi
mv labels* data/training_images/
python preprocess_data.py --tokenizer tokenizers/tokenizer_vocab_2000.json --data_dir data/training_images/ --data_split training --image_output_filename training --output_path data/ 
python preprocess_data.py --tokenizer tokenizers/tokenizer_vocab_2000.json --data_dir data/validation_images/ --data_split validation --image_output_filename validation --output_path data/
python preprocess_data.py --tokenizer tokenizers/tokenizer_vocab_2000.json --data_dir data/evaluation_images/ --data_split evaluation --image_output_filename evaluation --output_path data/ --process_img
echo('Done preprocessing data. You may move on to model training.')

python preprocess_data.py --tokenizer tokenizers/tokenizer_vocab_2000.json --data_dir data/evaluation_images/ --data_split evaluation --image_output_filename evaluation --output_path data/ --process_img --output_prefix vocab2000
#!/bin/bash
python src/RESNET101EncoderDecoder/main.py --do_train --do_eval --do_predict --lr 5e-5 --cuda --model_path models/model5e5_base --freeze_encoder
python src/RESNET101EncoderDecoder/main.py --do_train --do_eval --do_predict --lr 5e-5 --cuda --model_path model5e5_finetune
python src/RESNET101EncoderDecoder/main.py --do_predict --load_model --model_path models/model5e5_finetune_epoch_5.model --predict_list data/validation_images/labels_short.smi  --cuda --image_dir data/validation_images/ --output_file outputs/validation-5e5-finetune-predictions-epoch5.txt
python src/RESNET101EncoderDecoder/main.py --do_predict --load_model --model_path models/model5e5_base_epoch_5.model --predict_list data/validation_images/labels_short.smi  --cuda --image_dir data/validation_images/ --output_file outputs/validation-5e5-base-predictions-epoch5.txt
python src/evaluate.py --reference_file data/validation_images/labels_short.smi --candidate_file outputs/validation-5e5-base-predictions-epoch5.txt --output_file outputs/validation-5e5-results-epoch5.txt

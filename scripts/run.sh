#!/bin/bash
python src/transformer_based/main.py --do_train --do_eval --do_predict --lr 5e-5 --cuda --model_path models/model5e5_base --freeze_encoder
python src/transformer_based/main.py --do_train --do_eval --do_predict --lr 5e-5 --cuda --model_path model5e5_finetune
python src/transformer_based/main.py --do_predict --load_model --model_path models/model5e5_finetune_epoch_5.model --predict_list data/evaluation_labels.smi  --cuda --image_dir data/evaluation_images--output_file outputs/5e5finetune-predictions.txt
python src/transformer_based/main.py --do_predict --load_model --model_path models/model5e5_base_epoch_5.model --predict_list data/evaluation_labels.smi  --cuda --image_dir data/evaluation_images --output_file outputs/5e5base-predictions.txt
python src/run_pyosra.py
python src/get_eval_stgats_for_random.py
python src/evaluate.py --candidate_file outputs/5e5base-predictions.txt --output_file outputs/5e5base-results.txt
python src/evaluate.py --candidate_file outputs/5e5finetune-predictions.txt --output_file outputs/5e5fine-results.txt
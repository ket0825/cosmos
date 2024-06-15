#!/bin/bash

eval_fp="./resources_review/data/test/"
eval_batch_size=4
init_model_path=klue/bert-base
max_length=512
need_birnn=0
print_sample=1
sentiment_drop_ratio=0.3
aspect_drop_ratio=0.3
sentiment_in_feature=768
aspect_in_feature=768
base_path="./ckpt_review/result_model/"
label_info_file="meta.bin"
out_model_path="pytorch_model.bin"

python ./src_review/do_eval.py --eval_fp=$eval_fp --eval_batch_size=$eval_batch_size --init_model_path=$init_model_path --max_length=$max_length --need_birnn=$need_birnn --sentiment_drop_ratio=$sentiment_drop_ratio --aspect_drop_ratio=$aspect_drop_ratio --sentiment_in_feature=$sentiment_in_feature --aspect_in_feature=$aspect_in_feature --base_path=$base_path --label_info_file=$label_info_file --out_model_path=$out_model_path --print_sample=$print_sample

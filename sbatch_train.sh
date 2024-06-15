#!/bin/bash

#SBATCH -J keyboard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=8G
#SBATCH -p batch_ugrad
#SBATCH -t 10:0:0
#SBATCH -w aurora-g5
#SBATCH -o logs/slurm-%A.out

epochs=10
train_batch_size=4
valid_batch_size=4
init_model_path=klue/bert-base
max_length=512
need_birnn=0
sentiment_drop_ratio=0.3
aspect_drop_ratio=0.3
sentiment_in_feature=768
aspect_in_feature=768
stop_patience=3
train_fp="../../../../local_datasets/keyboard_dataset/train/"
valid_fp="../../../../local_datasets/keyboard_dataset/valid/"
base_path="./ckpt_review/model/"
label_info_file="meta.bin"
out_model_path="pytorch_model.bin"

# dataset path
fp="../../../../local_datasets/keyboard_dataset/"
save_p="../../../../local_datasets/keyboard_dataset/"
val_ratio=0.1
test_ratio=0.1
encoding=utf-8-sig

/data/$USER/anaconda3/bin/conda init
source ~/.bashrc
conda activate envs2

cd /data/lch0324/repos/cosmos/resources_review/data/
tar -cf keyboard_dataset.tar *.json
rm -rf /local_datasets/keyboard_dataset/
mkdir /local_datasets/keyboard_dataset/
tar -xf /data/lch0324/repos/cosmos/resources_review/data/keyboard_dataset.tar -C /local_datasets/keyboard_dataset/
echo "Data is ready!"

cd ../../

pwd
which python
hostname

python ./src_review/do_parsingData.py --fp=$fp --save_p=$save_p --val_ratio=$val_ratio --test_ratio=$test_ratio --encoding=$encoding
python ./src_review/do_train.py --epochs=$epochs --init_model_path=$init_model_path --train_batch_size=$train_batch_size --valid_batch_size=$valid_batch_size --max_length=$max_length --need_birnn=$need_birnn --sentiment_drop_ratio=$sentiment_drop_ratio --aspect_drop_ratio=$aspect_drop_ratio --sentiment_in_feature=$sentiment_in_feature --aspect_in_feature=$aspect_in_feature --stop_patience=$stop_patience --train_fp=$train_fp --valid_fp=$valid_fp --base_path=$base_path --label_info_file=$label_info_file --out_model_path=$out_model_path

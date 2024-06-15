#! /usr/bin/bash


#SBATCH -J p-review
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=16G
#SBATCH -p batch_ugrad
#SBATCH -t 10:0:0
#SBATCH -w aurora-g7
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
train_fp="./resources_review/parsing_data/train/"
valid_fp="./resources_review/parsing_data/valid/"
base_path="./ckpt_review/model/"
label_info_file="meta.bin"
out_model_path="pytorch_model.bin"

# dataset path
fp="../../../../local_datasets/p_review_dataset/"
save_p="./resources_review/parsing_data/"
val_ratio=0.1
test_ratio=0.1
encoding="utf-8-sig"
cd /data/ndn825/Preview_model/review_tagging/resources_review/data/
rm -rf p_review_dataset.tar
tar -cf p_review_dataset.tar *.json
rm -rf /local_datasets/p_review_dataset/
mkdir /local_datasets/p_review_dataset/
tar -xf /data/ndn825/Preview_model/review_tagging/resources_review/data/p_review_dataset.tar -C /local_datasets/p_review_dataset/
echo "Data is ready!"

cd /data/ndn825/Preview_model/review_tagging

/data/$USER/anaconda3/bin/conda init
source ~/.bashrc
conda activate preview_model

echo "PWD"
pwd
echo "WHICH PYTHON"
which python
echo "HOSTNAME"
hostname

# python ./src_review/do_parsingData.py --fp=$fp --save_p=$save_p --val_ratio=$val_ratio --test_ratio=$test_ratio --encoding=$encoding
python ./src_review/do_train.py --epochs=$epochs --init_model_path=$init_model_path --train_batch_size=$train_batch_size --valid_batch_size=$valid_batch_size --max_length=$max_length --need_birnn=$need_birnn --sentiment_drop_ratio=$sentiment_drop_ratio --aspect_drop_ratio=$aspect_drop_ratio --sentiment_in_feature=$sentiment_in_feature --aspect_in_feature=$aspect_in_feature --stop_patience=$stop_patience --train_fp=$train_fp --valid_fp=$valid_fp --base_path=$base_path --label_info_file=$label_info_file --out_model_path=$out_model_path

exit 0
#! /usr/bin/bash


#SBATCH -J p-review
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=16G
#SBATCH -p batch_ugrad
#SBATCH -t 10:0:0
#SBATCH -w aurora-g7
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
train_fp="./resources_review/parsing_data/train/"
valid_fp="./resources_review/parsing_data/valid/"
base_path="./ckpt_review/model/"
label_info_file="meta.bin"
out_model_path="pytorch_model.bin"

# dataset path
fp="../../../../local_datasets/p_review_dataset/"
save_p="./resources_review/parsing_data/"
val_ratio=0.1
test_ratio=0.1
encoding="utf-8-sig"
cd /data/ndn825/Preview_model/review_tagging/resources_review/data/
rm -rf p_review_dataset.tar
tar -cf p_review_dataset.tar *.json
rm -rf /local_datasets/p_review_dataset/
mkdir /local_datasets/p_review_dataset/
tar -xf /data/ndn825/Preview_model/review_tagging/resources_review/data/p_review_dataset.tar -C /local_datasets/p_review_dataset/
echo "Data is ready!"

cd /data/ndn825/Preview_model/review_tagging

/data/$USER/anaconda3/bin/conda init
source ~/.bashrc
conda activate preview_model

echo "PWD"
pwd
echo "WHICH PYTHON"
which python
echo "HOSTNAME"
hostname

python ./src_review/do_parsingData.py --fp=$fp --save_p=$save_p --val_ratio=$val_ratio --test_ratio=$test_ratio --encoding=$encoding
python ./src_review/do_train.py --epochs=$epochs --init_model_path=$init_model_path --train_batch_size=$train_batch_size --valid_batch_size=$valid_batch_size --max_length=$max_length --need_birnn=$need_birnn --sentiment_drop_ratio=$sentiment_drop_ratio --aspect_drop_ratio=$aspect_drop_ratio --sentiment_in_feature=$sentiment_in_feature --aspect_in_feature=$aspect_in_feature --stop_patience=$stop_patience --train_fp=$train_fp --valid_fp=$valid_fp --base_path=$base_path --label_info_file=$label_info_file --out_model_path=$out_model_path

exit 0

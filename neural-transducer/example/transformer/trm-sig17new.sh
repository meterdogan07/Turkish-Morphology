#!/bin/bash
lang=$1
arch=transformer

res=low
lr=0.001
scheduler=warmupinvsqr
max_steps=2000000
warmup=4000
beta2=0.98   # 0.999
label_smooth=0 # 0.0
total_eval=5
bs=1 # 256

# transformer
layers=2
hs=256
embed_dim=8
nb_heads=1
dropout=${2:-0.3}

data_dir=data
ckpt_dir=checkpoints/transformer

python src/train.py \
    --dataset sigmorphon17task1 \
    --train $data_dir/trmor/trmor/train_final.txt\
    --dev $data_dir/trmor/trmor/train_final.txt \
    --test $data_dir/trmor/trmor/train_final.txt \
    --model $ckpt_dir/$arch/sigmorphon17-task1-dropout$dropout/$lang-$res-$decode \
    --embed_dim $embed_dim --src_hs $hs --trg_hs $hs --dropout $dropout --nb_heads $nb_heads \
    --label_smooth $label_smooth --total_eval $total_eval \
    --src_layer $layers --trg_layer $layers --max_norm 1 --lr $lr --shuffle \
    --arch $arch --gpuid 0 --estop 1e-8 --bs $bs --max_steps $max_steps \
    --scheduler $scheduler --warmup_steps $warmup --cleanup_anyway --beta2 $beta2 --bestacc

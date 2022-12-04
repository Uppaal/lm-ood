#!/usr/bin/env bash

#METHOD=$1
#OUT_DATA=$2

METHOD='GradNorm'
OUT_DATA='iNaturalist'
ID_DATASET='imdb'
OOD_DATASET='sst2'
LM_MODEL_NAME=roberta_base_${ID_DATASET}

python test_ood_text.py \
--name test_${METHOD}_${OUT_DATA} \
--in_datadir /nobackup-fast/ILSVRC-2012/val \
--out_datadir dataset/ood_data/${OUT_DATA} \
--model_path checkpoints/pretrained_models/BiT-S-R101x1-flat-finetune.pth.tar \
--batch 256 \
--logdir checkpoints/test_log \
--score ${METHOD} \
--id_data ${ID_DATASET} \
--ood_data ${OOD_DATASET} \
--model_name ${LM_MODEL_NAME}

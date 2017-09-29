#!/bin/sh

# preprocessing
python3 -m squad.prepro -s ../training/09292020/data/squad/ -t ../training/09292020/data/squad

# check memory
python3 -m basic.cli --mode train --noload --debug --data_dir ../training/09292020/data/squad --out_base_dir ../training/09292020/out/ --use_char_emb false --batch_size 1

# traing
python3 -m basic.cli --mode train --noload --data_dir ../training/09292020/data/squad --out_base_dir ../training/09292020/out/ --use_char_emb false --len_opt --cluster --batch_size 1

# test
python3 -m basic.cli --data_dir ../training/09292020/data/squad --out_base_dir ../training/09292020/out/ --use_char_emb false --len_opt --cluster --batch_size 1


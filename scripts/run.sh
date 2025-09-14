#!/bin/bash

python scripts/rate_distortion_batch.py \
    --data_dir data/treeoflife.interactomes.max_cc.rw1000_adj \
    --log_dir_base logs/logs_interactomes.max_cc.rw1000_debug \
    --script scripts/rate_distortion_debug.py \
    --num_workers 40

python scripts/rate_distortion_batch.py \
    --data_dir data/treeoflife.interactomes.max_cc.rw2000_adj \
    --log_dir_base logs/logs_interactomes.max_cc.rw2000_debug \
    --script scripts/rate_distortion_debug.py \
    --num_workers 16
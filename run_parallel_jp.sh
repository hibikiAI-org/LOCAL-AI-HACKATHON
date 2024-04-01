#!/bin/bash

# 使用するGPUの数を指定
NUM_GPUS=7

# マスターアドレスとポートを設定
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"

# プロセスIDを格納する配列
pids=()

# プロセスを並列に実行
for ((i=0; i<$NUM_GPUS; i++)); do
    export RANK=$i
    export LOCAL_RANK=$i
    export WORLD_SIZE=$NUM_GPUS

    python train_ms_jp_extra.py &
    pids+=($!)
done

# Ctrl+Cが押されたときの処理
trap 'kill "${pids[@]}"' INT

# すべてのプロセスが終了するまで待機
wait
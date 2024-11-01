export CUDA_VISIBLE_DEVICES=7
export DATA_PATH=./sim_data/
export OUTPUT_PATH=./output/
export LD_LIBRARY_PATH=/home/users/tmp/tagging/dependency/openssl/lib:$LD_LIBRARY_PATH

      python3 train_copy.py \
    --data_dir ${DATA_PATH} \
    --output_dir ${OUTPUT_PATH} \
    --max_seq_len 20 \
    --learning_rate 5e-5 \
    --num_train_epochs 2 \
    --logging_steps 5 \
    --save_steps 100 \
    --batch_size 2 \
    --warmup_proportion 0.1 \
    --seed 42 \
    --device cpu
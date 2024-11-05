INPUT_FILE=$1
# export LD_LIBRARY_PATH=./miniconda3/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/home/users/tmp/tagging/dependency/openssl/lib:$LD_LIBRARY_PATH

python3 data_process_sim.py $INPUT_FILE

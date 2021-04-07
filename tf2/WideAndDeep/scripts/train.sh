export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${JAVA_HOME}/jre/lib/amd64/server
CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob)
# export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
# export KMP_BLOCKTIME=30
# export OMP_NUM_THREADS=36
# export KMP_SETTINGS=1
set -x
set -e

# horovodrun -np 8 -H sr112:2,sr610:2,sr612:2,sr613:2 --start-timeout 300 --timeline-filename timeline.json \
# --mpi-args="-x OMP_NUM_THREADS=18 --allow-run-as-root --map-by socket --report-bindings --oversubscribe" \
# /root/sw/miniconda3/envs/wd2/bin/python main.py \
#   --train_data_pattern hdfs://sr112:9001/outbrain2/tfrecords/train/part* \
#   --eval_data_pattern hdfs://sr112:9001/outbrain2/tfrecords/eval/part* \
#   --model_dir /mnt/nvm6/wd/checkpoints2 \
#   --transformed_metadata_path /outbrain2/tfrecords \
#   --cpu \
#   --num_epochs 10

# time mpirun -x OMP_NUM_THREADS=18 --allow-run-as-root --map-by socket --report-bindings --oversubscribe -np 2 -H sr112:2 \
# /root/sw/miniconda3/envs/wd2/bin/python main.py \
#   --train_data_pattern hdfs://sr112:9001/outbrain2/tfrecords/train/part* \
#   --eval_data_pattern hdfs://sr112:9001/outbrain2/tfrecords/eval/part* \
#   --model_dir /mnt/nvm6/wd/checkpoints2 \
#   --transformed_metadata_path /outbrain2/tfrecords \
#   --cpu \
#   --num_epochs 1 \
#   --deep_warmup_epochs 0 \
#   --deep_hidden_units 1024 512 256 \
#   --training_set_size 13107200
#   --amp

time mpirun -x OMP_NUM_THREADS=18 --allow-run-as-root --map-by socket --report-bindings --oversubscribe -np 2 -H sr113:2 \
/root/sw/miniconda3/envs/wd2/bin/python main.py \
  --train_data_pattern hdfs://sr113:9001/outbrain2/tfrecords/train/part* \
  --eval_data_pattern hdfs://sr113:9001/outbrain2/tfrecords/eval/part* \
  --model_dir /mnt/nvm6/wd/checkpoints2 \
  --transformed_metadata_path /outbrain2/tfrecords \
  --cpu

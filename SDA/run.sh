# WnD
python main.py --python_executable /root/sw/miniconda3/envs/wd2/bin/python --iface eth3 --ccl_worker_num 2 --train_data_pattern "/mnt/sdd/outbrain2/tfrecords/train/part*" --eval_data_pattern "/mnt/sdd/outbrain2/tfrecords/eval/part*" --model_dir /mnt/nvm6/wd/checkpoints2 --transformed_metadata_path /outbrain2/tfrecords --linear_learning_rate 0.8

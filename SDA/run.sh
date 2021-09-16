# WnD
/root/sw/miniconda3/envs/wd2/bin/python main.py --python_executable /root/sw/miniconda3/envs/wd2/bin/python --iface eth3 --ccl_worker_num 2 \
--train_data_pattern "/mnt/sdd/outbrain2/tfrecords/train/part*" --eval_data_pattern "/mnt/sdd/outbrain2/tfrecords/eval/part*" \
--model_dir /mnt/nvm6/wd/checkpoints2 --transformed_metadata_path /outbrain2/tfrecords \
--global_batch_size 524288 --eval_batch_size 524288 --num_epochs 20 \
--deep_warmup_epochs 6 --deep_hidden_units 1024 512 256 --deep_dropout 0.1 --deep_learning_rate 0.00048 --linear_learning_rate 0.8 \

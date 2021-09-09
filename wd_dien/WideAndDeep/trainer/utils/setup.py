# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os

# import horovod.tensorflow.keras as hvd
import tensorflow as tf
import tensorflow_transform as tft
from data.outbrain.dataloader import RawBinaryDataset, DatasetMetadata, data_input_fn, CriteoBinDataset
from trainer.utils.data_iterator import DataIterator

TRAIN_SET_SIZE = 13170 * 256

def init_cpu(args, logger):
    # hvd.init()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    init_logger(
        full=True,
        args=args,
        logger=logger
    )

    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(32)
    
    if args.amp:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.experimental.set_policy(policy)

    if args.xla:
        tf.config.optimizer.set_jit(True)
    
    logger.warning('--gpu flag not set, running computation on CPU')


def init_logger(args, full, logger):
    if full:
        logger.setLevel(logging.INFO)
        log_path = os.path.join(args.results_dir, args.log_filename)
        os.makedirs(args.results_dir, exist_ok=True)
        logger.warning('command line arguments: {}'.format(json.dumps(vars(args))))
        if not os.path.exists(args.results_dir):
            os.mkdir(args.results_dir)

        with open('{}/args.json'.format(args.results_dir), 'w') as f:
            json.dump(vars(args), f, indent=4)
    else:
        logger.setLevel(logging.ERROR)

    # dllogger.log(data=vars(args), step='PARAMETER')


def create_input_pipelines(train_dataset_path, eval_dataset_path, train_batch_size, eval_batch_size, 
        train_file="/mnt/nvm1/dien_raw_data/local_train_splitByUser",
        test_file="/mnt/nvm1/dien_raw_data/local_test_splitByUser",
        uid_voc="/mnt/nvm1/dien_raw_data/uid_voc.pkl",
        mid_voc="/mnt/nvm1/dien_raw_data/mid_voc.pkl",
        cat_voc="/mnt/nvm1/dien_raw_data/cat_voc.pkl",
        maxlen=100):
    train_data = DataIterator(
        train_file, uid_voc, mid_voc, cat_voc, train_batch_size, maxlen, shuffle_each_epoch=True)
    test_data = DataIterator(
        test_file, uid_voc, mid_voc, cat_voc, eval_batch_size, maxlen)
    n_uid, n_mid, n_cat = train_data.get_n()

    return train_data, test_data, n_uid, n_mid, n_cat


def create_config(args):
    # assert not (args.cpu and args.amp), \
    #     'Automatic mixed precision conversion works only with GPU'
    assert not args.benchmark or args.benchmark_warmup_steps < args.benchmark_steps, \
        'Number of benchmark steps must be higher than warmup steps'
    logger = logging.getLogger('tensorflow')

    if args.cpu:
        init_cpu(args, logger)

    # num_gpus = hvd.size()
    # gpu_id = hvd.rank()
    # train_batch_size = args.global_batch_size // num_gpus
    # eval_batch_size = args.eval_batch_size // num_gpus
    steps_per_epoch = TRAIN_SET_SIZE / args.global_batch_size
    # steps_per_epoch = 13500
    eval_point = 100

    feature_spec = tft.TFTransformOutput(
        '/root'
    ).transformed_feature_spec()
    train_dataset, test_dataset, n_uid, n_mid, n_cat = create_input_pipelines(args.train_data_pattern, args.eval_data_pattern, args.global_batch_size, args.eval_batch_size)

    config = {
        'steps_per_epoch': steps_per_epoch,
        'train_dataset': train_dataset,
        'eval_dataset': test_dataset, 
        'eval_point': eval_point
    }

    return config

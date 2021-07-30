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
from sys import getdefaultencoding
from threading import local

import dllogger
import horovod.tensorflow.keras as hvd
import tensorflow as tf
import tensorflow_transform as tft
from data.outbrain.dataloader import CriteoBinDataset, DataParallelSplitter, get_device_mapping
from data.outbrain.features import PREBATCH_SIZE


def init_cpu(args, logger):
    hvd.init()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    init_logger(
        full=hvd.rank() == 0,
        args=args,
        logger=logger
    )

    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(16)
    
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



def create_input_pipelines(train_dataset_path, eval_dataset_path, train_batch_size, eval_batch_size, data_splitter):
    # global_table_sizes= [7912889, 33823, 17139, 7339, 20046, 4, 7105, 1382, 63, 5554114, 582469, 245828, 11, 2209, 10667, 104, 4, 968, 15, 8165896, 2675940, 7156453, 302516, 12022, 97, 35]
    global_table_sizes = [39884406,39043,17289,7420,20263,3,7120,1543,63,38532951,2953546,403346,10,2208,11938,155,4,976,14,39979771,25641295, 39664984,585935,12972,108,36]
    feature_mapping = get_device_mapping(global_table_sizes, hvd.size())
    
    rank_table_sizes = [sum([global_table_sizes[i] for i in bucket]) for bucket in feature_mapping.rank_to_categorical_ids]
    print(feature_mapping)
    print(rank_table_sizes)

    local_tables = feature_mapping.rank_to_categorical_ids[hvd.rank()]
    numerical_features = 13

    train_dataset = CriteoBinDataset(train_dataset_path, train_batch_size, numerical_features, local_tables, global_table_sizes, data_splitter)
    test_dataset = CriteoBinDataset(eval_dataset_path, eval_batch_size, numerical_features, local_tables, global_table_sizes, data_splitter)
    return train_dataset, test_dataset, global_table_sizes, feature_mapping


def create_config(args):
    logger = logging.getLogger('tensorflow')

    if args.cpu:
        init_cpu(args, logger)

    splitter = DataParallelSplitter(args.global_batch_size)
    train_dataset, test_dataset, embedding_sizes, feature_mapping = create_input_pipelines(args.train_dataset_path, args.eval_dataset_path, args.global_batch_size, args.eval_batch_size, splitter)
    steps_per_epoch = train_dataset.__len__()

    config = {
        'steps_per_epoch': steps_per_epoch,
        'train_dataset': train_dataset,
        'eval_dataset': test_dataset, 
        'embedding_sizes': embedding_sizes,
        'feature_mapping': feature_mapping

    }

    return config

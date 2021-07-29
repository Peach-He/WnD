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
#


from textwrap import indent
import time
from pyspark.sql.functions import size
import tensorflow as tf

import concurrent
import math
import os
import queue
import json
from collections import deque, namedtuple

import numpy as np
from typing import Optional, Sequence, Tuple, Any, Dict

from functools import partial
import struct
from multiprocessing import cpu_count
import horovod.tensorflow.keras as hvd

def argsort(sequence, reverse: bool = False):
    idx_pairs = [(x, i) for i, x in enumerate(sequence)]
    sorted_pairs = sorted(idx_pairs, key=lambda pair: pair[0], reverse=reverse)
    return [i for _, i in sorted_pairs]


def distribute_to_buckets(sizes, buckets_num):
    def sum_sizes(indices):
        return sum(sizes[i] for i in indices)

    max_bucket_size = math.ceil(len(sizes) / buckets_num)
    idx_sorted = deque(argsort(sizes, reverse=True))
    buckets = [[] for _ in range(buckets_num)]
    final_buckets = []

    while idx_sorted:
        bucket = buckets[0]
        bucket.append(idx_sorted.popleft())

        if len(bucket) == max_bucket_size:
            final_buckets.append(buckets.pop(0))

        buckets.sort(key=sum_sizes)

    final_buckets += buckets

    return final_buckets


MultiGpuMetadata = namedtuple('MultiGpuMetadata',
                              ['rank_to_categorical_ids','rank_to_feature_count'])


def get_device_mapping(embedding_sizes, num_gpus):


    # gpu_buckets = distribute_to_buckets(embedding_sizes, num_gpus)
    # set cat features per rank manually
    gpu_buckets = [[19], [0], [21], [9], [20], [10], [22], [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25]]

    vectors_per_gpu = [len(bucket) for bucket in gpu_buckets]

    return MultiGpuMetadata(rank_to_categorical_ids=gpu_buckets,
                            rank_to_feature_count=vectors_per_gpu)


class DataParallelSplitter:
    def __init__(self, batch_size):
        local_batch_size = batch_size // hvd.size()
        batch_size_per_worker = [local_batch_size] * hvd.size()
        indices = tuple(np.cumsum([0] + list(batch_size_per_worker)))
        self.begin_idx = indices[hvd.rank()]
        self.end_idx = indices[hvd.rank() + 1]

    def __call__(self, input):
        output = input[self.begin_idx: self.end_idx]
        return output

class CriteoBinDataset:
    """Binary version of criteo dataset."""

    def __init__(self, 
                data_file,
                batch_size=1, 
                numerical_features=0,
                categorical_features=None,
                categorical_features_sizes=[],
                data_splitter=None,
                bytes_per_feature=4):
        # dataset
        self.label = 1   # single target
        self.dense_features = numerical_features  # dense  features
        self.sparse_features = len(categorical_features_sizes)  # sparse features
        self.total_features = self.label + self.dense_features + self.sparse_features
        self.data_splitter = data_splitter
        self.categorical_features = categorical_features

        self.batch_size = batch_size
        self.bytes_per_batch = (bytes_per_feature * self.total_features * batch_size)

        data_file_size = os.path.getsize(data_file)
        self.num_batches = math.ceil(data_file_size / self.bytes_per_batch)

        bytes_per_sample = bytes_per_feature * self.total_features
        self.num_samples = data_file_size // bytes_per_sample

        if hvd.size() > 1:
            self.bytes_per_rank = self.bytes_per_batch // hvd.size()
        else:
            self.bytes_per_rank = self.bytes_per_batch

        if hvd.size() > 1 and self.num_batches * self.bytes_per_batch > data_file_size:
            last_batch_size = (data_file_size % self.bytes_per_batch) // bytes_per_sample
            self.bytes_last_batch = last_batch_size // hvd.size() * bytes_per_sample
        else:
            self.bytes_last_batch = self.bytes_per_rank

        if self.bytes_last_batch == 0:
            self.num_batches = self.num_batches - 1
            self.bytes_last_batch = self.bytes_per_rank

        print('data file:', data_file, 'number of batches:', self.num_batches)
        self.file = open(data_file, 'rb')

        # self.counts= [7912889, 33823, 17139, 7339, 20046, 4, 7105, 1382, 63, 5554114, 582469, 245828, 11, 2209, 10667, 104, 4, 968, 15, 8165896, 2675940, 7156453, 302516, 12022, 97, 35]
        # self.counts = [39884406,39043,17289,7420,20263,3,7120,1543,63,38532951,2953546,403346,10,2208,11938,155,4,976,14,39979771,25641295, 39664984,585935,12972,108,36]

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if idx >= self.num_batches:
            raise IndexError()
        my_rank = hvd.rank() if hvd.size() > 1 else 0
        rank_size = self.bytes_last_batch if idx == (self.num_batches - 1) else self.bytes_per_rank 
        self.file.seek(idx * self.bytes_per_batch, 0)
        raw_data = self.file.read(self.bytes_per_batch)

        array = np.frombuffer(raw_data, dtype=np.int32).reshape(-1, self.total_features)

        if self.dense_features == 0:
            numerical_features = -1
        else:
            numerical_features = array[:, 1:self.dense_features].view(dtype=np.float32)
            numerical_features = tf.convert_to_tensor(numerical_features)
            numerical_features = self.data_splitter(numerical_features)

        categorical_features = []
        for index in self.categorical_features:
            categorical_features.append(tf.convert_to_tensor(array[:, index+14]))  #if index start from 0

        click = tf.convert_to_tensor(array[:, 0], dtype=tf.float32)
        click = self.data_splitter(click)
        
        # time.sleep(1)
        return (numerical_features, categorical_features), click
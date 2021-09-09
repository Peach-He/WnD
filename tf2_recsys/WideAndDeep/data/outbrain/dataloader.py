# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
# Modifications copyright Intel
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

import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np
import os

from data.outbrain.features import get_features_keys, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, target, feature_list

class RecsysBinDataset:

    def __init__(self, data_file, batch_size=1, bytes_per_feature=4):
        # dataset
        self.tar_fea = target   # single target
        self.den_fea = NUMERIC_COLUMNS  # 13 dense  features
        self.spa_fea = CATEGORICAL_COLUMNS  # 26 sparse features
        self.tot_fea = len(feature_list)

        self.batch_size = batch_size
        self.bytes_per_batch = (bytes_per_feature * self.tot_fea * batch_size)

        data_file_size = os.path.getsize(data_file)
        self.num_batches = data_file_size // self.bytes_per_batch

        bytes_per_sample = bytes_per_feature * self.tot_fea
        self.num_samples = data_file_size // bytes_per_sample

        if hvd.size() > 1:
            self.bytes_per_rank = self.bytes_per_batch // hvd.size()
        else:
            self.bytes_per_rank = self.bytes_per_batch

        print('data file:', data_file, 'number of batches:', self.num_batches)
        self.file = open(data_file, 'rb')

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if idx >= self.num_batches:
            raise IndexError()
        my_rank = hvd.rank() if hvd.size() > 1 else 0
        rank_size = self.bytes_per_rank 
        self.file.seek(idx * self.bytes_per_batch + rank_size * my_rank, 0)
        raw_data = self.file.read(rank_size)

        array = np.frombuffer(raw_data, dtype=np.int32).reshape(-1, self.tot_fea)
        features = {}
        for i, f in enumerate(self.den_fea):
            numerical_feature = array[:, i+32].view(dtype=np.float32)
            features[f] = tf.convert_to_tensor(numerical_feature)

        for i, f in enumerate(self.spa_fea):
            features[f] = tf.convert_to_tensor(array[:, i+4])
        click = tf.convert_to_tensor(array[:, 1])
        
        return features, click

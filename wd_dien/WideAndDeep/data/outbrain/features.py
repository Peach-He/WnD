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

import logging

import tensorflow as tf


LABEL_COLUMN = 'target'

CATEGORICAL_COLUMNS = ['uid', 'mid', 'cat', 'mid_his', 'cat_his']

HASH_BUCKET_SIZES = {
    'uid': 1686578, 
    'mid': 2052973,
    'cat' : 2661, 
    'mid_his': 2052973, 
    'cat_his': 2661, 
    'noclk_mid': 2052973, 
    'noclk_cat': 2661
}

EMBEDDING_DIMENSION = 32


def get_feature_columns():
    logger = logging.getLogger('tensorflow')
    wide_columns, deep_columns = [], []

    for column_name in CATEGORICAL_COLUMNS:
        categorical_column = tf.feature_column.categorical_column_with_identity(
            column_name, num_buckets=HASH_BUCKET_SIZES[column_name])
        # categorical_column = tf.feature_column.categorical_column_with_hash_bucket(column_name, 1000, tf.int32)
        # wrapped_wide_column = tf.feature_column.embedding_column(categorical_column, 1)
        wrapped_column = tf.feature_column.embedding_column(
            categorical_column,
            dimension=EMBEDDING_DIMENSION,
            combiner='mean')
        # wide_columns.append(wrapped_wide_column)
        wide_columns.append(categorical_column)
        deep_columns.append(wrapped_column)

    logger.warning('deep columns: {}'.format(len(deep_columns)))
    logger.warning('wide columns: {}'.format(len(wide_columns)))
    logger.warning('wide&deep intersection: {}'.format(len(set(wide_columns).intersection(set(deep_columns)))))

    return wide_columns, deep_columns

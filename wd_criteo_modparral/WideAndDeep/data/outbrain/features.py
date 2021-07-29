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


LABEL_COL = 0
INT_COLS = list(range(1, 14))
CAT_COLS = list(range(14, 40))

PREBATCH_SIZE = 4096

LABEL_COLUMN = 'c0'

NUMERIC_COLUMNS = ['c%d' % i for i in INT_COLS]

CATEGORICAL_COLUMNS = ['c%d' % i for i in CAT_COLS]

### spark feature size
counts = [7912889, 33823, 17139, 7339, 20046, 4, 7105, 1382, 63, 5554114, 582469, 245828, 11, 2209, 10667, 104, 4, 968, 15, 8165896, 2675940, 7156453, 302516, 12022, 97, 35]
HASH_BUCKET_SIZES = {
    'c14': 7912889,
    'c15': 33823,
    'c16': 17139,
    'c17': 7339,
    'c18': 20046,
    'c19': 4,
    'c20': 7105,
    'c21': 1382,
    'c22': 63,
    'c23': 5554114,
    'c24': 582469,
    'c25': 245828,
    'c26': 11,
    'c27': 2209,
    'c28': 10667,
    'c29': 104,
    'c30': 4,
    'c31': 968,
    'c32': 15,
    'c33': 8165896,
    'c34': 2675940,
    'c35': 7156453,
    'c36': 302516,
    'c37': 12022,
    'c38': 97,
    'c39': 35
}

### binary numpy feature size
# counts = [39884406,39043,17289,7420,20263,3,7120,1543,63,38532951,2953546,403346,10,2208,11938,155,4,976,14,39979771,25641295, 39664984,585935,12972,108,36]
# HASH_BUCKET_SIZES = {
#     'c14': 39884406,
#     'c15': 39043,
#     'c16': 17289,
#     'c17': 7420,
#     'c18': 20263,
#     'c19': 3,
#     'c20': 7120,
#     'c21': 1543,
#     'c22': 63,
#     'c23': 38532951,
#     'c24': 2953546,
#     'c25': 403346,
#     'c26': 10,
#     'c27': 2208,
#     'c28': 11938,
#     'c29': 155,
#     'c30': 4,
#     'c31': 976,
#     'c32': 14,
#     'c33': 39979771,
#     'c34': 25641295,
#     'c35': 39664984,
#     'c36': 585935,
#     'c37': 12972,
#     'c38': 108,
#     'c39': 36
# }


HASH_BUCKET_SIZE = 10000
EMBEDDING_DIMENSION = 16
# EMBEDDING_DIMENSION = {
#     'c14': 64,
#     'c15': 16,
#     'c16': 16,
#     'c17': 16,
#     'c18': 16,
#     'c19': 2,
#     'c20': 16,
#     'c21': 8,
#     'c22': 4,
#     'c23': 64,
#     'c24': 32,
#     'c25': 32,
#     'c26': 2,
#     'c27': 8,
#     'c28': 16,
#     'c29': 4,
#     'c30': 2,
#     'c31': 8,
#     'c32': 2,
#     'c33': 64,
#     'c34': 64,
#     'c35': 64,
#     'c36': 32,
#     'c37': 16,
#     'c38': 4,
#     'c39': 4
# }

def get_feature_columns():
    logger = logging.getLogger('tensorflow')
    wide_columns, deep_columns = [], []

    numerics = [tf.feature_column.numeric_column(column_name, shape=(1,), dtype=tf.float32)
                for column_name in NUMERIC_COLUMNS]

    wide_columns.extend(numerics)
    deep_columns.extend(numerics)

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

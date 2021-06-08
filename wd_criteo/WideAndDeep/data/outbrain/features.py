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

sparse_embedding = [
    {"name": "C14", "indice": 0, "voc_len": 179729, "dim": 212},
    {"name": "C15", "indice": 1, "voc_len": 12325, "dim": 56},
    {"name": "C16", "indice": 2, "voc_len": 11780, "dim": 55},
    {"name": "C17", "indice": 3, "voc_len": 4156, "dim": 33},
    {"name": "C18", "indice": 4, "voc_len": 10576, "dim": 52},
    {"name": "C22", "indice": 8, "voc_len": 5850, "dim": 39},
    {"name": "C23", "indice": 9, "voc_len": 1139, "dim": 17},
    {"name": "C62", "indice": 48, "voc_len": 136422, "dim": 185},
    {"name": "C63", "indice": 49, "voc_len": 33820, "dim": 92},
    {"name": "C64", "indice": 50, "voc_len": 34916, "dim": 94},
    {"name": "C75", "indice": 61, "voc_len": 1841, "dim": 22},
    {"name": "C76", "indice": 62, "voc_len": 5445, "dim": 37},
    {"name": "C137", "indice": 123, "voc_len": 615, "dim": 13},
    {"name": "C152", "indice": 138, "voc_len": 187781, "dim": 217},
    {"name": "C153", "indice": 139, "voc_len": 80021, "dim": 142},
    {"name": "C154", "indice": 140, "voc_len": 1655497, "dim": 644},
    {"name": "C155", "indice": 141, "voc_len": 29741, "dim": 87},
    {"name": "C156", "indice": 142, "voc_len": 7693, "dim": 44},
]

counts= [7912889, 33823, 17139, 7339, 20046, 4, 7105, 1382, 63, 5554114, 582469, 245828, 11, 2209, 10667, 104, 4, 968, 15, 8165896, 2675940, 7156453, 302516, 12022, 97, 35]
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

EMBEDDING_DIMENSIONS = {
    'doc_event_id': 128,
    'ad_id': 128,
    'doc_id': 128,
    'doc_ad_source_id': 64,
    'doc_event_source_id': 64,
    'event_geo_location': 64,
    'ad_advertiser': 64,
    'event_country_state': 64,
    'doc_ad_publisher_id': 64,
    'doc_event_publisher_id': 64,
    'event_country': 64,
    'event_platform': 16,
    'campaign_id': 128
}

# EMBEDDING_TABLE_SHAPES = {
#     column: (HASH_BUCKET_SIZES[column], EMBEDDING_DIMENSIONS[column]) for column in CATEGORICAL_COLUMNS
# }

HASH_BUCKET_SIZE = 10000
EMBEDDING_DIMENSION = 64

def get_features_keys():
    return CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + [DISPLAY_ID_COLUMN]


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
        wrapped_column = tf.feature_column.embedding_column(
            categorical_column,
            dimension=EMBEDDING_DIMENSION,
            combiner='mean')

        wide_columns.append(categorical_column)
        deep_columns.append(wrapped_column)



    logger.warning('deep columns: {}'.format(len(deep_columns)))
    logger.warning('wide columns: {}'.format(len(wide_columns)))
    logger.warning('wide&deep intersection: {}'.format(len(set(wide_columns).intersection(set(deep_columns)))))

    return wide_columns, deep_columns

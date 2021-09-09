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

import tensorflow as tf

from data.outbrain.features import get_feature_columns, CATEGORICAL_COLUMNS, HASH_BUCKET_SIZES, EMBEDDING_DIMENSION

def wide_deep_model_orig(args):

    wide_weighted_outputs = []
    numeric_dense_inputs = []
    wide_columns_dict = {}
    deep_columns_dict = {}
    features = {}

    for col in CATEGORICAL_COLUMNS:
        if 'his' in col:
            features[col] = tf.keras.Input(shape=(None,), batch_size=None, name=col, dtype=tf.int32, sparse=False)
        else:
            features[col] = tf.keras.Input(shape=(1,),
                                           batch_size=None,
                                           name=col,
                                           dtype=tf.int32,
                                           sparse=False)
    linear_embs = {}
    deep_embs = {}
    # linear_embs['uid'] = tf.keras.layers.Embedding(HASH_BUCKET_SIZES['uid'], 1, input_length=1)
    # linear_embs['mid'] = tf.keras.layers.Embedding(HASH_BUCKET_SIZES['mid'], 1, input_length=1)
    # linear_embs['cat'] = tf.keras.layers.Embedding(HASH_BUCKET_SIZES['cat'], 1, input_length=1)
    deep_embs['uid'] = tf.keras.layers.Embedding(HASH_BUCKET_SIZES['uid'], EMBEDDING_DIMENSION)
    deep_embs['mid'] = tf.keras.layers.Embedding(HASH_BUCKET_SIZES['mid'], EMBEDDING_DIMENSION)
    deep_embs['cat'] = tf.keras.layers.Embedding(HASH_BUCKET_SIZES['cat'], EMBEDDING_DIMENSION)
    # deep_embs['mid_his'] = tf.keras.layers.Embedding(HASH_BUCKET_SIZES['mid_his'], EMBEDDING_DIMENSION)
    # deep_embs['cat_his'] = tf.keras.layers.Embedding(HASH_BUCKET_SIZES['cat_his'], EMBEDDING_DIMENSION)

    uid_emb = tf.keras.layers.Flatten()(deep_embs['uid'](features['uid']))
    mid_emb = tf.keras.layers.Flatten()(deep_embs['mid'](features['mid']))
    cat_emb = tf.keras.layers.Flatten()(deep_embs['cat'](features['cat']))
    mid_his_emb = deep_embs['mid'](features['mid_his'])
    cat_his_emb = deep_embs['cat'](features['cat_his'])
    
    mid_his_emb_sum = tf.reduce_sum(mid_his_emb, 1)
    cat_his_emb_sum = tf.reduce_sum(cat_his_emb, 1)

    # tf.print(f'uid_emb: {uid_emb.shape}')
    # tf.print(f'mid_emb: {mid_emb.shape}')
    # tf.print(f'cat_emb: {cat_emb.shape}')
    # tf.print(f'mid_his_emb: {mid_his_emb.shape}')
    # tf.print(f'cat_his_emb: {cat_his_emb.shape}')
    # tf.print(f'item_his_emb_sum: {item_his_emb_sum.shape}')

    categorical_output_contrib = tf.keras.layers.add([uid_emb, mid_emb, cat_emb, mid_his_emb_sum, cat_his_emb_sum],
                                                     name='categorical_output')

    # dnn = ScalarDenseFeatures(deep_columns, name='deep_embedded')(features)
    dnn = tf.keras.layers.concatenate([uid_emb, mid_emb, cat_emb, mid_his_emb_sum, cat_his_emb_sum])
    for unit_size in args.deep_hidden_units:
        dnn = tf.keras.layers.Dense(units=unit_size, activation='relu')(dnn)
        dnn = tf.keras.layers.Dropout(rate=args.deep_dropout)(dnn)
        dnn = tf.keras.layers.BatchNormalization()(dnn)
    dnn = tf.keras.layers.Dense(units=2)(dnn)
    dnn_model = tf.keras.Model(inputs=features,
                               outputs=dnn)
    linear_output = tf.keras.layers.Dense(2)(categorical_output_contrib)

    linear_model = tf.keras.Model(inputs=features,
                                  outputs=linear_output)

    model = tf.keras.experimental.WideDeepModel(
        linear_model, dnn_model, activation='sigmoid')

    return model
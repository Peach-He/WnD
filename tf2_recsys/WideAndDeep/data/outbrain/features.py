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

feature_list = ['reply',
 'retweet',
 'retweet_comment',
 'like',
 'b_follows_a',
 'a_follower_count',
 'a_following_count',
 'b_follower_count',
 'b_following_count',
 'dt_dow',
 'dt_minute',
 'len_hashtags',
 'len_links',
 'len_domains',
 'tw_len_media',
 'tw_len_photo',
 'tw_len_video',
 'tw_len_gif',
 'tw_len_quest',
 'tw_len_token',
 'tw_count_capital_words',
 'tw_count_excl_quest_marks',
 'tw_count_special1',
 'tw_count_hash',
 'tw_last_quest',
 'tw_len_retweet',
 'tw_len_rt',
 'tw_count_at',
 'tw_count_words',
 'tw_count_char',
 'tw_rt_count_words',
 'tw_rt_count_char',
 'TE_a_user_id_reply',
 'TE_a_user_id_retweet',
 'TE_a_user_id_retweet_comment',
 'TE_a_user_id_like',
 'TE_b_user_id_reply',
 'TE_b_user_id_retweet',
 'TE_b_user_id_retweet_comment',
 'TE_b_user_id_like',
 'TE_tweet_type_reply',
 'TE_tweet_type_retweet',
 'TE_tweet_type_retweet_comment',
 'TE_tweet_type_like',
 'TE_language_reply',
 'TE_language_retweet',
 'TE_language_retweet_comment',
 'TE_language_like',
 'TE_media_reply',
 'TE_media_retweet',
 'TE_media_retweet_comment',
 'TE_media_like',
 'TE_tw_word0_reply',
 'TE_tw_word0_retweet',
 'TE_tw_word0_retweet_comment',
 'TE_tw_word0_like',
 'TE_a_user_id_language_tweet_type_reply',
 'TE_a_user_id_language_tweet_type_retweet',
 'TE_a_user_id_language_tweet_type_retweet_comment',
 'TE_a_user_id_language_tweet_type_like',
 'TE_b_user_id_language_tweet_type_reply',
 'TE_b_user_id_language_tweet_type_retweet',
 'TE_b_user_id_language_tweet_type_retweet_comment',
 'TE_b_user_id_language_tweet_type_like',
 'TE_language_tweet_type_media_reply',
 'TE_language_tweet_type_media_retweet',
 'TE_language_tweet_type_media_retweet_comment',
 'TE_language_tweet_type_media_like',
 'TE_a_user_id_b_user_id_reply',
 'TE_a_user_id_b_user_id_retweet',
 'TE_a_user_id_b_user_id_retweet_comment',
 'TE_a_user_id_b_user_id_like',
 'ff_a_ratio',
 'ff_b_ratio']

target = feature_list[1]
CATEGORICAL_COLUMNS = feature_list[4:32]
NUMERIC_COLUMNS = feature_list[32:74]
CAT_VOC = [2, 202952, 50897, 68755, 28560, 7, 1440, 48, 11, 11, 5, 5, 5, 3, 161, 482, 166, 232, 145, 55, 5, 8, 19, 53, 293, 1004, 6, 20]

# EMBEDDING_SIZES = dict(zip(CATEGORICAL_COLUMNS, CAT_VOC))
# EMBEDDING_DIMENSION = 16
EMBEDDING_SIZES = {
    'b_follows_a': 2, 
    'a_follower_count': 202952, 
    'a_following_count': 50897, 
    'b_follower_count': 68755, 
    'b_following_count': 28560, 
    'dt_dow': 7, 
    'dt_minute': 1440, 
    'len_hashtags': 48, 
    'len_links': 11, 
    'len_domains': 11, 
    'tw_len_media': 5, 
    'tw_len_photo': 5, 
    'tw_len_video': 5, 
    'tw_len_gif': 3, 
    'tw_len_quest': 161, 
    'tw_len_token': 482, 
    'tw_count_capital_words': 166, 
    'tw_count_excl_quest_marks': 232, 
    'tw_count_special1': 145, 
    'tw_count_hash': 55, 
    'tw_last_quest': 5, 
    'tw_len_retweet': 8, 
    'tw_len_rt': 19, 
    'tw_count_at': 53, 
    'tw_count_words': 293, 
    'tw_count_char': 1004, 
    'tw_rt_count_words': 6, 
    'tw_rt_count_char': 20}

EMBEDDING_DIMENSION = {
    'b_follows_a': 4, 
    'a_follower_count': 32, 
    'a_following_count': 16, 
    'b_follower_count': 32, 
    'b_following_count': 16, 
    'dt_dow': 4, 
    'dt_minute': 8, 
    'len_hashtags': 4, 
    'len_links': 4, 
    'len_domains': 4, 
    'tw_len_media': 4, 
    'tw_len_photo': 4, 
    'tw_len_video': 4, 
    'tw_len_gif': 4, 
    'tw_len_quest': 4, 
    'tw_len_token': 8, 
    'tw_count_capital_words': 4, 
    'tw_count_excl_quest_marks': 4, 
    'tw_count_special1': 4, 
    'tw_count_hash': 4, 
    'tw_last_quest': 4, 
    'tw_len_retweet': 4, 
    'tw_len_rt': 4, 
    'tw_count_at': 4, 
    'tw_count_words': 8, 
    'tw_count_char': 8, 
    'tw_rt_count_words': 4, 
    'tw_rt_count_char': 4}

def get_features_keys():
    return CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def get_feature_columns():
    logger = logging.getLogger('tensorflow')
    wide_columns, deep_columns = [], []

    for column_name in CATEGORICAL_COLUMNS:
        if column_name in EMBEDDING_SIZES:
            categorical_column = tf.feature_column.categorical_column_with_identity(
                column_name, num_buckets=EMBEDDING_SIZES[column_name])
            wrapped_column = tf.feature_column.embedding_column(
                categorical_column,
                dimension=EMBEDDING_DIMENSION[column_name],
                combiner='mean')
        else:
            raise ValueError(f'Unexpected categorical column found {column_name}')

        wide_columns.append(categorical_column)
        deep_columns.append(wrapped_column)

    numerics = [tf.feature_column.numeric_column(column_name, shape=(1,), dtype=tf.float32)
                for column_name in NUMERIC_COLUMNS]

    wide_columns.extend(numerics)
    deep_columns.extend(numerics)

    logger.warning('deep columns: {}'.format(len(deep_columns)))
    logger.warning('wide columns: {}'.format(len(wide_columns)))
    logger.warning('wide&deep intersection: {}'.format(len(set(wide_columns).intersection(set(deep_columns)))))

    return wide_columns, deep_columns

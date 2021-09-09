import os
import requests
import json
import tensorflow as tf
import tensorflow_transform as tft
from data.outbrain.dataloader import train_input_fn, eval_input_fn
from data.outbrain.features import PREBATCH_SIZE
from data.outbrain.features import DISPLAY_ID_COLUMN
import time


predict_url = 'http://sr112:8501/v1/models/wnd:predict'
status_url = 'http://sr112:8501/v1/models/wnd'
metadata_url = 'http://sr112:8501/v1/models/wnd/metadata'
eval_data_pattern = '/mnt/sdd/outbrain2/tfrecords/eval/part*'
transformed_metadata_path = '/outbrain2/tfrecords'
eval_batch_size = 524288 #524288


def make_predict():
    feature_spec = tft.TFTransformOutput(
        transformed_metadata_path
    ).transformed_feature_spec()

    input_fn = eval_input_fn(
        num_gpus=1,
        id=0,
        repeat=1,
        filepath_pattern=eval_data_pattern,
        feature_spec=feature_spec,
        records_batch_size=eval_batch_size // PREBATCH_SIZE
    )
    # preds = []
    # targets = []
    # display_ids = []
    display_id_counter = tf.Variable(0., trainable=False, dtype=tf.float64)
    streaming_map = tf.Variable(0., name='STREAMING_MAP', trainable=False, dtype=tf.float64)
    for step, (x, y) in enumerate(input_fn):
        # print(f'step: {step}, {y.shape}')
        # targets.append(y)
        # display_ids.append(x[DISPLAY_ID_COLUMN])
        sample = {}
        for k,v in x.items():
            sample[k] = v.numpy().tolist()
        request_data = json.dumps({"inputs": sample})

        s_time = time.time()
        response = requests.post(predict_url, data=request_data)
        pred = json.loads(response.text)['outputs']
        cal_map(pred, y, x[DISPLAY_ID_COLUMN], display_id_counter, streaming_map)
        est_time = time.time() - s_time
        throughput = eval_batch_size / est_time
        print(f'step: {step}, inference throughput: {throughput}')
        # preds.append(pred)
    map_metric = tf.divide(streaming_map, display_id_counter)
    return map_metric

def cal_map(preds, targets, display_ids, display_id_counter, streaming_map):
    bce = tf.keras.losses.BinaryCrossentropy()
    predictions = tf.convert_to_tensor(preds)
    loss = bce(targets, predictions)

    # display_id_counter = tf.Variable(0., trainable=False, dtype=tf.float64)
    # streaming_map = tf.Variable(0., name='STREAMING_MAP', trainable=False, dtype=tf.float64)
    predictions = tf.reshape(predictions, [-1])
    predictions = tf.cast(predictions, tf.float64)
    display_ids = tf.reshape(display_ids, [-1])
    labels = tf.reshape(targets, [-1])
    sorted_ids = tf.argsort(display_ids)
    display_ids = tf.gather(display_ids, indices=sorted_ids)
    predictions = tf.gather(predictions, indices=sorted_ids)
    labels = tf.gather(labels, indices=sorted_ids)
    _, display_ids_idx, display_ids_ads_count = tf.unique_with_counts(display_ids, out_idx=tf.int64)
    pad_length = 30 - tf.reduce_max(display_ids_ads_count)
    preds = tf.RaggedTensor.from_value_rowids(predictions, display_ids_idx).to_tensor()
    labels = tf.RaggedTensor.from_value_rowids(labels, display_ids_idx).to_tensor()
    labels_mask = tf.math.reduce_max(labels, 1)
    preds_masked = tf.boolean_mask(preds, labels_mask)
    labels_masked = tf.boolean_mask(labels, labels_mask)
    labels_masked = tf.argmax(labels_masked, axis=1, output_type=tf.int32)
    labels_masked = tf.reshape(labels_masked, [-1, 1])
    preds_masked = tf.pad(preds_masked, [(0, 0), (0, pad_length)])
    _, predictions_idx = tf.math.top_k(preds_masked, 12)
    indices = tf.math.equal(predictions_idx, labels_masked)
    indices_mask = tf.math.reduce_any(indices, 1)
    masked_indices = tf.boolean_mask(indices, indices_mask)
    res = tf.argmax(masked_indices, axis=1)
    ap_matrix = tf.divide(1, tf.add(res, 1))
    ap_sum = tf.reduce_sum(ap_matrix)
    shape = tf.cast(tf.shape(indices)[0], tf.float64)
    display_id_counter.assign_add(shape)
    streaming_map.assign_add(ap_sum)

    # map_metric = tf.divide(streaming_map, display_id_counter)
    return loss

def main():
    response = requests.get(status_url)
    print(response.text)
    response = requests.get(metadata_url)
    print(response.text)
    map = make_predict()

    # map = cal_map(preds, targets, display_ids)
    print(f'MAP: {map}')


if __name__ == '__main__':
    main()

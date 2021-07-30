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
import os
import time

import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from trainer.utils.schedulers import get_schedule, LearningRateScheduler

metrics_print_interval = 1
profiler_start_step = 5
profiler_stop_step = 10
os.environ['HOROVOD_CYCLE_TIME'] = '0.1'

def compute_eval_points(train_batches, evals_per_epoch):
    eval_points = np.linspace(0, train_batches - 1, evals_per_epoch + 1)[1:]
    eval_points = np.round(eval_points).tolist()
    return eval_points

def train(args, model, config):
    logger = logging.getLogger('tensorflow')

    train_dataset = config['train_dataset']
    eval_dataset = config['eval_dataset']
    steps = int(config['steps_per_epoch'])
    logger.info(f'Steps per epoch: {steps}')

    writer = tf.summary.create_file_writer(os.path.join(args.model_dir, 'event_files' + str(hvd.local_rank())))

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=args.learning_rate
    )
    if args.amp:
        optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
            optimizer,
            loss_scale='dynamic'
        )
    scheduler = LearningRateScheduler([optimizer],
                                      warmup_steps=args.warmup_steps,
                                      base_lr=args.learning_rate,
                                      decay_start_step=args.decay_start_step,
                                      decay_steps=args.decay_steps)

    eval_points = compute_eval_points(train_batches=steps,
                                      evals_per_epoch=args.evals_per_epoch)
                                      
    compiled_loss = tf.keras.losses.BinaryCrossentropy()
    eval_loss = tf.keras.metrics.Mean()

    metrics = [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.AUC()
    ]

    def scale_grad(grad, factor):
        if isinstance(grad, tf.IndexedSlices):
            # sparse gradient
            grad._values = grad._values * factor
            return grad
        else:
            # dense gradient
            return grad * factor

    def embedding_weight_update(model, optimizer, unscaled_gradients):
        embedding_gradients = model.extract_embedding_gradients(unscaled_gradients)
        if hvd.size() > 1:
            # need to correct for allreduced gradients being averaged and model-parallel ones not
            embedding_gradients = [scale_grad(g, 1 / hvd.size()) for g in embedding_gradients]
        optimizer.apply_gradients(zip(embedding_gradients, model.embedding_variables))
    def mlp_weight_update(model, optimizer, unscaled_gradients):
        mlp_gradients = model.extract_mlp_gradients(unscaled_gradients)
        if hvd.size() > 1:
            mlp_gradients = [hvd.allreduce(g, name="top_gradient_{}".format(i), op=hvd.Average,
                                           compression=hvd.compression.NoneCompressor) for i, g in
                             enumerate(mlp_gradients)]
        optimizer.apply_gradients(zip(mlp_gradients, model.mlp_variables))

    @tf.function
    def train_step(num_feature, cat_feature, y):
        with tf.GradientTape() as tape:
            y_pred = model(inputs=(num_feature, cat_feature), training=True)
            unscaled_loss = compiled_loss(y, y_pred)
            # tf keras doesn't reduce the loss when using a Custom Training Loop
            # unscaled_loss = tf.math.reduce_mean(unscaled_loss)
            scaled_loss = optimizer.get_scaled_loss(unscaled_loss) if args.amp else unscaled_loss

        scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)

        if args.amp:
            unscaled_gradients = optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            unscaled_gradients = scaled_gradients

        embedding_weight_update(model, optimizer, unscaled_gradients)
        mlp_weight_update(model, optimizer, unscaled_gradients)

        if hvd.size() > 1:
            # compute mean loss for all workers for reporting
            mean_loss = hvd.allreduce(unscaled_loss, name="mean_loss", op=hvd.Average)
        else:
            mean_loss = unscaled_loss

        return mean_loss

    @tf.function
    def evaluation_step(num_feature, cat_feature, y):
        y_pred = model((num_feature, cat_feature), training=False)
        # if hvd.size() > 1:
        #     y_pred = hvd.allgather(y_pred)
        # if hvd.rank() == 0:
        for metric in metrics:
            metric.update_state(y, y_pred)
        loss = compiled_loss(y, y_pred)
        
        return loss


    with writer.as_default():
        time_metric_start = time.time()
        for epoch in range(1, args.num_epochs + 1):
            for step, ((num_feature, cat_feature), y) in enumerate(train_dataset):
                # if step == profiler_start_step and hvd.rank() == 0:
                #     tf.profiler.experimental.start(os.path.join(args.model_dir, 'profile'))
                # if step == profiler_stop_step and hvd.rank() == 0:
                #     tf.profiler.experimental.stop()
                scheduler()

                loss = train_step(num_feature, cat_feature, y)
                if hvd.rank() == 0:
                    tf.summary.scalar('loss', loss, step=step + steps * (epoch - 1))
                    tf.summary.scalar('schedule', K.get_value(optimizer.lr), step=step + steps * (epoch - 1))
                    writer.flush()

                if step % metrics_print_interval == 0:
                    time_metric_end = time.time()
                    train_data = {'loss': f'{loss.numpy():.4f}', 'time': f'{(time_metric_end - time_metric_start):.4f}'}
                    logger.info(f'step: {step + steps * (epoch - 1)}, {train_data}')
                    time_metric_start = time.time()

                if step in eval_points:
                    for metric in metrics:
                        metric.reset_states()
                    eval_loss.reset_states()
                    stime = time.time()
                    for eval_step, ((num_feature, cat_feature), y) in enumerate(eval_dataset):
                        loss = evaluation_step(num_feature, cat_feature, y)
                        eval_loss.update_state(loss)
                    logger.info(f'eval time: {time.time() - stime}')

                    eval_loss_reduced = hvd.allreduce(eval_loss.result())
                    metrics_reduced = {
                        f'{metric.name}_val': hvd.allreduce(metric.result()) for metric in metrics
                    }
                    if hvd.rank() == 0:
                        for name, result in metrics_reduced.items():
                            tf.summary.scalar(f'{name}', result, step=step + steps * (epoch - 1))
                        tf.summary.scalar('loss_val', eval_loss_reduced, step=step + steps * (epoch - 1))
                        writer.flush()
                    eval_data = {name: f'{result.numpy():.4f}' for name, result in metrics_reduced.items()}
                    eval_data.update({
                        'loss_val': f'{eval_loss_reduced.numpy():.4f}'
                    })
                    logger.info(f'step: {step + steps * (epoch - 1)}, {eval_data}')
            logger.info(f'epoch {epoch} finished')
            for metric in metrics:
                metric.reset_states()
            eval_loss.reset_states()

            for eval_step, ((num_feature, cat_feature), y) in enumerate(eval_dataset):
                loss = evaluation_step(num_feature, cat_feature, y)
                eval_loss.update_state(loss)

            eval_loss_reduced = hvd.allreduce(eval_loss.result())
            metrics_reduced = {
                f'{metric.name}_val': hvd.allreduce(metric.result()) for metric in metrics
            }
            if hvd.rank() == 0:
                for name, result in metrics_reduced.items():
                    tf.summary.scalar(f'{name}', result, step=steps * epoch)
                tf.summary.scalar('loss_val', eval_loss_reduced, step=steps * epoch)
                writer.flush()

            eval_data = {name: f'{result.numpy():.4f}' for name, result in metrics_reduced.items()}
            eval_data.update({
                'loss_val': f'{eval_loss_reduced.numpy():.4f}'
            })
            logger.info(f'step: {steps * epoch}, {eval_data}')

            if hvd.rank() == 0:
                manager.save()
            # if map_metric >= 0.6553:
            #     logger.info(f'early stop at streaming_map_val: {map_metric}')
            #     break
        if hvd.rank() == 0:
            logger.info(f'Final eval result: {eval_data}')
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

from tensorflow.python.keras import backend as K
import tensorflow as tf


def get_schedule(args, steps_per_epoch):
    # assert args.deep_warmup_epochs <= args.num_epochs, 'Number of warmup epochs cannot be higher than training epochs'
    base_lr = args.deep_learning_rate
    # warmup_steps = args.deep_warmup_epochs * steps_per_epoch
    warmup_steps = args.deep_warmup_steps
    bound_epoch = args.deep_warmup_epochs + (args.num_epochs - args.deep_warmup_epochs) / 2
    boundaries = [bound_epoch * steps_per_epoch]
    values = [base_lr / 4, base_lr / 8]

    def schedule(optimizer, current_step):
        current_step = max(1, current_step)

        if current_step < warmup_steps:
            warmup_lr = base_lr * current_step / warmup_steps
            K.set_value(optimizer.lr, K.get_value(warmup_lr))
        else:
            # for index, bound in enumerate(boundaries):
            #     if current_step <= bound:
            #         K.set_value(optimizer.lr, K.get_value(values[index]))
            #         return
            K.set_value(optimizer.lr, K.get_value(base_lr))
        return

    return schedule

class LearningRateScheduler:
    """
    LR Scheduler combining Polynomial Decay with Warmup at the beginning.
    TF-based cond operations necessary for performance in graph mode.
    """

    def __init__(self, optimizers, base_lr, warmup_steps, decay_start_step, decay_steps):
        self.optimizers = optimizers
        self.warmup_steps = tf.constant(warmup_steps, dtype=tf.int32)
        self.decay_start_step = tf.constant(decay_start_step, dtype=tf.int32)
        self.decay_steps = tf.constant(decay_steps)
        self.decay_end_step = decay_start_step + decay_steps
        self.poly_power = 2
        self.base_lr = base_lr
        with tf.device('/CPU:0'):
            self.step = tf.Variable(0)

    @tf.function
    def __call__(self):
        with tf.device('/CPU:0'):
            # used for the warmup stage
            warmup_step = tf.cast(1 / self.warmup_steps, tf.float32)
            lr_factor_warmup = 1 - tf.cast(self.warmup_steps - self.step, tf.float32) * warmup_step
            lr_factor_warmup = tf.cast(lr_factor_warmup, tf.float32)

            # used for the constant stage
            lr_factor_constant = tf.cast(1., tf.float32)

            # used for the decay stage
            lr_factor_decay = (self.decay_end_step - self.step) / self.decay_steps
            lr_factor_decay = tf.math.pow(lr_factor_decay, self.poly_power)
            lr_factor_decay = tf.cast(lr_factor_decay, tf.float32)

            poly_schedule = tf.cond(self.step < self.decay_start_step, lambda: lr_factor_constant,
                                    lambda: lr_factor_decay)

            lr_factor = tf.cond(self.step < self.warmup_steps, lambda: lr_factor_warmup,
                                lambda: poly_schedule)

            lr = self.base_lr * lr_factor
            for optimizer in self.optimizers:
                optimizer.lr.assign(lr)

            self.step.assign(self.step + 1)
import tensorflow as tf
import horovod.tensorflow as hvd
import math

from data.outbrain.features import get_feature_columns, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, HASH_BUCKET_SIZES, HASH_BUCKET_SIZE, EMBEDDING_DIMENSION

class BroadcastingInitializer(tf.keras.initializers.Initializer):
    def __init__(self, wrapped):
        self.wrapped = wrapped
    
    def __call__(self, *args, **kwargs):
        weights = self.wrapped(*args, **kwargs)
        weights = hvd.broadcast(weights, root_rank=0, name='BroadcastingInitializer')
        return weights

    def get_config(self):
        return {}

class Trainer:
    def __init__(self, model, optimizer, amp, lr_scheduler):
        self.model = model
        self.optimizer = optimizer
        self.amp = amp
        self.lr_scheduler = lr_scheduler
        self.bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                                      from_logits=True)

    def _embedding_weight_update(self, unscaled_gradients):
        embedding_gradients = self.model.extract_embedding_gradients(unscaled_gradients)

        if hvd.size() > 1:
            # need to correct for allreduced gradients being averaged and model-parallel ones not
            embedding_gradients = [scale_grad(g, 1 / hvd.size()) for g in embedding_gradients]

        self.optimizer.apply_gradients(zip(embedding_gradients, self.model.embedding_variables))

    def _mlp_weight_update(self, unscaled_gradients):
        mlp_gradients = self.model.extract_mlp_gradients(unscaled_gradients)

        if hvd.size() > 1:
            mlp_gradients = [hvd.allreduce(g, name="top_gradient_{}".format(i), op=hvd.Average,
                                           compression=hvd.compression.NoneCompressor) for i, g in
                             enumerate(mlp_gradients)]

        self.optimizer.apply_gradients(zip(mlp_gradients, self.model.mlp_variables))

    @tf.function
    def train_step(self, numerical_features, categorical_features, labels):
        self.lr_scheduler()

        with tf.GradientTape() as tape:
            predictions = self.model(inputs=(numerical_features, categorical_features),
                                    training=True)

            unscaled_loss = self.bce(labels, predictions)
            # tf keras doesn't reduce the loss when using a Custom Training Loop
            unscaled_loss = tf.math.reduce_mean(unscaled_loss)
            scaled_loss = self.optimizer.get_scaled_loss(unscaled_loss) if self.amp else unscaled_loss

        scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)

        if self.amp:
            unscaled_gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            unscaled_gradients = scaled_gradients

        self._embedding_weight_update(unscaled_gradients)
        self._mlp_weight_update(unscaled_gradients)

        if hvd.size() > 1:
            # compute mean loss for all workers for reporting
            mean_loss = hvd.allreduce(unscaled_loss, name="mean_loss", op=hvd.Average)
        else:
            mean_loss = unscaled_loss

        return mean_loss

class wide_deep_model(tf.keras.Model):
    def __init__(self, args, embedding_sizes, distributed_metadata):
        super(wide_deep_model, self).__init__()
        local_table_ids = distributed_metadata.rank_to_categorical_ids[hvd.rank()]
        self.table_sizes = [embedding_sizes[i] for i in local_table_ids]
        self.rank_to_feature_count = distributed_metadata.rank_to_feature_count
        self.distributed = hvd.size() > 1
        # self.batch_size = args.batch_size
        self.num_all_categorical_features = len(embedding_sizes)

        self.amp = args.amp
        self.embedding_dim = 64

        self.local_embedding_dim = self.embedding_dim

        self.deep_hidden_units = args.deep_hidden_units

        self.top_mlp_padding = None
        self.bottom_mlp_padding = None

        self.variables_partitioned = False

        # self.num_numerical_features = args.num_numerical_features
        self.num_numerical_features = 13
        # override in case there's no numerical features in the dataset
        if self.num_numerical_features == 0:
            self.running_bottom_mlp = False

        self._create_embeddings()
        self._create_wide_mlp()
        self._create_deep_mlp()

        # write embedding checkpoints of 1M rows at a time
        self.embedding_checkpoint_batch = 1024 * 1024


    def _create_wide_mlp(self):
        self.wide_mlp_layers = []
        dim = 1
        kernel_initializer = tf.keras.initializers.GlorotNormal()
        bias_initializer = tf.keras.initializers.RandomNormal(stddev=math.sqrt(1. / dim))
        kernel_initializer = BroadcastingInitializer(kernel_initializer)
        bias_initializer = BroadcastingInitializer(bias_initializer)
        l = tf.keras.layers.Dense(dim, activation='relu',
                                  kernel_initializer=kernel_initializer,
                                  bias_initializer=bias_initializer)
        self.wide_mlp_layers.append(l)

    def _create_deep_mlp(self):
        self.deep_mlp_laryers = []
        for i, dim in enumerate(self.deep_hidden_units):
            if i == len(self.deep_hidden_units) - 1:
                # final layer
                activation = 'linear'
            else:
                activation = 'relu'

            kernel_initializer = tf.keras.initializers.GlorotNormal()
            bias_initializer = tf.keras.initializers.RandomNormal(stddev=math.sqrt(1. / dim))
            kernel_initializer = BroadcastingInitializer(kernel_initializer)
            bias_initializer = BroadcastingInitializer(bias_initializer)

            l = tf.keras.layers.Dense(dim, activation=activation,
                                      kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer)
            self.deep_mlp_laryers.append(l)

    def _create_embeddings(self):
        self.embedding_layers = []
        for i, table_size in enumerate(self.table_sizes):
            l = tf.keras.layers.Embedding(input_dim=table_size,
                          output_dim=self.local_embedding_dim)
            self.embedding_layers.append(l)

    @tf.function
    def call(self, inputs):
        numerical_features, cat_features = inputs
        embedding_outputs = self._call_embeddings(cat_features)
        embedding_alltoall = self._call_alltoall(embedding_outputs)

        linear_output = self._call_linear_part(embedding_alltoall, numerical_features)
        dnn_output = self._call_dnn_part(embedding_alltoall, numerical_features)
        logit_out = tf.keras.layers.Add()([dnn_output, linear_output])

        output = tf.keras.activations.sigmoid(logit_out)
        return output

    def _call_linear_part(self, embeddings, numerical_features):
        if self.amp:
            numerical_features = tf.cast(numerical_features, dtype=tf.float16)
        # for l in self.wide_mlp_layers:
        #     numerical_features = l(numerical_features)
        # embedding_output = []
        # for embedding in embeddings:
        #     embedding_output.append(tf.keras.layers.Flatten()(embedding))
        # categorical_output = tf.keras.layers.add(embedding_output)
        # linear_output = categorical_output + numerical_features

        x = tf.keras.layers.concatenate([numerical_features, tf.keras.layers.Flatten()(embeddings)])
        with tf.name_scope("linear"):
            for l in self.wide_mlp_layers:
                x = l(x)
            linear_output = x
        return linear_output


    def _call_dnn_part(self, embeddings, numerical_features):
        if self.amp:
            numerical_features = tf.cast(numerical_features, dtype=tf.float16)
        x = tf.keras.layers.concatenate([numerical_features, tf.keras.layers.Flatten()(embeddings)])

        with tf.name_scope("dnn"):
            for l in self.deep_mlp_laryers:
                x = l(x)
            dnn_output = x
        return dnn_output

    def _call_embeddings(self, cat_features):

        with tf.name_scope("embedding"):
            embedding_outputs = []
            if self.table_sizes:
                for i, l in enumerate(self.embedding_layers):
                    indices = tf.cast(cat_features[i], tf.int32)
                    out = l(indices)
                    embedding_outputs.append(out)
        if self.amp:
            embedding_outputs = [tf.cast(e, dtype=tf.float16) for e in embedding_outputs]
        return embedding_outputs

    def _call_alltoall(self, embedding_outputs):
        num_tables = len(self.table_sizes)
        
        embedding_outputs_concat = tf.concat(embedding_outputs, axis=1)
        global_batch = tf.shape(embedding_outputs_concat)[0]
        hvd_size = hvd.size()
        local_batch = global_batch // hvd_size
        embedding_dim = self.embedding_dim

        alltoall_input = tf.reshape(embedding_outputs_concat,
                                    shape=[global_batch * num_tables,
                                           embedding_dim])
        # split tensor to all rank equally
        splits = [tf.shape(alltoall_input)[0] // hvd_size] * hvd_size

        alltoall_output = hvd.alltoall(tensor=alltoall_input, splits=splits, ignore_name_scope=True, name='alltoall')
        # split alltoall embeddings to individual embedding
        vectors_per_worker = [x * local_batch for x in self.rank_to_feature_count]
        alltoall_output = tf.split(alltoall_output,
                                   num_or_size_splits=vectors_per_worker,
                                   axis=0)
        embedding_alltoall = [tf.reshape(x, shape=[local_batch, -1, embedding_dim]) for x in alltoall_output]


        embedding_alltoall = tf.concat(embedding_alltoall, axis=1)  # shape=[local_batch, num_vectors, vector_dim]
        return embedding_alltoall
    
    # def _call_alltoall(self, embedding_outputs):
    #     num_tables = len(self.table_sizes)
        
    #     global_batch = tf.shape(embedding_outputs[0])[0]
    #     hvd_size = hvd.size()
    #     local_batch = global_batch // hvd_size
    #     embedding_dim = self.embedding_dim

    #     splits = [global_batch // hvd_size] * hvd_size
    #     e = []
    #     for embedding in embedding_outputs:
    #         splitted_embedding = tf.split(embedding, num_or_size_splits=splits, axis=0)
    #         e.append(splitted_embedding)
    #     joined_embeddings = []
    #     for i in range(hvd_size):
    #         joined_embeddings.append(tf.concat([e[j][i] for j in range(num_tables)], axis=0))
        
    #     alltoall_input = tf.concat(joined_embeddings, axis=0)
    #     # split tensor to all rank equally
    #     ata_splits = [tf.shape(alltoall_input)[0] // hvd_size] * hvd_size

    #     alltoall_output = hvd.alltoall(tensor=alltoall_input, splits=ata_splits, ignore_name_scope=True, name='alltoall')
    #     # split alltoall embeddings to individual embedding
    #     vectors_per_worker = [x * local_batch for x in self.rank_to_feature_count]
    #     alltoall_output = tf.split(alltoall_output,
    #                                num_or_size_splits=vectors_per_worker,
    #                                axis=0)
    #     embedding_alltoall = [tf.reshape(x, shape=[local_batch, -1, embedding_dim]) for x in alltoall_output]


    #     embedding_alltoall = tf.concat(embedding_alltoall, axis=1)  # shape=[local_batch, num_vectors, vector_dim]
    #     return embedding_alltoall

    def _partition_variables(self):
        self.embedding_variables = [v for v in self.trainable_variables if 'embedding' in v.name]
        self.embedding_variable_indices = [i for i,v in enumerate(self.trainable_variables) if 'embedding' in v.name]
        self.mlp_variables = [v for v in self.trainable_variables if 'embedding' not in v.name]
        self.mlp_variable_indices = [i for i,v in enumerate(self.trainable_variables) if 'embedding' not in v.name]
        self.variables_partitioned = True

    def extract_embedding_gradients(self, all_gradients):
        if not self.variables_partitioned:
            self._partition_variables()
        return [all_gradients[i] for i in self.embedding_variable_indices]
    
    def extract_mlp_gradients(self, all_gradients):
        if not self.variables_partitioned:
            self._partition_variables()
        return [all_gradients[i] for i in self.mlp_variable_indices]
import tensorflow as tf
import horovod.tensorflow as hvd
import math

from data.outbrain.features import get_feature_columns, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, HASH_BUCKET_SIZES, HASH_BUCKET_SIZE, EMBEDDING_DIMENSION


class wide_deep_model(tf.keras.Model):
    def __init__(self, args, dataset_metadata, distributed_metadata):
        super(wide_deep_model, self).__init__()
        local_table_ids = distributed_metadata.rank_to_categorical_ids[hvd.rank()]
        self.table_sizes = [dataset_metadata.categorical_cardinalities[i] for i in local_table_ids]
        self.rank_to_feature_count = distributed_metadata.rank_to_feature_count
        self.distributed = hvd.size() > 1
        self.batch_size = args.batch_size
        self.num_all_categorical_features = len(dataset_metadata.categorical_cardinalities)

        self.amp = args.amp
        self.dataset_metadata = dataset_metadata

        self.embedding_dim = args.embedding_dim

        self.experimental_columnwise_split = args.experimental_columnwise_split

        if self.experimental_columnwise_split:
            self.local_embedding_dim = self.embedding_dim // hvd.size()
        else:
            self.local_embedding_dim = self.embedding_dim

        self.embedding_type = args.embedding_type
        self.embedding_trainable = args.embedding_trainable

        self.deep_hidden_units = args.deep_hidden_units

        self.top_mlp_padding = None
        self.bottom_mlp_padding = None

        self.variables_partitioned = False

        self.num_numerical_features = args.num_numerical_features
        # override in case there's no numerical features in the dataset
        if self.num_numerical_features == 0:
            self.running_bottom_mlp = False

        self._create_embeddings()
        self._create_linear_part()
        self._create_wide_part()

        # write embedding checkpoints of 1M rows at a time
        self.embedding_checkpoint_batch = 1024 * 1024


    def _create_wide_mlp(self):
        self.wide_mlp_layers = []
        dim = 1
        kernel_initializer = tf.keras.initializers.GlorotNormal()
        bias_initializer = tf.keras.initializers.RandomNormal(stddev=math.sqrt(1. / dim))
        kernel_initializer = hvd.broadcast(kernel_initializer, root_rank=0)
        bias_initializer = hvd.broadcast(bias_initializer, root_rank=0)
        l = tf.keras.layers.Dense(dim, activation='relu',
                                  kernel_initializer=kernel_initializer,
                                  bias_initializer=bias_initializer)
        self.wide_mlp_layers.append(l)

    def _create_deep_mlp(self):
        self.deep_mlp_laryers = []
        for i, dim in enumerate(self.deep_hidden_units):
            if i == len(self.top_mlp_dims) - 1:
                # final layer
                activation = 'linear'
            else:
                activation = 'relu'

            kernel_initializer = tf.keras.initializers.GlorotNormal()
            bias_initializer = tf.keras.initializers.RandomNormal(stddev=math.sqrt(1. / dim))
            kernel_initializer = hvd.broadcast(kernel_initializer, root_rank=0)
            bias_initializer = hvd.broadcast(bias_initializer, root_rank=0)

            l = tf.keras.layers.Dense(dim, activation=activation,
                                      kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer)
            self.deep_mlp_laryers.append(l)

    def _create_embeddings(self):
        self.embedding_layers = []
        for i, table_size in enumerate(self.table_sizes):
            l = tf.keras.layers.Embedding(input_dim=table_size,
                          output_dim=self.local_embedding_dim,
                          trainable=self.embedding_trainable)
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
        numerical_tensor = tf.keras.layers.concatenate(numerical_features)
        for l in self.wide_mlp_layers:
            numerical_tensor = l(numerical_tensor)
        embedding_output = []
        for embedding in embeddings:
            embedding_output.append(tf.keras.layers.Flatten()(embedding))
        categorical_output = tf.keras.layers.add(embedding_output)

        linear_output = categorical_output + numerical_tensor

        return linear_output


    def _call_dnn_part(self, embeddings, numerical_features):
        if self.amp:
            numerical_features = tf.cast(numerical_features, dtype=tf.float16)
        x = tf.keras.layers.concatenate([numerical_features, embeddings])

        with tf.name_scope("deep part"):
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

        splits = [tf.shape(alltoall_input)[0] // hvd_size] * hvd_size

        alltoall_output = hvd.alltoall(tensor=alltoall_input, splits=splits, ignore_name_scope=True)

        vectors_per_worker = [x * local_batch for x in self.rank_to_feature_count]
        alltoall_output = tf.split(alltoall_output,
                                   num_or_size_splits=vectors_per_worker,
                                   axis=0)
        embedding_alltoall = [tf.reshape(x, shape=[local_batch, -1, embedding_dim]) for x in alltoall_output]


        embedding_alltoall = tf.concat(embedding_alltoall, axis=1)  # shape=[local_batch, num_vectors, vector_dim]
        return embedding_alltoall

    @staticmethod
    def _get_variable_path(checkpoint_path, v, i=0):
        checkpoint_path = checkpoint_path + f'_rank_{hvd.rank()}'
        name = v.name.replace('/', '_').replace(':', '_')
        return checkpoint_path + '_' + name + f'_{i}' + '.npy'

    def maybe_save_checkpoint(self, checkpoint_path):
        if checkpoint_path is None:
            return

        dist_print('Saving a checkpoint...')
        for v in self.trainable_variables:
            filename = self._get_variable_path(checkpoint_path, v)
            if 'embedding' not in v.name:
                np.save(arr=v.numpy(), file=filename)
                continue
            print(f'saving embedding {v.name}')
            chunks = math.ceil(v.shape[0] / self.embedding_checkpoint_batch)
            for i in range(chunks):
                filename = self._get_variable_path(checkpoint_path, v, i)
                end = min((i + 1) * self.embedding_checkpoint_batch, v.shape[0])

                indices = tf.range(start=i * self.embedding_checkpoint_batch,
                                   limit=end,
                                   dtype=tf.int32)

                arr = tf.gather(params=v, indices=indices, axis=0)
                arr = arr.numpy()
                np.save(arr=arr, file=filename)

        dist_print('Saved a checkpoint to ', checkpoint_path)

    def maybe_restore_checkpoint(self, checkpoint_path):
        if checkpoint_path is None:
            return

        dist_print('Restoring a checkpoint...')
        self.force_initialization()

        for v in self.trainable_variables:
            filename = self._get_variable_path(checkpoint_path, v)
            if 'embedding' not in v.name:
                numpy_var = np.load(file=filename)
                v.assign(numpy_var)
                continue

            chunks = math.ceil(v.shape[0] / self.embedding_checkpoint_batch)
            for i in range(chunks):
                filename = self._get_variable_path(checkpoint_path, v, i)
                start = i * self.embedding_checkpoint_batch
                numpy_arr = np.load(file=filename)
                indices = tf.range(start=start,
                                   limit=start + numpy_arr.shape[0],
                                   dtype=tf.int32)
                update = tf.IndexedSlices(values=numpy_arr, indices=indices, dense_shape=v.shape)
                v.scatter_update(sparse_delta=update)

        dist_print('Restored a checkpoint from', checkpoint_path)


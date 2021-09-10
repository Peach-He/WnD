import horovod.tensorflow.keras as hvd
import tensorflow_transform as tft

from data.outbrain.dataloader import train_input_fn, eval_input_fn, BinDataset

def create_dataset(args):
    num_gpus = hvd.size()
    gpu_id = hvd.rank()
    train_batch_size = args.global_batch_size // num_gpus
    eval_batch_size = args.eval_batch_size // num_gpus
    steps_per_epoch = args.training_set_size / args.global_batch_size
    if args.dataset_format == "tfrecords":
        feature_spec = tft.TFTransformOutput(
            args.transformed_metadata_path
        ).transformed_feature_spec()
    
        train_spec_input_fn = train_input_fn(
            num_gpus=num_gpus,
            id=gpu_id,
            filepath_pattern=args.train_data_pattern,
            feature_spec=feature_spec,
            records_batch_size=train_batch_size // args.prebatch_size,
        )
    
        eval_spec_input_fn = eval_input_fn(
            num_gpus=num_gpus,
            id=gpu_id,
            filepath_pattern=args.eval_data_pattern,
            feature_spec=feature_spec,
            records_batch_size=eval_batch_size // args.prebatch_size
        )
    elif args.dataset_format == "binary":
        train_dataset = BinDataset(train_dataset_path, metadata, train_batch_size)
        test_dataset = BinDataset(eval_dataset_path, metadata, eval_batch_size)
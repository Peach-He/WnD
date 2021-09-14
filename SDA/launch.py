import subprocess
import argparse


def launch_dlrm(config):
    args = parse_args()
    dataset = config['dataset']
    model = config['model']
    hosts = config['cluster']
    args.nnodes = len(hosts)

    cmd = ["cd DLRM"]
    cmd.append(object)

    print(f'args: {args}')

    # if args.nnodes > 1:
    #     args.distributed = True
    
    # if args.distributed:
    #     mpi_dist_launch(args)
    # else:
    #     launch(args)


def launch_wnd(config):
    def parse_args():
        '''
        model specific argument, parse from SDA command line
        '''
        parser = argparse.ArgumentParser(
            description='WideAndDeep Model',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            add_help=True,
        )

        parser.add_argument('--dataset_format', type=str, default='TFRecords', help='train/test dataset format, support TFRecords and binary')

        parser.add_argument('--train_data_pattern', type=str, default='/outbrain/tfrecords/train/part*', 
                               help='Pattern of training file names. For example if training files are train_000.tfrecord, '
                                    'train_001.tfrecord then --train_data_pattern is train_*')

        parser.add_argument('--eval_data_pattern', type=str, default='/outbrain/tfrecords/eval/part*', 
                               help='Pattern of eval file names. For example if eval files are eval_000.tfrecord, '
                                    'eval_001.tfrecord then --eval_data_pattern is eval_*')

        parser.add_argument('--transformed_metadata_path', type=str, default='/outbrain/tfrecords',
                               help='Path to transformed_metadata for feature specification reconstruction, only available for TFRecords')
        
        parser.add_argument('--prebatch_size', type=int, default=4096, help='Dataset prebatch size, only available for TFRecords')

        parser.add_argument('--model_dir', type=str, default='/outbrain/checkpoints',
                               help='Destination where model checkpoint will be saved')

        parser.add_argument('--training_set_size', type=int, default=59761827,
                                     help='Number of samples in the training set')

        parser.add_argument('--global_batch_size', type=int, default=131072,
                                     help='Total size of training batch')

        parser.add_argument('--eval_batch_size', type=int, default=131072,
                                     help='Total size of evaluation batch')

        parser.add_argument('--num_epochs', type=int, default=1,
                                     help='Number of training epochs')

        parser.add_argument('--linear_learning_rate', type=float, default=-1,
                                     help='Learning rate for linear model')

        parser.add_argument('--deep_learning_rate', type=float, default=-1,
                                     help='Learning rate for deep model')

        parser.add_argument('--deep_warmup_epochs', type=float, default=-1,
                                     help='Number of learning rate warmup epochs for deep model')

        parser.add_argument('--deep_hidden_units', type=int, default=[], nargs="+",
                                        help='Hidden units per layer for deep model, separated by spaces')

        parser.add_argument('--deep_dropout', type=float, default=-1,
                                        help='Dropout regularization for deep model')

        parser.add_argument('--iface', type=str, default='eth0', help='mpi interface')
        parser.add_argument('--ccl_worker_num', type=int, default=1, help='OneCCL worker number')
        parser.add_argument('--python_executable', type=str, default='python', help='python interpreter')
        return parser.parse_args()
    
    args = parse_args()
    dataset = config['dataset']
    hosts = config['cluster']['hosts']
    ppn = config['cluster']['ppn']
    cores = config['cluster']['cores']
    omp_threads = cores // 2 // ppn - args.ccl_worker_num
    ranks = len(hosts) * ppn
    
    # construct WnD launch command with mpi
    cmd = "cd WnD; "
    cmd += f"time mpirun -genv OMP_NUM_THREADS={omp_threads} -map-by socket -n {ranks} -ppn {ppn} -hosts {','.join(hosts)} -print-rank-map "
    cmd += f"-genv I_MPI_PIN_DOMAIN=socket -genv OMP_PROC_BIND=true -genv KMP_BLOCKTIME=1 -genv KMP_AFFINITY=granularity=fine,compact,1,0 "
    cmd += f"-iface {args.iface} "
    cmd += f"{args.python_executable} main.py "
    cmd += f"--dataset_format {args.dataset_format} --prebatch_size {args.prebatch_size} " \
        + f"--train_data_pattern {args.train_data_pattern} --eval_data_pattern {args.eval_data_pattern} --transformed_metadata_path {args.transformed_metadata_path} " \
        + f"--global_batch_size {args.global_batch_size} --eval_batch_size {args.eval_batch_size} --num_epochs {args.num_epochs} "
    if args.linear_learning_rate != -1:
        cmd += f"--linear_learning_rate {args.linear_learning_rate} "
    if args.deep_learning_rate != -1:
        cmd += f"--deep_learning_rate {args.deep_learning_rate} "
    if args.deep_warmup_epochs != -1:
        cmd += f"--deep_warmup_epochs {args.deep_warmup_epochs} "
    if len(args.deep_hidden_units) != 0:
        cmd += f"--deep_hidden_units {' '.join([str(item) for item in args.deep_hidden_units])} "
    if args.deep_dropout != -1:
        cmd += f"--deep_dropout {args.deep_dropout} "
    print(f'training launch command: {cmd}')
    process = subprocess.Popen(cmd, shell=True)
    process.wait()

def launch_dien(config):
    pass
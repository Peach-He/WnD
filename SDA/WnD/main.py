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

from trainer.model.widedeep import wide_deep_model
from trainer.run import train
from trainer.utils.arguments import parse_args
from trainer.utils.setup import create_config
import horovod.tensorflow as hvd

from sigopt import Connection
import logging
import yaml
import time

def exclude_params(args, parameters):
    all_assignments = {}
    merged_parameters = []
    for param in parameters:
        all_assignments[param['name']] = param
    
    if len(args.deep_hidden_units) == 0:
        merged_parameters += [all_assignments['dnn_hidden_unit1'], all_assignments['dnn_hidden_unit2'], all_assignments['dnn_hidden_unit3']]
    if args.deep_learning_rate == -1:
        merged_parameters.append(all_assignments['deep_learning_rate'])
    if args.linear_learning_rate == -1:
        merged_parameters.append(all_assignments['linear_learning_rate'])
    if args.deep_warmup_epochs == -1:
        merged_parameters.append(all_assignments['deep_warmup_epochs'])
    if args.deep_dropout == -1:
        merged_parameters.append(all_assignments['deep_dropout'])
    return merged_parameters

def assign_params(args, assignments):
    if 'dnn_hidden_unit1' in assignments and 'dnn_hidden_unit2' in assignments and 'dnn_hidden_unit3' in assignments:
        args.deep_hidden_units = [assignments["dnn_hidden_unit1"], assignments["dnn_hidden_unit2"], assignments["dnn_hidden_unit3"]]
    if 'deep_learning_rate' in assignments:
        args.deep_learning_rate = assignments["deep_learning_rate"]
    if 'linear_learning_rate' in assignments:
        args.linear_learning_rate = assignments["linear_learning_rate"]
    if 'deep_warmup_epochs' in assignments:
        args.deep_warmup_epochs = assignments["deep_warmup_epochs"]
    if 'deep_dropout' in assignments:
        args.deep_dropout = assignments["deep_dropout"]

def create_experiment(args, yaml_file="sigopt.yaml"):
    with open(yaml_file) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    conn = Connection(client_token='NEKPNYXPRQRRUMTGRMVEKLRAZCUWGMCHTQDSILUOGQMLKZWD')
    conn.set_proxies(
        {
            "http": "http://child-prc.intel.com:913",
            "https": "http://child-prc.intel.com:913",
        }
    )
    parameters = exclude_params(args, data["sigopt"]["parameters"])
    metrics = data["sigopt"]["metrics"]
    observation_budget = data["sigopt"]["observation_budget"]
    experiment_name = data["sigopt"]["experiment"]
    project = data["sigopt"]["project"]
    parallel_bandwidth = data["sigopt"]["parallel_bandwidth"]
    metadata = data["sigopt"]["metadata"]

    experiment = conn.experiments().create(
        name=experiment_name,
        parameters=parameters,
        metrics=metrics,
        parallel_bandwidth=parallel_bandwidth,
        observation_budget=observation_budget,
        project=project,
        metadata = metadata
    )
    return conn, experiment


def main():
    args = parse_args()
    config = create_config(args)

    logger = logging.getLogger('tensorflow')

    assignments = {}
    experiment_id = 0
    suggestion_id = 0
    if hvd.rank() == 0:
        conn, experiment = create_experiment(args)
        experiment_id = experiment.id
        experiment_id = hvd.broadcast_object(experiment_id, root_rank=0)
        logger.info(f'experiment.observation_budget: {experiment.observation_budget}')
    else:
        experiment_id = hvd.broadcast_object(experiment_id, root_rank=0)

    conn = Connection(client_token='NEKPNYXPRQRRUMTGRMVEKLRAZCUWGMCHTQDSILUOGQMLKZWD')
    conn.set_proxies(
        {
            "http": "http://child-prc.intel.com:913",
            "https": "http://child-prc.intel.com:913",
        }
    )
    experiment = conn.experiments(experiment_id).fetch()
    for _ in range(experiment.observation_budget):

        if hvd.rank() == 0:
            suggestion = conn.experiments(experiment.id).suggestions().create()
            assignments = suggestion.assignments
            assignments = hvd.broadcast_object(assignments, root_rank=0)
        else:
            assignments = hvd.broadcast_object(assignments, root_rank=0)
        logger.info(f'{hvd.rank()}:assignment: {assignments}')
        assign_params(args, assignments)
        logger.info(f'The training parameters: dnn_units: {args.deep_hidden_units}, \
            linear_LR: {args.linear_learning_rate}, deep_LR: {args.deep_learning_rate}, warmup: {args.deep_warmup_epochs}, drop_out: {args.deep_dropout}')

        s_time = time.time()
        model = wide_deep_model(args)
        map = train(args, model, config)
        map = 0
        training_time = time.time() - s_time
        if hvd.rank() == 0:
            for i in range(5):
                try:
                    conn.experiments(experiment.id).observations().create(
                        suggestion=suggestion.id,
                        value=map,
                    )
                    experiment = conn.experiments(experiment.id).fetch()
                    break
                except Exception as e:
                    logger.info(f'Met exception when creating observation, retried {i} times. The exception is: {e}')
                    time.sleep(5)

    if hvd.rank() == 0:
        all_best_assignments = conn.experiments(experiment.id).best_assignments().fetch()
        best_assignments = all_best_assignments.data[0].assignments
        logger.info("Best Assignments: " + str(best_assignments))


def test():
    args = parse_args()
    config = create_config(args)

    logger = logging.getLogger('tensorflow')

    model = wide_deep_model(args)
    map = train(args, model, config)

if __name__ == '__main__':
    main()
    # test()

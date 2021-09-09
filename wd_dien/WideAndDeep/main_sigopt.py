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

from trainer.model.widedeep import wide_deep_model_orig
from trainer.run import train
from trainer.utils.arguments import parse_args
from trainer.utils.setup import create_config

from sigopt import Connection
import logging

def create_experiment():
    conn = Connection(client_token='NEKPNYXPRQRRUMTGRMVEKLRAZCUWGMCHTQDSILUOGQMLKZWD')
    conn.set_proxies(
        {
            "http": "http://child-prc.intel.com:913",
            "https": "http://child-prc.intel.com:913",
        }
    )
    parameters = []
    parameters.append(
        dict(name="dnn_hidden1", type="int", bounds=dict(min=256, max=1024))
    )
    parameters.append(
        dict(name="dnn_hidden2", type="int", bounds=dict(min=256, max=1024))
    )
    parameters.append(
        dict(name="dnn_hidden3", type="int", bounds=dict(min=256, max=1024))
    )
    parameters.append(
        dict(name="learning_rate", type="double", bounds=dict(min=1e-4, max=1e-1), transformation="log")
    )
    parameters.append(
        dict(name="deep_warmup_steps", type="int", bounds=dict(min=1000, max=12000))
    )

    experiment = conn.experiments().create(
        name="WnD dien",
        parameters=parameters,
        metrics=[dict(name="AUC", objective="maximize")],
        parallel_bandwidth=1,
        observation_budget=30,
        project="wideanddeep",
    )
    return conn, experiment


def main():
    args = parse_args()
    config = create_config(args)
    logger = logging.getLogger('tensorflow')

    conn, experiment = create_experiment()
    logger.info(f'experiment.observation_budget: {experiment.observation_budget}')

    while experiment.progress.observation_count < experiment.observation_budget:
        suggestion = conn.experiments(experiment.id).suggestions().create()

        assignments = suggestion.assignments
        logger.info(f'assignment: {assignments}')
        args.deep_hidden_units = [assignments["dnn_hidden1"], assignments["dnn_hidden2"], assignments["dnn_hidden3"]]
        args.deep_learning_rate = assignments["learning_rate"]
        args.deep_warmup_steps = assignments["deep_warmup_steps"]
        model = wide_deep_model_orig(args)
        value = train(args, model, config)

        conn.experiments(experiment.id).observations().create(
            suggestion=suggestion.id,
            value=value,
        )

    # Update the experiment object
    experiment = conn.experiments(experiment.id).fetch()

    # Fetch the best configuration and explore your experiment
    all_best_assignments = conn.experiments(experiment.id).best_assignments().fetch()
    # Returns a list of dict-like Observation objects
    best_assignments = all_best_assignments.data[0].assignments
    logger.info("Best Assignments: " + str(best_assignments))



if __name__ == '__main__':
    main()

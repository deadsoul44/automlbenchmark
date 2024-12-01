import logging
from importlib.metadata import version

from perpetual import PerpetualBooster

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.results import save_predictions
from amlb.utils import Timer

import pandas as pd

from frameworks.shared.callee import measure_inference_times

log = logging.getLogger(__name__)

# python3 runbenchmark.py perpetualbooster regression 1h8c_gp3 -m aws -p 3
# python3 runbenchmark.py autogluon_bestquality openml/t/10102 ag_gp3 -m aws -p 3


def run(dataset: Dataset, config: TaskConfig):
    log.info(f"\n**** Perpetual [v{version('perpetual')}] ****\n")

    is_classification = config.type == 'classification'

    if is_classification:
        X_train = dataset.train.X
        y_train = dataset.train.y_enc
        X_test = dataset.test.X
        y_test = dataset.test.y_enc
    else:
        X_train = dataset.train.X
        y_train = dataset.train.y.to_numpy().flatten()
        X_test = dataset.test.X
        y_test = dataset.test.y.to_numpy().flatten()

    objective = "LogLoss" if is_classification else "SquaredLoss"
    timeout = config.max_runtime_seconds
    memory_limit = config.max_mem_size_mb / 1000

    with Timer() as training:
        model = PerpetualBooster(objective=objective)
        model.fit(X_train, y_train, budget=0.5, timeout=timeout, memory_limit=memory_limit, iteration_limit=10000)
    log.info(f"Finished fit in {training.duration}s.")
    log.info(f"Number of trees: {model.number_of_trees}.")

    with Timer() as predict:
        predictions = model.predict(X_test)

    if is_classification:
        probabilities = model.predict_proba(X_test)
    else:
        probabilities = None

    log.info(f"Finished predict in {predict.duration}s.")

    def infer(data):
        data = pd.read_parquet(data) if isinstance(data, str) else data
        return model.predict(data)

    inference_times = {}
    if config.measure_inference_time:
        inference_times["file"] = measure_inference_times(infer, dataset.inference_subsample_files(fmt="parquet"))
        test_data = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test)
        inference_times["df"] = measure_inference_times(
            infer,
            [(1, test_data.sample(1, random_state=i)) for i in range(100)],
        )
        log.info(f"Finished inference time measurements.")

    save_predictions(dataset=dataset,
                     output_file=config.output_predictions_file,
                     probabilities=probabilities,
                     predictions=predictions,
                     truth=y_test,
                     target_is_encoded=is_classification)

    return dict(
        models_count=1,
        training_duration=training.duration,
        predict_duration=predict.duration,
        inference_times=inference_times,
    )

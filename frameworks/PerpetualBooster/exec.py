import logging
from importlib.metadata import version

from perpetual import PerpetualBooster, __version__

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import impute_array
from amlb.results import save_predictions
from amlb.utils import Timer, unsparsify

import pandas as pd

log = logging.getLogger(__name__)

# python3 runbenchmark.py perpetualbooster example 1h4c -m aws -f 0


def run(dataset: Dataset, config: TaskConfig):
    log.info(f"\n**** Perpetual [v{__version__}] ****\n")
    log.info(f"\n**** Perpetual [v{version('perpetual')}] ****\n")

    is_classification = config.type == 'classification'

    X_train, X_test = impute_array(*unsparsify(dataset.train.X_enc, dataset.test.X_enc, fmt='array'))
    y_train, y_test = unsparsify(dataset.train.y_enc, dataset.test.y_enc, fmt='array')

    objective = "LogLoss" if is_classification else "SquaredLoss"
    predictor = PerpetualBooster(objective=objective)

    with Timer() as training:
        predictor.fit(X_train, y_train, budget=1.0)
    log.info(f"Finished fit in {training.duration}s.")

    with Timer() as predict:
        predictions = predictor.predict(X_test)
    probabilities = predictor.predict_proba(X_test) if is_classification else None
    log.info(f"Finished predict in {predict.duration}s.")

    save_predictions(dataset=dataset,
                     output_file=config.output_predictions_file,
                     probabilities=probabilities,
                     predictions=predictions,
                     truth=y_test,
                     target_is_encoded=is_classification)

    return dict(
        models_count=1,
        training_duration=training.duration,
        predict_duration=predict.duration
    )

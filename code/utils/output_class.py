import os
import sys
import subprocess
import time
from utils.enums import Task, Metric, Classifier, DataType
from preprocessing.dataset import DatapointKey


class Output:
    metrics = ['accuracy', 'auroc', 'f1']
    definitions = ['CP07', 'U&T', 'U65']
    tasks = ['prediction', 'classification']
    data_types = ['real', 'simulated']

    def __init__(self, classifier, task, data_type, definition, cutoff_point,
                 feature_interval='-', prediction_start_day='-',
                 prediction_interval='-', metric='-', scores='-',
                 path=None):
        """Constructor of the output class

        :classifier: The classifier that has been used for the task
        :task: The task (either prediction or classification)
        :data_type: The type of the data (either real or simulated)
        :definition: The definition used (CP07, U&T, U65)
        :cutoff_point: The maximum cutoff point where you will have access to
        :feature_interval: The number of days in the past where you will look
                           before your cutoff point
        :prediction_interval: The number of days in the future you want to
                              predict for
        :metric: The metric used (accuracy, auroc, f1)
        :scores: The scores of the experiment as a python list
        """
        assert task in self.tasks, (
            "The available tasks are 'prediction' and 'classification'")
        assert data_type in self.data_types, (
            "The available data types are 'real' and 'simulated'")
        assert definition in self.definitions, (
            "The available definitions are 'CP07', 'U&T', 'U65'")
        assert metric in self.metrics, (
            "The available metrics are 'accuracy','auroc', 'f1'")

        self.classifier = classifier
        self.task = task
        self.data_type = data_type
        self.definition = definition
        self.cutoff_point = cutoff_point
        self.feature_interval = feature_interval
        self.prediction_start_day = prediction_start_day
        self.prediction_interval = prediction_interval
        self.metric = metric
        self.scores = scores
        if path is None:
            path = os.getenv("DSLAB_RESULT_FILE")
            if path is None:
                print("Your output path is not set correctly. Please set the "
                      "DSLAB_RESULT_FILE env variable or the path variable "
                      "to the 'results/results.csv' file in the repo")
                sys.exit(1)
            else:
                self.path = path
        else:
            self.path = path

    def write_output(self):
        """Writes the output of the experiment into a CSV file"""
        scores = ','.join(str(e) for e in self.scores)
        ts = time.ctime()
        label = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
        output_string = "{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                ts,
                label,
                self.classifier,
                self.task,
                self.data_type,
                self.definition,
                self.cutoff_point,
                self.feature_interval,
                self.prediction_start_day,
                self.prediction_interval,
                self.metric,
                scores
                )
        with open(self.path, 'a') as csv_file:
            csv_file.write(output_string)


if __name__ == "__main__":
    test = Output(Classifier.xgboost, Task.prediction,
                  DataType.simulated, DatapointKey.CP07, 120,
                  60, 30, 7, Metric.auroc, 5*[0.78])
    test.write_output()

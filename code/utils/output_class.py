import os
import sys
from enums import Task, Metric, Classifier
from preprocessing.dataset import DatapointKey


class Output:
    metrics = ['accuracy', 'auroc', 'f1']
    definitions = ['CP07', 'U&T', 'U65']
    tasks = ['prediction', 'classification']

    def __init__(self, classifier, task, definition, cutoff_point,
                 feature_interval, prediction_interval, metric, scores,
                 path=None):
        """Constructor of the output class

        :classifier: The classifier that has been used for the task
        :task: The task (either prediction or classification)
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
        assert definition in self.definitions, (
                "The available definitions are 'CP07', 'U&T', 'U65'")
        assert metric in self.metrics, (
                "The available metrics are 'accuracy','auroc', 'f1'")

        self.classifier = classifier
        self.task = task
        self.definition = definition
        self.cutoff_point = cutoff_point
        self.feature_interval = feature_interval
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
        output_string = "{},{},{},{},{},{},{},{}\n".format(
                self.classifier,
                self.task,
                self.definition,
                self.cutoff_point,
                self.feature_interval,
                self.prediction_interval,
                self.metric,
                scores
                )
        with open(self.path, 'a') as csv_file:
            csv_file.write(output_string)


if __name__ == "__main__":
    test = Output(Classifier.xgboost, Task.prediction, DatapointKey.CP07, 120,
                  60, 30, Metric.auroc, 5*[0.78])
    test.write_output()

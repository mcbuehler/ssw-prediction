import os
import csv
import sys


class Output:
    metrics = ['accuracy', 'auroc', 'f1']
    definitions = ['CP07', 'U&T', 'U65']
    tasks = ['prediction', 'classification']

    def __init__(self, classifier, task, definition, cutoff_point,
                 feature_interval, prediction_interval, metric, result):
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
        :result: The result of the experiment
        """
        assert task in self.tasks, ("The available tasks are 'prediction' and"
                                    "'classification'")
        assert definition in self.definitions, ("The available definitions are"
                                                "'CP07', 'U&T', 'U65'")
        assert metric in self.metrics, ("The available metrics are 'accuracy',"
                                        "'auroc', 'f1'")

        self.classifier = classifier
        self.task = task
        self.definition = definition
        self.cutoff_point = cutoff_point
        self.feature_interval = feature_interval
        self.prediction_interval = prediction_interval
        self.metric = metric
        self.result = result

    def write_output(self):
        """Writes the output of the experiment into a CSV file"""
        path_result_file = os.getenv("DSLAB_RESULT_FILE")
        if path_result_file is None:
            print("DSLAB_RESULT_FILE env variable is not set! Please set it "
                  "to the 'results/results.csv' file in the repo")
            sys.exit(1)

        # Checks for types and converts if necessary
        if isinstance(self.cutoff_point, int):
            self.cutoff_point = str(self.cutoff_point)
        if isinstance(self.feature_interval, int):
            self.feature_interval = str(self.feature_interval)
        if isinstance(self.prediction_interval, int):
            self.prediction_interval = str(self.prediction_interval)
        if isinstance(self.result, float):
            self.result = str(self.result)

        output_string = (
                self.classifier + "," +
                self.task + "," +
                self.cutoff_point + "," +
                self.feature_interval + "," +
                self.prediction_interval + "," +
                self.metric + "," +
                self.result + "\n"
                )
        with open(path_result_file, 'a') as csv_file:
            csv_file.write(output_string)


if __name__ == "__main__":
    test = Output('XGBoost', 'prediction', 'CP07', 120, 60, 30, 'auroc', 0.78)
    test.write_output()

class Task:
    """An enum class for the task"""
    prediction = "prediction"
    classification = "classification"


class Metric:
    """An enum class for the evaluation metric"""
    auroc = "auroc"
    f1 = "f1"
    accuracy = "accuracy"


class Classifier:
    """An enum class for the classifier used"""
    xgboost_tsfresh = "xgboost_tsfresh"
    xgboost_auto = "xgboost_autoencoders"
    cnn = "cnn"
    rnn = "rnn"
    randomforest = "randomforest"
    cnn_max_pool = "cnn_max_pool"


class DataType:
    """An enum class for the data type"""
    real = "real"
    simulated = "simulated"

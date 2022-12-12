import abc


class Metric:
    """
    Base class for prediction metric/loss function
    """
    @abc.abstractmethod
    def __init__(self, config):
        raise NotImplementedError()


    @abc.abstractmethod
    def compute(self, model_outputs, data):
        """
        Main function that computes the metric
        :param model_outputs: Predictions generated by the model
        :param data: raw data contains ground truth infos
        :return metric: Tensor with computed value of metric.
        """
        raise NotImplementedError()
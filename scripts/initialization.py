from metrics.lane_select import LaneSelectionLoss
from metrics.pred_loss import PredictionLoss

def initialize_metric(metric_type):
    """
    Initialize appropriate metric by type.
    """
    metric_mapping = {
        'lane_select': LaneSelectionLoss,
        'prediction_loss': PredictionLoss
    }
    return metric_mapping[metric_type]()

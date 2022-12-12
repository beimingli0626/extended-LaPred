from metrics.lane_select import LaneSelectionLoss
from metrics.pred_loss import PredictionLoss
from metrics.mod_select import ModSelectionLoss

def initialize_metric(metric_type, config):
    """
    Initialize appropriate metric by type.
    """
    metric_mapping = {
        'lane_select': LaneSelectionLoss,
        'prediction_loss': PredictionLoss,
        'mod_select': ModSelectionLoss 
    }
    return metric_mapping[metric_type](config)

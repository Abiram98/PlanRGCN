
import numpy as np
def relative_error(preds, gt):
    diff = gt-preds
    abs_diff = np.absolute(diff)
    return abs_diff/gt

def relative_error_mean(preds, gt):
    re_s = relative_error(preds,gt)
    return np.mean(re_s)
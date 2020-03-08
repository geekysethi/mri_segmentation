import torch.nn as nn
import numpy as np

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc

    
    

def calc_IOU(predictionTensor,targetTensor):
    predictionTensor = predictionTensor.view(-1)
    targetTensor = targetTensor.view(-1)
    
    prediction = predictionTensor.cpu().detach().numpy()
    target = targetTensor.cpu().detach().numpy()

    prediction = np.round(prediction).astype(int)

    # print(prediction)
    # print(target)
    intersection = np.logical_and(prediction, target)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score


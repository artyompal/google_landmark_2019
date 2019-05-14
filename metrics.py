''' Add all the necessary metrics here. '''

import torch
from debug import dprint

def F_score(predict: torch.Tensor, label: torch.Tensor, beta: int,
            threshold: float = 0.5) -> torch.Tensor:
    predict = predict > threshold
    label = label > threshold

    TP = (predict & label).sum(1).float()
    TN = ((~predict) & (~label)).sum(1).float()
    FP = (predict & (~label)).sum(1).float()
    FN = ((~predict) & label).sum(1).float()

    precision = TP / (TP + FP + 1e-12)
    recall = TP / (TP + FN + 1e-12)
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    return F2.mean(0)

def GAP(predicts: torch.Tensor, confs: torch.Tensor, targets: torch.Tensor) -> float:
    ''' Computes GAP@1 '''
    if len(predicts.shape) != 1:
        dprint(predicts.shape)
        assert False

    if len(confs.shape) != 1:
        dprint(confs.shape)
        assert False

    if len(targets.shape) != 1:
        dprint(targets.shape)
        assert False

    assert predicts.shape == confs.shape and confs.shape == targets.shape

    sorted_confs, indices = torch.sort(confs, descending=True)

    confs = confs.cpu().numpy()
    predicts = predicts[indices].cpu().numpy()
    targets = targets[indices].cpu().numpy()

    res, true_pos = 0.0, 0

    for i, (c, p, t) in enumerate(zip(confs, predicts, targets)):
        rel = int(p == t)
        true_pos += rel

        res += true_pos / (i + 1) * rel

    res /= targets.shape[0] # FIXME: incorrect, not all test images depict landmarks
    return res

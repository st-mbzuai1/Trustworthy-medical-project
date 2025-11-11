
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

def softmax_np(x):
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)

def ece(probs, labels, n_bins=15):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece_val = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i+1])
        prop = in_bin.mean()
        if prop > 0:
            ece_val += np.abs(accuracies[in_bin].mean() - confidences[in_bin].mean()) * prop
    return float(ece_val)

def sens_at_spec(y_true, y_score, target_spec=0.95):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    spec = 1 - fpr
    valid = spec >= target_spec
    if valid.any(): return float(tpr[valid].max())
    return float(0.0)

def multiclass_metrics(y_true, logits, n_classes=7):
    probs = softmax_np(logits)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).mean()
    aurocs, auprcs, sens95 = [], [], []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        if len(set(y_bin))>1:
            aurocs.append(roc_auc_score(y_bin, probs[:,c]))
            auprcs.append(average_precision_score(y_bin, probs[:,c]))
            sens95.append(sens_at_spec(y_bin, probs[:,c], 0.95))
    from numpy import nanmean
    return {
        "acc": float(acc),
        "macro_auroc": float(nanmean(aurocs)),
        "macro_auprc": float(nanmean(auprcs)),
        "macro_sens_at_sp95": float(nanmean(sens95)),
        "ece": float(ece(probs, y_true))
    }

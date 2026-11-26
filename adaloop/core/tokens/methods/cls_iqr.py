"""cls_iqr: keep the tokens whose CLS attention is an outlier (TRIM's IQR rule), so the count differs per image."""
# Ported from FreedomIntelligence/TRIM:llava/model/multimodal_encoder/clip_encoder.py @ f2032236583b (Apache-2.0),
# the IQR block of `token_reduction` (l.93-107). `outlier_detection` is unchanged except the multiplier 1.5, which is
# the knob `m` here. Changes around it:
#   - score = CLS attention per unit (core/tokens/vision.py); upstream scores softmax(-patch·text similarity). The
#     attention is used as it is: no further softmax or sign change
#   - count = round(ratio * N), the number of outliers; upstream's int(ratio * N) (l.108) can lose one to float rounding
#   - at least 1 token (the highest score)
#   - no mean token of the unselected: our decoder never saw one
# This is not TRIM: the rule only.
import numpy as np
import torch


def cls_iqr_select(attention: torch.Tensor, m: float) -> tuple[torch.Tensor, int]:
    """(sorted indices of the kept tokens, number of outliers) for the scores `attention` (N,) and multiplier `m`."""

    def outlier_detection(sim):
        sim_np = sim.to(dtype=torch.float32).cpu().numpy().flatten()
        Q1 = np.percentile(sim_np, 25)
        Q3 = np.percentile(sim_np, 75)
        IQR = Q3 - Q1
        upper_bound = Q3 + m * IQR
        outlier_indices = np.where((sim_np > upper_bound))[0]
        ratio = len(outlier_indices) / len(sim_np)
        return ratio

    n_outliers = round(outlier_detection(attention) * len(attention))
    return attention.topk(max(1, n_outliers)).indices.sort().values, n_outliers

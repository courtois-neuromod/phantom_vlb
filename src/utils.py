"""
Source:
https://github.com/courtois-neuromod/phantom_LLM/blob/dev_beluga/phantom_LLM/src/ridge_align.py
"""

import numpy as np
import torch
import torch.nn as nn
from nilearn.glm.first_level import compute_regressor


def get_hrf_weight(time_diff: float) -> float:
    """
    Computes estimated hrf weights for a series of input tokens (language or visual),
    based on their temporal distance to the target BOLD frame it predicts.

    hrf weights are used to apply a weighted sum on the input token's
    latent (hidden) space embeddings in a manner that models the hemodynamic response function.

    Args: time_diff (np.array): absolute time difference between input token times and target TR time, in seconds.
        Note that target TR time is assigned to the middle of a BOLD slice. It corresponds to TR_onset + (TR_length/2)
    Output:
        hrf_weights (np.arrays): weights that estimate the relative impact of an input token on a target TR
        based on their respective timing and the shape of the hemodynamic response,
        as estimated with Nilearn's Glover function.
    """

    hrf_regressors, regressor_names = compute_regressor(
        exp_condition=np.array([[0], [1], [1]]),        # Dummy onsets, duration, amplitude
        #hrf_model="glover + derivative + dispersion",
        hrf_model="glover",
        frame_times=np.array([0.0, time_diff])
    )

    return hrf_regressors[-1, 0]


class HRFConvolveLayer(nn.Module):
    def __init__(self):
        super(HRFConvolveLayer, self).__init__()

    def forward(self, embeddings, hrf_weights):
        """
        Weight sum of input token latent embeddings according to their hrf_weight,
        which is based on their temporal distance to the target BOLD slice (a single TR)

        Args:
            embedding : torch.array, dim = (batch size, sequence lenght, embedding size)
            hrf_weights : torch.array, dim = (batch size, sequence lenght)
        Output:
            _ torch.array, dim = (batch size, encoding size)
        """

        return torch.einsum('bse,bs->be', embeddings, hrf_weights)


class RidgeRegressionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, l2_lambda=0.01):
        super(RidgeRegressionLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.l2_lambda = l2_lambda

    def forward(self, x, add_regularization=True):
        output = self.linear(x)
        output = self.layer_norm(output)  # Normalize the output
        if add_regularization:
            l2_reg = self.l2_lambda * torch.norm(self.linear.weight, p=2) ** 2
            return output, l2_reg
        else:
            return output

""" Module containing the loss function implementations. """
from torch import nn
import torch


def divide_no_nan(a, b):
    """
    Auxiliary funtion to handle divide by 0
    """
    div = a / b
    div[div != div] = 0.0
    div[div == float("inf")] = 0.0
    return div


#############################################################################
# FORECASTING LOSSES
#############################################################################


def MAPELoss(y, y_hat, mask=None):
    """MAPE Loss

    Calculates Mean Absolute Percentage Error between
    y and y_hat. MAPE measures the relative prediction
    accuracy of a forecasting method by calculating the
    percentual deviation of the prediction and the true
    value at a given time and averages these devations
    over the length of the series.

    Parameters
    ----------
    y: tensor (batch_size, output_size)
        actual values in torch tensor.
    y_hat: tensor (batch_size, output_size)
        predicted values in torch tensor.
    mask: tensor (batch_size, output_size)
        specifies date stamps per serie
        to consider in loss

    Returns
    -------
    mape:
    Mean absolute percentage error.
    """
    mask = divide_no_nan(mask, torch.abs(y))
    mape = torch.abs(y - y_hat) * mask
    mape = torch.mean(mape)
    return mape


def cos_loss(output, target):
    """
    Loss based on vector angle (needs n_out>=2)

    Args:
        output (RNN prediction), Tensor size [batch_size, seq_len, n_out]
        target, Tensor size [batch_size, seq_len, n_out]

    Returns:
        loss

    """
    cos = nn.CosineSimilarity(dim=1)
    loss = 1 - cos(output, target).mean()
    return loss


def pearson_loss(output, target):
    """
    Loss based on the pearson correlation coefficient (needs n_out>=2)
    As pearson correlation is between -1 and 1, the loss is 1-pearson

    Args:
        output (RNN prediction), Tensor size [batch_size, seq_len, n_out]
        target, Tensor size [batch_size, seq_len, n_out]

    Returns:
        loss
    """
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    pearson = cos(output - output.mean(dim=1, keepdim=True), target - target.mean(dim=1, keepdim=True)).mean()
    loss = 1 - pearson
    return loss


def mse_loss(output, target, mask=None):
    """
    Loss based on the mean squared error (needs n_out>=2)

    Args:
        output (RNN prediction), Tensor size [batch_size, seq_len, n_out]
        target, Tensor size [batch_size, seq_len, n_out]

    Returns:
        loss

    """
    if mask is not None:
        output = output * mask
        target = target * mask

    mse = nn.MSELoss()
    loss = mse(output, target)
    return loss


def smoothL1_loss(output, target, beta=0.1, mask=None):
    """
    Loss based on the mean squared error (needs n_out>=2)

    Args:
        output (RNN prediction), Tensor size [batch_size, seq_len, n_out]
        target, Tensor size [batch_size, seq_len, n_out]

    Returns:
        loss

    """
    if mask is not None:
        output = output * mask
        target = target * mask
    mse = nn.SmoothL1Loss(reduction="mean", beta=beta)
    loss = mse(output, target)
    return loss


def combined_loss(output, target, loss_type="mse", mask=None):
    """
    Loss based on the mean squared error and pearson correlation(needs n_out>=2)

    Args:
        output (RNN prediction), Tensor size [batch_size, seq_len, n_out]
        target, Tensor size [batch_size, seq_len, n_out]

    Returns:
        loss

    """
    if loss_type.upper() == "MAPE":
        mse = MAPELoss
    elif loss_type.upper() == "MSE":
        mse = nn.MSELoss(reduction="mean")
    elif loss_type.upper() == "HUBER":
        mse = smoothL1_loss
    else:
        raise ValueError(f"{loss_type} is not implemented")

    if mask is not None:
        output = output * mask
        target = target * mask

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    pearson = cos(output - output.mean(dim=1, keepdim=True), target - target.mean(dim=1, keepdim=True)).mean()
    # Add MSE and Pearson loss together
    loss = mse(output, target) + 1 - pearson
    return loss

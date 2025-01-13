import torch
from pathlib import Path

import flyvision

def save_model(model_path, model, model_name):
    """ Saves the model in a given path """
    state_dict_file = Path(model_path) / f"{model_name}model_state_dict.pkl"
    torch.save(model.state_dict(), state_dict_file)
    return

def load_model(model_path, model, model_name):
    """ Load the model from a given path. """
    device = flyvision.device
    state_dict_file = model_path / f"{model_name}model_state_dict.pkl"
    model.load_state_dict(torch.load(state_dict_file, map_location=torch.device(device)))
    return model

def predict(
    rnn,
    input,
    loss_fn=None,
    target=None,
    mask=None,
    x0=None,
):
    """
    Do a forward pass with an RNN
    Utility function to call outside of training

    Args:
        rnn: Initialized RNN
        input: input tensor of size [batch_size, seq_len, n_inp]
        loss_fn (optional): loss function
        target (optional), tensor of size [batch_size, seq_len, n_out]
        mask(optional), tensor of size [batch_size, seq_len, n_out]
        x0(optional), tensor of size [batch_size, n_rec]

    Returns:
        rates: tensor of size [batch_size, seq_len, n_rec]
        predict: tensor of size [batch_size, seq_len, n_out]

    """
    rnn.eval()
    # if single trial, add batch dimension
    if input.dim() < 3:
        input = input.unsqueeze(0)

    device = rnn.rnn.w_inp.device
    input = input.to(device=device)

    if loss_fn is not None:
        if target.dim() < 3:
            target = target.unsqueeze(0)
        if mask.dim() < 3:
            mask = mask.unsqueeze(0)
        target = target.to(device=device)
        mask = mask.to(device=device)

    with torch.no_grad():
        rates, predict = rnn(input, x0=x0)

        if loss_fn is not None:
            loss = loss_fn(predict, target, mask)
            print("test loss:", loss.item())
            print("==========================")

    return rates.cpu().detach().numpy(), predict.cpu().detach().numpy()

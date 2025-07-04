import torch
import torch.nn as nn


class TimeDistributed(nn.Module):

    """
    Applies a layer to a sequence of timesteps.

    This is done by collapsing the temporal dimension into the batch dimension and returning to the original shape after
    the layer has been applied.

    Assumes the input x is either 1) a list of tensors where each tensor represents a timestep and has shape (bs, ...)
    or 2) a single tensor with shape (bs, timesteps, ...)

    - layer: the layer to be applied to each timestep
    - to_list: boolean, if True out is converted to a list of tensor with length timesteps, if false out is a 
    single tensor with shape (bs, timesteps, ...) 
    """

    def __init__(self, layer, to_list=True):
        super(TimeDistributed, self).__init__()
        self.layer = layer
        self.to_list = to_list

    def forward(self, x):
        bs = x[0].size()[0] if isinstance(x, list) else x.size()[0]
        n_steps = len(x) if isinstance(x, list) else x.size()[1]
        x_collapsed = collapse_time_into_batch_dim(x)
        out_collapsed = self.layer(x_collapsed)
        out = decollapse_time_from_batch_dim(
            out_collapsed, n_steps, bs, to_list=self.to_list)
        return out


def collapse_time_into_batch_dim(x, mode="interleave_batches"):
    """
    x: either list of timesteps or tensor with batch first
    mode: str ("interleave", "stack"). If interleave time steps come together,
    if stack batches come together.
    e.g.: 2 batches, 3 time steps
    - interleave_batches: [b0t0, b1t0, b0t1, b1t1, b0t2, b1t2]
    - stack_batches: [b0t0, b0t1, b0t2, b1t0, b1t1, b1t2]
    """
    assert mode in ["interleave_batches", "stack_batches"]
    if isinstance(x, list):
        if mode == "interleave_batches":
            x = torch.stack(x, dim=0)
        else:
            x = torch.stack(x, dim=1)
    else:  # x is tensor of shape (bs, t, ...)
        if mode == "interleave_batches":
            x = torch.swapaxes(x, 0, 1)
    new_first_dim = x.size()[0] * x.size()[1]
    new_size = tuple([new_first_dim] + list(x.size())[2:])
    return x.contiguous().view(*new_size)


def decollapse_time_from_batch_dim(x, time_steps, bs, mode="interleave_batches", to_list=True):
    """
    to_list (bool): if True the output tensor is converted to a list of timesteps
                    if False the output is a tensor of shape (bs, t, ...)
    """

    assert mode in ["interleave_batches", "stack_batches"]
    if mode == "interleave_batches":
        new_size = tuple([time_steps, bs] + list(x.size())[1:])
        x = x.contiguous().view(*new_size)
        x = torch.swapaxes(x, 0, 1)
    else:
        new_size = tuple([bs, time_steps] + list(x.size())[1:])
        x = x.contiguous().view(*new_size)
    if to_list:
        return list(torch.unbind(x, dim=1))
    return x

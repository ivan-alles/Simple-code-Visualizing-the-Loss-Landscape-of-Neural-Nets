""" Implementation of the loss landscape visualization for a neural network from the paper

    Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer and Tom Goldstein. Visualizing the Loss Landscape of Neural Nets. NIPS, 2018.

    Based on:
    https://github.com/Leebh-kor/Simple-code-Visualizing-the-Loss-Landscape-of-Neural-Nets
    https://github.com/marcellodebernardi/loss-landscapes
"""

import torch


def create_random_direction(model, normalize=True):
    """
    Create random direction in the parameter space of the model.
    Code from original paper:
    https://github.com/tomgoldstein/loss-landscape/blob/64ef4d57f8dabe79b57a637819c44e48eda98f33/net_plotter.py#L196
    """
    parameters = [p.data for p in model.parameters()]
    direction = [torch.randn_like(w, device=w.device) for w in parameters]
    if normalize:
        normalize_direction(direction, parameters)
    return direction


def normalize_direction(direction, parameters, ignore="bias"):
    """ Filter-wise normalization.
    Code from original repo (section 4 in the paper):
        https://github.com/tomgoldstein/loss-landscape/blob/64ef4d57f8dabe79b57a637819c44e48eda98f33/net_plotter.py#L132

    :param direction: list of tensors comprising a direction
    :param parameters: list of model parameters, must contain the same tensor shapes as direction
    :param ignore: ignore bias (also batch norm) parameters.
    """
    assert (len(direction) == len(parameters))
    for d, p in zip(direction, parameters):
        if d.ndim <= 1:
            # Assume a bias parameter
            if ignore=="bias":
                d.fill_(0)
            else:
                d.copy_(p)
        else:
            # Weight parameter have shape (out_channels, in_channels, ...).
            # ... are kernels for conv layers.
            # TODO: transposed convolutions have (in_channels, out_channels, ...) convention, do we need to take it into account?
            for d_filter, p_filter in zip(d, p):
                assert d_filter.shape == p_filter.shape
                # Torch linalg.norm corresponds to a Frobenius norm for a flattened kernel of shape (num_elements, 1)
                d_filter *= torch.linalg.norm(p_filter) / (torch.linalg.norm(d_filter) + 1e-10)

def update_model(model, params0, directions, coeff):
    """
    Update the model parameters in the direction of the given coefficients.
    I.e. move the model to a new point in the parameter space.
    """
    alpha = coeff[0]
    beta = coeff[1]
    for p, p0, delta, eta in zip(model.parameters(), params0, directions[0], directions[1]):
        p.data.copy_(p0 + alpha * delta + beta * eta)


@torch.no_grad()
def compute_loss(model, loss_fn, device, data_loader, num_batches):
    """ Compute the loss at the point in the parameter space given by model's weight. """
    loss_sum = 0
    num_samples = 0

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        batch_size = inputs.size(0)
        num_samples += batch_size
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss_sum += loss.item() * batch_size
        if batch_idx + 1 >= num_batches:
            break

    loss = loss_sum / num_samples
    return loss


@torch.no_grad()
def compute_loss_landscape(model, loss_fn,  data_loader, num_batches=1, directions=None, min_val=-1, max_val=1, num_points=20, device=None):
    """
    directions: a tuple of 2 directions in the model parameter space. See also create_random_directions().
    :return: delta, eta - (num_points,) tensors with coordinates along directions, loss - (num_points, num_points) tensor with loss values.
    """
    device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    if directions is None:
        directions = [create_random_direction(model) for _ in range(2)]

    if len(directions) != 2:
        raise ValueError("directions must be a tuple of 2 directions")

    delta = torch.linspace(min_val, max_val, num_points)
    eta = torch.linspace(min_val, max_val, num_points)

    shape = (len(delta), len(eta))
    loss = torch.zeros(shape)

    params0 = [p.data.clone() for p in model.parameters()]  # Initial weights for (0, 0) point in delta, eta space

    for i in range(len(delta)):
        for j in range(len(eta)):
            # Move to the new point in the parameter space
            update_model(model, params0, directions, (delta[i].item(), eta[j].item()))

            # Evaluate the model at a given point in the parameter space
            l = compute_loss(model, loss_fn, device, data_loader, num_batches)
            loss[i, j] = l

    return delta, eta, loss






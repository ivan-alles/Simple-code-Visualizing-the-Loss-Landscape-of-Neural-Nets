""" Implementation of the loss landscape visualization for a neural network from the paper

    Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer and Tom Goldstein. Visualizing the Loss Landscape of Neural Nets. NIPS, 2018.

    Based on:
    https://github.com/Leebh-kor/Simple-code-Visualizing-the-Loss-Landscape-of-Neural-Nets
    https://github.com/marcellodebernardi/loss-landscapes
"""

import torch

def create_random_directions(model):
    x_direction = create_random_direction(model)
    y_direction = create_random_direction(model)
    return [x_direction, y_direction]


def create_random_direction(model):
    weights = get_weights(model)
    direction = get_random_weights(weights)
    normalize_directions_for_weights(direction, weights)

    return direction


def get_weights(model):
    return [p.data for p in model.parameters()]


def get_random_weights(weights):
    return [torch.randn(w.size()) for w in weights]


def normalize_direction(direction, weights):
    for d, w in zip(direction, weights):
        d.mul_(w.norm() / (d.norm() + 1e-10))


def normalize_directions_for_weights(directions, weights):
    assert (len(directions) == len(weights))
    for i in range(len(directions)):
        w = weights[i]
        d = directions[i].to(w.device)  # Ensure the direction is on the same device as the weights
        if d.dim() <= 1:
            d.fill_(0)
        normalize_direction(d, w)
        directions[i] = d  # Update the direction in the list

def overwrite_weights(model, init_weights, directions, step):
    dx = directions[0]  # Direction vector present in the scale of weights
    dy = directions[1]
    changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)]  # αδ + βη

    for (p, w, d) in zip(model.parameters(), init_weights, changes):
        p.data = w + d  # θ^* + αδ + βη

@torch.no_grad()
def compute_loss(model, loss_fn, device, data_loader, num_batches):
    """ Compute the loss at the given point in model parameter space. """
    total_loss = 0
    num_samples = 0

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        batch_size = inputs.size(0)
        num_samples += batch_size
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        total_loss += loss.item() * batch_size
        if batch_idx + 1 >= num_batches:
            break

    return total_loss / num_samples


@torch.no_grad()
def compute_loss_landscape(model, loss_fn, data_loader, num_batches=1, min_val=-1, max_val=1, num_points=50, device=None):
    """
    directions : filter-wise normalized directions(d = (d / d.norm) * w.norm, d is random vector from gausian distribution)
    To make d have the same norm as w.
    """
    device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()

    directions = create_random_directions(model)  # Create random directions

    x_coordinates = torch.linspace(min_val, max_val, num_points)
    y_coordinates = torch.linspace(min_val, max_val, num_points)

    shape = (len(x_coordinates), len(y_coordinates))
    losses = torch.zeros(shape)

    init_weights = [p.data for p in model.parameters()]  # pretrained weights

    for xi in range(len(x_coordinates)):
        for yi in range(len(y_coordinates)):
            # Move to the new point in the parameter space
            overwrite_weights(model, init_weights, directions, (x_coordinates[xi].item(), y_coordinates[yi].item()))

            # Evaluate the model at a given point in the parameter space
            loss = compute_loss(model, loss_fn, device, data_loader, num_batches)
            losses[xi, yi] = loss

    return x_coordinates, y_coordinates, losses




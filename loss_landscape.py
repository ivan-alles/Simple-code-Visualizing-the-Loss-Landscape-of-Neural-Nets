""" Implementation of the loss landscape visualization for a neural network from the paper

    Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer and Tom Goldstein. Visualizing the Loss Landscape of Neural Nets. NIPS, 2018.

    Based on:
    https://github.com/Leebh-kor/Simple-code-Visualizing-the-Loss-Landscape-of-Neural-Nets
    https://github.com/marcellodebernardi/loss-landscapes
"""

import torch

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
def compute_loss_landscape(model, loss_fn, device, data_loader, num_batches, directions, min_val=-1, max_val=1, num_points=50):
    """
    directions : filter-wise normalized directions(d = (d / d.norm) * w.norm, d is random vector from gausian distribution)
    To make d have the same norm as w.
    """

    model.eval()

    x_coordinates = torch.linspace(min_val, max_val, num_points)
    y_coordinates = torch.linspace(min_val, max_val, num_points)

    shape = (len(x_coordinates), len(y_coordinates))
    losses = torch.zeros(shape=shape)

    init_weights = [p.data for p in model.parameters()]  # pretrained weights

    for xi in range(len(x_coordinates)):
        for yi in range(len(y_coordinates)):
            # Move to the new point in the parameter space
            overwrite_weights(model, init_weights, directions, (x_coordinates[xi].item(), y_coordinates[yi].item()))

            # Evaluate the model at a given point in the parameter space
            loss = compute_loss(model, loss_fn, device, data_loader, num_batches)
            losses[xi, yi] = loss

    return x_coordinates, y_coordinates, losses




""" Implementation of the loss landscape visualization for a neural network from the paper

    Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer and Tom Goldstein. Visualizing the Loss Landscape of Neural Nets. NIPS, 2018.

    Based on:
    https://github.com/Leebh-kor/Simple-code-Visualizing-the-Loss-Landscape-of-Neural-Nets
    https://github.com/marcellodebernardi/loss-landscapes
"""

import torch
import numpy as np
import h5py
import os
from src.test_model import eval_loss
from src.dataloader import dataloader


def calulate_loss_landscape(args, model, directions, save_path):
    """
    directions : filter-wise normalized directions(d = (d / d.norm) * w.norm, d is random vector from gausian distribution)
    To make d have the same norm as w.
    """
    surface_path = setup_surface_file(args, save_path)
    init_weights = [p.data for p in model.parameters()]  # pretrained weights

    with h5py.File(surface_path, 'r+') as f:
        xcoordinates = f['xcoordinates'][:]
        ycoordinates = f['ycoordinates'][:]
        losses = f["train_loss"][:]
        accuracies = f["train_acc"][:]

        inds, coords = get_indices(losses, xcoordinates, ycoordinates)

        _, _, land_loader = dataloader()

        for count, ind in enumerate(inds):
            print("ind...%s" % ind)
            coord = coords[count]
            overwrite_weights(model, init_weights, directions, coord)

            loss, acc = eval_loss(model, land_loader)
            print(loss, acc)

            losses.ravel()[ind] = loss
            accuracies.ravel()[ind] = acc

            print('Evaluating %d/%d  (%.1f%%)  coord=%s' % (
                ind, len(inds), 100.0 * count / len(inds), str(coord)))

            f["train_loss"][:] = losses
            f["train_acc"][:] = accuracies
            f.flush()

            # if ind % 300 == 0:
            #    break

    return surface_path


def setup_surface_file(args, save_path):
    surface_path = f"{save_path}/3d_surface_file.h5"

    with h5py.File(surface_path, 'w') as f:
        print("Create new 3d_sureface_file.h5")

        xcoordinates = np.linspace(args.xmin, args.xmax, args.xnum)
        f['xcoordinates'] = xcoordinates

        ycoordinates = np.linspace(args.ymin, args.ymax, args.ynum)
        f['ycoordinates'] = ycoordinates

        shape = (len(xcoordinates), len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = np.ones(shape=shape)

        f["train_loss"] = losses
        f["train_acc"] = accuracies

        return surface_path


def get_indices(vals, xcoordinates, ycoordinates):
    inds = np.array(range(vals.size))
    inds = inds[vals.ravel() <= 0]

    xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
    s1 = xcoord_mesh.ravel()[inds]
    s2 = ycoord_mesh.ravel()[inds]

    return inds, np.c_[s1, s2]


def overwrite_weights(model, init_weights, directions, step):
    dx = directions[0]  # Direction vector present in the scale of weights
    dy = directions[1]
    changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)]  # αδ + βη

    for (p, w, d) in zip(model.parameters(), init_weights, changes):
        p.data = w + d  # θ^* + αδ + βη

def main():
    """ Test the loss landscape visualization on a pretrained model. """
    import torchvision

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((224, 224)),
        # Normalize for resnet model

    ])

    test_dataset = torchvision.datasets.CIFAR10(root='./data/', train=False, transform=transforms.ToTensor(),
                                                download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False, num_workers=2)
    land_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=False, num_workers=2, persistent_workers=True)



if __name__ == "main":
    main()

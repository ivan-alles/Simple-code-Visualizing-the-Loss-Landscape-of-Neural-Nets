import torch
from torch import nn

def eval_loss(model, data_loader, num_batches=10):
    criterion = nn.CrossEntropyLoss()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    correct = 0
    total_loss = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            batch_size = inputs.size(0)
            total += batch_size
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets).sum().item()

            if batch_idx >= num_batches:
                break

            # if batch_idx == 100:
                # break

    return total_loss / total, 100. * correct / total

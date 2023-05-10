import torch

def train_epoch(model, device, train_loader, criteria, optimizer):
    model.train()

    running_loss = 0.
    n_ep_it_loss = 300

    for i, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)

        output = model(X)
        loss = criteria(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

def eval_epoch(model, device, test_loader, criteria):
    model.eval()
    losses = []
    for i, (X, y) in enumerate(test_loader):
        X = X.to(device)
        y = y.to(device)

        with torch.no_grad():
            output = model(X)
        loss = criteria(output, y)
        losses.append(loss)

    mean_loss = sum(losses) / len(losses)

    return mean_loss
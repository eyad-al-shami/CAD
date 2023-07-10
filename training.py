import torch
import tqdm

def train(model, train_dataloader, test_dataloader, cfg):
    model.to(cfg.DEVICE)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.MODEL.LR, momentum=0.9)
    for epoch in tqdm.tqdm(range(cfg.TRAIN.EPOCHS)):
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output['predictions'], target)
            loss.backward()
            optimizer.step()
        test(model, test_dataloader, cfg)


def test(model, dataloader, cfg):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(dataloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss,))

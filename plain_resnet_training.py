import torch
import tqdm
import wandb
import torch.nn.functional as F



def train(model, train_dataloader, test_dataloader, wandb, cfg):
    model.to(cfg.DEVICE)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.MODEL.LR, momentum=0.9)
    
    for epoch in range(1, cfg.TRAIN.EPOCHS+1):
        correct = 0
        for data, target in tqdm.tqdm(train_dataloader):
            data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        print('Train set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(train_dataloader.dataset), 100. * correct / len(train_dataloader.dataset)))
        wandb.log({'epoch': epoch, 'loss': loss.item(), 'train_accuracy': correct / len(train_dataloader.dataset)})
        test(model, test_dataloader, cfg)


def test(model, dataloader, cfg, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in tqdm.tqdm(dataloader):
            data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(dataloader.dataset)
    wandb.log({'epoch': epoch, 'test_loss': test_loss, 'test_accuracy': correct / len(dataloader.dataset)})
    print('Test set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(dataloader.dataset), 100. * correct / len(dataloader.dataset)))

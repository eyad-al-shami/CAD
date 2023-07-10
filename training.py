import torch
import tqdm
import wandb
import torch.nn.functional as F



def train(model, train_dataloader, test_dataloader, wandb, cfg):
    model.to(cfg.DEVICE)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.MODEL.LR, momentum=0.9)

    activation = {}
    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activation[name] = output
        return hook

    model.mask_estimator.register_forward_hook(getActivation('mask'))
    
    for epoch in range(1, cfg.TRAIN.EPOCHS+1):
        correct = 0
        for data, target in tqdm.tqdm(train_dataloader):
            data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)
            optimizer.zero_grad()
            output = model(data)
            
            mask = activation['mask']
            mask = F.gumbel_softmax(mask, tau=1, hard=True, dim=1)
            mask_low_res_active = mask[:,1:2,:,:].sum() / mask[:,1:2,:,:].numel()
            loss_mask = (mask_low_res_active - 0.3).square()

            loss = criterion(output['predictions'], target) + loss_mask
            loss.backward()
            optimizer.step()
            pred = output['predictions'].argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        print('Train set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(train_dataloader.dataset), 100. * correct / len(train_dataloader.dataset)))
        wandb.log({'epoch': epoch, 'loss': loss.item(), 'train_accuracy': correct / len(train_dataloader.dataset)})
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
    wandb.log({'test_loss': test_loss, 'test_accuracy': correct / len(dataloader.dataset)})
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss,))

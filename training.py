import torch
import tqdm
# import wandb
import torch.nn.functional as F
from utils import save_model
import os


class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader, wandb, cfg):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.wandb = wandb
        self.cfg = cfg
        self.highest_test_accuracy = 0
        self.output_dir = os.path.join(self.cfg.OUTPUT_DIR, self.cfg.LOGGING.EXPERIMENT_NAME)

    def train(self):
        self.model.to(self.cfg.DEVICE)
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.MODEL.LR, momentum=0.9)

        activation = {}
        def getActivation(name):
            # the hook signature
            def hook(model, input, output):
                activation[name] = output
            return hook

        self.model.mask_estimator.register_forward_hook(getActivation('mask'))
        
        for epoch in range(1, self.cfg.TRAIN.EPOCHS+1):
            correct = 0
            for data, target in tqdm.tqdm(self.train_dataloader):
                data, target = data.to(self.cfg.DEVICE), target.to(self.cfg.DEVICE)
                optimizer.zero_grad()
                predictions, mask = self.model(data)
                
                # mask = activation['mask']
                mask = F.gumbel_softmax(mask, tau=1, hard=True, dim=1)
                mask_low_res_active = mask[:,1:2,:,:].sum() / mask[:,1:2,:,:].numel()
                loss_mask = (mask_low_res_active - self.cfg.MODEL.MASK_LOW_RES_ACTIVE).square()

                loss = criterion(predictions, target) + loss_mask
                loss.backward()
                optimizer.step()
                pred = predictions.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            print('At Epoch  {}:  Train set: Accuracy: {}/{} ({:.0f}%)\n'.format(epoch, correct, len(self.train_dataloader.dataset), 100. * correct / len(self.train_dataloader.dataset)))
            self.wandb.log({'loss': loss.item(), 'train_accuracy': correct / len(self.train_dataloader.dataset)})
            self.test(epoch)
        return self.model, self.highest_test_accuracy


    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for data, target in tqdm.tqdm(self.test_dataloader):
                data, target = data.to(self.cfg.DEVICE), target.to(self.cfg.DEVICE)
                output = self.model(data)
                test_loss += criterion(output['predictions'], target).item()
                pred = output['predictions'].argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_dataloader.dataset)
        self.wandb.log({'test_loss': test_loss, 'test_accuracy': correct / len(self.test_dataloader.dataset)})
        print('At Epoch  {}:  Test set: Accuracy: {}/{} ({:.0f}%)\n'.format(epoch, correct, len(self.test_dataloader.dataset), 100. * correct / len(self.test_dataloader.dataset)))
        save_model(self.model, epoch, correct / len(self.test_dataloader.dataset), self.cfg)

    def save_model_(self, accuracy, epoch):
        if (accuracy > self.highest_test_accuracy):
            if (os.path.exists(self.output_dir) == False):
                os.makedirs(self.output_dir)
            model_path = os.path.join(self.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'accuracy': accuracy,
                'cfg': self.cfg
            }, model_path)
            self.model.save(os.path.join(self.wandb.run.dir, "best_model.pth"))

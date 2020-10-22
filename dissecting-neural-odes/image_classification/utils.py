import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

def get_cifar_dloaders(batch_size=64, size=32, path='../data/cifar10_data', num_workers=20):
    batch_size=batch_size
    size=size
    path_to_data=path

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = datasets.CIFAR10(path_to_data, train=True, download=True,
                                  transform=transform_train)
    test_data = datasets.CIFAR10(path_to_data, train=False,
                                 transform=transform_test)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    return trainloader, testloader
    
    
class CIFARLearner(pl.LightningModule):
    def __init__(self, model:nn.Module, trainloader, testloader):
        super().__init__()
        self.model = model
        self.iters = 0.
        self.trainloader, self.testloader = trainloader, testloader
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        self.iters += 1.
        x, y = batch   
        self.model[2].nfe = 0
        y_hat = self.model(x)   
        loss = nn.CrossEntropyLoss()(y_hat, y)
        epoch_progress = self.iters / self.loader_len
        acc = accuracy(y_hat, y)
        nfe = self.model[2].nfe ; self.model[2].nfe = 0
        tqdm_dict = {'train_loss': loss, 'accuracy': acc, 'NFE': nfe}
        return {'loss': loss, 'progress_bar': tqdm_dict}   

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        acc = accuracy(y_hat, y)
        return {'test_loss': nn.CrossEntropyLoss()(y_hat, y), 'test_accuracy': acc}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_accuracy'] for x in outputs]).mean()
        return {'avg_test_loss': avg_loss, 'avg_test_accuracy': avg_acc}
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=5e-4)
        sched = {'scheduler': torch.optim.lr_scheduler.StepLR(opt, 5, gamma=0.9),
                 'monitor': 'loss', 
                 'interval': 'epoch'}
        return [opt], [sched]

    def train_dataloader(self):
        self.loader_len = len(self.trainloader)
        return self.trainloader

    def test_dataloader(self):
        self.test_loader_len = len(self.testloader)
        return self.testloader

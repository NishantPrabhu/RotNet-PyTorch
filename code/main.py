
import torch 
import models 
import data_utils
import train_utils
import torch.nn as nn
from tqdm import tqdm
from torchvision import datasets
import torch.nn.functional as F 
import torch.optim as optim
from termcolor import cprint
import wandb
import yaml
import argparse
import numpy as np


def collate_fn(data):
    """ Collate function for dataloaders """
    batch_size = len(data)
    in_size = data[0]['img'].size()
    tensor, labels = [], []
    for d in data:
        tensor.append(d['img'])
        labels.append(d['target'])
    tensor = torch.cat(tensor, dim=0)
    labels = torch.cat(labels, dim=0)
    return {'img': tensor.float(), 'target': labels.long()}


class Trainer:

    def __init__(self, config):
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Models and optimizers
        self.encoder = models.Encoder(**config['encoder']).to(self.device)
        self.clf_head = models.RotnetClassifier(in_dim=self.encoder.backbone_dim, n_classes=4).to(self.device)
        self.optim = train_utils.get_optimizer(config['optimizer'], list(self.encoder.parameters())+list(self.clf_head.parameters()))
        self.scheduler, _ = train_utils.get_scheduler(config['scheduler'], self.optim)
        
        # Loss and others
        self.criterion = nn.NLLLoss()
        self.best_acc = 0
        wandb.init('rotnet-test')


    def train_one_epoch(self, epoch, data_loader):
        
        self.encoder.train()
        self.clf_head.train()
        pbar = tqdm(total=len(data_loader), desc=f'[Train epoch] {epoch}')
        losses = []

        for batch in data_loader:
            img, labels = batch['img'].to(self.device), batch['target'].to(self.device)
            out = self.clf_head(self.encoder(img))
            loss = self.criterion(out, labels)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            pbar.update(1)
            losses.append(loss.item())
            wandb.log({'NLL loss': loss.item()})
        
        avg_loss = np.mean(losses)
        pbar.set_description(f"[Train epoch] {epoch} - [Average NLL Loss] {round(avg_loss, 4)}")
        pbar.close()
        return avg_loss


    def validate(self, epoch, val_loader):

        self.encoder.eval()
        self.clf_head.eval()
        pbar = tqdm(total=len(val_loader), desc=f'[Train epoch] {epoch}')
        total_correct = 0

        with torch.no_grad():
            for batch in val_loader:
                img, labels = batch['img'].to(self.device), batch['target'].to(self.device)
                preds = self.clf_head(self.encoder(img)).argmax(dim=-1)   
                total_correct += preds.eq(target.view_as(preds)).sum().item() 
                pbar.update(1)
        
        acc = total_correct/len(val_loader.dataset)
        wandb.log({'Validation accuracy': acc})
        pbar.set_description(f"[Eval epoch] {epoch} - [Accuracy] {round(acc, 4)}")
        pbar.close()
        return acc


    def linear_eval(self, train_loader, val_loader):
        
        self.encoder.eval()
        linear_clf = models.LinearClassifier(in_dim=self.encoder.backbone_dim, n_classes=self.config['dataset']['num_classes']).to(self.device)
        clf_optim = train_utils.get_optimizer(config['clf_optimizer'], linear_clf.parameters())
        clf_scheduler, _ = train_utils.get_scheduler(config['clf_scheduler'], clf_optim)

        for epoch in self.config['linear_eval_epochs']:
            train_losses = []
            train_correct = 0
            pbar = tqdm(total=len(train_loader), desc=f'[Train epoch] {epoch+1}')

            for batch in train_loader:
                img, trg = batch[0].to(self.device), batch[1].to(self.device)
                with torch.no_grad():
                    vecs = self.encoder(img).detach()
                
                clf_out = linear_clf(vecs)
                loss = F.nll_loss(clf_out, trg, reduction='mean')
                clf_optim.zero_grad()
                loss.backward()
                clf_optim.step()
                
                train_losses.append(loss.item())
                pred = clf_out.argmax(dim=-1)
                train_correct += pred.eq(trg.view_as(pred)).sum().item()
                pbar.update(1)

            train_loss_avg = np.mean(train_losses)
            train_acc = train_correct/len(train_loader.dataset)
            pbar.set_description(f'[Train epoch] {epoch+1} - [Average loss] {round(train_loss_avg, 4)} - [Accuracy] {round(train_acc, 4)}')
            pbar.close()
            wandb.log({'Linear eval loss': train_loss_avg, 'Linear eval train acc': train_acc, 'Epoch': epoch+1})

            if (epoch+1) % self.config['linear_eval_valid_every']:
                val_correct = 0

                for batch in val_loader:
                    img, trg = batch[0].to(self.device), batch[1].to(self.device)
                    with torch.no_grad():
                        vecs = self.encoder(img).detach()
                    
                    clf_out = linear_clf(vecs)
                    pred = clf_out.argmax(dim=-1)
                    val_correct += pred.eq(trg.view_as(pred)).sum().item()
                    pbar.update(1) 

                val_acc = val_correct/len(val_loader.dataset)
                pbar.set_description(f'[Val epoch] {epoch+1} - [Val accuracy] {round(val_acc, 4)}')
                pbar.close()
                wandb.log({'Linear eval val acc': val_acc, 'Epoch': epoch+1})

            clf_scheduler.step()


    def train(self, train_loader, val_loader):

        for epoch in range(self.config['epochs']):
            avg_loss = self.train_one_epoch(epoch+1, train_loader)   
            
            if (epoch+1) % self.config['eval_every'] == 0:
                acc = self.validate(epoch+1, val_loader)
                
                if acc > self.best_acc:
                    cprint(f"[INFO] Accuracy improved from {round(self.best_acc, 4)} -> {round(acc, 4)}; saving data", 'yellow')
                    self.best_acc = acc
                    torch.save(self.encoder.state_dict(), '../saved_data/models/best_encoder.ckpt')
                    torch.save(self.clf_head.state_dict(), '../saved_data/models/best_clf_head.ckpt') 
            
            self.scheduler.step()

        
if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True, help='path to config file')
    args = vars(ap.parse_args())

    config = yaml.safe_load(open(args['config'], 'r'))
    print()
    trainer = Trainer(config)

    # Datasets and loaders
    train_dset = data_utils.RotnetDataset(config['dataset'], split='train')
    val_dset = data_utils.RotnetDataset(config['dataset'], split='val')
    trainset = data_utils.get_dataset(config['dataset'], split='train')
    valset = data_utils.get_dataset(config['dataset'], split='val')

    train_loader = data_utils.get_dataloader(train_dset, config['batch_size'], config['num_workers'], shuffle=True, weigh=False, collate_fn=collate_fn)
    val_loader = data_utils.get_dataloader(val_dset, config['batch_size'], config['num_workers'], shuffle=False, weigh=False, collate_fn=collate_fn)
    trainloader = data_utils.get_dataloader(trainset, config['batch_size'], config['num_workers'], shuffle=True, weigh=False)
    valloader = data_utils.get_dataloader(valset, config['batch_size'], config['num_workers'], shuffle=False, weigh=False)

    # Train
    trainer.train(train_loader, val_loader)

    # Linear evaluation
    trainer.linear_eval(trainloader, valloader)
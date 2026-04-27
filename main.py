import torch
import warnings
import random
import argparse
from utils import PoisonedTestSetCIFAR10, PoisonedTrainSetCIFAR10
from utils import PoisonedTrainSetMiniImageNet, PoisonedTestSetMiniImageNet, PoisonedTestSetCIFAR100, PoisonedTrainSetCIFAR100, PoisonedTrainSetCaltech101, PoisonedTestSetCaltech101
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.models as models
import numpy as np
from tqdm import tqdm
warnings.filterwarnings("ignore", category=UserWarning)


parser = argparse.ArgumentParser()

parser.add_argument('--learning', type=str, default='trans', choices=['trans', 'e2e'],
                    help='Learning mode.')
parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'MiniImageNet', 'Caltech101'],
                    help='Dataset name.')
parser.add_argument('--loc', type=str, default='AdaPT_BadNets', choices=['BadNets', 'Blend', 'AdaPT_BadNets', 'AdaPT_Blend'],
                    help='Attack type.')
parser.add_argument('--surrogate_models', nargs='+', type=str, default=['resnet50'], choices=['resnet50', 'resnet101', 'regnet_x_3_2gf'],
                    help='List of surrogate models used to compute Grad-NAM.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs.')
parser.add_argument('--p', type=float, default=0.1,
                    help='Poisoning rate.')

args = parser.parse_args()

DATASET_ROOTS = {
    "CIFAR10": r"/data/zc/datasets/CIFAR10",
    "CIFAR100": r"/data/zc/datasets/CIFAR100",
    "MiniImageNet": r"/data/zc/datasets/mini-imagenet",
    "Caltech101": r"/data/zc/datasets/caltech-101/101_ObjectCategories",
}

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class Runner:
    def __init__(self, learning, data, loc, surrogate_models=['resnet50']):
        self.device = 'cuda'
        self.learning = learning
        self.data = data
        self.loc = loc
        self.surrogate_models = surrogate_models

        # 数据
        if self.data == 'CIFAR10':
            self.out_dim = 10
        if self.data == 'MiniImageNet':
            self.out_dim = 100
        if self.data == 'CIFAR100':
            self.out_dim = 100
        if self.data == 'Caltech101':
            self.out_dim = 101

        if self.learning == 'trans':
            # 网络(迁移学习)
            self.network = models.resnet50(pretrained=True)
            self.network.fc = torch.nn.Linear(self.network.fc.in_features, self.out_dim)
            for param in self.network.parameters():
                param.requires_grad = False
            for param in self.network.fc.parameters():
                param.requires_grad = True
            self.network.to(self.device)

            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.SGD(self.network.fc.parameters(), lr=0.005)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=20, gamma=0.8, last_epoch=-1
            )

        if self.learning == 'e2e':
            # 网络(端到端学习)
            self.network = models.resnet18(pretrained=False)
            self.network.fc = torch.nn.Linear(self.network.fc.in_features, self.out_dim)
            self.network.to(self.device)

            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001, weight_decay=1e-4)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=20, gamma=0.8, last_epoch=-1
            )

    def load_train_data(self, batch_size, p, loc):
        if self.data == 'CIFAR10':
            dataset = PoisonedTrainSetCIFAR10(
                root=DATASET_ROOTS[self.data],
                p_rate=p,
                loc=loc,
                attack_target=0,
                surrogate_models=self.surrogate_models
            )

        if self.data == 'CIFAR100':
            dataset = PoisonedTrainSetCIFAR100(
                root=DATASET_ROOTS[self.data],
                p_rate=p,
                loc=loc,
                attack_target=0,
                surrogate_models=self.surrogate_models
            )

        if self.data == 'MiniImageNet':
            dataset = PoisonedTrainSetMiniImageNet(
                root=DATASET_ROOTS[self.data],
                p_rate=p,
                loc=loc,
                attack_target=0,
                surrogate_models=self.surrogate_models
            )

        if self.data == 'Caltech101':
            dataset = PoisonedTrainSetCaltech101(
                root=DATASET_ROOTS[self.data],
                p_rate=p,
                loc=loc,
                attack_target=0,
                surrogate_models=self.surrogate_models
            )

        indices = list(range(len(dataset)))
        random.shuffle(indices)
        split = int(len(dataset) * 0.1)
        val_indices, train_indices = indices[:split], indices[split:]
        val_dataset = Subset(dataset, val_indices)
        train_dataset = Subset(dataset, train_indices)

        dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        return dataloader_train, dataloader_val

    def load_test_data(self, batch_size, p, loc):
        if self.data == 'CIFAR10':
            dataset = PoisonedTestSetCIFAR10(
                root=DATASET_ROOTS[self.data],
                p_rate=p,
                loc=loc,
                attack_target=0,
                surrogate_models=self.surrogate_models
            )

        if self.data == 'CIFAR100':
            dataset = PoisonedTestSetCIFAR100(
                root=DATASET_ROOTS[self.data],
                p_rate=p,
                loc=loc,
                attack_target=0,
                surrogate_models=self.surrogate_models
            )

        if self.data == 'MiniImageNet':
            dataset = PoisonedTestSetMiniImageNet(
                root=DATASET_ROOTS[self.data],
                p_rate=p,
                loc=loc,
                attack_target=0,
                surrogate_models=self.surrogate_models
            )

        if self.data == 'Caltech101':
            dataset = PoisonedTestSetCaltech101(
                root=DATASET_ROOTS[self.data],
                p_rate=p,
                loc=loc,
                attack_target=0,
                surrogate_models=self.surrogate_models
            )

        dataloader_test = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader_test

    def train(self, epochs, p):
        loc = self.loc
        asr_log, bac_log = [], []
        dataloader_train, dataloader_val = self.load_train_data(batch_size=256, p=p, loc=loc)

        for i in range(epochs):
            # 1. train
            for j, (x, y) in enumerate(dataloader_train):
                self.network.train()
                x, y = x.to(self.device), y.to(self.device)
                out = self.network(x)
                loss = self.criterion(out, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_acc = torch.eq(out.argmax(1), y).float().mean().item()
                print('epoch:', i, 'iteration:', '[{}/{}]'.format(j, len(dataloader_train)),
                      'train loss:', round(loss.item(), 4), 'train acc:', round(train_acc, 4))

            # 2. eval
            self.network.eval()
            pred_list = []
            true_list = []
            with torch.no_grad():
                for j, (x, y) in enumerate(dataloader_val):
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.network(x)
                    pred_list.append(out.argmax(1))
                    true_list.append(y)
            pred_tensor = torch.cat(pred_list, dim=0).to('cpu')
            true_tensor = torch.cat(true_list, dim=0).to('cpu')
            eval_acc = torch.eq(pred_tensor, true_tensor).float().mean().item()
            print('epoch:', i, 'eval acc:', round(eval_acc, 4))

            # 3. save
            if i % 50 == 49:
                torch.save(
                    self.network.state_dict(),
                    'checkpoints/Data[{}]_Learning[{}]_Poison[{}]_Position[{}]_Epoch[{}]_EvalAcc[{}].pth'
                    .format(self.data, self.learning, p, loc, i, round(eval_acc, 4))
                )

            print('computing ba...')
            bac, _, _, _ = self.tst(p=0)
            print('computing asr...')
            _, asr, _, _ = self.tst(p=1)
            print('epoch:', i, 'tst ba:', round(bac, 4), 'tst asr:', round(asr, 4))

            bac_log.append(round(bac, 4))
            asr_log.append(round(asr, 4))
            np.savetxt(
                'results/Log-BA_Data[{}]_Learning[{}]_Position[{}]'.format(self.data, self.learning, loc),
                bac_log, fmt='%.4f'
            )
            np.savetxt(
                'results/Log-ASR_Data[{}]_Learning[{}]_Position[{}]'.format(self.data, self.learning, loc),
                asr_log, fmt='%.4f'
            )

    def tst(self, p):
        loc = self.loc
        dataloader_test = self.load_test_data(batch_size=128, p=p, loc=loc)

        self.network.eval()
        pred_list = []
        true_list = []
        with torch.no_grad():
            for i, (x, y) in tqdm(enumerate(dataloader_test)):
                x, y = x.to(self.device), y.to(self.device)
                out = self.network(x)
                pred_list.append(out.argmax(1))
                true_list.append(y)

        pred_tensor = torch.cat(pred_list, dim=0).to('cpu')
        true_tensor = torch.cat(true_list, dim=0).to('cpu')
        acc = torch.eq(pred_tensor, true_tensor).float().mean().item()

        y_true = true_tensor.numpy()
        y_pred = pred_tensor.numpy()

        n_att_all = (y_true != 0).sum()
        n_att_suc = ((y_true != 0) & (y_pred == 0)).sum()
        asr = n_att_suc / n_att_all

        return acc, asr, y_true, y_pred


if __name__ == '__main__':
    runner = Runner(
        learning=args.learning,
        data=args.data,
        loc=args.loc,
        surrogate_models=args.surrogate_models
    )
    runner.train(epochs=args.epochs, p=args.p)

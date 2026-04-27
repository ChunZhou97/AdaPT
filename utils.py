import os
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
import torchvision.transforms as transforms


class PoisonedTrainSetCIFAR10(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,
                 p_rate=0.1, loc='BadNets', attack_target=0, surrogate_models=['resnet50']):
        super(PoisonedTrainSetCIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.p_rate = p_rate
        self.attack_target = attack_target
        self.loc = loc
        self.surrogate_models = surrogate_models

        position_path = get_position_path("CIFAR10", self.surrogate_models, "train")
        self.positions = np.loadtxt(position_path, dtype=int)

    def __getitem__(self, item):
        x, y = super(PoisonedTrainSetCIFAR10, self).__getitem__(item)
        position = self.positions[item]

        transform_size = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        transforms_poison = AddTrigger(
            p=self.p_rate,
            tigger_type=self.loc,
            position=position,
            attack_target=self.attack_target,
            mode='train'
        )

        x = transform_size(x)
        x, y = transforms_poison(x, y, item)
        return x, y


class PoisonedTestSetCIFAR10(CIFAR10):
    def __init__(self, root, train=False, transform=None, target_transform=None, download=False,
                 p_rate=1, loc='BadNets', attack_target=0, surrogate_models=['resnet50']):
        super(PoisonedTestSetCIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.p_rate = p_rate
        self.attack_target = attack_target
        self.loc = loc
        self.surrogate_models = surrogate_models

        position_path = get_position_path("CIFAR10", self.surrogate_models, "test")
        self.positions = np.loadtxt(position_path, dtype=int)

    def __getitem__(self, item):
        x, y = super(PoisonedTestSetCIFAR10, self).__getitem__(item)
        position = self.positions[item]

        transform_size = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        transforms_poison = AddTrigger(
            p=self.p_rate,
            tigger_type=self.loc,
            position=position,
            attack_target=self.attack_target,
            mode='test'
        )

        x = transform_size(x)
        x, y = transforms_poison(x, y, item)
        return x, y


class PoisonedTrainSetCIFAR100(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,
                 p_rate=0.1, loc='BadNets', attack_target=0, surrogate_models=['resnet50']):
        super(PoisonedTrainSetCIFAR100, self).__init__(root, train, transform, target_transform, download)
        self.p_rate = p_rate
        self.attack_target = attack_target
        self.loc = loc
        self.surrogate_models = surrogate_models

        position_path = get_position_path("CIFAR100", self.surrogate_models, "train")
        self.positions = np.loadtxt(position_path, dtype=int)

    def __getitem__(self, item):
        x, y = super(PoisonedTrainSetCIFAR100, self).__getitem__(item)
        position = self.positions[item]

        transform_size = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        transforms_poison = AddTrigger(
            p=self.p_rate,
            tigger_type=self.loc,
            position=position,
            attack_target=self.attack_target,
            mode='train'
        )

        x = transform_size(x)
        x, y = transforms_poison(x, y, item)
        return x, y


class PoisonedTestSetCIFAR100(CIFAR100):
    def __init__(self, root, train=False, transform=None, target_transform=None, download=False,
                 p_rate=1, loc='BadNets', attack_target=0, surrogate_models=['resnet50']):
        super(PoisonedTestSetCIFAR100, self).__init__(root, train, transform, target_transform, download)
        self.p_rate = p_rate
        self.attack_target = attack_target
        self.loc = loc
        self.surrogate_models = surrogate_models

        position_path = get_position_path("CIFAR100", self.surrogate_models, "test")
        self.positions = np.loadtxt(position_path, dtype=int)

    def __getitem__(self, item):
        x, y = super(PoisonedTestSetCIFAR100, self).__getitem__(item)
        position = self.positions[item]

        transform_size = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        transforms_poison = AddTrigger(
            p=self.p_rate,
            tigger_type=self.loc,
            position=position,
            attack_target=self.attack_target,
            mode='test'
        )

        x = transform_size(x)
        x, y = transforms_poison(x, y, item)
        return x, y


class PoisonedTrainSetMiniImageNet(Dataset):
    def __init__(self, root, p_rate=0.1, loc='BadNets', attack_target=0, surrogate_models=['resnet50']):
        self.p_rate = p_rate
        self.attack_target = attack_target
        self.loc = loc
        self.surrogate_models = surrogate_models
        self.root = root

        position_path = get_position_path("MiniImageNet", self.surrogate_models, "train")
        self.positions = np.loadtxt(position_path, dtype=int)

        fig_names = os.listdir(self.root)
        class_names = [fig_names[i][:12] for i in range(len(fig_names))]
        class_set = list(set(class_names))
        self.class_dict = {class_set[i]: i for i in range(100)}

        fig_names = np.array(fig_names).reshape(100, 600)
        self.train_names = fig_names[:, :550].reshape(-1)
        self.test_names = fig_names[:, 550:].reshape(-1)

    def __getitem__(self, item):
        position = self.positions[item]
        img_name = self.train_names[item]
        fig_path = os.path.join(self.root, img_name)

        x = Image.open(fig_path).convert("RGB")
        y = self.class_dict[img_name[:12]]

        transform_size = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        transforms_poison = AddTrigger(
            p=self.p_rate,
            tigger_type=self.loc,
            position=position,
            attack_target=self.attack_target,
            mode='train'
        )

        x = transform_size(x)
        x, y = transforms_poison(x, y, item)
        return x, y

    def __len__(self):
        return len(self.train_names)


class PoisonedTestSetMiniImageNet(Dataset):
    def __init__(self, root, p_rate=1, loc='BadNets', attack_target=0, surrogate_models=['resnet50']):
        self.p_rate = p_rate
        self.attack_target = attack_target
        self.loc = loc
        self.surrogate_models = surrogate_models
        self.root = root

        position_path = get_position_path("MiniImageNet", self.surrogate_models, "test")
        self.positions = np.loadtxt(position_path, dtype=int)

        fig_names = os.listdir(self.root)
        class_names = [fig_names[i][:12] for i in range(len(fig_names))]
        class_set = list(set(class_names))
        self.class_dict = {class_set[i]: i for i in range(100)}

        fig_names = np.array(fig_names).reshape(100, 600)
        self.train_names = fig_names[:, :550].reshape(-1)
        self.test_names = fig_names[:, 550:].reshape(-1)

    def __getitem__(self, item):
        position = self.positions[item]
        img_name = self.test_names[item]
        fig_path = os.path.join(self.root, img_name)

        x = Image.open(fig_path).convert("RGB")
        y = self.class_dict[img_name[:12]]

        transform_size = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        transforms_poison = AddTrigger(
            p=self.p_rate,
            tigger_type=self.loc,
            position=position,
            attack_target=self.attack_target,
            mode='test'
        )

        x = transform_size(x)
        x, y = transforms_poison(x, y, item)
        return x, y

    def __len__(self):
        return len(self.test_names)


class PoisonedTrainSetCaltech101(Dataset):
    def __init__(self, root, p_rate=0.1, loc='BadNets', attack_target=0, surrogate_models=['resnet50']):
        self.p_rate = p_rate
        self.attack_target = attack_target
        self.loc = loc
        self.surrogate_models = surrogate_models
        self.root = root

        position_path = get_position_path("Caltech101", self.surrogate_models, "train")
        self.positions = np.loadtxt(position_path, dtype=int)

        caltech101_dataset = ImageFolder(root=self.root)
        train_size = int(0.8 * len(caltech101_dataset))
        test_size = len(caltech101_dataset) - train_size
        torch.manual_seed(11)
        self.train_set, self.test_set = random_split(caltech101_dataset, [train_size, test_size])

    def __getitem__(self, item):
        x, y = self.train_set[item]
        position = self.positions[item]

        transform_size = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        transforms_poison = AddTrigger(
            p=self.p_rate,
            tigger_type=self.loc,
            position=position,
            attack_target=self.attack_target,
            mode='train'
        )

        x = transform_size(x)
        x, y = transforms_poison(x, y, item)
        return x, y

    def __len__(self):
        return len(self.train_set)


class PoisonedTestSetCaltech101(Dataset):
    def __init__(self, root, p_rate=1, loc='BadNets', attack_target=0, surrogate_models=['resnet50']):
        self.p_rate = p_rate
        self.attack_target = attack_target
        self.loc = loc
        self.surrogate_models = surrogate_models
        self.root = root

        position_path = get_position_path("Caltech101", self.surrogate_models, "test")
        self.positions = np.loadtxt(position_path, dtype=int)

        caltech101_dataset = ImageFolder(root=self.root)
        train_size = int(0.8 * len(caltech101_dataset))
        test_size = len(caltech101_dataset) - train_size
        torch.manual_seed(11)
        self.train_set, self.test_set = random_split(caltech101_dataset, [train_size, test_size])

    def __getitem__(self, item):
        x, y = self.test_set[item]
        position = self.positions[item]

        transform_size = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        transforms_poison = AddTrigger(
            p=self.p_rate,
            tigger_type=self.loc,
            position=position,
            attack_target=self.attack_target,
            mode='test'
        )

        x = transform_size(x)
        x, y = transforms_poison(x, y, item)
        return x, y

    def __len__(self):
        return len(self.test_set)

class AddTrigger(torch.nn.Module):
    def __init__(self, p=0.1, tigger_type='BadNets', position=(0, 0), attack_target=0, mode='train'):
        super(AddTrigger, self).__init__()
        self.tigger_type = tigger_type
        self.attack_target = attack_target
        self.position = position
        self.mode = mode
        self.p = p

    def forward(self, x, y, item):
        # Ensure that whether the same item is poisoned is non-random across each epoch.
        random.seed(item)
        n = random.random()
        if (n <= self.p) and (y != self.attack_target):
            # BadNets
            if self.tigger_type == 'BadNets':
                x = attack_BadNets(x, 0.02)
            # AdaPT_BadNets
            elif self.tigger_type == 'AdaPT_BadNets':
                x = attack_AdaPT_BadNets(x, 0.02, self.position)
            # Blend
            elif self.tigger_type == 'Blend':
                x = attack_Blend(x, 0.06)
            # AdaPT_Blend
            elif self.tigger_type == 'AdaPT_Blend':
                x = attack_AdaPT_Blend(x, 0.06, self.position)

            # label
            if self.mode == 'train':
                y = self.attack_target
            elif self.mode == 'test':
                pass
            return x, y
        else:
            return x, y



def get_position_path(dataset_name, surrogate_models, mode):
    if not isinstance(surrogate_models, (list, tuple)) or len(surrogate_models) == 0:
        raise ValueError("surrogate_models must be a non-empty list, e.g. ['resnet101', 'googlenet'].")

    if mode not in ["train", "test"]:
        raise ValueError("mode must be 'train' or 'test'")

    surrogate_tag = "_".join(surrogate_models)
    position_path = os.path.join(
        "positions",
        dataset_name,
        surrogate_tag,
        f"positions_{mode}.txt"
    )

    if not os.path.exists(position_path):
        raise FileNotFoundError(f"Position file not found: {position_path}")

    return position_path


def attack_BadNets(x, s):
    w = x.shape[2]
    s = int(s * w)
    r = torch.full((s, s), 0, dtype=torch.float32)
    g = torch.full((s, s), 0, dtype=torch.float32)
    b = torch.full((s, s), 0, dtype=torch.float32)

    x[0, w - s:w, w - s:w] = r
    x[1, w - s:w, w - s:w] = g
    x[2, w - s:w, w - s:w] = b

    x[0, w - s:w, w - 3*s:w-2*s] = r
    x[1, w - s:w, w - 3*s:w-2*s] = g
    x[2, w - s:w, w - 3*s:w-2*s] = b

    x[0, w - 2*s:w - s, w - 2*s:w - s] = r
    x[1, w - 2*s:w - s, w - 2*s:w - s] = g
    x[2, w - 2*s:w - s, w - 2*s:w - s] = b

    x[0, w - 3*s:w-2*s, w - s:w] = r
    x[1, w - 3*s:w-2*s, w - s:w] = g
    x[2, w - 3*s:w-2*s, w - s:w] = b
    return x


def attack_AdaPT_BadNets(x, s, pos):
    w = x.shape[2]
    s = int(s * w)
    max_row, max_col = pos

    # Trigger超出边界拉回
    if max_row+3*s > w:
        max_row = w-3*s
    if max_col+3*s > w:
        max_col = w-3*s

    r = torch.full((s, s), 0, dtype=torch.float32)
    g = torch.full((s, s), 0, dtype=torch.float32)
    b = torch.full((s, s), 0, dtype=torch.float32)

    w1 = max_row+3*s
    w2 = max_col+3*s

    x[0, w1 - s:w1, w2 - s:w2] = r
    x[1, w1 - s:w1, w2 - s:w2] = g
    x[2, w1 - s:w1, w2 - s:w2] = b

    x[0, w1 - s:w1, w2 - 3*s:w2-2*s] = r
    x[1, w1 - s:w1, w2 - 3*s:w2-2*s] = g
    x[2, w1 - s:w1, w2 - 3*s:w2-2*s] = b

    x[0, w1 - 2*s:w1 - s, w2 - 2*s:w2 - s] = r
    x[1, w1 - 2*s:w1 - s, w2 - 2*s:w2 - s] = g
    x[2, w1 - 2*s:w1 - s, w2 - 2*s:w2 - s] = b

    x[0, w1 - 3*s:w1-2*s, w2 - s:w2] = r
    x[1, w1 - 3*s:w1-2*s, w2 - s:w2] = g
    x[2, w1 - 3*s:w1-2*s, w2 - s:w2] = b
    return x


def attack_Blend(x, s):
    w = int(x.shape[1] * s)
    torch.manual_seed(11)
    trigger = torch.rand((3, w, w))

    # 局部blend
    p = (x.shape[2] - w, x.shape[2] - w)
    x[:, p[0]:p[0]+w, p[1]:p[1]+w] = 0.8 * x[:, p[0]:p[0]+w, p[1]:p[1]+w] + 0.2 * trigger
    return x


def attack_AdaPT_Blend(x, s, pos):
    w = x.shape[2]
    s = int(x.shape[1] * s)

    torch.manual_seed(11)
    trigger = torch.rand((3, s, s))

    # 局部blend
    max_row, max_col = pos
    if max_row + s > w:
        max_row = w - s
    if max_col + s > w:
        max_col = w - s

    x[:, max_row:max_row+s, max_col:max_col+s] = 0.8 * x[:, max_row:max_row+s, max_col:max_col+s] + 0.2 * trigger
    return x



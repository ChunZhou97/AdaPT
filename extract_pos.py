import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from utils import (
    PoisonedTrainSetCIFAR10,
    PoisonedTestSetCIFAR10,
    PoisonedTrainSetCIFAR100,
    PoisonedTestSetCIFAR100,
    PoisonedTrainSetMiniImageNet,
    PoisonedTestSetMiniImageNet,
    PoisonedTrainSetCaltech101,
    PoisonedTestSetCaltech101,
)


parser = argparse.ArgumentParser()

parser.add_argument('--surrogate_models', nargs='+', type=str,
                    default=['resnet101'],
                    choices=['resnet101', 'resnet50', 'regnet_x_3_2gf'],
                    help='List of surrogate models used to compute Grad-NAM.')

parser.add_argument('--dataset_name', type=str,
                    default='CIFAR10',
                    choices=['CIFAR10', 'CIFAR100', 'MiniImageNet', 'Caltech101'],
                    help='Choose the dataset to extract trigger positions from.')

parser.add_argument('--mode', type=str,
                    default='train',
                    choices=['train', 'test'],
                    help='Choose whether to process the training set or test set.')

parser.add_argument('--trigger_size', type=int,
                    default=13,
                    help='Trigger size for Trigger-Conv. A square kernel of size trigger_size x trigger_size is used. int(224x0.06)=13')

parser.add_argument('--num_workers', type=int,
                    default=1,
                    help='Number of worker processes for DataLoader.')

parser.add_argument('--save_dir', type=str,
                    default='positions',
                    help='Directory used to save the extracted position txt file.')

parser.add_argument('--device', type=str,
                    default=None,
                    choices=['cuda', 'cpu'],
                    help='Device to use. If not set, the code will automatically choose cuda if available, otherwise cpu.')

args = parser.parse_args()



DATASET_ROOTS = {
    "CIFAR10": r"/data/zc/datasets/CIFAR10",
    "CIFAR100": r"/data/zc/datasets/CIFAR100",
    "MiniImageNet": r"/data/zc/datasets/mini-imagenet",
    "Caltech101": r"/data/zc/datasets/caltech-101/101_ObjectCategories",
}


def build_model(model_name: str, device: str):
    model_name = model_name.lower()

    if model_name == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        target_layers = [model.layer4[-1]]
        feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
        fc_layer = model.fc

    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        target_layers = [model.layer4[-1]]
        feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
        fc_layer = model.fc

    elif model_name == "regnet_x_3_2gf":
        model = models.regnet_x_3_2gf(weights="DEFAULT")
        target_layers = [model.trunk_output]
        feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
        fc_layer = model.fc

    else:
        raise ValueError(
            f"Unsupported surrogate model: {model_name}. "
            f"Supported models: ['resnet101', 'resnet50', 'regnet_x_3_2gf']"
        )

    model = model.to(device)
    model.eval()

    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    return model, target_layers, feature_extractor, fc_layer


def build_dataset(dataset_name: str, mode: str):
    if dataset_name not in DATASET_ROOTS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if mode not in ["train", "test"]:
        raise ValueError("mode must be 'train' or 'test'")

    root = DATASET_ROOTS[dataset_name]

    if dataset_name == "CIFAR10":
        dataset_cls = PoisonedTrainSetCIFAR10 if mode == "train" else PoisonedTestSetCIFAR10
        dataset = dataset_cls(root=root, p_rate=0)

    elif dataset_name == "CIFAR100":
        dataset_cls = PoisonedTrainSetCIFAR100 if mode == "train" else PoisonedTestSetCIFAR100
        dataset = dataset_cls(root=root, p_rate=0)

    elif dataset_name == "MiniImageNet":
        dataset_cls = PoisonedTrainSetMiniImageNet if mode == "train" else PoisonedTestSetMiniImageNet
        dataset = dataset_cls(root=root, p_rate=0)

    elif dataset_name == "Caltech101":
        dataset_cls = PoisonedTrainSetCaltech101 if mode == "train" else PoisonedTestSetCaltech101
        dataset = dataset_cls(root=root, p_rate=0)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return dataset


def compute_grad_nam(model, target_layers, feature_extractor, fc_layer, input_tensor, device):
    x = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        h = feature_extractor(x).flatten(1).squeeze(0)

    with torch.no_grad():
        old_fc_weight = fc_layer.weight.detach().clone()
        fc_layer.weight[0].copy_(2 * h)

    try:
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(0)]
        grayscale_cam = cam(input_tensor=x, targets=targets)[0]
    finally:
        with torch.no_grad():
            fc_layer.weight.copy_(old_fc_weight)

    return grayscale_cam


def extract_nam_pos(
    surrogate_models,
    dataset_name,
    mode,
    trigger_size,
    num_workers,
    save_dir,
    device,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not isinstance(surrogate_models, (list, tuple)) or len(surrogate_models) == 0:
        raise ValueError("surrogate_models must be a non-empty list.")

    dataset = build_dataset(dataset_name, mode)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    model_infos = []
    for model_name in surrogate_models:
        model, target_layers, feature_extractor, fc_layer = build_model(model_name, device)
        model_infos.append({
            "name": model_name,
            "model": model,
            "target_layers": target_layers,
            "feature_extractor": feature_extractor,
            "fc_layer": fc_layer,
        })

    M = len(surrogate_models)
    positions = []

    for _, (x, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        input_tensor = x.squeeze(0).to(device)

        nam_maps = []
        for info in model_infos:
            grayscale_cam = compute_grad_nam(
                model=info["model"],
                target_layers=info["target_layers"],
                feature_extractor=info["feature_extractor"],
                fc_layer=info["fc_layer"],
                input_tensor=input_tensor,
                device=device,
            )
            nam_maps.append(torch.tensor(grayscale_cam, dtype=torch.float32, device=device))

        nam_tensor = torch.stack(nam_maps, dim=0).unsqueeze(0)

        kernel = torch.ones(
            (1, M, trigger_size, trigger_size),
            dtype=torch.float32,
            device=device,
        )

        score_map = F.conv2d(nam_tensor, kernel, stride=1).squeeze(0).squeeze(0)

        max_idx = torch.argmax(score_map)
        out_h, out_w = score_map.shape
        max_row = (max_idx // out_w).item()
        max_col = (max_idx % out_w).item()

        positions.append([max_row, max_col])

    positions_np = np.array(positions, dtype=int)

    surrogate_tag = "_".join(surrogate_models)
    out_dir = os.path.join(save_dir, dataset_name, surrogate_tag)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"positions_{mode}.txt")
    np.savetxt(out_path, positions_np, fmt="%d")

    print(f"Saved positions to: {out_path}")
    print(f"Total samples: {len(positions_np)}")

    return positions_np


if __name__ == "__main__":
    extract_nam_pos(
        surrogate_models=args.surrogate_models,
        dataset_name=args.dataset_name,
        mode=args.mode,
        trigger_size=args.trigger_size,
        num_workers=args.num_workers,
        save_dir=args.save_dir,
        device=args.device,
    )

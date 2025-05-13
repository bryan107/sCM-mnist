import os
import click
import torch
import torch.nn as nn
from unet import Unet #type: ignore
import wandb

import torchvision.transforms as transforms #type: ignore
import torchvision.models as models #type: ignore
import numpy as np #type: ignore
import torch.nn.functional as F

@click.command()
@click.option("--device", default="cuda", help="Device to train on")
@click.option("--epoch-start", default=105, help="starting epoch to compute IS")
@click.option("--epoch-end", default=105, help="Stop epoch (inclusive) to compute IS")
@click.option("--lr", default=3e-5, help="Learning rate")
@click.option("--enable-wandb", is_flag=True, help="Upload to Wandb", default=False)
def compute_is(device, epoch_start, epoch_end, lr, enable_wandb):
    if enable_wandb:
        wandb.init(
            project="MNIST_consistency_distillation_IS_new",
            entity="actrec",
            config={
                'lr': lr,
            }, 
            name=f"Consistency Training lr:{lr}"
        ) 
        
    sigma_data = 0.5
    sample_n = 10_000

    # Example: Assume we have a batch of generated MNIST images from a model
    # If you have a generative model, you can pass your generated images here
    # Otherwise, we will use the real MNIST dataset for demonstration purposes.

    for epoch in range(epoch_start, epoch_end + 1):
        # Setup model and optimizer
        model = Unet(256, 1, 1, base_dim=64, dim_mults=[2, 4]).to(device)
        model_path = f'models/consistency/model_consistency_{epoch}.pt'
        print(f"load model {model_path}")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path), strict=False)
            model.logvar_linear = nn.Linear(model.logvar_linear.in_features, model.logvar_linear.out_features).to(device)
            print(f"Loaded model from {model_path}, compute IS...")
        else:
            print("No model found, skip IS computation.")

        #########
        # IS Evaluation
        #########

        classifier = load_mnist_classifier(device)

        z = torch.randn(sample_n, 1, 28, 28, generator=torch.Generator().manual_seed(42)).to(device)
        t0 = 1.56454 * torch.ones(sample_n, device=device)
        t1 = 1.1 * torch.ones(sample_n, device=device)
        with torch.no_grad():
            pred_x0 = torch.clamp(-sigma_data * model(z, t0), min=-0.5, max=0.5)
            generated_imsages = pred_x0
            print("Computing IS score 1", end='...')
            is_mean_1, is_std_1 = mnist_score(generated_imsages, classifier, device=device, splits=10)
            print(f" done → IS: {is_mean_1:.2f} ± {is_std_1:.2f}")
            
            z = torch.randn(sample_n, 1, 28, 28, generator=torch.Generator().manual_seed(43)).to(device)
            t1_exp = t1.view(-1, 1, 1, 1)
            x_t = torch.sin(t1_exp) * z * sigma_data + torch.cos(t1_exp) * pred_x0
            pred_x0 = torch.clamp(torch.cos(t1_exp) * x_t - torch.sin(t1_exp) * sigma_data * model(x_t / sigma_data, t1), min=-0.5, max=0.5)
            generated_imsages = pred_x0
            print("Computing IS score 2", end='...')
            is_mean_2, is_std_2 = mnist_score(generated_imsages, classifier, device=device, splits=10)
            print(f" done → IS: {is_mean_2:.2f} ± {is_std_2:.2f}")
            

        if enable_wandb:
            wandb.log({'Train_CD_1_IS_mean': is_mean_1, 'Train_CD_1_IS_std': is_std_1}, step=epoch)
            wandb.log({'Train_CD_2_IS_mean': is_mean_2, 'Train_CD_2_IS_std': is_std_2}, step=epoch)

    if enable_wandb:
        wandb.finish()


# Assuming generated_images is a list or a batch of images (torch tensors) of size [batch_size, 1, 28, 28]
# for MNIST dataset, each image will be 1 channel, 28x28

def load_mnist_classifier(device, model_path="models/mnist_resnet18.pth"):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 10)  # MNIST has 10 classes
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def mnist_score(images, classifier, device='cuda', splits=10):
    """
    Compute MNIST Inception Score with multiple splits.
    images: [N, 1, 28, 28] or [N, 1, 224, 224]
    """
    classifier.eval()
    N = len(images)
    batch_size = 64

    # Preprocess
    images = (images + 0.5)  # from [-0.5, 0.5] to [0, 1]
    images = (images - 0.1307) / 0.3081
    images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

    # Get softmax probabilities
    preds = []
    for i in range(0, N, batch_size):
        batch = images[i:i+batch_size].to(device)
        with torch.no_grad():
            logits = classifier(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds.append(probs)
    preds = np.concatenate(preds, axis=0)

    # Compute split IS
    split_scores = []
    split_size = N // splits
    for k in range(splits):
        part = preds[k * split_size: (k + 1) * split_size]
        py = np.mean(part, axis=0)
        scores = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        scores = np.sum(scores, axis=1)
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


if __name__ == '__main__':
    compute_is()
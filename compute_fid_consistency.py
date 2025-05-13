import os
import random
import click
import torch
import torch.nn as nn
import torchvision # type: ignore
import torchvision.transforms as transforms # type: ignore
from unet import Unet
import wandb

from torchvision.transforms import Compose, Resize, Lambda
from torchmetrics.image.fid import FrechetInceptionDistance

@click.command()
@click.option("--device", default="cuda", help="Device to train on")
@click.option("--epoch-start", default=100, help="starting epoch to compute FID")
@click.option("--epoch-end", default=135, help="Stop epoch (inclusive) to compute FID")
@click.option("--batch-size", default=128, help="Batch size")
@click.option("--lr", default=3e-5, help="Learning rate")
@click.option("--extra-plots", is_flag=True, help="Generate additional training plots", default=True)
@click.option("--enable-wandb", is_flag=True, help="Upload to Wandb", default=False)
@click.option("--model-dir", default="models/consistency", help="model to evaluate")
def compute_fid(device, epoch_start, epoch_end, lr, enable_wandb, model_dir):
    if enable_wandb:
        wandb.init(
            project="MNIST_consistency_distillation_FID_new",
            entity="actrec",
            config={
                'lr': lr,
            }, 
            name=f"Consistency Training lr:{lr}"
        ) 
        
    sigma_data = 0.5
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (1.0))
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                         download=True, transform=transform)

    sample_n = 10_000
    indices = random.sample(range(len(dataset)), sample_n)
    real_images = torch.stack([dataset[i][0] for i in indices])

    for epoch in range(epoch_start, epoch_end + 1):
        # Setup model and optimizer
        model = Unet(256, 1, 1, base_dim=64, dim_mults=[2, 4]).to(device)
        model_path = os.path.join(model_dir, f"model_consistency_{epoch}.pt")
        print(f"load model {model_path}")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path), strict=False)
            model.logvar_linear = nn.Linear(model.logvar_linear.in_features, model.logvar_linear.out_features).to(device)
            print(f"Loaded model from {model_path}, compute FID...")
        else:
            print("No model found, skip FID computation.")
                
        #########
        # FID Evaluation
        #########

        z = torch.randn(sample_n, 1, 28, 28, generator=torch.Generator().manual_seed(42)).to(device)
        t0 = 1.56454 * torch.ones(sample_n, device=device)
        t1 = 1.1 * torch.ones(sample_n, device=device)
        with torch.no_grad():
            pred_x0 = torch.clamp(-sigma_data * model(z, t0), min=-0.5, max=0.5)
            generated_imsages = pred_x0
            fid_score_1 = calculate_fid(real_images, generated_imsages, device='cuda')
            
            z = torch.randn(sample_n, 1, 28, 28, generator=torch.Generator().manual_seed(43)).to(device)
            t1_exp = t1.view(-1, 1, 1, 1)
            x_t = torch.sin(t1_exp) * z * sigma_data + torch.cos(t1_exp) * pred_x0
            pred_x0 = torch.clamp(torch.cos(t1_exp) * x_t - torch.sin(t1_exp) * sigma_data * model(x_t / sigma_data, t1), min=-0.5, max=0.5)
            generated_imsages = pred_x0
            fid_score_2 = calculate_fid(real_images, generated_imsages, device='cuda')
            
            print(f"FID score 1: {fid_score_1}, FID score 2: {fid_score_2}")

        if enable_wandb:
            wandb.log({'Train_CD_1_FID': fid_score_1}, step=epoch)
            wandb.log({'Train_CD_2_FID': fid_score_2}, step=epoch)

    if enable_wandb:
        wandb.finish()

def calculate_fid(real_images, generated_images, device='cuda'):
    """
    Calculates FID between real and generated MNIST images.
    
    Args:
        real_images (Tensor): Real images (N, 1, H, W), values in [0, 1] or [-1, 1].
        generated_images (Tensor): Generated images (N, 1, H, W), same range as real.
        device (str): 'cuda' or 'cpu'.
        
    Returns:
        float: FID score.
    """
    # Convert to 3 channels and resize to 299x299 for InceptionV3
    transform = Compose([
        Lambda(lambda x: (x + 1) / 2 if x.min() < 0 else x),  # Rescale to [0, 1] if needed
        Lambda(lambda x: x.repeat(3, 1, 1)),  # (1, H, W) -> (3, H, W)
        Resize((299, 299))
    ])
    
    real_images = real_images.to(device)
    generated_images = generated_images.to(device)

    fid = FrechetInceptionDistance(feature=2048).to(device)

    with torch.no_grad():
        for img in real_images:
            img = transform(img)
            img = img.clamp(0, 1) * 255  # Ensure values in [0, 255]
            img = img.to(torch.uint8)
            fid.update(img.unsqueeze(0), real=True)

        for img in generated_images:
            img = transform(img)
            img = img.clamp(0, 1) * 255  # Ensure values in [0, 255]
            img = img.to(torch.uint8)
            fid.update(img.unsqueeze(0), real=False)

    return fid.compute().item()

if __name__ == '__main__':
    compute_fid()
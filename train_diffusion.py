import math
import os
import click
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from unet import Unet
import matplotlib.pyplot as plt

@click.command()
@click.option("--device", default="cuda", help="Device to train on")
@click.option("--epochs", default=200, help="Number of epochs to train")
@click.option("--batch-size", default=256, help="Batch size")
@click.option("--lr", default=1e-3, help="Learning rate")
def train(device, epochs, batch_size, lr):
    # Setup model and optimizer
    model = Unet(256, 1, 1, base_dim=64, dim_mults=[2, 4]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3, betas=(0.9, 0.99))
    sigma_data = 0.5
    
    P_mean = -0.4
    P_std = 1.4
    
    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (1.0))
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                         download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True, persistent_workers=True)
    
    step = 0
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch, _ in progress_bar:
            images = batch.to(device)
            
            # Sample noise from log-normal distribution
            sigma = torch.randn(images.shape[0], device=images.device).reshape(-1, 1, 1, 1)
            sigma = (sigma * P_std + P_mean).exp()  # Sample from proposal distribution
            t = torch.arctan(sigma / sigma_data)  # Convert to t using arctan
            
            # Generate z and x_t
            z = torch.randn_like(images) * sigma_data
            x_t = torch.cos(t) * images + torch.sin(t) * z
            
            # Next we have to calculate g and F_theta. We can do this simultaneously with torch.func.jvp
            pred, logvar = model(x_t / sigma_data, t.flatten(), return_logvar=True)
            logvar = logvar.view(-1, 1, 1, 1)
            
            pred_x0 = torch.cos(t) * x_t - torch.sin(t) * sigma_data * pred
            
            # Calculate loss with adaptive weighting
            weight = 1 / sigma**2 + 1 / sigma_data**2
            loss = (weight / torch.exp(logvar)) * torch.square(pred_x0 - images) + logvar
            loss = loss.mean()
            
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            progress_bar.set_postfix({"loss": loss.item(), "grad_norm": grad_norm.item()})
            step += 1
            
        torch.save(model.state_dict(), f'model.pt')
        
        # Plot one noise sample with different timesteps
        # Sample one image from the dataset for visualization
        sample_img, _ = dataset[0]
        sample_img = sample_img.unsqueeze(0).to(device)
        
        z = torch.randn(1, 1, 28, 28, generator=torch.Generator().manual_seed(42)).to(device) * sigma_data
        z = z.repeat(16, 1, 1, 1)  # Repeat the same noise 16 times
        t = torch.linspace(0, 1.56454, 16, device=device).view(-1, 1, 1, 1)  # Linearly spaced timesteps
        x_t = torch.cos(t) * sample_img + torch.sin(t) * z
        with torch.no_grad():
            pred_x0 = torch.clamp(torch.cos(t) * x_t - torch.sin(t) * sigma_data * model(x_t / 0.5, t.flatten()), min=-0.5, max=0.5)
            plt.figure(figsize=(12, 12))
            for i in range(16):
                plt.subplot(4, 4, i+1)
                plt.imshow(pred_x0[i, 0].cpu().numpy(), cmap='gray')
                plt.title(f't={t[i].item():.2f}')
                plt.axis('off')
            plt.tight_layout()
            os.makedirs('outputs_diffusion/timesteps', exist_ok=True)
            plt.savefig(f'outputs_diffusion/timesteps/epoch_{epoch:04d}.png')
            plt.close()
        

if __name__ == '__main__':
    train()
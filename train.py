import math
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
@click.option("--batch-size", default=128, help="Batch size")
@click.option("--lr", default=3e-5, help="Learning rate")
def train(device, epochs, batch_size, lr):
    # Setup model and optimizer
    model = Unet(256, 1, 1, base_dim=64, dim_mults=[2, 4]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.99))
    sigma_data = 0.5
    
    P_mean = -1.0
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
        
        for batch, labels in progress_bar:
            images = batch.to(device)
            
            # Sample noise from log-normal distribution
            sigma = torch.randn(images.shape[0], device=images.device).reshape(-1, 1, 1, 1)
            sigma = (sigma * P_std + P_mean).exp()  # Sample from proposal distribution
            t = torch.arctan(sigma / sigma_data)  # Convert to t using arctan
            t.requires_grad_(True)
            
            # Generate z and x_t
            z = torch.randn_like(images) * sigma_data
            x_t = torch.cos(t) * images + torch.sin(t) * z
            
            # Calculate dx_t/dt exactly (For consistency TRAINING)
            dxt_dt = torch.cos(t) * z - torch.sin(t) * images
            
            # For consistency DISTILLATION use something like this:
            # (model_pretrained is assumed to output v-predictions)
            # with torch.no_grad():
            #     pretrain_pred = model_pretrained(x_t / sigma_data, noise_labels=t.flatten())
            #     dxt_dt = sigma_data * pretrain_pred
            
            # Next we have to calculate g and loss. We can do this simultaneously with torch.func.jvp
            x_t.requires_grad_(True)  # Enable gradients for JVP
            def model_wrapper(scaled_x_t, t):
                pred, logvar = model(scaled_x_t, t.flatten(), return_logvar=True)
                return pred, logvar
            
            v_x = torch.cos(t) * torch.sin(t) * dxt_dt
            v_t = torch.cos(t) * torch.sin(t) * sigma_data
            pred, F_theta_grad, logvar = torch.func.jvp(
                model_wrapper, 
                (x_t / sigma_data, t),
                (v_x, v_t),
                has_aux=True
            )
            logvar = logvar.view(-1, 1, 1, 1)
            F_theta_grad = F_theta_grad.detach()
                
            F_theta = sigma_data * pred
            F_theta_minus = F_theta.detach()
            
            # Warmup steps. 10000 was used in the paper. I'm using 1000 for MNIST since it's an easier dataset.
            r = min(1.0, step / 10000)
            # Calculate gradient g using JVP rearrangement
            g = -torch.cos(t)**2 * (sigma_data * F_theta_minus - dxt_dt)
            # Note that F_theta_grad is already multiplied by sin(t) cos(t) from the tangents. Doing it early helps with stability.
            second_term = -r * torch.cos(t) * torch.sin(t) * x_t - r * sigma_data * F_theta_grad
            g = g + second_term
            g = g.detach()
            
            # Tangent normalization
            g_norm = torch.linalg.vector_norm(g, dim=(1, 2, 3), keepdim=True)
            g_norm = g_norm * np.sqrt(g_norm.numel() / g.numel())  # Multiplying by sqrt(numel(g_norm) / numel(g)) ensures that the norm is invariant to the spatial dimensions.
            g = g / (g_norm + 0.1)  # 0.1 is the constant c, can be modified but 0.1 was used in the paper
            
            # Tangent clipping (Only use this OR normalization)
            # g = torch.clamp(g, min=-1, max=1)
            
            # Calculate loss with adaptive weighting
            weight = 1 / sigma
            loss = (weight / torch.exp(logvar)) * torch.square(F_theta - F_theta_minus - g) + logvar
            loss = loss.mean()
            
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            progress_bar.set_postfix({"loss": loss.item(), "grad_norm": grad_norm.item()})
            step += 1
        
        z = torch.randn(16, 1, 28, 28, generator=torch.Generator().manual_seed(42)).to(device)
        t = 1.56454 * torch.ones(16, device=device)
        with torch.no_grad():
            pred_x0 = -sigma_data * model(z, t)
            plt.figure(figsize=(12, 12))
            for i in range(16):
                plt.subplot(4, 4, i+1)
                plt.imshow(pred_x0[i, 0].cpu().numpy(), cmap='gray')
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'sample_epoch_{epoch:04d}.png')
            plt.close()
        

if __name__ == '__main__':
    train()
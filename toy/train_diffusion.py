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
from toynet import Net
import matplotlib.pyplot as plt
from toy_utils import *

@click.command()
@click.option("--device", default="cuda", help="Device to train on")
@click.option("--epochs", default=4, help="Number of epochs to train")
@click.option("--batch-size", default=4096, help="Batch size")
@click.option("--lr", default=1e-3, help="Learning rate")
def train(device, epochs, batch_size, lr):
    # Setup model and optimizer
    model = Net(embedding_dim=256, hidden_dim=512).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.99))
    sigma_data = torch.tensor(0.5)
    
    P_mean = torch.tensor(-0.4)
    P_std = torch.tensor(1.4)
    
    size = 2048
    dataset = BimodalDataset(batch_size=batch_size, size=size, std=0.5, sigma_data=0.5)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, 
                            num_workers=4, pin_memory=True, persistent_workers=True)
    
    sampling_timesteps = torch.arctan(calc_karras_sigmas(sigma_min=0.002, sigma_max=80, steps=1000, rho=7) / sigma_data).to(device)
    sigma_data = sigma_data.to(device)
    P_mean = P_mean.to(device)
    P_std = P_std.to(device)
    
    step = 0
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        losses = []
        
        for batch in progress_bar:
            x_0 = batch.to(device)
            
            # Sample noise from log-normal distribution
            sigma = torch.randn(x_0.shape[0], device=x_0.device)
            sigma = (sigma * P_std + P_mean).exp()  # Sample from proposal distribution
            t = torch.arctan(sigma / sigma_data)  # Convert to t using arctan
            
            # Generate z and x_t
            z = torch.randn_like(x_0) * sigma_data
            x_t = torch.cos(t) * x_0 + torch.sin(t) * z
            
            pred, logvar = model(x_t / sigma_data, t, return_logvar=True)
            
            pred_x0 = torch.cos(t) * x_t - torch.sin(t) * sigma_data * pred
            exact_pred_x0 = exact_solution(x_t / sigma_data, t, sigma_data, dataset)
            
            # Calculate loss with adaptive weighting
            weight = 1 / sigma**2 + 1 / sigma_data**2
            loss = (weight / torch.exp(logvar)) * torch.square(pred_x0 - x_0) + logvar
            loss = loss.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.norm_logvar()
            
            losses.append(loss.item())
            progress_bar.set_postfix({"loss": np.mean(losses), "exact_loss": ((exact_pred_x0 - pred_x0)**2).mean().item()})
            step += 1
            
        torch.save(model.state_dict(), f'toy-model.pt')

        # Sample and plot ODE trajectory every epoch
        model.eval()
        with torch.no_grad():
            # Start from pure noise
            x_t = torch.linspace(-1.5, 1.5, 64, device=device)
            z = x_t
            
            # Store trajectory for plotting
            trajectory_x = []
            trajectory_t = []
            trajectory_x0 = []
            
            use_exact_solution = False
            
            # Gradually denoise
            for t in sampling_timesteps:
                trajectory_x.append(x_t.cpu().numpy())
                trajectory_t.append(t.cpu().numpy())
                
                t_repeated = t.repeat(x_t.shape[0])
                
                if not use_exact_solution:
                    # Get model prediction
                    pred = model(x_t / sigma_data, t_repeated)
                    
                    # Calculate predicted x0
                    pred_x0 = torch.cos(t) * x_t - torch.sin(t) * sigma_data * pred
                else:
                    pred_x0 = exact_solution(x_t / sigma_data, t, sigma_data, dataset)
                    pred = (torch.cos(t) * x_t - pred_x0) / (torch.sin(t) * sigma_data)
                
                trajectory_x0.append(pred_x0.cpu().numpy())
                
                next_t = sampling_timesteps[sampling_timesteps < t]
                if len(next_t) == 0:
                    next_t = 0
                else:
                    next_t = next_t[0]
                
                # Calculate x0 prediction and update x
                x_t = torch.cos(t - next_t) * x_t - torch.sin(t - next_t) * sigma_data * pred
                
            trajectory_x.append(x_t.cpu().numpy())
            trajectory_t.append(0)
            trajectory_x0.append(x_t.cpu().numpy())  # At t=0, x_t is the final x0 prediction
                
            # Now calculate pred_x0 for a grid of x_t and t
            n = 64
            x_t = torch.linspace(-1.5, 1.5, n, device=device)
            t = torch.linspace(0, torch.pi/2, n, device=device)
            x_t, t = torch.meshgrid(x_t, t, indexing='ij')
            x_t = x_t.flatten()
            t = t.flatten()
            if use_exact_solution:
                pred_x0 = exact_solution(x_t / sigma_data, t, sigma_data, dataset)
            else:
                pred = model(x_t / sigma_data, t)
                pred_x0 = torch.cos(t) * x_t - torch.sin(t) * sigma_data * pred
                
            # Reshape x_t and t into 64x64 grids for visualization
            pred_x0_grid = pred_x0.reshape(n, n)
                
            # Create figure with three subplots side by side
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            
            trajectory_x = np.array(trajectory_x)
            trajectory_t = np.array(trajectory_t)
            trajectory_x0 = np.array(trajectory_x0)
            
            # Plot ODE trajectories in left subplot
            for i in range(trajectory_x.shape[1]):
                ax1.plot(trajectory_t, trajectory_x[:, i], alpha=0.5)
            ax1.set_ylabel('x_t')
            ax1.set_xlabel('t')
            ax1.set_title(f'ODE Trajectories - Epoch {epoch}')
            
            # Plot predicted x0 trajectories in middle subplot
            im = ax2.imshow(trajectory_x0.T[:, ::-1], aspect='auto', origin='lower',
                        extent=[trajectory_t.min(), trajectory_t.max(),
                                trajectory_x[0].min(), trajectory_x[0].max()],
                        cmap='viridis', interpolation='nearest')
            ax2.set_title('Predicted x0 Trajectories')
            ax2.set_ylabel('Initial x value')
            ax2.set_xlabel('t')
            plt.colorbar(im, ax=ax2, label='Predicted x0')
            
            # Plot pred_x0 grid in right subplot
            im2 = ax3.imshow(pred_x0_grid.detach().cpu().numpy()[::-1, :], aspect='auto',
                         extent=[0, torch.pi/2, -1.5, 1.5],
                         cmap='viridis', interpolation='nearest')
            ax3.set_title('Predicted x0 Grid') 
            ax3.set_xlabel('t')
            ax3.set_ylabel('x_t')
            plt.colorbar(im2, ax=ax3, label='Predicted x0')
            
            plt.tight_layout()
            os.makedirs('plots/diffusion', exist_ok=True)
            plt.savefig(f'plots/diffusion/epoch_{epoch}.png')
            plt.close()
        
        

if __name__ == '__main__':
    train()
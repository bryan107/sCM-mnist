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
@click.option("--epochs", default=200, help="Number of epochs to train")
@click.option("--batch-size", default=4096, help="Batch size")
@click.option("--lr", default=1e-4, help="Learning rate")
@click.option("--swap-cos", default=True, help="Swap cos for sin in the gradient calculation")
def train(device, epochs, batch_size, lr, swap_cos):
    # Setup model and optimizer
    model = Net(embedding_dim=256, hidden_dim=512).to(device)
    model_pretrained = None
    
    if os.path.exists('toy-model.pt'):
        model.load_state_dict(torch.load('toy-model.pt'))
        
        model_pretrained = Net(embedding_dim=256, hidden_dim=512).to(device)
        model_pretrained.load_state_dict(torch.load('toy-model.pt'))
        print("Loaded model from model.pt, performing cosistency distillation.")
        consistency_training = False
    else:
        print("No model found, performing cosistency training.")
        consistency_training = True
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.99))
    sigma_data = 0.5
    
    P_mean = -1
    P_std = 1.6
    
    # Load CIFAR-10
    dataset = BimodalDataset(batch_size=batch_size, size=4096, std=0.5, sigma_data=0.5)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, 
                            num_workers=4, pin_memory=True, persistent_workers=True)
    
    sampling_timesteps = torch.arctan(calc_karras_sigmas(sigma_min=0.002, sigma_max=80, steps=1000, rho=7) / sigma_data).to(device)
    
    step = 0
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            x0 = batch.to(device)
            
            # Sample noise from log-normal distribution
            sigma = torch.randn(x0.shape[0], device=x0.device)
            sigma = (sigma * P_std + P_mean).exp()  # Sample from proposal distribution
            t = torch.arctan(sigma / sigma_data)  # Convert to t using arctan
            
            # Generate z and x_t
            z = torch.randn_like(x0) * sigma_data
            x_t = torch.cos(t) * x0 + torch.sin(t) * z
            
            if consistency_training:
                # Estimate dx_t/dt (For consistency TRAINING)
                dxt_dt = torch.cos(t) * z - torch.sin(t) * x0
            else:
                # For consistency DISTILLATION
                # (model_pretrained is assumed to output v-predictions)
                with torch.no_grad():
                    pretrain_pred = model_pretrained(x_t / sigma_data, t.flatten())
                    dxt_dt = sigma_data * pretrain_pred
            
            # Next we have to calculate g and F_theta. We can do this simultaneously with torch.func.jvp
            def model_wrapper(scaled_x_t, t):
                pred, logvar = model(scaled_x_t, t.flatten(), return_logvar=True)
                return pred, logvar
            
            # I'm likely just missing something, but I can't see why the paper multiplies F_theta_grad by cosine instead of sine (see eq. 8)
            # Upon experimentation, it seems like both work well. I've found sine to be marginally better but it's not clear if that scales & works in fp16.
            if not swap_cos:
                v_x = torch.cos(t) * torch.sin(t) * dxt_dt / sigma_data
                v_t = torch.cos(t) * torch.sin(t)
            else:
                v_x = torch.sin(t) * torch.sin(t) * dxt_dt / sigma_data
                v_t = torch.sin(t) * torch.sin(t)
            F_theta, F_theta_grad, logvar = torch.func.jvp(
                model_wrapper, 
                (x_t / sigma_data, t),
                (v_x, v_t),
                has_aux=True
            )
            F_theta_grad = F_theta_grad.detach()
            F_theta_minus = F_theta.detach()
            
            # Warmup steps. 1000 was used in the paper.
            r = min(1.0, step / 1000)
            # Calculate gradient g using JVP rearrangement
            if not swap_cos:
                g = -torch.cos(t) * torch.cos(t) * (sigma_data * F_theta_minus - dxt_dt)
                # Note that F_theta_grad is already multiplied by sin(t) cos(t) from the tangents. Doing it early helps with stability.
                second_term = -r * (torch.cos(t) * torch.sin(t) * x_t + sigma_data * F_theta_grad)
            else:
                g = -torch.cos(t) * torch.sin(t) * (sigma_data * F_theta_minus - dxt_dt)
                # Note that F_theta_grad is already multiplied by sin(t) cos(t) from the tangents. Doing it early helps with stability.
                second_term = -r * (torch.sin(t) * torch.sin(t) * x_t + sigma_data * F_theta_grad)
            g = g + second_term
            
            # Tangent normalization
            g_norm = torch.abs(g)
            g = g / (g_norm + 0.1)  # 0.1 is the constant c, can be modified but 0.1 was used in the paper
            
            # Tangent clipping (Only use this OR normalization)
            # g = torch.clamp(g, min=-1, max=1)
            
            # Calculate loss with adaptive weighting
            weight = 1 / sigma
            loss = (weight / torch.exp(logvar)) * torch.square(F_theta - F_theta_minus - g) + logvar
            loss = loss.mean()
            
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            model.norm_logvar()
            progress_bar.set_postfix({"loss": loss.item(), "grad_norm": grad_norm.item()})
            step += 1
        
        # Sample and plot ODE trajectory every epoch
        model.eval()
        # Start from pure noise
        x_t = torch.linspace(-1.5, 1.5, 1024, device=device)
        
        use_exact_trajectory = True
        
        # Store trajectory for plotting
        trajectory_x = []
        trajectory_t = []
        trajectory_x0 = []
        trajectory_grad = []
        
        with torch.no_grad():
            trajectory_guess = -sigma_data * model_wrapper(x_t / sigma_data, torch.full_like(x_t, sampling_timesteps[0]))[0]
        
        # Gradually denoise
        for t in sampling_timesteps:
            trajectory_x.append(x_t.cpu().numpy())
            trajectory_t.append(t.cpu().numpy())
            
            t_repeated = t.repeat(x_t.shape[0])
        
            # For visualization use EXACT dx_t/dt, usually impossible
            exact_pred_x0 = exact_solution(x_t / sigma_data, t, sigma_data, dataset)
            dxt_dt = (torch.cos(t) * x_t - exact_pred_x0) / torch.sin(t)
            
            v_x = torch.cos(t_repeated) * torch.sin(t_repeated) * dxt_dt / sigma_data
            v_t = torch.cos(t_repeated) * torch.sin(t_repeated)
            with torch.no_grad():
                # Calculate JVP using torch.func.jvp
                F_theta, F_theta_grad, logvar = torch.func.jvp(
                    model_wrapper, 
                    (x_t / sigma_data, t_repeated),
                    (v_x, v_t),
                    has_aux=True
                )
                F_theta_grad = F_theta_grad.detach()
                F_theta_minus = F_theta.detach()
                
                # Numerically verify JVP calculation
                # eps = 1e-3
                # x_perturbed = x_t / sigma_data + eps * v_x
                # t_perturbed = t_repeated + eps * v_t
                # 
                # F_theta_perturbed = model_wrapper(x_perturbed, t_perturbed)[0]
                # numerical_grad = (F_theta_perturbed - F_theta) / eps
                # grad_diff = torch.abs(numerical_grad - F_theta_grad).mean().item()
                # print(f"Difference: {grad_diff:.6f}")
            
            # Calculate predicted x0
            pred_x0 = torch.cos(t) * x_t - torch.sin(t) * sigma_data * F_theta
            trajectory_x0.append(pred_x0.detach().cpu().numpy())
            
            r = 1
            # Calculate gradient g using JVP rearrangement
            g = -torch.cos(t) * torch.cos(t) * (sigma_data * F_theta_minus - dxt_dt)
            # Note that F_theta_grad is already multiplied by sin(t) cos(t) from the tangents. Doing it early helps with stability.
            second_term = -r * (torch.cos(t) * torch.sin(t) * x_t + sigma_data * F_theta_grad)
            g = g + second_term
            trajectory_grad.append(torch.abs(g).cpu().numpy())
            
            next_t = sampling_timesteps[sampling_timesteps < t]
            if len(next_t) == 0:
                next_t = 0
            else:
                next_t = next_t[0]
            
            # Calculate x0 prediction and update x
            if not use_exact_trajectory:
                x_t = torch.cos(t - next_t) * x_t - torch.sin(t - next_t) * sigma_data * F_theta
            else:
                x_t = torch.cos(t - next_t) * x_t - torch.sin(t - next_t) * dxt_dt
            x_t = x_t.detach()
            
        trajectory_x.append(x_t.cpu().numpy())
        trajectory_t.append(0)
        trajectory_x0.append(x_t.cpu().numpy())  # At t=0, x_t is the final x0 prediction
        trajectory_grad.append(np.zeros_like(x_t.cpu().numpy()))  # At t=0, grad is 0
            
        consistency_error = np.abs(trajectory_guess.cpu().numpy() - trajectory_x0[-1]).mean()
        print(f"Consistency error: {consistency_error:.4f}")
            
        # Create figure with four subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))
        
        trajectory_x = np.array(trajectory_x)
        trajectory_t = np.array(trajectory_t)
        trajectory_x0 = np.array(trajectory_x0)
        trajectory_grad = np.array(trajectory_grad)
        
        # Calculate discrete gradient from trajectory_x0
        discrete_grad = np.zeros_like(trajectory_x0)
        discrete_grad[:-1] = -np.cos(trajectory_t[:-1, None]) * (trajectory_x0[1:] - trajectory_x0[:-1]) / (trajectory_t[1:, None] - trajectory_t[:-1, None])
        discrete_grad[:-1] = np.abs(discrete_grad[:-1])
        
        # Plot ODE trajectories in first subplot
        for i in range(trajectory_x.shape[1]):
            ax1.plot(trajectory_t, trajectory_x[:, i], alpha=0.5)
        ax1.set_ylabel('x_t')
        ax1.set_xlabel('t')
        ax1.set_title(f'ODE Trajectories - Epoch {epoch}')
        
        # Plot predicted x0 trajectories in second subplot
        im = ax2.imshow(trajectory_x0.T[:, ::-1], aspect='auto', origin='lower',
                    extent=[trajectory_t.min(), trajectory_t.max(),
                            trajectory_x[0].min(), trajectory_x[0].max()],
                    cmap='viridis', interpolation='nearest')
        ax2.set_title(f'Predicted x0 Trajectories (Error {consistency_error:.4f})')
        ax2.set_ylabel('Initial x value')
        ax2.set_xlabel('t')
        plt.colorbar(im, ax=ax2, label='Predicted x0')
        
        # Plot F_theta_grad trajectories in third subplot
        im = ax3.imshow(trajectory_grad.T[:, ::-1], aspect='auto', origin='lower',
                    extent=[trajectory_t.min(), trajectory_t.max(),
                            trajectory_x[0].min(), trajectory_x[0].max()],
                    cmap='viridis', interpolation='nearest')
        ax3.set_title('Trajectory grad')
        ax3.set_ylabel('Initial x value')
        ax3.set_xlabel('t')
        plt.colorbar(im, ax=ax3, label='Gradient')

        # Plot discrete gradient in fourth subplot
        im = ax4.imshow(discrete_grad.T[:, ::-1], aspect='auto', origin='lower',
                    extent=[trajectory_t.min(), trajectory_t.max(),
                            trajectory_x[0].min(), trajectory_x[0].max()],
                    cmap='viridis', interpolation='nearest')
        ax4.set_title('Discrete Gradient')
        ax4.set_ylabel('Initial x value')
        ax4.set_xlabel('t')
        plt.colorbar(im, ax=ax4, label='Discrete Gradient')

        plt.tight_layout()
        os.makedirs('plots/consistency', exist_ok=True)
        plt.savefig(f'plots/consistency/epoch_{epoch}.png')
        plt.close()
        

if __name__ == '__main__':
    train()
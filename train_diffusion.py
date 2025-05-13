import math
import os
import random
import click
import torch
from torch.utils.data import DataLoader
import torchvision # type: ignore
import torchvision.transforms as transforms # type: ignore
from tqdm import tqdm # type: ignore
from unet import Unet # type: ignore
import matplotlib.pyplot as plt
import wandb

from torchvision.transforms import Compose, Resize, Lambda # type: ignore
from torchmetrics.image.fid import FrechetInceptionDistance

@click.command()
@click.option("--device", default="cuda", help="Device to train on")
@click.option("--epochs", default=200, help="Number of epochs to train")
@click.option("--batch-size", default=128, help="Batch size")
@click.option("--lr", default=3e-4, help="Learning rate")
@click.option("--enable-wandb", is_flag=True , help="Enable wandb")
def train(device, epochs, batch_size, lr, enable_wandb):
    # Setup model and optimizer
    model = Unet(256, 1, 1, base_dim=64, dim_mults=[2, 4]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3, betas=(0.9, 0.99))
    sigma_data = 0.5
    
    P_mean = -1.73
    P_std = 1.4
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (1.0))
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                         download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True, persistent_workers=True)
    
    if enable_wandb:
        wandb.init(
            project="MNIST_diff_for_consistency_model_2",
            entity="actrec",
            config={
                    'lr': lr, 
            }, 
            name="MNIST diffusion for Consistency"
        )

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
            
            pred_v_t, logvar = model(x_t / sigma_data, t.flatten(), return_logvar=True)
            pred_v_t = pred_v_t * sigma_data
            logvar = logvar.view(-1, 1, 1, 1)
            
            v_t = torch.cos(t) * z - torch.sin(t) * images
            loss = (1 / torch.exp(logvar)) * torch.square((pred_v_t - v_t) / sigma_data) + logvar
            loss = loss.mean()
            
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            progress_bar.set_postfix({"loss": loss.item(), "grad_norm": grad_norm.item()})
            step += 1

        if enable_wandb:
            wandb.log({'Train_loss': loss.item()}, step=epoch)

        torch.save(model.state_dict(), f'models/diffusion/model_{epoch}.pt')
        
        
        # Plot one noise sample with different timesteps
        # Sample one image from the dataset for visualization
        print("Save time step example", end='...')

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

        print("DONE")

        #########
        # Generate 16 random samples
        #########
        print("Random Sample", end='...')

        z = torch.randn(16, 1, 28, 28, device=device, generator=torch.Generator(device=device).manual_seed(42)) * sigma_data
        x_t = z  # At t=max, x_t is just noise
        
        # Sample using 100 steps
        with torch.no_grad():
            # Create linearly spaced timesteps from max_t to 0
            # Use Karras timesteps for better sampling quality
            # Formula: sigma_i = sigma_min * (sigma_max/sigma_min)^(1 - (i/(num_steps-1))^rho) where rho=7.0
            rho = 7.0
            i = torch.arange(100, device=device)
            sigma = 0.002 * (80/0.002)**(1 - (i/(100-1))**rho)  # Karras sigma schedule
            timesteps = torch.atan(sigma / sigma_data)  # Convert to t-space
            
            # Iteratively sample using first order solver
            for i in range(len(timesteps)-1):
                s = timesteps[i]  # Current timestep
                t = timesteps[i+1]  # Next timestep
                
                # Get model prediction at current timestep
                pred = model(x_t / sigma_data, s.repeat(16), return_logvar=False)
                
                # Apply first order solver formula
                # xt = cos(s-t)*xs - sin(s-t)*sigma_d*F_theta(xs/sigma_d, s)
                x_t = torch.cos(s - t) * x_t - torch.sin(s - t) * sigma_data * pred
            
            # fids = []
            # Plot final samples
            plt.figure(figsize=(12, 12))
            for i in range(16):
                plt.subplot(4, 4, i+1)
                plt.imshow(x_t[i, 0].cpu().numpy(), cmap='gray')
                plt.axis('off')
            plt.tight_layout()
            os.makedirs('outputs_diffusion/samples', exist_ok=True)
            plt.savefig(f'outputs_diffusion/samples/epoch_{epoch:04d}.png')
            plt.close()

        print("DONE")

        #########
        # FID Evaluation
        #########
        print("Compute FID", end="...")

        sample_n = 1_000
        z = torch.randn(sample_n, 1, 28, 28, device=device, generator=torch.Generator(device=device).manual_seed(42)) * sigma_data
        x_t = z  # At t=max, x_t is just noise
        
        # Sample using 100 steps
        with torch.no_grad():
            # Create linearly spaced timesteps from max_t to 0
            # Use Karras timesteps for better sampling quality
            # Formula: sigma_i = sigma_min * (sigma_max/sigma_min)^(1 - (i/(num_steps-1))^rho) where rho=7.0
            rho = 7.0
            i = torch.arange(100, device=device)
            sigma = 0.002 * (80/0.002)**(1 - (i/(100-1))**rho)  # Karras sigma schedule
            timesteps = torch.atan(sigma / sigma_data)  # Convert to t-space
            
            # Iteratively sample using first order solver
            for i in range(len(timesteps)-1):
                s = timesteps[i]  # Current timestep
                t = timesteps[i+1]  # Next timestep
                
                # Get model prediction at current timestep
                pred = model(x_t / sigma_data, s.repeat(sample_n), return_logvar=False)
                
                # Apply first order solver formula
                # xt = cos(s-t)*xs - sin(s-t)*sigma_d*F_theta(xs/sigma_d, s)
                x_t = torch.cos(s - t) * x_t - torch.sin(s - t) * sigma_data * pred
            
            indices = random.sample(range(len(dataset)), sample_n)
            real_images = torch.stack([dataset[i][0] for i in indices])
            generated_imsages = x_t
            fid_score = calculate_fid(real_images, generated_imsages, device='cuda')
            print(f"FID score: {fid_score}")
            
            if enable_wandb:
                wandb.log({'Train_FID': fid_score}, step=epoch)

        # Estimate average loss at different timesteps
        print("Save Loss fig", end='...')

        sample_img, _ = dataset[0]  # Use first image for loss estimation
        sample_img = sample_img.unsqueeze(0).to(device)
        
        # Test 50 different timesteps
        x = torch.logspace(math.log10(0.002), math.log10(80), 100, device=device)
        test_t = torch.atan(x / sigma_data)
        losses = []
        
        with torch.no_grad():
            for curr_t in test_t:
                # Generate 10 noisy versions for this timestep
                z = torch.randn(10, 1, 28, 28, device=device) * sigma_data
                t = torch.ones(10, device=device).view(-1, 1, 1, 1) * curr_t
                x_t = torch.cos(t) * sample_img + torch.sin(t) * z
                
                # Get predictions and logvar
                pred, logvar = model(x_t / sigma_data, t.flatten(), return_logvar=True)
                logvar = logvar.view(-1, 1, 1, 1)
                pred_x0 = torch.clamp(torch.cos(t) * x_t - torch.sin(t) * sigma_data * pred, min=-0.5, max=0.5)
                
                # Calculate loss with adaptive weighting
                sigma = torch.tan(curr_t) * sigma_data
                weight = 1 / sigma**2 + 1 / sigma_data**2
                batch_loss = weight * torch.square(pred_x0 - sample_img)
                losses.append(batch_loss.mean().item())
        
        # Plot loss curve
        plt.figure(figsize=(10, 5))
        plt.semilogx(torch.tan(test_t).cpu().numpy() * sigma_data, losses)
        plt.xlabel('tan(t)*sigma_data (log scale)')
        plt.ylabel('Mean Loss')
        plt.title('Loss vs Noise Level')
        os.makedirs('outputs_diffusion/losses', exist_ok=True)
        plt.savefig(f'outputs_diffusion/losses/epoch_{epoch:04d}.png')
        plt.close()

        print("DONE")

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
    train()

import math
import torch


class BimodalDataset:
    def __init__(self, batch_size, size=10000, std=0.5, sigma_data=0.5):
        """
        Creates a dataset sampling from a mixture of two Gaussians
        
        Args:
            size: Number of samples in dataset
            std: Standard deviation of the Gaussians. The gaussians have mean 1 or -1 before scaling.
            sigma_data: Standard deviation of the output; we scale the data to have this std.
        """
        self.size = size
        self.batch_size = batch_size
        self.std = std
        self.sigma_data = sigma_data
        self.norm_std = math.sqrt(std**2 + 1)

        self.true_mean = 1 / self.norm_std * self.sigma_data
        self.true_std = self.std / self.norm_std * self.sigma_data
        print("True Mean: ", self.true_mean)
        print("True Std: ", self.true_std)
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate sample from mixture of Gaussians on the fly
        # Generate batch_size samples from mixture of two Gaussians
        probs = torch.rand(self.batch_size)
        means = torch.where(probs < 0.5, 1.0, -1.0)
        sample = means + torch.randn(self.batch_size) * self.std
            
        return sample / self.norm_std * self.sigma_data

def calc_karras_sigmas(sigma_min, sigma_max, steps, rho=7):
    ramp = torch.linspace(0, 1, steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas

def exact_solution(x_t, t, sigma_data, dataset):
    a = torch.sin(t)
    b = torch.cos(t)
    # x_t = a * (z / sigma_data) + b * (x_0 / sigma_data)
    
    mu = dataset.true_mean / sigma_data
    sigma = dataset.true_std / sigma_data
    
    variance_y_given_z = a**2 + b**2 * sigma**2
    
    # Calculate the posterior probabilities P1 and P2
    exp1 = torch.exp(-((x_t - b * mu)**2) / (2 * variance_y_given_z))
    exp2 = torch.exp(-((x_t + b * mu)**2) / (2 * variance_y_given_z))
    
    P1 = exp1 / (exp1 + exp2)
    P2 = 1 - P1
    
    # Calculate the conditional expectation E[z | Y = y]
    term1 = P1 * (mu + (b * sigma**2 / variance_y_given_z) * (x_t - b * mu))
    term2 = P2 * (-mu + (b * sigma**2 / variance_y_given_z) * (x_t + b * mu))
    
    expectation = term1 + term2
    pred_x0 = expectation * sigma_data
    return pred_x0

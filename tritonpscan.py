import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import copy
from torch.utils.data import DataLoader

class STEQuantizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, linear_weight):
        _, indices = torch.topk(x, k=1, dim=-1)
        one_hot = F.one_hot(indices.squeeze(-1), num_classes=x.shape[-1]).float()
        ctx.save_for_backward(x, linear_weight, one_hot)
        output = F.linear(one_hot, linear_weight)
        return output

    @staticmethod
    def backward(ctx, g):
        x, linear_weight, one_hot = ctx.saved_tensors

        # --- Part 1: Gradient for the linear layer's weight (Standard) ---
        grad_linear_weight = g.T @ one_hot

        # --- Part 2: Gradient for the input x (STE Logic) ---
        # The STE simply passes the gradient that would have gone to the one_hot
        # vector straight through to x. This provides a direct, albeit approximate,
        # signal for the reconstruction loss.
        grad_x = g @ linear_weight

        return grad_x, grad_linear_weight

class STEQuantizer(nn.Module):
    def __init__(self, quant_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(quant_dim, output_dim, bias=False)

    def forward(self, x):
        return STEQuantizerFunction.apply(x, self.linear.weight)

class CustomGradientQuantizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, linear_weight):
        _, indices = torch.topk(x, k=1, dim=-1)

        one_hot = F.one_hot(indices.squeeze(-1), num_classes=x.shape[-1]).float()

        ctx.save_for_backward(x, linear_weight, one_hot)

        output = F.linear(one_hot, linear_weight)
        return output

    @staticmethod
    def backward(ctx, g):
        x, linear_weight, one_hot = ctx.saved_tensors

        grad_linear_weight = g.T @ one_hot 

        gl = g @ linear_weight

        _, target_indices = torch.topk(-gl, k=1, dim=-1)

        target_one_hot = F.one_hot(target_indices.squeeze(-1), num_classes=x.shape[-1]).float()
        
        # 4. Calculate the gradient for x. This is the mathematical gradient of
        #    CrossEntropy(x, target), which is (softmax(x) - target_one_hot).
        #    This is the core of your idea, substituted in place of the real gradient.
        grad_x = F.softmax(x, dim=-1) - target_one_hot

        return grad_x, grad_linear_weight


class CustomQuantizer(nn.Module):
    def __init__(self, quant_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(quant_dim, output_dim, bias=False)

    def forward(self, x):
        return CustomGradientQuantizerFunction.apply(x, self.linear.weight)


# --- The NEW Hybrid Gradient Quantizer ---
class HybridQuantizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, linear_weight, beta):
        _, indices = torch.topk(x, k=1, dim=-1)
        one_hot = F.one_hot(indices.squeeze(-1), num_classes=x.shape[-1]).float()
        ctx.save_for_backward(x, linear_weight, one_hot)
        ctx.beta = beta # Save the scalar beta
        output = F.linear(one_hot, linear_weight)
        return output

    @staticmethod
    def backward(ctx, g):
        x, linear_weight, one_hot = ctx.saved_tensors
        beta = ctx.beta

        # Part 1: Gradient for the linear layer's weight (Standard)
        grad_linear_weight = g.T @ one_hot

        # Part 2: Gradient for the input x (The Hybrid Logic)
        # 2a: The STE gradient (task-specific signal)
        ste_grad = g @ linear_weight
        
        # 2b: Your custom gradient (regularization signal)
        # Note: We use x.detach() because this part should only regularize,
        # not be influenced by the STE gradient path itself.
        with torch.no_grad():
            target_indices = torch.topk(-ste_grad, k=1, dim=-1)[1]
            target_one_hot = F.one_hot(target_indices.squeeze(-1), num_classes=x.shape[-1]).float()
        
        custom_grad_regularizer = F.softmax(x.detach(), dim=-1) - target_one_hot

        # 2c: Combine them!
        grad_x = ste_grad + beta * custom_grad_regularizer
        
        # Must return a grad for each input (x, linear_weight, beta). Grad for beta is None.
        return grad_x, grad_linear_weight, None

class HybridQuantizer(nn.Module):
    def __init__(self, quant_dim, output_dim, beta=0.25):
        super().__init__()
        self.linear = nn.Linear(quant_dim, output_dim, bias=False)
        self.beta = beta
    def forward(self, x):
        return HybridQuantizerFunction.apply(x, self.linear.weight, self.beta)

# --- The Autoencoder and Experiment Setup ---
class Autoencoder(nn.Module):
    def __init__(self, encoder, quantizer, decoder):
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder

    def forward(self, x):
        logits = self.encoder(x)
        
        quantized = self.quantizer(logits)
            
        reconstruction = self.decoder(quantized)
        return reconstruction, logits

def calculate_perplexity(logits):
    # Calculates the perplexity of the codebook usage for a batch
    probs = F.softmax(logits, dim=-1).mean(dim=0)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    perplexity = torch.exp(entropy)
    return perplexity.item()

# --- Main Experiment ---
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision('high')
    print(f"Using device: {device}")

    # Hyperparameters
    INPUT_DIM, HIDDEN_DIM, QUANT_DIM, EMBED_DIM = 28*28, 256, 64, 32
    BATCH_SIZE, LEARNING_RATE, STEPS = 256, 1e-3, 10000
    BETA, GAMMA = 0.25, 1.0

    # Data
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST("./", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    data_iterator = iter(train_loader)

    # Base models
    encoder_base = nn.Sequential(nn.Linear(INPUT_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, QUANT_DIM))
    decoder_base = nn.Sequential(nn.Linear(EMBED_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, INPUT_DIM))

    # Instantiate the three models
    models = {
        "STE": Autoencoder(copy.deepcopy(encoder_base), STEQuantizer(QUANT_DIM, EMBED_DIM), copy.deepcopy(decoder_base)),
        "Hybrid": Autoencoder(copy.deepcopy(encoder_base), HybridQuantizer(QUANT_DIM, EMBED_DIM, beta=BETA), copy.deepcopy(decoder_base)),
        "CER": Autoencoder(copy.deepcopy(encoder_base), CustomQuantizer(QUANT_DIM, EMBED_DIM), copy.deepcopy(decoder_base))
    }
    
    print("Compiling models...")
    models = {name: torch.compile(model).to(device) for name, model in models.items()}
    print("Compilation complete.")

    optimizers = {name: torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) for name, model in models.items()}
    
    # Tracking metrics
    metrics = {name: {"loss": [], "perplexity": []} for name in models.keys()}

    print("Starting training to test robustness against codebook collapse...")
    for step in range(STEPS):
        try:
            images, _ = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            images, _ = next(data_iterator)
        
        images = images.view(-1, INPUT_DIM).to(device)

        for name, model in models.items():
            optimizers[name].zero_grad()
            
            recon, logits = model(images)
            
            recon_loss = F.mse_loss(recon, images)
            total_loss = recon_loss
            
            total_loss.backward()
            optimizers[name].step()

            # Record metrics
            metrics[name]["loss"].append(recon_loss.item())
            metrics[name]["perplexity"].append(calculate_perplexity(logits))

        if step % 1000 == 0:
            print(f"--- Step {step:5d} ---")
            for name in models.keys():
                print(f"{name:>10} | Loss: {metrics[name]['loss'][-1]:.6f} | Perplexity: {metrics[name]['perplexity'][-1]:2.3f}/{QUANT_DIM}")

    # --- Plotting Results ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    for name in models.keys():
        ax1.plot(metrics[name]["loss"], label=f'{name} Loss')
        ax2.plot(metrics[name]["perplexity"], label=f'{name} Perplexity')

    ax1.set_title('Reconstruction Loss (MSE)')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    ax2.set_title('Codebook Usage Perplexity')
    ax2.set_ylabel(f'Perplexity (Max = {QUANT_DIM})')
    ax2.set_xlabel('Training Step')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
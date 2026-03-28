
import copy
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.adamw
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from parts import quantizer
import parts.modules as modules
import parts.optim as optim
import parts.utils as utils

class Hotmod(nn.Module):
    def __init__(self, hotfunc, k=None):
        super().__init__()
        self.hotfunc = hotfunc
        params = inspect.signature(self.hotfunc.forward).parameters
        self._needs_k = 'k' in params
        self.k = k

    def forward(self, x):
        
        if self._needs_k:
            return self.hotfunc.apply(x, self.k)
        return self.hotfunc.apply(x)

def acmod(ac):
    class Mod(nn.Module):
        def __init__(self, x=None):
            super().__init__()

        def forward(self, x):
            return ac(x)
    return Mod

class StochasticHotMod(nn.Module):
    def __init__(self, hotfunc, k):
        super().__init__()
        self.hotfunc = hotfunc
        params = inspect.signature(self.hotfunc.forward).parameters
        self._needs_k = 'k' in params
        self.k = k
        self.temp = 1.0

    def forward(self, x:torch.tensor):
        if self.training:
            uniform_samples = torch.rand_like(x)
            gumbels = -torch.log(-torch.log(uniform_samples + 1e-9) + 1e-9)
            gumbels = gumbels * torch.sqrt(x.norm(dim=-1,keepdim=True))
            noisy_x = (x + gumbels) 
            x_mod = noisy_x
        else:
            x_mod = x
        if self._needs_k:
            return self.hotfunc.apply(x_mod, self.k)
        return self.hotfunc.apply(x_mod)   

def calculate_perplexity(logits):
    # Calculates the perplexity of the codebook usage for a batch
    probs = F.softmax(logits, dim=-1).mean(dim=0)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    perplexity = torch.exp(entropy)
    return perplexity.mean().detach().item()

@torch.no_grad()
def run_validation(model, val_loader, device, num_batches=10, input_dim=28*28):
    model.eval()
    total_loss = 0
    total_perplexity = 0
    total_k_perplexity = 0
    total_k = 0
    batches_processed = 1
    for i, (images, _) in enumerate(val_loader):
        if i >= num_batches:
            break
        images = images.view(-1, input_dim).to(device)
        recon, logits, _, k = model(images)
        total_loss += F.mse_loss(recon, images).mean().detach().item()
        if(logits is not None):
            total_perplexity += calculate_perplexity(logits)
        if(k is not None):
            total_k_perplexity += k.var().detach().item()
            #k_indices = torch.argmax(k, dim=-1) + 1
            total_k += k.mean().detach().item()
        batches_processed += 1
    bp = batches_processed 
    avg_k = total_k / bp
    avg_loss = total_loss / bp
    avg_perplexity = total_perplexity / bp
    avg_k_perplexity = total_k_perplexity / bp
    model.train()
    return avg_loss, avg_perplexity, avg_k, avg_k_perplexity



def whiteloss( x, y):
    w = utils.zca_newton_schulz(torch.cat((x,y),dim=0))
    #x,y = torch.chunk(w,2,dim=0)
    x_w, y_w = w.split(x.size(0), dim=0)
    return utils.minmaxnorm(F.mse_loss(x_w , y_w,reduction="none")).sum()
Linear = nn.Linear
class debugAutoencoder(nn.Module):
    def __init__(self, input, hidden, embed, qdim, quantizer, Act , k = None):
        super().__init__()
        self.qdim = qdim
        
        params = inspect.signature(Act.__init__).parameters
        if('in_features' in params):
            ac = Act(hidden)
            ac2 = Act(hidden)
        else:
            ac=Act()
            ac2=Act()
        self.encoder = nn.Sequential(
            Linear(input, hidden), 
            ac, 
            nn.Linear(hidden, qdim))
        self.quantizer = Hotmod(quantizer, k)
        self.codebook = nn.Linear(self.qdim, embed, bias=False)
        self.decoder = nn.Sequential(
            Linear(embed, hidden), 
            ac2, 
            nn.Linear(hidden, input))
    
    def forward(self, x):
        k_hot, vq_loss = self.quant(x)
        reconstruction = self.dequant(k_hot)

        k = utils.k_from_hot(k_hot, self.qdim)
        return reconstruction, k_hot, vq_loss, k

    def quant(self, x):
        logits = self.encoder(x)
        vq_loss = 0.0

        k_hot = self.quantizer(logits)
        
        return k_hot, vq_loss
    

    def dequant(self, k_hot, norm = False):
        k = utils.k_from_hot(k_hot, self.qdim)
        if norm:
            k_hot = k_hot / k.detach()

        q = self.codebook(k_hot) #/ k.detach()
        #q = utils.rms(q)
        return self.decoder(q)

    
    def optimizer(self, LR):
        num_ae = sum(p.numel() for p in self.parameters())
        print(f"ae parameters: {num_ae}")
        return optim.Grms(params=self.parameters(), 
            lr=LR, 
            #nesterov    = True, 
            #momentum    = 0.9,
            fused       = True,
            #weight_decay= 0.01
            )
        #return torch.optim.adamw(params=self.parameters(), lr=LR, fused=True)
    


def quantizer_run():
    # Hyperparameters
    INPUT_DIM, HIDDEN_DIM, QUANT_DIM, EMBED_DIM = 28*28, 256, 64, 32
    BATCH_SIZE, LEARNING_RATE, STEPS = 256, 1e-3, 10001
    
    VAL_LOG_INTERVAL = 1000
    TRAIN_LOG_INTERVAL = 100
    VALIDATION_STEPS = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    full_train_dataset = torchvision.datasets.MNIST("./", train=True, download=True, transform=transform)
    train_size = 50000
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=True)
    data_iterator = iter(train_loader)

    # Base models
    # Instantiate the three models
    models = {
        #"bthot":  debugAutoencoder(INPUT_DIM, HIDDEN_DIM, EMBED_DIM, QUANT_DIM, BoltzmannThresHot, nn.LeakyReLU, 32),
        #"btkhot":  debugAutoencoder(INPUT_DIM, HIDDEN_DIM, EMBED_DIM, QUANT_DIM, BoltzmannTopKHot, nn.LeakyReLU, 32),
        #"tkhot":  debugAutoencoder(INPUT_DIM, HIDDEN_DIM, EMBED_DIM, QUANT_DIM, TopKHot, nn.LeakyReLU, 32),
        "thot":  debugAutoencoder(INPUT_DIM, HIDDEN_DIM, EMBED_DIM, QUANT_DIM, quantizer.ThresHot, acmod(utils.dead_rat_relu), 32),
    }
    
    print("Compiling models...")
    torch.set_float32_matmul_precision('medium')
    models = {name: torch.compile(model.to(device)) for name, model in models.items()}
    auxl = torch.compile(utils.wnormlearnloss(INPUT_DIM).to(device))
    print("Compilation complete.")
    
    optimizers = {name: model.optimizer(LEARNING_RATE) for name, model in models.items()}
    #optimizers = {name: torch.optim.Adam(model.k_predictor.parameters(), lr=LEARNING_RATE*0.001) for name, model in models.items()}
    metrics = {
        name: {
            "train_loss": [], 
            "train_perp": [], 
            "train_k_var": [], 
            "train_k_mean": [], 
            "val_loss": [], 
            "val_perp": [], 
            "val_k_var": [], 
            "val_k_mean": [], 
            } for name in models.keys()}

    print("Starting training with validation...")
    for step in range(STEPS):
        try:
            images, _ = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            images, _ = next(data_iterator)
        images = images.view(-1, INPUT_DIM).to(device)

        # Training Step
        for name, model in models.items():
            optimizers[name].zero_grad(set_to_none=True)
            recon, logits, aux, k = model(images)
            #else:
            #    recon, logits, aux = model(images)
            #    k = model.qdim//2

            loss_raw = F.mse_loss(recon, images)
            
            loss = loss_raw+aux
            #loss = loss_raw + whiteloss(recon, images)
            #loss = loss + auxl(recon)#*0.1
            
            #r2, rq, _, k2 = model(recon.detach())
            #loss = loss + F.mse_loss(recon.detach(), r2) # The target distribution (from the first pass)

            #target_dist = F.softmax(logits.detach() / k, dim=-1)
            #input_dist_log = F.log_softmax(rq / k2, dim=-1)
            #loss = loss + F.kl_div(input_dist_log, target_dist, reduction="batchmean")
            #loss = loss + F.kl_div(F.softmax(logits.detach()/k.detach(),dim=-1), F.softmax(rq/k2,dim=-1), reduction="batchmean") 
            #loss.backward()
            #lw = F.softmax(recon)-F.softmax(images)
            loss.backward()
            optimizers[name].step()
            
            if step % 100 == 0:
                metrics[name]["train_loss"].append(loss_raw.detach().cpu().item())
                if(k is not None):
                    metrics[name]["train_k_var"].append(k.var().detach().cpu().item())
                    metrics[name]["train_k_mean"].append(k.mean().detach().cpu().item())
                else:
                    metrics[name]["train_k_var"].append(0)
                    metrics[name]["train_k_mean"].append(QUANT_DIM//2)
                #metrics[name]["k_value"].append(k.detach().item())
                perp = calculate_perplexity(logits.detach()) if logits is not None else 0.0
                metrics[name]["train_perp"].append(perp)

        if step > 0 and step % VAL_LOG_INTERVAL == 0:
            print(f"--- Step {step:5d} ---")
            for name, model in models.items():
                val_loss, val_perp, kavg, kpavg = run_validation(model, val_loader, device, VALIDATION_STEPS)
                metrics[name]["val_loss"].append(val_loss)
                metrics[name]["val_perp"].append(val_perp)
                metrics[name]["val_k_var"].append(kpavg)
                metrics[name]["val_k_mean"].append(kavg)
                print(f"{name:>10} | Train Loss: {metrics[name]['train_loss'][-1]:.6f} | Val Loss: {val_loss:.6f} | Val Perp: {val_perp:.2f}| Val k: {kavg:.2f}| Val k var: {kpavg:.2f}")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    for name in models.keys():
        # --- Generate Correct X-Axes Based on Logged Data Length ---
        # This is more robust than assuming the training runs to full STEPS
        num_train_logs = len(metrics[name]["train_loss"])
        train_x_axis = np.arange(num_train_logs) * TRAIN_LOG_INTERVAL

        num_val_logs = len(metrics[name]["val_loss"])
        # The first validation happens at VAL_LOG_INTERVAL, not 0
        val_x_axis = (np.arange(num_val_logs) + 1) * VAL_LOG_INTERVAL

        #train_steps_x_axis = range(0, STEPS, TRAIN_LOG_INTERVAL) 
        #val_steps_x_axis = range(0, STEPS, VAL_LOG_INTERVAL) 

        # Plot training loss
        if num_train_logs > 0:
            ax1.plot(train_x_axis, metrics[name]["train_loss"], label=f'{name} Train Loss',       alpha=0.5)
            ax2.plot(train_x_axis, metrics[name]["train_perp"], label=f'{name} Train Perplexity', alpha=0.5)

            train_k_mean = np.array(metrics[name]["train_k_mean"])
            train_k_var = np.array(metrics[name]["train_k_var"])
            train_k_std = np.sqrt(train_k_var) 

            ax3.plot(train_x_axis, train_k_mean, label=f'{name} Train K', alpha=0.5)
            ax3.fill_between(train_x_axis, train_k_mean - train_k_std, train_k_mean + train_k_std, alpha=0.2)
    
        
        if num_val_logs > 0:
            ax1.plot(val_x_axis, metrics[name]["val_loss"], label=f'{name} Val Loss', linestyle='--', marker='o', markersize=4)
            ax2.plot(val_x_axis, metrics[name]["val_perp"], label=f'{name} Val Perplexity', linestyle='--', marker='o', markersize=4)

            val_k_mean = np.array(metrics[name]["val_k_mean"])
            val_k_var = np.array(metrics[name]["val_k_var"])
            val_k_std = np.sqrt(val_k_var)
            ax3.plot(val_x_axis, val_k_mean, label=f'{name} Train K', alpha=0.5)
            ax3.fill_between(val_x_axis, val_k_mean - val_k_std, val_k_mean + val_k_std, alpha=0.2)
    
    ax1.set_title('Training vs. Validation Loss')
    ax1.set_ylabel('Mean Squared Error (MSE) Loss')
    ax1.grid(True, which="both", ls="--")
    ax1.legend()
    ax1.set_yscale('log')
    ax1.set_xlim(0, STEPS)
    
    ax2.set_ylim(0, QUANT_DIM)
    ax2.set_title('Validation Codebook Perplexity')
    ax2.set_ylabel(f'Perplexity (Max = {QUANT_DIM})')
    ax2.set_xlabel('Training Step')
    ax2.grid(True, which="both", ls="--")
    ax2.legend()
    ax2.set_xlim(0, STEPS)

    # --- Formatting for the K Variance Plot (ax3) ---
    ax3.set_title('Variance of K (Codebook Usage)')
    ax3.set_ylabel('Variance')
    ax3.set_xlabel('Training Step')
    ax3.grid(True, which="both", ls="--")
    ax3.legend()
    ax3.set_ylim(bottom=0)

    # Apply shared settings
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0, STEPS)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    #run_seed_viz()
    #run_live_viz()
    quantizer_run()
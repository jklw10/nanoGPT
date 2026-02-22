
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, random_split
import time

import modules
import utils

# --- USER CONSTANTS ---
INPUT_DIM, HIDDEN_DIM, QUANT_DIM, EMBED_DIM = 28*28, 256, 64, 32
BATCH_SIZE, LEARNING_RATE, STEPS = 256, 1e-3, 5  # Reduced steps for demo, increase for real training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- THE META-LEARNER (PREDICTOR) ---
class GradientPredictor(nn.Module):
    """
    Takes (Current_Input_State, Incoming_Gradient)
    Outputs: Guess_For_Input_Gradient
    """
    def __init__(self, input_dim):
        super().__init__()
        # We concat input + grad_output, so input size is dim * 2
        # A simple MLP to map the relationship
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, input_dim)
        )
        
        # Initialize close to 0 so we don't explode immediately
        with torch.no_grad():
            self.net[-1].weight.data *= 0.01
            self.net[-1].bias.data.zero_()

    def forward(self, x, grad_output):
        # x: [Batch, Dim], grad_output: [Batch, Dim]
        combined = torch.cat([x, grad_output], dim=-1)
        return self.net(combined)
class SSMGradientPredictor(nn.Module):
    def __init__(self, input_dim, ssm_dim=64):
        super().__init__()
        
        self.adapter = nn.Sequential(
            nn.Linear(input_dim * 2, ssm_dim),
            nn.LayerNorm(ssm_dim),
            nn.Tanh()
        )
        
        self.ssm = modules.SSM(ssm_dim,0.0,False)
        
        self.head = nn.Sequential(
            nn.Linear(ssm_dim * 2, ssm_dim),
            nn.ReLU(),
            nn.Linear(ssm_dim, input_dim)
        )
        
        self.register_buffer('state', torch.zeros(1, ssm_dim))

    def forward(self, x, grad_output):
        combined = torch.cat([x, grad_output], dim=-1)
        local_emb = self.adapter(combined)
        
        global_input = local_emb.mean(dim=0, keepdim=True)
        
        prev_state = self.state.detach()
        
        ssm_out, new_state = self.ssm.nexts(prev_state, global_input)
        with torch.no_grad():
            self.state = new_state.detach()
        
        global_features = ssm_out.expand(x.shape[0], -1)
        
        final_features = torch.cat([local_emb, global_features], dim=-1)
        
        guess = (self.head(final_features))
        return guess

class GeneralizingGradientPredictor(nn.Module):
    def __init__(self, input_dim, mem_dim=128, mem_slots=8): 
        super().__init__()
        
        self.mem_slots = mem_slots
        self.mem_dim = mem_dim
        
        # Input Adapter: Maps (State + Scream) -> Embedding
        self.input_adapter = nn.Sequential(
            nn.Linear(input_dim * 2, mem_dim), 
            nn.LayerNorm(mem_dim),
            nn.Tanh()
        )

        # Persistent Memory Buffer (The "Old Mem")
        # Initialized with low variance noise
        self.register_buffer('memory', torch.randn(1, mem_slots, mem_dim) * 0.02)
        
        # Attention Blocks for Mixing
        # Block 1: Input interacting with Old Memory
        self.block_input_mem = nn.MultiheadAttention(mem_dim, num_heads=4, batch_first=True)
        self.ln1 = nn.LayerNorm(mem_dim)
        
        # Block 2: Generating the New Memory Candidate
        self.block_mem_gen = nn.MultiheadAttention(mem_dim, num_heads=4, batch_first=True)
        self.ln2 = nn.LayerNorm(mem_dim)
        
        # Output Heads
        self.head_grad = nn.Linear(mem_dim, input_dim)
        self.head_gate = nn.Sequential(
            nn.Linear(mem_dim, input_dim),
            nn.Sigmoid() 
        )
    def forward(self, x, grad_output):
        #shuffle, test
        batch_size = x.shape[0]
        idx = torch.randperm(batch_size, device=x.device)
        x_shuffled = x[idx]
        grad_shuffled = grad_output[idx]
        
        _, test_mem = self.mem_forward(x_shuffled, grad_shuffled, external_memory=None)
        test_mem = test_mem.mean(dim=0, keepdim=True) # [1, 1, Dim]
        malph = 1.0
        avg_update = test_mem.mean(dim=0, keepdim=True) 
        combined = self.memory*(1-malph) + avg_update*malph
        test_mem = utils.rms(combined)
        
        #unshuffle, real value
        x_out, new_mem = self.mem_forward(x, grad_output, external_memory=test_mem)
        self.update_persistent_memory(new_mem,malph)
        return x_out

    def mem_forward(self, x, grad_output, external_memory=None):
        batch_size = x.shape[0]
        
        combined = torch.cat([x, grad_output], dim=-1) 
        x_emb = self.input_adapter(combined).unsqueeze(1) # [B, 1, Mem_Dim]
        
        # 2. Select Memory Source
        # If external_memory (memtest) is provided, use it. Otherwise use persistent self.memory
        if external_memory is not None:
            # If external memory is [1, Slots, Dim], expand to batch
            if external_memory.shape[0] != batch_size:
                mem_context = external_memory.expand(batch_size, -1, -1)
            else:
                mem_context = external_memory
        else:
            mem_context = self.memory.expand(batch_size, -1, -1)

        # 3. Interaction: Input queries Memory
        # "Based on my current state (x_emb), what do I know from history?"
        # Standard Pre-LN Transformer flow
        q = self.ln1(x_emb)
        k = v = self.ln1(mem_context)
        attn_out, _ = self.block_input_mem(q, k, v)
        x_mixed = x_emb + attn_out # Residual
        
        # 4. Generate New Memory Candidate (memtest)
        # "What did I learn from this interaction?"
        q2 = self.ln2(x_mixed)
        attn_out2, _ = self.block_mem_gen(q2, q2, q2) # Self-attend to consolidate
        mem_candidate = x_mixed + attn_out2 # [B, 1, Mem_Dim]
        
        # 5. Decode Gradient
        features = x_mixed.squeeze(1)
        #raw_grad = self.head_grad(features)
        raw_grad = (self.head_grad(features))
        gate = self.head_gate(features)
        final_grad = raw_grad * gate
        
        #final_grad = F.normalize(final_grad)
        return final_grad, mem_candidate

    def update_persistent_memory(self, new_mem_content, m_alpha=1.0):
        with torch.no_grad():
            #known to be worse than dreamer in earlier tests
            avg_update = new_mem_content.mean(dim=0, keepdim=True) 
            combined = self.memory*(1-m_alpha) + avg_update*m_alpha
            self.memory.data = utils.rms(combined)

# --- THE CUSTOM AUTOGRAD FUNCTION ---
class NegotiatedSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, predictor, predictor_opt, simulation_lr=0.01):
        # We are simulating f(x) = x^2
        output = x ** 2
        
        # Save context for backward
        ctx.save_for_backward(x)
        ctx.predictor = predictor
        ctx.predictor_opt = predictor_opt
        ctx.simulation_lr = simulation_lr
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        predictor = ctx.predictor
        predictor_opt = ctx.predictor_opt
        sim_lr = ctx.simulation_lr

        # 1. Detach inputs to create a sandbox for the predictor
        # We don't want to backprop through the main network yet
        x_det = x.detach()
        grad_out_det = grad_output.detach()

        # --- THE "2x WORK" / SIMULATION STEP ---
        with torch.enable_grad():
            # A. Ask the predictor: "How should I update x to satisfy grad_output?"
            guessed_grad_input = predictor(x_det, grad_out_det)
            
            # B. SIMULATE the update locally
            # If we apply this gradient, where does x go?
            #guessed_grad_input = utils.rms(guessed_grad_input)
            x_new_sim = x_det - (sim_lr * guessed_grad_input)
            
            # C. Check the result of the function (x^2)
            y_new_sim = x_new_sim ** 2
            
            # D. Calculate the Target
            # The upstream layer (loss) wanted y to move by (lr * grad_output)
            y_initial = x_det ** 2
            y_target = y_initial - (sim_lr * grad_out_det)
            
            # E. Local Loss: Did the predictor's guess actually move y towards the target?
            # This accounts for the nonlinearity! If y_target is negative, 
            # y_new_sim (which is x^2) can never reach it.
            # The predictor will eventually learn to output 0 or a limited gradient 
            # to minimize this error, rather than exploding.
            pred_loss = F.mse_loss(y_new_sim, y_target)
            
            # F. Train the Predictor NOW (in the backward pass)
            predictor_opt.zero_grad()
            pred_loss.backward()
            # Gradient clipping to keep the predictor stable
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            predictor_opt.step()

        # Return the gradient to the main network
        # We detach it because the main network's optimizer will handle the actual update
        return guessed_grad_input.detach(), None, None, None

# --- THE LAYER WRAPPER ---
class SmartSquareLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.predictor = SSMGradientPredictor(dim)
        # The predictor needs its own optimizer, separate from the main model
        self.predictor_opt = optim.Adam(self.predictor.parameters(), lr=1e-2)
        self.dim = dim

    def forward(self, x):
        return NegotiatedSquare.apply(x, self.predictor, self.predictor_opt)

# --- THE MODEL ---
class SmartMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        
        # REPLACING Standard ReLU with our "Smart Square"
        # This layer learns its own backward pass!
        self.smart_act = SmartSquareLayer(HIDDEN_DIM)
        
        self.fc2 = nn.Linear(HIDDEN_DIM, 10)

    def forward(self, x):
        x = self.flat(x)
        x = self.fc1(x)
        x = self.smart_act(x) # x^2 with learned gradients
        x = self.fc2(x)
        return x
class BaselineMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        # We don't need a smart layer here, just the standard layers
        self.fc2 = nn.Linear(HIDDEN_DIM, 10)

    def forward(self, x):
        x = self.flat(x)
        x = self.fc1(x)
        x = x ** 2  # Standard PyTorch Backprop: Gradient is exactly 2*x
        x = self.fc2(x)
        return x
def train_model(model, train_loader, steps, name):
    print(f"\n--- Training {name} ---")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    start_time = time.time()
    history = []
    
    # Create an infinite iterator for the loop
    data_iter = iter(train_loader)
    
    for step in range(steps):
        try:
            data, target = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            data, target = next(data_iter)
            
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Check for NaN (Exploding Gradients common with x^2)
        if torch.isnan(loss):
            print(f"Step {step}: Loss is NaN! Model exploded.")
            history.append((step, float('nan'), 0.0))
            break
            
        loss.backward()
        
        # Standard grad clipping for baseline stability comparison
        # (Comment out to see if Baseline explodes faster than Smart)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()

        if step % 100 == 0 or step == steps - 1:
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            acc = 100. * correct / len(data)
            elapsed = time.time() - start_time
            print(f"Step {step:4d}: Loss {loss.item():.4f} | Accuracy {acc:.1f}% | Time {elapsed:.1f}s")
            history.append((step, loss.item(), acc))
            
    return history

def run_comparison():
    # Setup Data
    print(f"Preparing data on {DEVICE}...")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST("./", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 1. Train Smart Model (Negotiated Gradients)
    smart_model = SmartMNISTModel().to(DEVICE)
    smart_hist = train_model(smart_model, train_loader, 1000, "Smart Negotiated Model")
    
    # 2. Train Baseline Model (Standard Gradients)
    baseline_model = BaselineMNISTModel().to(DEVICE)
    base_hist = train_model(baseline_model, train_loader, 1000, "Baseline Standard Model")

    # 3. Final Comparison
    print("\n--- Final Results (Step 1000) ---")
    s_loss, s_acc = smart_hist[-1][1], smart_hist[-1][2]
    b_loss, b_acc = base_hist[-1][1], base_hist[-1][2]
    
    print(f"Smart Model:    Loss {s_loss:.4f} | Acc {s_acc:.1f}%")
    print(f"Baseline Model: Loss {b_loss:.4f} | Acc {b_acc:.1f}%")
    
    if s_loss < b_loss:
        print(">> The Negotiated Gradient found a better local minimum.")
    else:
        print(">> The Standard Gradient was more efficient.")

if __name__ == "__main__":
    run_comparison()
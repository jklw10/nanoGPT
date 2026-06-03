import torch
import torch.nn as nn
import torch.nn.functional as F
import parts.utils as utils
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class CognitiveTextGenerator(nn.Module):
    def __init__(self, core_model, vocab_size, dim, max_context_len=100):
        super().__init__()
        self.core = core_model  # StandardNodeNet2 instance
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_context_len = max_context_len
        
        # Token embedding and projection
        self.embedding = nn.Embedding(vocab_size, dim)
        self.unembed = nn.Linear(dim, vocab_size, bias=False)
        self.unembed.weight = self.embedding.weight
        
        # Simple sinusoidal or learned position embeddings for the context retrieval keys
        self.pos_embedding = nn.Embedding(max_context_len, dim)

    def retrieve_token(self, query, context_tokens):
        """
        Queries the context memory using the model's 'read wish' vector.
        """
        batch_size, seq_len = context_tokens.shape
        if seq_len == 0:
            return torch.zeros(batch_size, self.dim, device=query.device)
            
        context_embeds = self.embedding(context_tokens)
        positions = torch.arange(seq_len, device=query.device)
        pos_embeds = self.pos_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)
        
        keys = context_embeds + pos_embeds
        scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1) / (self.dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        retrieved_representation = torch.bmm(attn_weights.unsqueeze(1), context_embeds).squeeze(1)
        return retrieved_representation

    def forward_thinking_step(self, slot_0, retrieved_token):
        """
        Runs a single step of the Core StandardNodeNet2 model.
        """
        env_input = torch.cat([slot_0, retrieved_token], dim=-1)
        loss = self.core(env_input)
        return loss
    
def plot_loss_curve(distances_history, best_actor, dataset_name):
    """Generates and saves a cross-entropy loss curve graph over time."""
    history = torch.stack(distances_history).cpu().numpy()
    num_tokens, batch_size = history.shape
    
    plt.figure(figsize=(10, 5))
    avg_loss = history.mean(axis=1)
    plt.plot(avg_loss, label="Average Sweep Loss", color="gray", alpha=0.6, linestyle="--")
    
    best_loss = history[:, best_actor]
    window = max(1, num_tokens // 10)
    smoothed_best = [np.mean(best_loss[max(0, i-window+1):i+1]) for i in range(len(best_loss))]
    
    plt.plot(best_loss, label="Best Actor Loss (Raw)", color="royalblue", alpha=0.3)
    plt.plot(smoothed_best, label="Best Actor Loss (Smoothed)", color="royalblue", linewidth=2)
    
    plt.title(f"Cross-Entropy Loss Curve over Time ({dataset_name})")
    plt.xlabel("Tokens Processed")
    plt.ylabel("Cross-Entropy Loss (Nats)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig("out/loss_over_time.png", dpi=300)
    plt.close()
    print(f"[Visualization] Graph saved to 'out/loss_over_time.png'.")

def plot_first_step_thinking_loss(thinking_losses):
    """Plots and saves the loss trajectory within the very first step's thinking phase."""
    plt.figure(figsize=(8, 4))
    plt.plot(thinking_losses, marker='o', color='crimson', linewidth=2)
    plt.title("Thinking Phase Loss Settling (First Token Step)")
    plt.xlabel("Thinking Iteration")
    plt.ylabel("Core Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("out/first_step_thinking_loss.png", dpi=300)
    plt.close()
    print(f"[Visualization] First step thinking loss saved to 'out/first_step_thinking_loss.png'.")

def run_vis(time, environment, brain, optimizer, sweep, device):
    """
    Harness tracking model training via Cross-Entropy loss.
    """
    dataset_name = environment if environment else "openwebtext"
    if "MiniGrid" in dataset_name or "Pendulum" in dataset_name:
        dataset_name = "openwebtext"

    always_byte = (dataset_name == "openwebtext")
    decode, vocab_size = utils.get_meta(dataset_name, always_byte=always_byte) 
    dim = brain.core.D

    thinking_steps = 10  
    num_tokens = max(5, time // thinking_steps)
    eval_start_token = num_tokens // 2
    block_size = num_tokens + 1

    get_batch = utils.get_get_batch(
        block_size=block_size,
        batch_size=sweep.batch_size,
        dataset=dataset_name,
        device=device,
        always_byte=always_byte,
        single=True
    )

    raw_codebook = nn.Parameter(torch.randn(vocab_size, dim, device=device) / (dim ** 0.5))
    codebook_optimizer = torch.optim.AdamW([raw_codebook], lr=0.01, weight_decay=0.01)

    target_sequences = get_batch('train', size=block_size) 

    context_tokens = target_sequences[:, :1]
    distances_history = []
    predictions_history = []
    
    # Track core loss during thinking iterations of the first prediction step (t = 0)
    first_step_thinking_losses = []

    # 3. Active Inference Sequence Training Loop
    for t in range(num_tokens):
        current_tok = target_sequences[:, t]
        next_tok = target_sequences[:, t + 1]
        
        codebook = F.normalize(raw_codebook, p=2, dim=-1)
        slot_0 = codebook[current_tok].detach()
        next_tok_embed = codebook[next_tok].detach()
        R = torch.zeros_like(slot_0)
        
        # Thinking phase (MSE optimization of brain parameters/states)
        for think_step in range(thinking_steps):
            env_input = torch.cat([slot_0, R], dim=-1)
            
            if optimizer is not None:
                optimizer.zero_grad()
                loss, output = brain(env_input)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    loss, output = brain(env_input)
                    
            # Record the settling losses of the very first step
            if t == 0:
                first_step_thinking_losses.append(loss.item())
                    
            read_wish = output[:, 2, :].detach()
            context_embeds = codebook[context_tokens].detach()
            
            scores = torch.bmm(context_embeds, read_wish.unsqueeze(-1)).squeeze(-1) / (dim ** 0.5)
            attn_weights = F.softmax(scores, dim=-1)
            R = torch.bmm(attn_weights.unsqueeze(1), context_embeds).squeeze(1)

        # 4. Compute next-token prediction metrics via Cross-Entropy
        predicted_next_embed = output[:, 0, :].detach() 
        predictions_history.append(predicted_next_embed.detach())
        
        logits = torch.matmul(predicted_next_embed, codebook.t())
        ce_loss_mean = F.cross_entropy(logits, next_tok)
        
        codebook_optimizer.zero_grad()
        ce_loss_mean.backward()
        codebook_optimizer.step()
        
        with torch.no_grad():
            ce_loss_ind = F.cross_entropy(logits, next_tok, reduction='none')
            distances_history.append(ce_loss_ind)
        
        context_tokens = torch.cat([context_tokens, next_tok.unsqueeze(-1)], dim=-1)
        if context_tokens.shape[1] > 32:
            context_tokens = context_tokens[:, -32:]

        if (t + 1) % max(1, num_tokens // 4) == 0 or t == num_tokens - 1:
            dists_tensor = torch.stack(distances_history)
            current_mean_dists = dists_tensor.mean(dim=0)
            best_actor = torch.argmin(current_mean_dists).item()
            
            preds_tensor = torch.stack(predictions_history)
            best_preds = preds_tensor[:, best_actor, :]
            
            sims = torch.matmul(best_preds, codebook.t())
            decoded_ids = torch.argmax(sims, dim=-1).cpu().tolist()
            
            predicted_str = decode(decoded_ids)[1]
            target_ids = target_sequences[best_actor, 1:t+2].cpu().tolist()
            target_str = decode(target_ids)[1]
            
            print(f"Step {t+1}/{num_tokens} | Best Actor Loss: {current_mean_dists[best_actor].item():.4f} nats")

    # Save visualization of the first step's thinking progression
    if len(first_step_thinking_losses) > 0:
        plot_first_step_thinking_loss(first_step_thinking_losses)

    final_dists = torch.stack(distances_history)
    eval_distances = final_dists[eval_start_token:]
    
    mean_dist = eval_distances.mean(dim=0)
    var_dist = eval_distances.var(dim=0)
    
    fitness = torch.sigmoid(-mean_dist * 0.1 - var_dist * 0.01)
    best_actor = torch.argmax(fitness).item()

    plot_loss_curve(distances_history, best_actor, dataset_name)

    # 6. Autoregressive 200-Token Output Generation
    print("\n" + "="*60)
    print(f"Generating 200 Tokens Autoregressively from Best Agent ({best_actor})")
    print("="*60)
    
    gen_context = target_sequences[:, :1].clone()
    generated_ids = []

    for g_step in range(200):
        current_tok = gen_context[:, -1]
        slot_0 = codebook[current_tok].detach()
        R = torch.zeros_like(slot_0)
        
        # CHANGE: Always run inference under no_grad to freeze model weights
        with torch.no_grad():
            for think_step in range(thinking_steps):
                env_input = torch.cat([slot_0, R], dim=-1)
                
                # Retrieve loss & output without using model optimizer
                loss, output = brain(env_input)
                        
                read_wish = output[:, 2, :].detach()
                context_embeds = codebook[gen_context].detach()
                
                scores = torch.bmm(context_embeds, read_wish.unsqueeze(-1)).squeeze(-1) / (dim ** 0.5)
                attn_weights = F.softmax(scores, dim=-1)
                R = torch.bmm(attn_weights.unsqueeze(1), context_embeds).squeeze(1)

            predicted_next_embed = output[:, 0, :].detach()
            logits = torch.matmul(predicted_next_embed, codebook.t())
            
            probs = F.softmax(logits / 0.8, dim=-1)
            sampled_tok = torch.multinomial(probs, num_samples=1)
            
            gen_context = torch.cat([gen_context, sampled_tok], dim=-1)
            generated_ids.append(sampled_tok.squeeze(-1))
            
            if gen_context.shape[1] > 32:
                gen_context = gen_context[:, -32:]

    generated_tensor = torch.stack(generated_ids)
    best_generated_ids = generated_tensor[:, best_actor].cpu().tolist()
    #generated_text = decode(best_generated_ids)[1]
    generated_text = decode(best_generated_ids)
    
    print("\n--- AUTOREGRESSIVE GENERATION ---")
    print("\nRaw generated IDs:", best_generated_ids)
    print(generated_text)
    print("="*60 + "\n")

    return fitness, mean_dist


def generate_with_thinking(
    model, 
    prompt_tokens, 
    max_tokens_to_generate, 
    optimizer=None, # Passed if training online, but kept None for inference
    max_thinking_steps=100, 
    loss_threshold=0.01,
    temperature=1.0,
    top_k=50
):
    device = prompt_tokens.device
    batch_size = prompt_tokens.shape[0]
    
    context_tokens = prompt_tokens.clone()
    current_token = prompt_tokens[:, -1]
    slot_0 = model.embedding(current_token).detach()
    
    generated_sequence = []
    
    for t_step in range(max_tokens_to_generate):
        retrieved_token = torch.zeros(batch_size, model.dim, device=device)
        
        # --- THINKING PHASE ---
        # If optimizer is provided here, it will modify the weights of `model`
        # and cause generation collapse. For pure generation, pass optimizer=None.
        for think_step in range(max_thinking_steps):
            if optimizer is not None:
                model.train()
                optimizer.zero_grad()
                loss = model.forward_thinking_step(slot_0, retrieved_token)
                loss.backward()
                optimizer.step()
            else:
                model.eval()
                with torch.no_grad():
                    loss = model.forward_thinking_step(slot_0, retrieved_token)
            
            read_wish = model.core.output[:, 2, :].detach()
            retrieved_token = model.retrieve_token(read_wish, context_tokens)
            
            if loss.item() / batch_size < loss_threshold:
                break
                
        # --- DECISION PHASE ---
        with torch.no_grad():
            next_token_prediction = model.core.output[:, 0, :]
            logits = model.unembed(next_token_prediction)
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
        generated_sequence.append(next_token.unsqueeze(1))
        context_tokens = torch.cat([context_tokens, next_token.unsqueeze(1)], dim=1)
        
        if context_tokens.shape[1] > model.max_context_len:
            context_tokens = context_tokens[:, -model.max_context_len:]
            
        current_token = next_token
        slot_0 = model.embedding(current_token).detach()
        
    return torch.cat([prompt_tokens, torch.cat(generated_sequence, dim=1)], dim=1)
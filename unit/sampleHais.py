import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Attempt to load configuration and models from your codebase
try:
    from models.HAIS import GPTConfig, HRMAgent
except ImportError:
    print("[!] Could not import models.HAIS. Ensure the script is run from the directory containing models/")
    sys.exit(1)

# Check Pygame availability for the graphical environment
pygame_available = False
try:
    import pygame
    pygame_available = True
except ImportError:
    print("[*] Pygame not detected. Falling back to Terminal-only interactive mode.")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================================================================
# HELPER CLASSES & FUNCTION DEFINITIONS
# =========================================================================

class ByteStreamDecoder:
    """Helper class to accumulate and decode incoming UTF-8 bytes gracefully."""
    def __init__(self):
        self.byte_queue = []

    def decode_and_print(self, new_byte):
        self.byte_queue.append(new_byte)
        try:
            decoded_str = bytes(self.byte_queue).decode('utf-8')
            self.byte_queue.clear()
            return decoded_str
        except UnicodeDecodeError:
            if len(self.byte_queue) > 4:
                self.byte_queue.pop(0)
            return ""

def sample_logits(logits, temperature=0.7, top_k=40):
    """Applies temperature scaling and top-k filtering before sampling."""
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        if logits.dim() == 1:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = -float('Inf')
        else:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def clean_text(text):
    """Safely strips null bytes and null characters to prevent Pygame render crashes."""
    return str(text).replace('\x00', '').replace('\u0000', '')


# =========================================================================
# PHASE 2 RECURRENT ACTIVE-INFERENCE GENERATION (For Coop Checkpoints)
# =========================================================================

def init_recurrent_states(agent, prompt_bytes, keys=None, vals=None, mask=None):
    """
    Sequentially builds the recurrent state vector over prompt bytes.
    Retains parameter gradients for active-inference loops.
    """
    device = agent.W_talk.device
    states = [block.ssm.identity.unsqueeze(0).clone() for block in agent.blocks]
    
    for b in prompt_bytes:
        char_tensor = torch.tensor([[b]], device=device)
        x_one_hot = F.one_hot(char_tensor, num_classes=agent.config.vocab_size).float()
        x = agent.token_proj(x_one_hot) # [B=1, T=1, n_embd]
        
        new_states = []
        for l, block in enumerate(agent.blocks):
            norm_x = block.ln_1(x)
            norm_x_squeezed = norm_x.squeeze(1) # [B=1, n_embd]
            
            ssm_out_squeezed, next_s = block.ssm.nexts(states[l], norm_x_squeezed)
            new_states.append(next_s)
            
            ssm_out = ssm_out_squeezed.unsqueeze(1)
            x = x + ssm_out
            
            if keys is not None:
                x = x + block.memory(x, keys=keys, vals=vals, mask=mask)
            
            x = x + block.mlp_out(block.gelu(block.mlp_proj(block.ln_2(x))))
        states = new_states
    return states


def recurrent_step(agent, curr_byte, states, keys=None, vals=None, mask=None, temperature=0.7, top_k=40):
    """
    Advances the sequence model exactly 1 step using its causal recurrent transition weights.
    Gradients are kept active.
    """
    device = agent.W_talk.device
    char_tensor = torch.tensor([[curr_byte]], device=device)
    x_one_hot = F.one_hot(char_tensor, num_classes=agent.config.vocab_size).float()
    x = agent.token_proj(x_one_hot)
    
    new_states = []
    for l, block in enumerate(agent.blocks):
        norm_x = block.ln_1(x)
        norm_x_squeezed = norm_x.squeeze(1)
        
        ssm_out_squeezed, next_s = block.ssm.nexts(states[l], norm_x_squeezed)
        new_states.append(next_s)
        
        ssm_out = ssm_out_squeezed.unsqueeze(1)
        x = x + ssm_out
        
        if keys is not None:
            x = x + block.memory(x, keys=keys, vals=vals, mask=mask)
        
        x = x + block.mlp_out(block.gelu(block.mlp_proj(block.ln_2(x))))
    
    x_norm = agent.ln_f(x)
    logits = agent.lm_head(x_norm)[:, -1, :]
    
    next_token = sample_logits(logits, temperature=temperature, top_k=top_k)
    next_byte = next_token.item()
    
    # Update Dynamic Memory Blocks based on action selections
    if keys is not None:
        latent_act = x_norm[:, -1, :]
        mem_write_logits  = latent_act @ agent.W_mem_write
        mem_delete_logits = latent_act @ agent.W_mem_delete
        
        write_slot = torch.argmax(mem_write_logits, dim=-1)
        delete_slot = torch.argmax(mem_delete_logits, dim=-1)
        
        mask[0, delete_slot] = 0.0
        keys[0, write_slot] = latent_act
        vals[0, write_slot] = latent_act
        mask[0, write_slot] = 1.0
        
    return next_byte, new_states


# =========================================================================
# ONLINE WEIGHT ADAPTATION SYSTEM (With Constant Shape Padding)
# =========================================================================
def train_on_sequence(agent, optimizer, seq_bytes, mem_keys, mem_vals, mem_mask):
    """
    Performs online weight adaptation on the sequence of bytes using Phase 2 Active Inference.
    Pads inputs to constant config.block_size to bypass SymPy dynamic compiler exceptions.
    """
    if len(seq_bytes) < 2:
        return 0.0

    agent.train()
    optimizer.zero_grad(set_to_none=True)

    block_size = agent.config.block_size
    seq_bytes = seq_bytes[:block_size]

    inputs = seq_bytes[:-1]
    targets = seq_bytes[1:]
    actual_len = len(inputs)

    padded_inputs = [0] * block_size
    padded_targets = [-100] * block_size

    padded_inputs[:actual_len] = inputs
    padded_targets[:actual_len] = targets

    inputs_tensor = torch.tensor([padded_inputs], device=device)
    targets_tensor = torch.tensor([padded_targets], device=device)

    x_one_hot = F.one_hot(inputs_tensor, num_classes=agent.config.vocab_size).float()
    
    out = agent(
        x_one_hot, 
        keys=mem_keys, 
        vals=mem_vals, 
        mask=mem_mask, 
        is_text=False
    )

    logits = out['proposal_logits']
    aux_loss = out['aux_loss']
    boredom_scale = out['boredom_scale']

    with torch.no_grad():
        clamped_targets = torch.clamp(targets_tensor, min=0)
        target_one_hot = F.one_hot(clamped_targets, num_classes=agent.config.vocab_size).float()
        latent_target = agent.token_proj(target_one_hot)

    loss_mask = (targets_tensor != -100).float().unsqueeze(-1)
    pred_loss = F.mse_loss(out['pred_state'] * loss_mask, latent_target * loss_mask)

    refined_logits = logits.detach().clone().requires_grad_(True)
    for tick in range(agent.config.diffusion_ticks):
        energy = F.cross_entropy(
            refined_logits.view(-1, agent.config.vocab_size), 
            targets_tensor.view(-1), 
            ignore_index=-100
        )
        energy_grad = torch.autograd.grad(energy, refined_logits)[0]
        
        langevin_noise_scale = torch.sqrt(boredom_scale.detach()) * 0.01
        refined_logits = refined_logits - agent.config.langevin_lr * energy_grad + langevin_noise_scale * torch.randn_like(refined_logits)

    proposal_loss = F.cross_entropy(
        logits.view(-1, agent.config.vocab_size), 
        targets_tensor.view(-1), 
        ignore_index=-100
    )
    amortization_loss = F.mse_loss(logits * loss_mask, refined_logits.detach() * loss_mask)
    
    total_loss = pred_loss + 0.1 * aux_loss + proposal_loss + 0.1 * amortization_loss
    total_loss.backward()
    
    optimizer.step()

    mem_keys.detach_()
    mem_vals.detach_()
    mem_mask.detach_()

    return total_loss.item()


def train_on_sequence_autoregressive(agent, optimizer, seq_bytes):
    """
    Standard autoregressive online adaptation for Phase 1 baseline models.
    """
    if len(seq_bytes) < 2:
        return 0.0
    agent.train()
    optimizer.zero_grad(set_to_none=True)
    
    block_size = agent.config.block_size
    seq_bytes = seq_bytes[:block_size]
    
    inputs = seq_bytes[:-1]
    targets = seq_bytes[1:]
    actual_len = len(inputs)
    
    padded_inputs = [0] * block_size
    padded_targets = [-100] * block_size
    padded_inputs[:actual_len] = inputs
    padded_targets[:actual_len] = targets
    
    inputs_tensor = torch.tensor([padded_inputs], device=device)
    targets_tensor = torch.tensor([padded_targets], device=device)
    
    _, loss = agent(inputs_tensor, targets=targets_tensor, is_text=True)
    loss.backward()
    optimizer.step()
    return loss.item()


# =========================================================================
# GRAPHICAL INTERACTIVE PYGAME ENVIRONMENT
# =========================================================================
def run_pygame_session(agents, optimizers, mem_keys, mem_vals, mem_mask, is_coop_active, active_idx=0):
    pygame.init()
    pygame.key.set_repeat(250, 40)
    
    width, height = 1100, 700
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("HAIS Dynamic Active Inference Swarm Interface")
    clock = pygame.time.Clock()

    try:
        font_main = pygame.font.SysFont("Consolas", 15)
        font_bold = pygame.font.SysFont("Consolas", 18, bold=True)
        font_title = pygame.font.SysFont("Consolas", 24, bold=True)
    except Exception:
        font_main = pygame.font.Font(None, 20)
        font_bold = pygame.font.Font(None, 24)
        font_title = pygame.font.Font(None, 32)

    energies = [0.8, 0.75, 0.65, 0.85]
    boredom_rates = [0.1, 0.2, 0.15, 0.05]
    latest_loss = 0.0
    
    messages = [
        {"sender": "System", "text": "Active Inference Workspace initialized."},
        {"sender": "System", "text": "Chat to update weights and trigger online learning dynamically."}
    ]
    input_text = ""
    agent_typing = False
    agent_response_bytes = []
    recurrent_states = []
    generation_context = []
    curr_byte = 10
    max_response_len = 384
    share_pulses = []

    def wrap_text(text, rect, font):
        text = clean_text(text)
        words = text.split(' ')
        lines = []
        curr_line = []
        for word in words:
            test_line = ' '.join(curr_line + [word])
            if font.size(test_line)[0] < rect.width:
                curr_line.append(word)
            else:
                lines.append(' '.join(curr_line))
                curr_line = [word]
        if curr_line:
            lines.append(' '.join(curr_line))
        return lines

    running = True
    while running:
        screen.fill((11, 15, 25))
        tick_time = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if not agent_typing:
                    if event.key == pygame.K_RETURN:
                        if input_text.strip():
                            user_msg = clean_text(input_text)
                            messages.append({"sender": "You", "text": user_msg})
                            input_text = ""
                            
                            prompt_bytes = list(user_msg.encode('utf-8')) + [10]
                            
                            active_agent = agents[active_idx]
                            opt = optimizers[active_idx]
                            
                            if is_coop_active:
                                latest_loss = train_on_sequence(
                                    active_agent, opt, prompt_bytes, 
                                    mem_keys[active_idx], mem_vals[active_idx], mem_mask[active_idx]
                                )
                            else:
                                latest_loss = train_on_sequence_autoregressive(
                                    active_agent, opt, prompt_bytes
                                )
                            
                            agent_typing = True
                            agent_response_bytes = []
                            generation_context = list(prompt_bytes)
                            
                            if is_coop_active:
                                active_agent.eval()
                                recurrent_states = init_recurrent_states(
                                    active_agent, prompt_bytes,
                                    mem_keys[active_idx], mem_vals[active_idx], mem_mask[active_idx]
                                )
                                curr_byte = prompt_bytes[-1]
                            
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        input_text += event.unicode

        if tick_time % 60 == 0:
            for idx in range(4):
                energies[idx] = max(0.1, energies[idx] - 0.001)
                if idx != active_idx and math.sin(tick_time * 0.005 + idx) > 0.8:
                    energies[idx] = min(1.0, energies[idx] + 0.05)
                    share_pulses.append({
                        "start": idx, 
                        "dest": (idx + 1) % 4, 
                        "progress": 0.0, 
                        "val": 0.03
                    })
                    energies[(idx + 1) % 4] = min(1.0, energies[(idx + 1) % 4] + 0.01)

        if agent_typing:
            active_agent = agents[active_idx]
            active_agent.eval()
            
            if is_coop_active:
                for _ in range(3):
                    if not agent_typing:
                        break
                    
                    next_byte, recurrent_states = recurrent_step(
                        active_agent,
                        curr_byte,
                        recurrent_states,
                        mem_keys[active_idx],
                        mem_vals[active_idx],
                        mem_mask[active_idx]
                    )
                    
                    boredom_rates[active_idx] = 0.0
                    agent_response_bytes.append(next_byte)
                    curr_byte = next_byte

                    if len(agent_response_bytes) >= max_response_len or next_byte == 10:
                        agent_typing = False
                        
                        # Detach memory values from the autograd history tracking
                        mem_keys[active_idx].detach_()
                        mem_vals[active_idx].detach_()
                        mem_mask[active_idx].detach_()
                        
                        decoded_res = bytes(agent_response_bytes).decode('utf-8', errors='ignore').strip()
                        messages.append({"sender": "Agent", "text": clean_text(decoded_res)})
            else:
                for _ in range(3):
                    if not agent_typing:
                        break
                    
                    actual_len = len(generation_context)
                    block_size = active_agent.config.block_size
                    
                    # Pad context sequence to constant shape (block_size)
                    padded_context = [0] * block_size
                    if actual_len > block_size:
                        padded_context = generation_context[-block_size:]
                        actual_len = block_size
                    else:
                        padded_context[:actual_len] = generation_context
                    
                    context_tensor = torch.tensor([padded_context], device=device)
                    
                    with torch.no_grad():
                        logits, _ = active_agent(context_tensor, is_text=True)
                        next_token_logits = logits[0, actual_len - 1, :]
                        next_token = sample_logits(next_token_logits, temperature=0.7, top_k=40)
                        next_byte = next_token.item()
                        
                    agent_response_bytes.append(next_byte)
                    generation_context.append(next_byte)

                    if len(agent_response_bytes) >= max_response_len or next_byte == 10:
                        agent_typing = False
                        decoded_res = bytes(agent_response_bytes).decode('utf-8', errors='ignore').strip()
                        messages.append({"sender": "Agent", "text": clean_text(decoded_res)})

        pygame.draw.rect(screen, (21, 29, 48), (20, 20, 480, 660))
        pygame.draw.rect(screen, (21, 29, 48), (520, 20, 560, 660))
        
        screen.blit(font_title.render("HAIS SWARM VISUALIZATION", True, (0, 242, 254)), (40, 40))
        screen.blit(font_main.render(clean_text(f"ACTIVE RECOVERY: ONLINE  | DEVICE: {device.upper()}"), True, (157, 78, 221)), (40, 70))
        
        agent_positions = [
            (260, 180),
            (380, 300),
            (260, 420),
            (140, 300)
        ]
        
        for pulse in share_pulses[:]:
            p1 = agent_positions[pulse["start"]]
            p2 = agent_positions[pulse["dest"]]
            pulse["progress"] += 0.05
            if pulse["progress"] >= 1.0:
                share_pulses.remove(pulse)
            else:
                x = p1[0] + (p2[0] - p1[0]) * pulse["progress"]
                y = p1[1] + (p2[1] - p1[1]) * pulse["progress"]
                pygame.draw.circle(screen, (0, 245, 212), (int(x), int(y)), 6)

        for idx, pos in enumerate(agent_positions):
            if idx == active_idx:
                glow_color = (0, 242, 254)
                pygame.draw.circle(screen, glow_color, pos, 45, 3)
            else:
                pygame.draw.circle(screen, (74, 85, 104), pos, 38, 1)

            pygame.draw.circle(screen, (15, 23, 42), pos, 35)
            
            energy_angle = 360 * energies[idx]
            rect = pygame.Rect(pos[0]-35, pos[1]-35, 70, 70)
            pygame.draw.arc(screen, (0, 245, 212), rect, 0, math.radians(energy_angle), 3)

            lbl = font_bold.render(clean_text(f"A{idx}"), True, (226, 232, 240))
            screen.blit(lbl, (pos[0]-10, pos[1]-15))
            
            lbl_nrg = font_main.render(clean_text(f"{int(energies[idx]*100)}%"), True, (148, 163, 184))
            screen.blit(lbl_nrg, (pos[0]-15, pos[1]+5))

            mem_offset_x = pos[0] + 55 if idx in [0, 1] else pos[0] - 95
            mem_offset_y = pos[1] - 25
            
            slots = mem_mask[idx][0].cpu().numpy()
            for s_idx in range(16):
                col = (0, 242, 254) if slots[s_idx] > 0.5 else (40, 50, 70)
                px = mem_offset_x + (s_idx % 4) * 8
                py = mem_offset_y + (s_idx // 4) * 8
                pygame.draw.rect(screen, col, (px, py, 6, 6))

        status_y = 530
        pygame.draw.rect(screen, (15, 23, 42), (40, status_y, 440, 120))
        screen.blit(font_bold.render("DYNAMIC TRAINING MONITOR:", True, (226, 232, 240)), (50, status_y + 10))
        
        loss_text = f"Last Step Objective Loss: {latest_loss:.6f}" if latest_loss > 0 else "System idle. Awaiting prompts."
        screen.blit(font_main.render(clean_text(loss_text), True, (0, 245, 212)), (50, status_y + 40))
        
        boredom_text = f"Active Agent Boredom Level: {boredom_rates[active_idx]:.4f}"
        screen.blit(font_main.render(clean_text(boredom_text), True, (241, 95, 120)), (50, status_y + 65))
        
        status_text = "WEIGHTS ADAPTING LIVE" if agent_typing else "GRADIENTS STANDING BY"
        screen.blit(font_main.render(clean_text(f"Optimization State: {status_text}"), True, (157, 78, 221)), (50, status_y + 90))

        screen.blit(font_title.render("NEURAL CHAT TERMINAL", True, (0, 242, 254)), (540, 40))
        
        chat_box = pygame.Rect(540, 80, 520, 480)
        pygame.draw.rect(screen, (15, 23, 42), chat_box)
        
        current_y = 540
        for msg in reversed(messages):
            sender = clean_text(msg["sender"])
            text = clean_text(msg["text"])
            
            col_sender = (0, 242, 254) if sender == "You" else (157, 78, 221) if sender == "Agent" else (148, 163, 184)
            wrapped_lines = wrap_text(f"{sender}: {text}", pygame.Rect(0, 0, 490, 480), font_main)
            
            for line in reversed(wrapped_lines):
                current_y -= 18
                if current_y < 90:
                    break
                screen.blit(font_main.render(clean_text(line), True, (226, 232, 240) if sender != "System" else col_sender), (550, current_y))

        if agent_typing:
            gen_text = bytes(agent_response_bytes).decode('utf-8', errors='ignore')
            wrapped_lines = wrap_text(f"Agent: {gen_text}█", pygame.Rect(0, 0, 490, 480), font_main)
            gen_y = 540 - (18 * len(wrapped_lines))
            for i, line in enumerate(wrapped_lines):
                if gen_y + i * 18 >= 90:
                    screen.blit(font_main.render(clean_text(line), True, (157, 78, 221)), (550, gen_y + i * 18))

        input_box = pygame.Rect(540, 580, 520, 80)
        pygame.draw.rect(screen, (15, 23, 42), input_box)
        pygame.draw.rect(screen, (0, 242, 254) if not agent_typing else (74, 85, 104), input_box, 1)
        
        if agent_typing:
            screen.blit(font_main.render("Generating response and optimizing network...", True, (148, 163, 184)), (550, 595))
        else:
            cursor = "_" if (tick_time // 400) % 2 == 0 else ""
            txt_surface = font_main.render(clean_text(f"You: {input_text}{cursor}"), True, (226, 232, 240))
            screen.blit(txt_surface, (550, 595))
            screen.blit(font_main.render("[Press Return/Enter to Submit]", True, (74, 85, 104)), (550, 635))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


# =========================================================================
# TERMINAL-ONLY INTERACTIVE MODE (Fallback Engine)
# =========================================================================
def run_terminal_session(agents, optimizers, mem_keys, mem_vals, mem_mask, is_coop_active, active_idx=0):
    agent = agents[active_idx]
    optimizer = optimizers[active_idx]
    
    print("\n" + "="*50)
    print(" Active Inference Terminal Session (Fallback)")
    print(" Network gradients and weights adapt in real-time.")
    print(" Type 'exit' or 'quit' to end the session.")
    print("="*50 + "\n")
    
    stream_decoder = ByteStreamDecoder()
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                break
                
            if not user_input.strip():
                continue
                
            prompt_bytes = list(user_input.encode('utf-8')) + [10]
            
            # --- 1. ONLINE PROPAGATION UPDATE (Learning Phase) ---
            print("[*] Propagating sequences down energy surface... Updating weights.")
            if is_coop_active:
                loss = train_on_sequence(
                    agent, optimizer, prompt_bytes, 
                    mem_keys[active_idx], mem_vals[active_idx], mem_mask[active_idx]
                )
            else:
                loss = train_on_sequence_autoregressive(
                    agent, optimizer, prompt_bytes
                )
            print(f"[*] Gradient Step complete. Calculated Objective Loss: {loss:.6f}\n")
            
            # --- 2. RESPONSE GENERATION ---
            agent.eval()
            print("Agent: ", end='', flush=True)
            
            if is_coop_active:
                recurrent_states = init_recurrent_states(
                    agent, prompt_bytes,
                    mem_keys[active_idx], mem_vals[active_idx], mem_mask[active_idx]
                )
                curr_byte = prompt_bytes[-1]
                max_response_len = 384
                generated_bytes = []
                
                for _ in range(max_response_len):
                    next_byte, recurrent_states = recurrent_step(
                        agent,
                        curr_byte,
                        recurrent_states,
                        mem_keys[active_idx],
                        mem_vals[active_idx],
                        mem_mask[active_idx]
                    )
                    
                    print(stream_decoder.decode_and_print(next_byte), end='', flush=True)
                    generated_bytes.append(next_byte)
                    curr_byte = next_byte
                    
                    if next_byte == 10:
                        break
                print()
                
                mem_keys[active_idx].detach_()
                mem_vals[active_idx].detach_()
                mem_mask[active_idx].detach_()
            else:
                generation_context = list(prompt_bytes)
                max_response_len = 384
                generated_bytes = []
                
                for _ in range(max_response_len):
                    actual_len = len(generation_context)
                    block_size = agent.config.block_size
                    
                    # Pad context sequence to constant shape (block_size)
                    padded_context = [0] * block_size
                    if actual_len > block_size:
                        padded_context = generation_context[-block_size:]
                        actual_len = block_size
                    else:
                        padded_context[:actual_len] = generation_context
                    
                    context_tensor = torch.tensor([padded_context], device=device)
                    
                    with torch.no_grad():
                        logits, _ = agent(context_tensor, is_text=True)
                        next_token_logits = logits[0, actual_len - 1, :]
                        next_token = sample_logits(next_token_logits, temperature=0.7, top_k=40)
                        next_byte = next_token.item()
                        
                    print(stream_decoder.decode_and_print(next_byte), end='', flush=True)
                    generated_bytes.append(next_byte)
                    generation_context.append(next_byte)
                    
                    if next_byte == 10:
                        break
                print()

        except KeyboardInterrupt:
            print("\nSession paused.")
            break


# =========================================================================
# CORE EXECUTION INITIALIZER
# =========================================================================
def run_session():
    gptconf = GPTConfig()
    
    latest_coop_path = 'out/latest_coop_checkpoint.pt'
    baseline_path = 'out/baseline_ckpt.pt'
    
    agents = nn.ModuleList([HRMAgent(gptconf).to(device) for _ in range(4)])
    optimizers = [torch.optim.AdamW(a.parameters(), lr=1e-4) for a in agents]
    
    is_coop_active = False
    
    if os.path.exists(latest_coop_path):
        print(f"[*] Loading active-inference cooperative checkpoint: {latest_coop_path}")
        checkpoint = torch.load(latest_coop_path, map_location=device)
        agents.load_state_dict(checkpoint['agents_state_dict'])
        for idx, opt_a in enumerate(optimizers):
            opt_a.load_state_dict(checkpoint['optimizers_state_dicts'][idx])
        is_coop_active = True
    elif os.path.exists(baseline_path):
        print(f"[*] Cooperative checkpoint not found. Scaling up Phase 1 Baseline: {baseline_path}")
        base_state = torch.load(baseline_path, map_location=device)
        for a in agents:
            a.load_state_dict(base_state, strict=False)
            with torch.no_grad():
                a.token_proj.weight.copy_(a.wte.weight.t())
        is_coop_active = False
    else:
        print("[!] No trained checkpoints found. Please run baseline or active pretraining first.")
        sys.exit(1)

    mem_keys = [torch.zeros(1, 16, gptconf.n_embd, device=device) for _ in range(4)]
    mem_vals = [torch.zeros(1, 16, gptconf.n_embd, device=device) for _ in range(4)]
    mem_mask = [torch.zeros(1, 16, device=device) for _ in range(4)]
    
    if pygame_available:
        try:
            run_pygame_session(agents, optimizers, mem_keys, mem_vals, mem_mask, is_coop_active, active_idx=0)
        except pygame.error as e:
            print(f"\n[!] Pygame setup failed ({e}). Reverting to terminal framework...")
            run_terminal_session(agents, optimizers, mem_keys, mem_vals, mem_mask, is_coop_active, active_idx=0)
    else:
        run_terminal_session(agents, optimizers, mem_keys, mem_vals, mem_mask, is_coop_active, active_idx=0)

if __name__ == "__main__":
    run_session()
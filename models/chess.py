import os
import sys
import glob
import math
import pygame
import torch
import torch.nn.functional as F

# Import model architecture & environment from your training script
# (Assuming your training file is named 'agent.py' or similar; adjust import if needed)

from models.nodechess9 import (
    ChessNodeNet, 
    GuardrailedTorchChess, 
    HyperParamsConfig,
    noise_cap
)


# ==========================================
# 0. CONFIGURATION & CONSTANTS
# ==========================================
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 640
BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8
SIDEBAR_WIDTH = WINDOW_WIDTH - BOARD_SIZE

LIGHT_SQ_COLOR = (240, 217, 181)
DARK_SQ_COLOR  = (181, 136, 99)
HIGHLIGHT_SRC  = (205, 210, 106)
HIGHLIGHT_DST  = (170, 162, 58)
LEGAL_DOT_CLR  = (100, 110, 64, 160)
BG_SIDEBAR     = (32, 33, 36)
TEXT_COLOR     = (220, 220, 220)
ACCENT_COLOR   = (76, 175, 80)
WARN_COLOR     = (244, 67, 54)

# Unicode Chess Symbols: 0:empty, 1..6: White P,N,B,R,Q,K | 7..12: Black p,n,b,r,q,k
PIECE_GLYPHS = {
    0: "",
    1: "♙", 2: "♘", 3: "♗", 4: "♖", 5: "♕", 6: "♔",
    7: "♟", 8: "♞", 9: "♝", 10: "♜", 11: "♛", 12: "♚"
}

# ==========================================
# 1. CHECKPOINT LOADER
# ==========================================
def find_latest_checkpoint():
    ckpts = glob.glob("out/ckpt(*).pt")
    if not ckpts:
        return None
    # Sort numerically by batch index
    def extract_idx(p):
        try:
            return int(p.split("ckpt(")[1].split(")")[0])
        except Exception:
            return -1
    ckpts.sort(key=extract_idx)
    return ckpts[-1]

def load_agent(device, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()

    cfg = HyperParamsConfig(device)
    model = ChessNodeNet(num_nodes=128, dim=32, input_dim=100, device=device).to(device)
    model.apply_hyperparams(cfg)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[+] Loading model checkpoint from: {checkpoint_path}")
        data = torch.load(checkpoint_path, map_location=device)
        
        state_dict = data['model_state_dict']
        
        # --- Strip '_orig_mod.' prefix created by torch.compile ---
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            clean_key = k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k
            cleaned_state_dict[clean_key] = v

        model.load_state_dict(cleaned_state_dict)
        print(f"[+] Loaded successfully! (Batch index: {data.get('batch_idx', '?')})")
    else:
        print("[!] No checkpoint found in out/. Running with randomly initialized agent.")

    model.eval()
    return model

# ==========================================
# 2. PYGAME GUI ENGINE
# ==========================================
class ChessGUI:
    def __init__(self, human_plays_white=True, thinking_steps=32, ckpt_path=None):
        pygame.init()
        pygame.font.init()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Active Inference Chess vs Human")

        self.human_plays_white = human_plays_white
        self.thinking_steps = thinking_steps
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_piece = pygame.font.SysFont(["segoeuisymbol", "dejavusans", "applesymbols", "arial"], 52)
        self.font_ui = pygame.font.SysFont("helvetica", 16)
        self.font_ui_bold = pygame.font.SysFont("helvetica", 18, bold=True)
        self.font_title = pygame.font.SysFont("helvetica", 22, bold=True)

        # Agent & Environment
        self.model = load_agent(self.device, ckpt_path)
        self.env = GuardrailedTorchChess(batch_size=1, device=self.device, halfmove_limit=100)
        self.reset_game()

    def reset_game(self):
        self.env.reset()
        self.dones = torch.zeros(1, dtype=torch.bool, device=self.device)
        self.winners = torch.zeros(1, dtype=torch.int8, device=self.device)
        
        # Internal states of the recurrent thought-scape
        self.state, self.output, self.err_ema, self.lp_ema = self.model.init_states(batch_size=1)
        self.selected_sq = None
        self.valid_destinations = []
        self.last_move = None
        self.last_omega = 0.0
        self.last_threat = 0.5
        self.status_msg = "Game Started. Your turn!" if self.human_plays_white else "Agent thinking..."
        self.ply_count = 0

    def sq_to_screen(self, r, c):
        """Converts board (r, c) to display (screen_x, screen_y) based on player orientation."""
        if self.human_plays_white:
            sx = c * SQUARE_SIZE
            sy = (7 - r) * SQUARE_SIZE
        else:
            sx = (7 - c) * SQUARE_SIZE
            sy = r * SQUARE_SIZE
        return sx, sy

    def screen_to_sq(self, x, y):
        """Converts mouse click (x, y) to board (r, c)."""
        if x >= BOARD_SIZE:
            return None, None
        c_disp = x // SQUARE_SIZE
        r_disp = y // SQUARE_SIZE
        if self.human_plays_white:
            return 7 - r_disp, c_disp
        else:
            return r_disp, 7 - c_disp

    def get_legal_actions_dict(self):
        legal_mask = self.env.compute_legal_masks()[0] # (4096,)
        legal_indices = torch.nonzero(legal_mask > -1e8).squeeze(-1).tolist()
        
        moves_by_from = {}
        for act in legal_indices:
            f_sq = act // 64
            t_sq = act % 64
            moves_by_from.setdefault(f_sq, []).append(t_sq)
        return moves_by_from

    def step_agent(self):
        legal_mask = self.env.compute_legal_masks()
        legal_acts = torch.nonzero(legal_mask[0] > -1e8).squeeze(-1).tolist()

        if len(legal_acts) == 0:
            self.dones[0] = True
            self.winners[0] = 1 if self.env.players[0] == 1 else 2
            return

        env_input, threat = self.env.get_obs()
        self.last_threat = threat.item()

        # Deliberate over internal attractors
        with torch.no_grad():
            for _ in range(self.thinking_steps):
                _, self.state, self.output, self.err_ema, self.lp_ema, omega = self.model(
                    env_input, threat, self.state, self.output, self.err_ema, self.lp_ema
                )
            self.last_omega = omega.mean().item()

            raw_logits = self.model.get_action_logits(self.state).squeeze(0)
            masked_logits = raw_logits + legal_mask[0]
            act = torch.argmax(masked_logits).item()

        f_sq = act // 64
        t_sq = act % 64
        self.last_move = (f_sq, t_sq)

        self.dones, self.winners = self.env.step(
            torch.tensor([act], device=self.device), self.dones, self.winners
        )
        self.ply_count += 1

    def handle_human_click(self, r, c):
        if r is None or self.dones[0].item():
            return

        clicked_sq = r * 8 + c
        legal_moves = self.get_legal_actions_dict()

        if self.selected_sq is not None:
            if clicked_sq in self.valid_destinations:
                # Execute move
                act = self.selected_sq * 64 + clicked_sq
                self.last_move = (self.selected_sq, clicked_sq)
                self.dones, self.winners = self.env.step(
                    torch.tensor([act], device=self.device), self.dones, self.winners
                )
                self.ply_count += 1
                self.selected_sq = None
                self.valid_destinations = []
                return

            if clicked_sq in legal_moves:
                self.selected_sq = clicked_sq
                self.valid_destinations = legal_moves[clicked_sq]
                return

            self.selected_sq = None
            self.valid_destinations = []
        else:
            if clicked_sq in legal_moves:
                self.selected_sq = clicked_sq
                self.valid_destinations = legal_moves[clicked_sq]

    def draw_board(self):
        board_tensor = self.env.boards[0]

        for r in range(8):
            for c in range(8):
                sx, sy = self.sq_to_screen(r, c)
                sq_idx = r * 8 + c
                is_light = (r + c) % 2 != 0

                color = LIGHT_SQ_COLOR if is_light else DARK_SQ_COLOR

                # Highlight last move
                if self.last_move and sq_idx in self.last_move:
                    color = HIGHLIGHT_DST if sq_idx == self.last_move[1] else HIGHLIGHT_SRC
                # Highlight selected square
                if sq_idx == self.selected_sq:
                    color = HIGHLIGHT_SRC

                pygame.draw.rect(self.screen, color, (sx, sy, SQUARE_SIZE, SQUARE_SIZE))

                # Highlight legal destination dots
                if sq_idx in self.valid_destinations:
                    center = (sx + SQUARE_SIZE // 2, sy + SQUARE_SIZE // 2)
                    dot_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                    pygame.draw.circle(dot_surface, LEGAL_DOT_CLR, (SQUARE_SIZE // 2, SQUARE_SIZE // 2), 12)
                    self.screen.blit(dot_surface, (sx, sy))

                # Draw piece glyph
                p = int(board_tensor[r, c].item())
                if p != 0:
                    glyph = PIECE_GLYPHS[p]
                    # White pieces rendered with dark outline/tint
                    p_color = (255, 255, 255) if p <= 6 else (20, 20, 20)
                    text_surf = self.font_piece.render(glyph, True, p_color)
                    rect = text_surf.get_rect(center=(sx + SQUARE_SIZE // 2, sy + SQUARE_SIZE // 2))
                    self.screen.blit(text_surf, rect)

    def draw_sidebar(self):
        sidebar_rect = pygame.Rect(BOARD_SIZE, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, BG_SIDEBAR, sidebar_rect)

        x_margin = BOARD_SIZE + 20
        y = 25

        # Title
        title_surf = self.font_title.render("Active Inference Chess", True, (255, 255, 255))
        self.screen.blit(title_surf, (x_margin, y))
        y += 45

        # Turn indicator
        turn_white = (self.env.players[0].item() == 0)
        is_human_turn = (turn_white == self.human_plays_white)
        turn_text = "White to Move" if turn_white else "Black to Move"
        turn_clr = ACCENT_COLOR if is_human_turn else (100, 180, 255)
        self.screen.blit(self.font_ui_bold.render(f"Turn: {turn_text}", True, turn_clr), (x_margin, y))
        y += 28

        role_str = "You are: White" if self.human_plays_white else "You are: Black"
        self.screen.blit(self.font_ui.render(role_str, True, (180, 180, 180)), (x_margin, y))
        y += 35

        # Telemetry
        self.screen.blit(self.font_ui_bold.render("--- Agent Telemetry ---", True, (160, 160, 160)), (x_margin, y))
        y += 25

        self.screen.blit(self.font_ui.render(f"Ply Step: {self.ply_count}", True, TEXT_COLOR), (x_margin, y))
        y += 22

        halfmove = int(self.env.halfmove_clock[0].item())
        self.screen.blit(self.font_ui.render(f"50-Move Clock: {halfmove}/100", True, TEXT_COLOR), (x_margin, y))
        y += 22

        rep_count = int(self.env.history_len[0].item())
        self.screen.blit(self.font_ui.render(f"Rep Window Len: {rep_count}", True, TEXT_COLOR), (x_margin, y))
        y += 22

        # Threat Level Bar
        self.screen.blit(self.font_ui.render(f"Somatic Threat (H): {self.last_threat:.3f}", True, TEXT_COLOR), (x_margin, y))
        y += 20
        bar_w, bar_h = 200, 12
        pygame.draw.rect(self.screen, (60, 60, 60), (x_margin, y, bar_w, bar_h))
        pygame.draw.rect(self.screen, WARN_COLOR if self.last_threat > 0.6 else ACCENT_COLOR, (x_margin, y, int(bar_w * self.last_threat), bar_h))
        y += 25

        # Epistemic Drive (Omega)
        self.screen.blit(self.font_ui.render(f"Epistemic Drive (Ω): {self.last_omega:.4f}", True, TEXT_COLOR), (x_margin, y))
        y += 40

        # Outcome Status
        if self.dones[0].item():
            w = self.winners[0].item()
            if w == 1:
                res_txt = "WHITE WINS!"
                res_clr = (255, 215, 0)
            elif w == 2:
                res_txt = "BLACK WINS!"
                res_clr = (255, 215, 0)
            else:
                res_txt = "DRAW (Repetition / 50-Move)"
                res_clr = (200, 200, 200)
            
            res_surf = self.font_title.render(res_txt, True, res_clr)
            self.screen.blit(res_surf, (x_margin, y))
            y += 35

        # Help Instructions
        y = WINDOW_HEIGHT - 60
        self.screen.blit(self.font_ui.render("Press 'R' to Restart", True, (140, 140, 140)), (x_margin, y))
        y += 20
        self.screen.blit(self.font_ui.render("Press 'F' to Flip Perspective", True, (140, 140, 140)), (x_margin, y))

    def run(self):
        running = True
        while running:
            self.clock.tick(60)

            # Check if it's the bot's turn
            turn_white = (self.env.players[0].item() == 0)
            is_bot_turn = (turn_white != self.human_plays_white) and not self.dones[0].item()

            if is_bot_turn:
                # Render once so the user sees their move before the bot thinks
                self.draw_board()
                self.draw_sidebar()
                pygame.display.flip()

                self.step_agent()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_f:
                        self.human_plays_white = not self.human_plays_white
                        self.reset_game()

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if not is_bot_turn:
                        mx, my = pygame.mouse.get_pos()
                        r, c = self.screen_to_sq(mx, my)
                        self.handle_human_click(r, c)

            self.draw_board()
            self.draw_sidebar()
            pygame.display.flip()

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    # Optional CLI arguments: python play.py [black|white] [ckpt_path]
    human_white = True
    custom_ckpt = None

    if len(sys.argv) > 1 and sys.argv[1].lower() == "black":
        human_white = False
    if len(sys.argv) > 2:
        custom_ckpt = sys.argv[2]

    gui = ChessGUI(human_plays_white=human_white, thinking_steps=32, ckpt_path=custom_ckpt)
    gui.run()
import math
import numpy as np
import pyray as rl
from raylib import rl as raw_rl

# -------------------------------------------------------------
# 1. 12-Parameter XOR Model & Closed-Form Solution
# -------------------------------------------------------------
PARAM_RADIUS = 3.6
X_data = np.array(
    [[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]], dtype=np.float32
)
Y_data = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float32)

# Exact analytical zero-error solution vector (Unit Normalized)
raw_sol = np.array(
    [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 0.5, 0.5, -0.5, -0.5],
    dtype=np.float32,
)
EXACT_SOL = raw_sol / np.linalg.norm(raw_sol)


def get_sample_errors_numpy(theta):
    """theta: (..., 12) -> returns (..., 4)"""
    orig_shape = theta.shape[:-1]
    flat_theta = theta.reshape(-1, 12)
    B = flat_theta.shape[0]

    W1 = flat_theta[:, 0:8].reshape(B, 2, 4)
    W2 = flat_theta[:, 8:12].reshape(B, 4, 1)

    # (1, 4, 2) @ (B, 2, 4) -> (B, 4, 4)
    H = np.maximum(0.0, X_data[None, :, :] @ W1)
    preds = (H @ W2).squeeze(-1)  # (B, 4)

    errors = np.abs(preds - Y_data[None, :])
    return errors.reshape(*orig_shape, 4)


def make_givens_12d(i, j, angle):
    G = np.eye(12, dtype=np.float32)
    c, s = math.cos(angle), math.sin(angle)
    G[i, i] = c
    G[j, j] = c
    G[i, j] = -s
    G[j, i] = s
    return G


DIAL_PLANES = [
    (0, 3),
    (1, 4),
    (2, 5),
    (0, 6),
    (1, 7),
    (2, 8),
    (0, 9),
    (1, 10),
    (2, 11),
]

LEVEL_TITLES = [
    "The 4th Dimension (Tutorial)",
    "Dual Axis Warp",
    "Tri-Manifold",
    "Tetra-Rotation",
    "Pentatope Drift",
    "Hexagonal Slice",
    "Septenary Tension",
    "Octa-Hypersphere",
    "Enneachoron Core",
    "Deep Curvature",
    "Hyper-Coupling",
    "SO(12) Singularity (FINAL BOSS)",
]

# -------------------------------------------------------------
# 2. Vectorized NumPy Ray-Sphere Render
# -------------------------------------------------------------
RES_W, RES_H = 200, 150
FOV = 1.3
CAM_DIST = 2.4

y_grid, x_grid = np.meshgrid(
    np.linspace(-1.0, 1.0, RES_H, dtype=np.float32),
    np.linspace(
        -1.0 * (RES_W / RES_H), 1.0 * (RES_W / RES_H), RES_W, dtype=np.float32
    ),
    indexing="ij",
)

SAMPLE_COLORS = np.array(
    [
        [0.45, 0.05, 0.15],  # 1. Ruby
        [0.05, 0.45, 0.15],  # 2. Emerald
        [0.10, 0.15, 0.55],  # 3. Cobalt
        [0.40, 0.35, 0.15],  # 4. Amber
    ],
    dtype=np.float32,
)


def render_hypersphere_numpy(R_12D):
    rays_d = np.stack(
        [x_grid, -y_grid, np.full_like(x_grid, FOV)], axis=-1
    )
    rays_d /= np.linalg.norm(rays_d, axis=-1, keepdims=True)

    oz = -CAM_DIST
    b = 2.0 * (oz * rays_d[..., 2])
    c = oz**2 - 1.0
    disc = b**2 - 4.0 * c
    hit_mask = disc > 0.0

    t = (-b - np.sqrt(np.maximum(0.0, disc))) / 2.0
    hit_pos_3d = np.zeros((RES_H, RES_W, 3), dtype=np.float32)
    hit_pos_3d[..., 0] = t * rays_d[..., 0]
    hit_pos_3d[..., 1] = t * rays_d[..., 1]
    hit_pos_3d[..., 2] = oz + t * rays_d[..., 2]

    norm_3d = hit_pos_3d / (
        np.linalg.norm(hit_pos_3d, axis=-1, keepdims=True) + 1e-6
    )

    p_12d = np.zeros((RES_H, RES_W, 12), dtype=np.float32)
    p_12d[..., :3] = norm_3d

    theta_unit = p_12d @ R_12D.T
    theta = theta_unit * PARAM_RADIUS

    errors = get_sample_errors_numpy(theta)

    TOLERANCE = 0.65
    intensity = np.clip(1.0 - (errors / TOLERANCE) ** 2, 0.0, 1.0)
    ribbon_rgb = np.sum(
        intensity[..., None] * SAMPLE_COLORS[None, None, :, :], axis=-2
    )

    solved_mask = (errors < 0.32).astype(np.float32)
    overlap_count = np.sum(solved_mask, axis=-1)
    all_solved = (overlap_count == 4.0).astype(np.float32)[..., None]
    max_err = np.max(errors, axis=-1, keepdims=True)

    white_nexus = (
        np.clip(1.0 - max_err / 0.32, 0.0, 1.0) ** 2
        * all_solved
        * np.array([2.5, 2.5, 2.5], dtype=np.float32)
    )
    ribbon_rgb += white_nexus

    view_dir = -rays_d
    fresnel = (
        np.clip(
            1.0 - np.sum(norm_3d * view_dir, axis=-1, keepdims=True), 0.0, 1.0
        )
        ** 2.5
    )
    spec = (
        np.clip(
            norm_3d[..., 0] * 0.4
            + norm_3d[..., 1] * 0.7
            - norm_3d[..., 2] * 0.6,
            0.0,
            1.0,
        )
        ** 24
    )

    crystal_body = np.array([0.04, 0.05, 0.08], dtype=np.float32)
    surface_rgb = (
        crystal_body
        + ribbon_rgb * 1.3
        + fresnel * 0.35
        + spec[..., None] * 0.5
    )

    bg_r = 0.03 + 0.02 * np.abs(y_grid)
    bg_g = 0.04 + 0.02 * np.abs(y_grid)
    bg_b = 0.07 + 0.04 * np.abs(y_grid)
    bg_rgb = np.stack([bg_r, bg_g, bg_b], axis=-1)

    final_rgb = np.where(hit_mask[..., None], surface_rgb, bg_rgb)
    alpha = np.ones((*final_rgb.shape[:-1], 1), dtype=np.float32)
    final_rgba = np.concatenate([final_rgb, alpha], axis=-1)

    pixels = (np.clip(final_rgba, 0.0, 1.0) * 255).astype(np.uint8)
    max_visible_overlap = (
        np.max(overlap_count[hit_mask]) if np.any(hit_mask) else 0.0
    )
    return pixels, int(max_visible_overlap)


# -------------------------------------------------------------
# 3. Main Loop
# -------------------------------------------------------------
def main():
    Q, _ = np.linalg.qr(
        np.column_stack([EXACT_SOL, np.random.randn(12, 11).astype(np.float32)])
    )
    if np.dot(Q[:, 0], EXACT_SOL) < 0:
        Q[:, 0] *= -1.0

    SCREEN_W, SCREEN_H = 880, 620
    rl.init_window(
        SCREEN_W, SCREEN_H, "12D Alignment Campaign: Hypersphere S^11"
    )
    rl.set_target_fps(60)

    img = rl.gen_image_color(RES_W, RES_H, rl.BLACK)
    tex = rl.load_texture_from_image(img)
    rl.unload_image(img)

    level = 1
    selected_dial = 0
    dial_angles = [0.0] * 9

    def get_level_scramble(lvl):
        R = Q.copy()
        scramble_angles = [
            0.75,
            -0.85,
            0.65,
            -0.90,
            0.70,
            -0.80,
            0.75,
            -0.65,
            0.85,
        ]
        if lvl <= 9:
            for d in range(lvl):
                i, j = DIAL_PLANES[d]
                R = R @ make_givens_12d(i, j, scramble_angles[d])
        elif lvl == 10:
            for d in range(9):
                i, j = DIAL_PLANES[d]
                R = R @ make_givens_12d(i, j, scramble_angles[d] * 1.4)
        elif lvl == 11:
            for d in range(9):
                i, j = DIAL_PLANES[d]
                R = R @ make_givens_12d(i, j, scramble_angles[d])
            R = (
                R
                @ make_givens_12d(3, 4, 0.6)
                @ make_givens_12d(5, 6, -0.7)
                @ make_givens_12d(7, 8, 0.5)
            )
        else:
            np.random.seed(42)
            rand_m = np.random.randn(12, 12).astype(np.float32)
            Q_rand, _ = np.linalg.qr(rand_m)
            R = Q @ Q_rand
        return R

    R_12D = get_level_scramble(level)

    while not rl.window_should_close():
        dt = rl.get_frame_time()
        rot_speed = 1.3 * dt

        # Orbit
        if rl.is_mouse_button_down(
            rl.MOUSE_BUTTON_LEFT
        ) or rl.is_mouse_button_down(rl.MOUSE_BUTTON_RIGHT):
            delta = rl.get_mouse_delta()
            R_12D = R_12D @ make_givens_12d(0, 2, delta.x * 0.006)
            R_12D = R_12D @ make_givens_12d(1, 2, -delta.y * 0.006)

        # Select Dial
        num_active_dials = min(level, 9)
        if rl.is_key_pressed(rl.KEY_UP) or rl.is_key_pressed(rl.KEY_W):
            selected_dial = (selected_dial - 1) % num_active_dials
        if rl.is_key_pressed(rl.KEY_DOWN) or rl.is_key_pressed(rl.KEY_S):
            selected_dial = (selected_dial + 1) % num_active_dials

        # Rotate Dial
        active_plane = DIAL_PLANES[selected_dial]
        if rl.is_key_down(rl.KEY_RIGHT) or rl.is_key_down(rl.KEY_D):
            R_12D = R_12D @ make_givens_12d(
                active_plane[0], active_plane[1], rot_speed
            )
            dial_angles[selected_dial] += rot_speed
        if rl.is_key_down(rl.KEY_LEFT) or rl.is_key_down(rl.KEY_A):
            R_12D = R_12D @ make_givens_12d(
                active_plane[0], active_plane[1], -rot_speed
            )
            dial_angles[selected_dial] -= rot_speed

        # Crosshair check: sphere surface normal directly under camera center is (0, 0, -1)
        center_norm_3d = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        center_p12d = np.zeros(12, dtype=np.float32)
        center_p12d[:3] = center_norm_3d
        center_theta = (center_p12d @ R_12D.T) * PARAM_RADIUS
        center_errs = get_sample_errors_numpy(center_theta[None, :])[0]

        samples_matched = np.sum(center_errs < 0.32)
        puzzle_solved = samples_matched == 4

        # Enter to advance to next level when puzzle is solved at the crosshair
        if puzzle_solved and (
            rl.is_key_pressed(rl.KEY_ENTER) or rl.is_key_pressed(rl.KEY_KP_ENTER)
        ):
            if level < 12:
                level += 1
                selected_dial = 0
                dial_angles = [0.0] * 9
                R_12D = get_level_scramble(level)

        # Manual level override / Reset controls
        if rl.is_key_pressed(rl.KEY_RIGHT_BRACKET) and level < 12:
            level += 1
            selected_dial = 0
            dial_angles = [0.0] * 9
            R_12D = get_level_scramble(level)
        if rl.is_key_pressed(rl.KEY_LEFT_BRACKET) and level > 1:
            level -= 1
            selected_dial = 0
            dial_angles = [0.0] * 9
            R_12D = get_level_scramble(level)
        if rl.is_key_pressed(rl.KEY_R):
            dial_angles = [0.0] * 9
            R_12D = get_level_scramble(level)

        for k in range(rl.KEY_ONE, rl.KEY_NINE + 1):
            if rl.is_key_pressed(k):
                level = k - rl.KEY_ONE + 1
                selected_dial = 0
                dial_angles = [0.0] * 9
                R_12D = get_level_scramble(level)

        # Render
        frame, max_overlap = render_hypersphere_numpy(R_12D)
        raw_rl.UpdateTexture(tex, frame.tobytes())

        # Draw
        rl.begin_drawing()
        rl.clear_background(rl.BLACK)

        rl.draw_texture_pro(
            tex,
            rl.Rectangle(0, 0, RES_W, RES_H),
            rl.Rectangle(10, 10, 580, 560),
            rl.Vector2(0, 0),
            0.0,
            rl.WHITE,
        )

        vp_cx, vp_cy = 10 + 580 // 2, 10 + 560 // 2
        ch_color = rl.GOLD if puzzle_solved else rl.WHITE
        rl.draw_line(vp_cx - 14, vp_cy, vp_cx + 14, vp_cy, ch_color)
        rl.draw_line(vp_cx, vp_cy - 14, vp_cx, vp_cy + 14, ch_color)
        rl.draw_circle_lines(vp_cx, vp_cy, 18, ch_color)

        rl.draw_text(
            f"LEVEL {level}/12: {LEVEL_TITLES[level-1]}",
            25,
            25,
            20,
            rl.RAYWHITE,
        )
        rl.draw_text("RESONANCE:", 25, 55, 16, rl.LIGHTGRAY)
        gauge_color = [rl.GRAY, rl.ORANGE, rl.YELLOW, rl.GOLD, rl.WHITE][
            max_overlap
        ]
        rl.draw_text(
            f"{max_overlap} / 4 SAMPLES ALIGNED", 130, 54, 18, gauge_color
        )

        colors = [rl.RED, rl.GREEN, rl.SKYBLUE, rl.YELLOW]
        labels = ["[1, 1]->+1", "[-1, 1]->-1", "[1, -1]->-1", "[-1, -1]->+1"]
        for idx in range(4):
            err = center_errs[idx]
            is_ok = err < 0.32
            rl.draw_rectangle(25, 470 + idx * 24, 12, 12, colors[idx])
            stat = "ALIGNED" if is_ok else f"Err: {err:.2f}"
            rl.draw_text(
                f"S{idx+1} {labels[idx]}: {stat}",
                45,
                469 + idx * 24,
                15,
                rl.WHITE if is_ok else rl.GRAY,
            )

        # Right Panel
        panel_x = 605
        rl.draw_rectangle(
            panel_x, 10, 265, 560, rl.Color(20, 24, 32, 255)
        )
        rl.draw_rectangle_lines(
            panel_x, 10, 265, 560, rl.DARKGRAY
        )
        rl.draw_text("12D DIAL CONSOLE", panel_x + 35, 25, 18, rl.GOLD)
        rl.draw_text(
            "[W/S] Select  |  [A/D] Turn",
            panel_x + 32,
            50,
            14,
            rl.LIGHTGRAY,
        )
        rl.draw_line(
            panel_x + 10, 72, panel_x + 255, 72, rl.GRAY
        )

        dial_names = [
            "4D (W_3)",
            "5D (W_4)",
            "6D (W_5)",
            "7D (W_6)",
            "8D (W_7)",
            "9D (W_8)",
            "10D (W_9)",
            "11D (W_10)",
            "12D (W_11)",
        ]

        for d in range(9):
            y_pos = 85 + d * 48
            is_active = d == selected_dial
            is_unlocked = d < num_active_dials

            if is_active:
                rl.draw_rectangle(
                    panel_x + 8,
                    y_pos - 4,
                    249,
                    42,
                    rl.Color(40, 50, 75, 255),
                )
                rl.draw_rectangle_lines(
                    panel_x + 8, y_pos - 4, 249, 42, rl.GOLD
                )

            prefix = "> " if is_active else "  "
            name_col = (
                rl.WHITE
                if is_active
                else (rl.LIGHTGRAY if is_unlocked else rl.DARKGRAY)
            )
            rl.draw_text(
                f"{prefix}Dial {d+1}: {dial_names[d]}",
                panel_x + 14,
                y_pos,
                15,
                name_col,
            )

            if is_unlocked:
                ang = dial_angles[d] % (2 * math.pi)
                rl.draw_text(
                    f"{ang:.2f} rad",
                    panel_x + 185,
                    y_pos,
                    14,
                    rl.YELLOW if is_active else rl.GRAY,
                )
                rl.draw_rectangle(
                    panel_x + 30, y_pos + 20, 200, 6, rl.Color(30, 30, 40, 255)
                )
                bar_fill = int((ang / (2 * math.pi)) * 200)
                rl.draw_rectangle(
                    panel_x + 30,
                    y_pos + 20,
                    bar_fill,
                    6,
                    rl.GOLD if is_active else rl.SKYBLUE,
                )
            else:
                rl.draw_text(
                    "[LOCKED]", panel_x + 185, y_pos, 14, rl.DARKGRAY
                )

        rl.draw_text(
            "[Mouse Drag] 3D Orbit  |  [ENTER] Next Level (when aligned)  |  [R] Reset",
            20,
            SCREEN_H - 30,
            15,
            rl.LIGHTGRAY,
        )

        if puzzle_solved:
            box_w, box_h = 440, 80
            rl.draw_rectangle(
                vp_cx - box_w // 2, vp_cy - box_h // 2, box_w, box_h, rl.Color(0, 0, 0, 225)
            )
            rl.draw_rectangle_lines(
                vp_cx - box_w // 2, vp_cy - box_h // 2, box_w, box_h, rl.GOLD
            )
            
            title_txt = "NEXUS TARGET ACQUIRED!"
            rl.draw_text(
                title_txt,
                vp_cx - rl.measure_text(title_txt, 22) // 2,
                vp_cy - 26,
                22,
                rl.GOLD,
            )
            
            next_txt = (
                f"PRESS [ENTER] FOR LEVEL {level+1}"
                if level < 12
                else "CONGRATULATIONS! ALL 12 DIMENSIONS CONQUERED!"
            )
            rl.draw_text(
                next_txt,
                vp_cx - rl.measure_text(next_txt, 16) // 2,
                vp_cy + 6,
                16,
                rl.WHITE,
            )

        rl.end_drawing()

    rl.close_window()


if __name__ == "__main__":
    main()
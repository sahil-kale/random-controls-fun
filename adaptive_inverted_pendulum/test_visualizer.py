import math
import pygame
import numpy as np

# >>> change this import if your class file is named differently
from multi_link_pendulum import MultiLinkPendulum

# -----------------------
# Tunables
# -----------------------
SCREEN_W, SCREEN_H = 1000, 600
M_PER_PX = 1.0 / 80.0      # meters per pixel (world scale)
DT_SIM = 1.0 / 240.0       # integration step (s) — small for stability
FPS = 60                   # visual refresh rate
FORCE_GAIN = 40.0          # N applied when holding A/D
CART_WIDTH_M = 0.6
CART_HEIGHT_M = 0.25
RAIL_Y_M = 0.0             # cart rides along y = 0 in "world" meters
GROUND_LINE_OFFSET_PX = 40 # visual offset below rail for a ground line

# -----------------------
# Helpers
# -----------------------
def world_to_screen(x_m, y_m):
    """World coords: x right (+), y up (+). Screen y goes down (+)."""
    x_px = SCREEN_W // 2 + int(x_m / M_PER_PX)
    y_px = SCREEN_H // 2 - int(y_m / M_PER_PX)
    return x_px, y_px

def link_com_positions(x_cart, thetas, link_lengths):
    """
    Replicates the kinematics used in the dynamics code:
    pos_link_x = previous_x + l_i * sin(theta_i)
    pos_link_y = previous_y - l_i * cos(theta_i)
    Returns a list of (x, y) COM positions in meters.
    """
    pos = []
    prev_x, prev_y = x_cart, RAIL_Y_M
    for i, th in enumerate(thetas):
        x = prev_x + link_lengths[i] * math.sin(th)
        y = prev_y - link_lengths[i] * math.cos(th)
        pos.append((x, y))
        prev_x, prev_y = x, y
    return pos

def reset_state(pendulum):
    """Small perturbation from upright (theta=pi) or hanging (theta=0). Choose what you want."""
    # Hanging-down initial condition (theta ~ 0)
    th0 = np.array([0.2 for _ in range(pendulum.num_links)], dtype=float)
    dth0 = np.zeros_like(th0)
    x0, dx0 = 0.0, 0.0
    state = np.zeros(2 + 2 * pendulum.num_links, dtype=float)
    state[0] = x0
    state[1] = dx0
    state[2:2+pendulum.num_links] = th0
    state[2+pendulum.num_links:] = dth0
    pendulum.state = state

# -----------------------
# Main
# -----------------------
def main():
    pygame.init()
    pygame.display.set_caption("Multi-Link Pendulum on a Cart")
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 18)

    # --- Build the model (edit these to taste) ---
    num_links = 2
    link_lengths = [1.7] * num_links  # meters
    link_masses = [1.0] * num_links   # kg
    cart_mass = 5.0                   # kg
    pend = MultiLinkPendulum(num_links, link_lengths, link_masses, cart_mass)
    reset_state(pend)

    running = True
    force_cmd = 0.0

    while running:
        # -------- events / input --------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False
        if keys[pygame.K_r]:
            reset_state(pend)

        # A / D apply force left/right
        force_cmd = 0.0
        if keys[pygame.K_a]:
            force_cmd -= FORCE_GAIN
        if keys[pygame.K_d]:
            force_cmd += FORCE_GAIN

        # -------- simulate (substeps) --------
        # integrate at DT_SIM; run as many substeps as needed for this frame's elapsed time
        # here we simply run a fixed number of substeps per visual frame for stability
        substeps = max(1, int((1.0 / FPS) / DT_SIM))
        for _ in range(substeps):
            pend.step_in_time(force_cmd, DT_SIM)

        # -------- draw --------
        screen.fill((245, 245, 245))

        # world info
        x = float(pend.state[0])
        thetas = pend.state[2:2+pend.num_links]
        coms = link_com_positions(x, thetas, pend.link_lengths)

        # draw ground rail line
        rail_px = world_to_screen(0.0, RAIL_Y_M)[1]
        pygame.draw.line(screen, (20, 20, 20), (0, rail_px), (SCREEN_W, rail_px), 2)
        pygame.draw.line(screen, (180, 180, 180),
                         (0, rail_px + GROUND_LINE_OFFSET_PX),
                         (SCREEN_W, rail_px + GROUND_LINE_OFFSET_PX), 1)

        # draw cart
        cart_w_px = int(CART_WIDTH_M / M_PER_PX)
        cart_h_px = int(CART_HEIGHT_M / M_PER_PX)
        cart_x_px, cart_y_px = world_to_screen(x, RAIL_Y_M)
        cart_rect = pygame.Rect(0, 0, cart_w_px, cart_h_px)
        cart_rect.center = (cart_x_px, cart_y_px - cart_h_px // 2)
        pygame.draw.rect(screen, (40, 90, 200), cart_rect, border_radius=8)
        # wheels (purely cosmetic)
        w_r = max(6, cart_h_px // 4)
        pygame.draw.circle(screen, (50, 50, 50), (cart_rect.left + w_r, rail_px + w_r//2), w_r)
        pygame.draw.circle(screen, (50, 50, 50), (cart_rect.right - w_r, rail_px + w_r//2), w_r)

        # draw links (lines from previous joint to COM point used by model)
        prev_joint = world_to_screen(x, RAIL_Y_M)
        for i, (cx, cy) in enumerate(coms):
            com_pt = world_to_screen(cx, cy)
            pygame.draw.line(screen, (10, 10, 10), prev_joint, com_pt, 3)
            pygame.draw.circle(screen, (200, 60, 60), com_pt, 6)  # COM marker
            prev_joint = com_pt

        # HUD text
        hud_lines = [
            "Controls: A/D = push left/right, R = reset, ESC = quit",
            f"Force = {force_cmd:+.1f} N   |   DT(sim) = {DT_SIM*1000:.1f} ms   substeps/frame = {substeps}",
            f"x = {pend.state[0]:+.3f} m   xdot = {pend.state[1]:+.3f} m/s",
        ]
        for k, th in enumerate(thetas, 1):
            hud_lines.append(f"theta_{k} = {th:+.3f} rad")

        y0 = 8
        for line in hud_lines:
            surf = font.render(line, True, (30, 30, 30))
            screen.blit(surf, (10, y0))
            y0 += 18

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()

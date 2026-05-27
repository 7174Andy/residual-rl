"""Pygame renderer for UnicycleGoalEnv. Handles both 'human' and 'rgb_array' modes."""

from __future__ import annotations

import os

import numpy as np


class PygameRenderer:
    def __init__(self, env, render_mode: str, window_size: int = 600):
        if render_mode == "rgb_array":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        import pygame

        self.pygame = pygame
        self.env = env
        self.render_mode = render_mode
        self.window_size = window_size

        pygame.init()
        if render_mode == "human":
            pygame.display.init()
            self.surface = pygame.display.set_mode((window_size, window_size))
            pygame.display.set_caption("TwoWheelGoal")
            self.clock = pygame.time.Clock()
        elif render_mode == "rgb_array":
            self.surface = pygame.Surface((window_size, window_size))
            self.clock = None
        else:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        # Bundled font avoids fc-list enumeration (slow / warns on macOS).
        self.font = pygame.font.Font(None, 22)
        self._trail: list[tuple[float, float]] = []
        self._last_step = -1

    def _world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        (xmin, xmax), (ymin, ymax) = self.env.workspace_bounds
        sx = (x - xmin) / (xmax - xmin) * self.window_size
        sy = (ymax - y) / (ymax - ymin) * self.window_size  # flip y for screen coords
        return int(sx), int(sy)

    def render(self):
        pygame = self.pygame

        if self.render_mode == "human":
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.close()
                    raise SystemExit

        # Trail bookkeeping
        if self.env.step_idx == 0:
            self._trail = []
        if self.env.step_idx != self._last_step:
            self._trail.append((float(self.env.state[0]), float(self.env.state[1])))
            self._last_step = self.env.step_idx

        self.surface.fill((250, 250, 250))

        # Workspace border
        pygame.draw.rect(
            self.surface,
            (200, 200, 200),
            (0, 0, self.window_size, self.window_size),
            width=2,
        )

        # Trail
        if len(self._trail) >= 2:
            pts = [self._world_to_screen(x, y) for x, y in self._trail]
            pygame.draw.lines(self.surface, (160, 160, 220), False, pts, 2)

        # Goal: outer tolerance ring + filled center
        gx, gy = self.env.goal
        gscr = self._world_to_screen(gx, gy)
        x_extent = self.env.workspace_bounds[0, 1] - self.env.workspace_bounds[0, 0]
        tol_px = max(4, int(self.env.goal_tolerance / x_extent * self.window_size))
        pygame.draw.circle(self.surface, (100, 200, 100), gscr, tol_px, width=2)
        pygame.draw.circle(self.surface, (50, 150, 50), gscr, 5)

        # Robot triangle (tip = heading direction)
        x, y, delta = self.env.state
        size = 0.6  # world units
        tip = (x + size * np.cos(delta), y + size * np.sin(delta))
        left = (
            x + 0.45 * size * np.cos(delta + 2.4),
            y + 0.45 * size * np.sin(delta + 2.4),
        )
        right = (
            x + 0.45 * size * np.cos(delta - 2.4),
            y + 0.45 * size * np.sin(delta - 2.4),
        )
        tri = [
            self._world_to_screen(*tip),
            self._world_to_screen(*left),
            self._world_to_screen(*right),
        ]
        pygame.draw.polygon(self.surface, (60, 60, 200), tri)
        pygame.draw.circle(self.surface, (30, 30, 100), self._world_to_screen(x, y), 3)

        # HUD
        dist = float(np.linalg.norm(self.env.state[:2] - self.env.goal))
        v, w = self.env.last_action
        hud = (
            f"step {self.env.step_idx}/{self.env.max_steps}  "
            f"dist {dist:.2f}  v {v:+.2f}  w {w:+.2f}"
        )
        text_surf = self.font.render(hud, True, (40, 40, 40))
        self.surface.blit(text_surf, (8, 8))

        if self.render_mode == "human":
            pygame.display.flip()
            assert self.clock is not None  # set in __init__ for human mode
            self.clock.tick(self.env.metadata["render_fps"])
            return None
        else:
            arr = pygame.surfarray.array3d(self.surface)
            return np.transpose(arr, (1, 0, 2))  # pygame uses (W, H, 3); return (H, W, 3)

    def close(self):
        pygame = getattr(self, "pygame", None)
        if pygame is None:
            return
        if self.render_mode == "human":
            pygame.display.quit()
        pygame.quit()

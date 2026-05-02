import sys
import os
import pygame
from pygame.locals import *
from typing import Optional, Tuple

from minisim import TrafficSim, TileType, import_map

maps_settings = {
    "size": {"small": (10, 8), "medium": (20, 16), "large": (30, 24)},
    "budget": {"small": [300, 100], "medium": [500, 300], "large": [1200, 800]},
    "landscape": {
        "small": {
            "water_params": {
                "num_rivers": 1,
                "river_width": 1,
                "num_lakes": 1,
                "lake_size": (2, 5),
            },
            "mountain_params": {"range_size": (5, 10)},
            "building_params": {
                "num_blocks": 1,
                "min_width": 2,
                "max_width": 3,
                "min_height": 1,
                "max_height": 3,
            },
        },
        "medium": {
            "water_params": {
                "num_rivers": 1,
                "river_width": 2,
                "num_lakes": 1,
                "lake_size": (2, 5),
            },
            "mountain_params": {"range_size": (5, 12)},
            "building_params": {
                "num_blocks": 1,
                "min_width": 2,
                "max_width": 4,
                "min_height": 1,
                "max_height": 4,
            },
        },
        "large": {
            "water_params": {
                "num_rivers": 2,
                "river_width": 1,
                "num_lakes": 2,
                "lake_size": (2, 5),
            },
            "mountain_params": {"range_size": (5, 10)},
            "building_params": {
                "num_blocks": 2,
                "min_width": 2,
                "max_width": 3,
                "min_height": 1,
                "max_height": 3,
            },
        },
    },
    "seed": {"small": [42, 42], "medium": [42, 42], "large": [42, 42]},
}

#  Config 
picked = "large"
budget_range = "sufficient"  # 0 'sufficient', 1 'tight
DEFAULT_SIZE = maps_settings["size"][picked]
W, H = DEFAULT_SIZE
SEED = maps_settings["seed"][picked][0 if budget_range == "sufficient" else 1]
INITIAL_BUDGET = maps_settings["budget"][picked][
    0 if budget_range == "sufficient" else 1
]
WITH_WATER = True
WITH_MOUNTAIN = True
WITH_BUILDINGS = True
MAP_LANDSCAPE = maps_settings["landscape"][picked]

SCALE = 28
FPS = 60
FONT_NAME = None
BG_COLOR = (30, 30, 30)
GRID_COLOR = (40, 40, 40)
PANEL_H = 90
ASSET_DIR = "gui_assets"
EXPORT_PATH = f"./maps/ori_map_{picked}_{budget_range}.pkl"
LOAD_PATH = f"./maps/optimized_map_{picked}_{budget_range}.pkl"
ROAD_CAPACITY_OPTIONS = [1, 2, 3, 4, 5]

COLORS = {
    "grass": (180, 220, 180),
    "water": (80, 120, 200),
    "road": (60, 60, 60),
    "mountain": (100, 80, 60),
    "building": (120, 120, 120),
    "car": (230, 90, 60),
    "out": (240, 200, 60),
    "in": (70, 200, 90),
    "grey": (160, 160, 160),
    "white": (240, 240, 240),
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class App:
    def __init__(self):
        pygame.init()
        self.sim = TrafficSim(
            W,
            H,
            seed=SEED,
            initial_budget=INITIAL_BUDGET,
            with_water=WITH_WATER,
            with_mountain=WITH_MOUNTAIN,
            with_buildings=WITH_BUILDINGS,
            water_params=MAP_LANDSCAPE["water_params"],
            mountain_params=MAP_LANDSCAPE["mountain_params"],
            building_params=MAP_LANDSCAPE["building_params"],
        )
        self.screen = pygame.display.set_mode((W * SCALE, H * SCALE + PANEL_H))
        pygame.display.set_caption("TrafficSim v2")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(FONT_NAME, 18)
        self.small = pygame.font.Font(FONT_NAME, 14)

        self.running_sim = False
        self.drag_left = False
        self.drag_right = False
        self.mode = "road"
        self.default_depot_amt = 10

        self.road_capacity_idx = 0
        self.current_road_capacity = ROAD_CAPACITY_OPTIONS[self.road_capacity_idx]

        self.tile_surfaces = {}
        self.car_surface = None
        self._load_tiles()

    #  Asset loading 
    def _load_tiles(self, asset_dir: str = ASSET_DIR):
        for name in ("grass", "water", "road", "mountain", "building", "out", "in"):
            path = os.path.join(asset_dir, f"{name}.png")
            try:
                img = pygame.image.load(path).convert_alpha()
                self.tile_surfaces[name] = pygame.transform.scale(img, (SCALE, SCALE))
            except Exception:
                self.tile_surfaces[name] = None

        try:
            car_img = pygame.image.load(
                os.path.join(asset_dir, "car.png")
            ).convert_alpha()
            self.car_surface = pygame.transform.scale(car_img, (SCALE, SCALE))
        except Exception:
            self.car_surface = None

    def _grid_pos(self, mx: int, my: int) -> Optional[Tuple[int, int]]:
        if my >= self.sim.h * SCALE:
            return None
        gx, gy = mx // SCALE, my // SCALE
        if 0 <= gx < self.sim.w and 0 <= gy < self.sim.h:
            return (gx, gy)
        return None

    def _handle_click(self, pos, button):
        if pos is None:
            return
        if button == 1:
            if self.mode == "road":
                self.sim.add_road(pos, self.current_road_capacity)
            elif self.mode == "out":
                self.sim.add_depot(pos, "out", self.default_depot_amt)
            elif self.mode == "in":
                self.sim.add_depot(pos, "in", self.default_depot_amt)
        elif button == 3:
            if self.sim.grid[pos[0]][pos[1]] == TileType.ROAD:
                self.sim.remove_road(pos)

    def _export(self, path: str = EXPORT_PATH):
        self.sim.export_map(path)

    def _load(self, path: str = LOAD_PATH):
        if not os.path.exists(path):
            return
        self.sim = import_map(path)
        self.sim.export_path = path
        if self.sim.w != W or self.sim.h != H:
            self.screen = pygame.display.set_mode(
                (self.sim.w * SCALE, self.sim.h * SCALE + PANEL_H)
            )

    def _reset_sim(self):
        """Reset simulation. Requires a loaded optimizer schedule."""
        if self.sim.saved_departures is not None:
            self.sim.reset_sim(self.sim.saved_departures)

    def _draw_world(self):
        car_positions = {c.pos for c in self.sim.cars.values()}
        for y in range(self.sim.h):
            for x in range(self.sim.w):
                p = (x, y)
                t = self.sim.grid[x][y]

                if t == TileType.DEPOT_OUT:
                    key = "out"
                elif t == TileType.DEPOT_IN:
                    key = "in"
                else:
                    key = {
                        TileType.GRASS: "grass",
                        TileType.WATER: "water",
                        TileType.ROAD: "road",
                        TileType.MOUNTAIN: "mountain",
                        TileType.BUILDING: "building",
                    }.get(t, "grey")

                surf = self.tile_surfaces.get(key)
                if surf:
                    self.screen.blit(surf, (x * SCALE, y * SCALE))
                else:
                    color = COLORS.get(key, COLORS["grey"])
                    self.screen.fill(
                        color, pygame.Rect(x * SCALE, y * SCALE, SCALE, SCALE)
                    )

                if t == TileType.ROAD and p in car_positions:
                    if self.car_surface:
                        self.screen.blit(self.car_surface, (x * SCALE, y * SCALE))
                    else:
                        pad = max(1, SCALE // 5)
                        pygame.draw.ellipse(
                            self.screen,
                            COLORS["car"],
                            (
                                x * SCALE + pad,
                                y * SCALE + pad,
                                SCALE - 2 * pad,
                                SCALE - 2 * pad,
                            ),
                        )

                # Draw capacity number on road tiles
                if t == TileType.ROAD:
                    cap = self.sim.road_capacity.get(p, 1)
                    cap_text = self.small.render(str(cap), True, COLORS["white"])
                    text_x = x * SCALE + SCALE - cap_text.get_width() - 2
                    text_y = y * SCALE + 2
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        outline_text = self.small.render(str(cap), True, (0, 0, 0))
                        self.screen.blit(outline_text, (text_x + dx, text_y + dy))
                    self.screen.blit(cap_text, (text_x, text_y))

        for x in range(self.sim.w + 1):
            pygame.draw.line(
                self.screen, GRID_COLOR, (x * SCALE, 0), (x * SCALE, self.sim.h * SCALE)
            )
        for y in range(self.sim.h + 1):
            pygame.draw.line(
                self.screen, GRID_COLOR, (0, y * SCALE), (self.sim.w * SCALE, y * SCALE)
            )

    def _draw_panel(self, hover: Optional[Tuple[int, int]]):
        panel_rect = pygame.Rect(0, self.sim.h * SCALE, self.sim.w * SCALE, PANEL_H)
        self.screen.fill(BG_COLOR, panel_rect)

        state = "RUN" if self.running_sim else "PAUSE"
        total_cars = len(self.sim.cars)
        cars_on_road = self.sim.cars_on_road()
        has_schedule = "YES" if self.sim.saved_departures else "NO"
        line1 = (
            f"[{state}] tick={self.sim.tick_count} cars={total_cars} (road={cars_on_road}) "
            f"supply={self.sim.total_supply()} demand={self.sim.total_demand()} "
            f"schedule={has_schedule}"
        )
        line2 = f"edit_mode={self.mode} cap={self.current_road_capacity} depot_amt={self.default_depot_amt}"
        line3 = (
            "LMB=add RMB=remove | R=road 1=OUT 2=IN | SPACE=run/pause S/F=step | "
            "E=export L=load | C=reset"
        )

        self.screen.blit(
            self.font.render(line1, True, COLORS["white"]), (10, self.sim.h * SCALE + 8)
        )
        self.screen.blit(
            self.small.render(line2, True, COLORS["white"]),
            (10, self.sim.h * SCALE + 32),
        )
        self.screen.blit(
            self.small.render(line3, True, COLORS["white"]),
            (10, self.sim.h * SCALE + 50),
        )

        if hover:
            x, y = hover
            t = self.sim.grid[x][y]
            info = f"({x},{y}) {t.name}"
            if t == TileType.ROAD:
                cap = self.sim.get_road_capacity((x, y))
                occ = self.sim.get_road_occupancy((x, y))
                info += f"  cap={cap} occ={occ}/{cap}"
            elif t in (TileType.DEPOT_OUT, TileType.DEPOT_IN):
                did = self.sim.depot_at.get((x, y))
                if did is not None:
                    d = self.sim.depots[did]
                    info += f"  depot#{did} {d.kind} amount={d.amount}"
            self.screen.blit(
                self.small.render(info, True, COLORS["white"]),
                (10, self.sim.h * SCALE + 72),
            )

    def run(self):
        while True:
            self.clock.tick(FPS)
            hover = None

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit(0)

                elif event.type == KEYDOWN:
                    if event.key == K_SPACE:
                        self.running_sim = not self.running_sim
                    elif event.key == K_s:
                        self.sim.step()
                    elif event.key == K_f:
                        for _ in range(10):
                            self.sim.step()
                    elif event.key == K_c:
                        self._reset_sim()
                        self.running_sim = False
                    elif event.key == K_r:
                        self.mode = "road"
                    elif event.key == K_1:
                        self.mode = "out"
                    elif event.key == K_2:
                        self.mode = "in"
                    elif event.key == K_LEFTBRACKET:
                        self.default_depot_amt = clamp(
                            self.default_depot_amt - 1, 0, 9999
                        )
                    elif event.key == K_RIGHTBRACKET:
                        self.default_depot_amt = clamp(
                            self.default_depot_amt + 1, 0, 9999
                        )
                    elif event.key == K_UP:
                        self.road_capacity_idx = (self.road_capacity_idx + 1) % len(
                            ROAD_CAPACITY_OPTIONS
                        )
                        self.current_road_capacity = ROAD_CAPACITY_OPTIONS[
                            self.road_capacity_idx
                        ]
                    elif event.key == K_DOWN:
                        self.road_capacity_idx = (self.road_capacity_idx - 1) % len(
                            ROAD_CAPACITY_OPTIONS
                        )
                        self.current_road_capacity = ROAD_CAPACITY_OPTIONS[
                            self.road_capacity_idx
                        ]
                    elif event.key == K_e:
                        self._export()
                    elif event.key == K_l:
                        self._load()

                elif event.type == MOUSEBUTTONDOWN:
                    pos = self._grid_pos(*event.pos)
                    if event.button == 1:
                        self.drag_left = True
                        self._handle_click(pos, 1)
                    elif event.button == 3:
                        self.drag_right = True
                        self._handle_click(pos, 3)

                elif event.type == MOUSEBUTTONUP:
                    if event.button == 1:
                        self.drag_left = False
                    elif event.button == 3:
                        self.drag_right = False

                elif event.type == MOUSEMOTION:
                    pos = self._grid_pos(*event.pos)
                    hover = pos
                    if self.drag_left:
                        self._handle_click(pos, 1)
                    if self.drag_right:
                        self._handle_click(pos, 3)

            if self.running_sim:
                self.sim.step()

            self.screen.fill(BG_COLOR)
            self._draw_world()
            if hover is None:
                hover = self._grid_pos(*pygame.mouse.get_pos())
            self._draw_panel(hover)
            pygame.display.flip()
if __name__ == "__main__":
    App().run()

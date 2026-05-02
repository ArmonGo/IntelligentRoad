from __future__ import annotations

import os
from collections import deque
from typing import Dict, List, Optional, Tuple
from minisim import TrafficSim, TileType

# Fallback colours used when tile PNGs are not available
_COLORS = {
    "grass": (180, 220, 180),
    "water": (80, 120, 200),
    "road": (60, 60, 60),
    "mountain": (100, 80, 60),
    "building": (120, 120, 120),
    "out": (240, 200, 60),
    "in": (70, 200, 90),
    "car": (230, 90, 60),
}


def render_image(
    sim: TrafficSim,
    asset_dir: str = "gui_assets",
    scale: int = 28,
    show_cars: bool = True,
    show_capacity: bool = True,
):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise ImportError(
            "Pillow is required for render_image. " "Install with: pip install Pillow"
        )

    # Load tile sprites (None if missing)
    sprites: Dict[str, Optional[Image.Image]] = {}
    for name in ("grass", "water", "road", "mountain", "building", "out", "in", "car"):
        path = os.path.join(asset_dir, f"{name}.png")
        try:
            sprites[name] = Image.open(path).convert("RGBA").resize((scale, scale))
        except Exception:
            sprites[name] = None

    img = Image.new("RGBA", (sim.w * scale, sim.h * scale))
    draw = ImageDraw.Draw(img)

    # Try to load a small font for capacity numbers
    try:
        font = ImageFont.truetype("arial.ttf", max(10, scale // 3))
    except Exception:
        font = ImageFont.load_default()

    car_positions = {c.pos for c in sim.cars.values()} if show_cars else set()

    tile_key_map = {
        TileType.GRASS: "grass",
        TileType.WATER: "water",
        TileType.ROAD: "road",
        TileType.MOUNTAIN: "mountain",
        TileType.BUILDING: "building",
    }

    for y in range(sim.h):
        for x in range(sim.w):
            p = (x, y)
            t = sim.grid[x][y]

            if t == TileType.DEPOT_OUT:
                key = "out"
            elif t == TileType.DEPOT_IN:
                key = "in"
            else:
                key = tile_key_map.get(t, "grass")

            sprite = sprites.get(key)
            px, py = x * scale, y * scale
            if sprite:
                img.paste(sprite, (px, py), sprite)
            else:
                color = _COLORS.get(key, (128, 128, 128))
                draw.rectangle([px, py, px + scale - 1, py + scale - 1], fill=color)

            # Overlay car
            if t == TileType.ROAD and p in car_positions:
                car_sprite = sprites.get("car")
                if car_sprite:
                    img.paste(car_sprite, (px, py), car_sprite)
                else:
                    pad = max(1, scale // 5)
                    draw.ellipse(
                        [px + pad, py + pad, px + scale - pad, py + scale - pad],
                        fill=_COLORS["car"],
                    )

            # Show capacity number on road tiles
            if show_capacity and t == TileType.ROAD:
                cap = sim.road_capacity.get(p, 1)
                text = str(cap)
                # Draw in top-right corner with white text and black outline
                text_x = px + scale - 8
                text_y = py + 2
                # Black outline
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            draw.text(
                                (text_x + dx, text_y + dy),
                                text,
                                fill=(0, 0, 0),
                                font=font,
                            )
                # White text
                draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)

    return img.convert("RGB")

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from typing import Dict, List, Optional, Tuple
import copy
import pickle

Pos = Tuple[int, int]
DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

TERRAIN_ROAD_COST = {
    "grass": 1,
    "water": 5,
    "mountain": 10,
    "building": float("inf"),
}

TERRAIN_MAX_CAPACITY = {
    "grass": 5,
    "water": 3,
    "mountain": 2,
    "building": 0,
}


class TileType(Enum):
    GRASS = 0
    WATER = 1
    ROAD = 2
    DEPOT_IN = 3
    DEPOT_OUT = 4
    MOUNTAIN = 5
    BUILDING = 6


@dataclass
class Depot:
    id: int
    kind: str  # "out" or "in"
    amount: int
    pos: Tuple[int, int]


@dataclass
class Car:
    id: int
    pos: Tuple[int, int]
    target_depot_id: int
    path: List[Tuple[int, int]] = field(default_factory=list)


def import_map(path: str) -> "TrafficSim":
    with open(path, "rb") as f:
        data = pickle.load(f)

    params = data.get("map_params", {})
    sim = TrafficSim(
        params.get("width", 10),
        params.get("height", 10),
        seed=params.get("seed"),
        initial_budget=params.get("initial_budget"),
        with_water=params.get("with_water", False),
        with_mountain=params.get("with_mountain", False),
        with_buildings=params.get("with_buildings", False),
        export_path=None,
    )

    sim.grid = data.get("grid", data.get("grids", sim.grid))

    if "init_depots" in data:
        sim.init_depots = data["init_depots"]
        sim.depots = copy.deepcopy(sim.init_depots)
    elif "depots" in data:
        sim.depots = data["depots"]
        sim.init_depots = copy.deepcopy(sim.depots)

    sim.depot_at = data.get("depot_at", {})
    sim.road_capacity = data.get("road_capacity", {})
    sim.road_terrain = data.get("road_terrain", {})
    sim.road_occupancy = {p: 0 for p in sim.road_capacity}

    # Load optimizer output: car plan and departure schedule
    sim.car_plan = data.get("car_plan", [])
    sim.saved_departures = data.get("departures", None)

    if sim.saved_departures is not None and sim.car_plan:
        sim.reset_sim(sim.saved_departures)

    return sim


class TrafficSim:
    """
    Deterministic traffic simulation on a grid world.

    Replay mode: requires a car_plan and departure schedule from an external
    optimizer.  Use import_map() to load a saved map with optimizer output,
    then call reset_sim() / step() to replay.

    Movement model (simultaneous with iterative chain resolution):
    - Each tick, all cars declare their desired move.
    - Moves are approved iteratively: when a car vacates a tile, it frees
      capacity for another car to enter that tile in the SAME tick.
    - Ties are broken by car ID (lower ID gets priority).
    - This matches CP-SAT cumulative-constraint semantics (half-open
      intervals), ensuring optimizer and simulator agree on makespan.
    """

    def __init__(
        self,
        width: int,
        height: int,
        seed: Optional[int] = None,
        initial_budget: Optional[int] = None,
        with_water: bool = False,
        with_mountain: bool = False,
        with_buildings: bool = False,
        water_params: Optional[Dict] = None,
        mountain_params: Optional[Dict] = None,
        building_params: Optional[Dict] = None,
        export_path: Optional[str] = "./exported_map.pkl",
    ):
        self.w = width
        self.h = height
        self.seed = seed

        import random

        self.rng = random.Random(seed)

        self.grid: List[List[TileType]] = [
            [TileType.GRASS for _ in range(self.h)] for _ in range(self.w)
        ]

        self.depot_at: Dict[Pos, int] = {}
        self.depots: Dict[int, Depot] = {}
        self.init_depots: Dict[int, Depot] = {}
        self.next_depot_id = 1

        self.road_capacity: Dict[Pos, int] = {}
        self.road_terrain: Dict[Pos, str] = {}
        self.road_occupancy: Dict[Pos, int] = {}

        self.cars: Dict[int, Car] = {}
        self.car_plan: List[Dict] = []
        self._pending: List[Tuple[int, int, Dict]] = []

        self.tick_count = 0
        self.export_path = export_path

        self.budget = initial_budget
        self.initial_budget = initial_budget

        self.saved_departures: Optional[Dict[int, int]] = None

        self.with_water = with_water
        self.with_mountain = with_mountain
        self.with_buildings = with_buildings

        if with_water:
            self._generate_water(**(water_params or {}))
        if with_mountain:
            self._generate_mountains(**(mountain_params or {}))
        if with_buildings:
            self._generate_buildings(**(building_params or {}))

    def _in_bounds(self, p: Pos) -> bool:
        return 0 <= p[0] < self.w and 0 <= p[1] < self.h

    def _neighbors4(self, p: Pos):
        x, y = p
        for dx, dy in DIR4:
            q = (x + dx, y + dy)
            if self._in_bounds(q):
                yield q

    @staticmethod
    def _terrain_name(tile: TileType) -> str:
        return {
            TileType.GRASS: "grass",
            TileType.WATER: "water",
            TileType.MOUNTAIN: "mountain",
            TileType.BUILDING: "building",
        }.get(tile, "grass")

    def _terrain_at(self, p: Pos) -> str:
        if not self._in_bounds(p):
            return "building"
        return self._terrain_name(self.grid[p[0]][p[1]])

    def _generate_water(
        self,
        num_rivers: int = 1,
        river_width: int = 1,
        drift_prob: float = 0.35,
        num_lakes: int = 1,
        lake_size: Tuple[int, int] = (4, 12),
        lake_compactness: float = 0.85,
    ):
        half = river_width // 2

        for _ in range(num_rivers):
            horizontal = self.rng.random() < 0.5
            if horizontal:
                pos = self.rng.randint(0, self.h - 1)
                for step in range(self.w):
                    for d in range(-half, half + 1):
                        ny = pos + d
                        if 0 <= ny < self.h:
                            self.grid[step][ny] = TileType.WATER
                    if self.rng.random() < drift_prob:
                        pos = max(0, min(self.h - 1, pos + self.rng.choice([-1, 1])))
            else:
                pos = self.rng.randint(0, self.w - 1)
                for step in range(self.h):
                    for d in range(-half, half + 1):
                        nx = pos + d
                        if 0 <= nx < self.w:
                            self.grid[nx][step] = TileType.WATER
                    if self.rng.random() < drift_prob:
                        pos = max(0, min(self.w - 1, pos + self.rng.choice([-1, 1])))

        for _ in range(num_lakes):
            cx = self.rng.randrange(self.w)
            cy = self.rng.randrange(self.h)
            size = self.rng.randint(lake_size[0], lake_size[1])
            q: deque = deque([(cx, cy)])
            visited = {(cx, cy)}
            placed = 0
            while q and placed < size:
                cur = q.popleft()
                self.grid[cur[0]][cur[1]] = TileType.WATER
                placed += 1
                for nb in self._neighbors4(cur):
                    if nb not in visited and self.rng.random() < lake_compactness:
                        visited.add(nb)
                        q.append(nb)

    def _generate_mountains(
        self,
        num_ranges: Optional[int] = None,
        range_size: Tuple[int, int] = (15, 60),
        compactness: float = 0.75,
    ):
        if num_ranges is None:
            num_ranges = max(2, (self.w * self.h) // 400)

        for _ in range(num_ranges):
            cx = self.rng.randrange(self.w)
            cy = self.rng.randrange(self.h)
            if self.grid[cx][cy] != TileType.GRASS:
                continue
            size = self.rng.randint(range_size[0], range_size[1])
            q: deque = deque([(cx, cy)])
            visited = {(cx, cy)}
            placed = 0
            while q and placed < size:
                cur = q.popleft()
                if self.grid[cur[0]][cur[1]] == TileType.GRASS:
                    self.grid[cur[0]][cur[1]] = TileType.MOUNTAIN
                    placed += 1
                for nb in self._neighbors4(cur):
                    if nb not in visited and self.rng.random() < compactness:
                        visited.add(nb)
                        q.append(nb)

    def _generate_buildings(
        self,
        num_blocks: Optional[int] = None,
        min_width: int = 2,
        max_width: int = 6,
        min_height: int = 2,
        max_height: int = 5,
    ):
        if num_blocks is None:
            num_blocks = max(3, (self.w * self.h) // 250)

        for _ in range(num_blocks):
            bw = self.rng.randint(min_width, max_width)
            bh = self.rng.randint(min_height, max_height)
            bx = self.rng.randint(0, max(0, self.w - bw))
            by = self.rng.randint(0, max(0, self.h - bh))
            for x in range(bx, min(bx + bw, self.w)):
                for y in range(by, min(by + bh, self.h)):
                    self.grid[x][y] = TileType.BUILDING

    def calculate_road_cost(self, p: Pos, capacity: int) -> int:
        terrain = self.road_terrain.get(p) or self._terrain_at(p)
        base = TERRAIN_ROAD_COST[terrain]
        return int(base * capacity)

    def can_afford(self, cost: int) -> bool:
        return self.budget is None or self.budget >= cost

    def spend(self, cost: int) -> bool:
        if self.budget is None:
            return True
        if self.can_afford(cost):
            self.budget -= cost
            return True
        return False

    def refund(self, amount: int):
        if self.budget is not None:
            self.budget += amount

    def add_road(self, p: Pos, capacity: int = 1) -> Tuple[bool, str]:
        if not self._in_bounds(p):
            return False, "out of bounds"

        tile = self.grid[p[0]][p[1]]
        if tile in (TileType.DEPOT_IN, TileType.DEPOT_OUT):
            return False, "cannot build on depot"

        terrain = self.road_terrain.get(p) or self._terrain_at(p)
        if terrain == "building":
            return False, "cannot build on building"
        if capacity <= 0:
            return False, "capacity must be positive"
        if capacity > TERRAIN_MAX_CAPACITY.get(terrain, 0):
            return False, "capacity exceeds terrain max"

        if tile == TileType.ROAD:
            old_cap = self.road_capacity.get(p, 1)
            if capacity == old_cap:
                return True, "already set"
            old_cost = self.calculate_road_cost(p, old_cap)
            new_cost = self.calculate_road_cost(p, capacity)
            delta = new_cost - old_cost
            if delta > 0 and not self.spend(delta):
                return False, "budget exceeded"
            if delta < 0:
                self.refund(-delta)
            self.road_capacity[p] = capacity
            return True, "updated"

        cost = self.calculate_road_cost(p, capacity)
        if not self.spend(cost):
            return False, "budget exceeded"

        self.grid[p[0]][p[1]] = TileType.ROAD
        self.road_capacity[p] = capacity
        self.road_terrain[p] = terrain
        self.road_occupancy[p] = 0
        return True, "built"

    def remove_road(self, p: Pos) -> Tuple[bool, str]:
        if not self._in_bounds(p):
            return False, "out of bounds"
        if self.grid[p[0]][p[1]] != TileType.ROAD:
            return False, "not a road"
        if self.road_occupancy.get(p, 0) > 0:
            return False, "road occupied"

        terrain = self.road_terrain.get(p, "grass")
        refund_amt = self.calculate_road_cost(p, self.road_capacity.get(p, 1))

        revert = {
            "grass": TileType.GRASS,
            "water": TileType.WATER,
            "mountain": TileType.MOUNTAIN,
            "building": TileType.BUILDING,
        }
        self.grid[p[0]][p[1]] = revert.get(terrain, TileType.GRASS)

        self.road_capacity.pop(p, None)
        self.road_terrain.pop(p, None)
        self.road_occupancy.pop(p, None)
        self.refund(refund_amt)
        return True, "removed"

    def add_depot(self, p: Pos, kind: str, amount: int) -> Optional[int]:
        if kind not in ("out", "in"):
            return None
        if not self._in_bounds(p):
            return None
        if self.grid[p[0]][p[1]] not in (
            TileType.GRASS,
            TileType.WATER,
            TileType.MOUNTAIN,
        ):
            return None

        depot_id = self.next_depot_id
        self.next_depot_id += 1
        depot = Depot(depot_id, kind, max(0, int(amount)), p)
        self.depots[depot_id] = depot
        self.init_depots[depot_id] = copy.deepcopy(depot)
        self.depot_at[p] = depot_id
        self.grid[p[0]][p[1]] = (
            TileType.DEPOT_OUT if kind == "out" else TileType.DEPOT_IN
        )
        return depot_id

    def reset_sim(self, departures: Dict[int, int]):
        """Reset simulation state using an external optimizer schedule.
        """
        self.cars.clear()
        self.road_occupancy = {p: 0 for p in self.road_capacity}
        self.depots = copy.deepcopy(self.init_depots)
        self.tick_count = 0
        self._pending = []

        for idx, plan in enumerate(self.car_plan):
            car_id = idx + 1
            if idx in departures:
                self._pending.append((departures[idx], car_id, plan))
            else:
                self.cars[car_id] = Car(
                    id=car_id,
                    pos=plan["depot_out_pos"],
                    target_depot_id=plan["depot_in_id"],
                    path=list(plan["path"]),
                )
        self._pending.sort(key=lambda x: (x[0], x[1]))

    def _spawn_pending(self):
        while self._pending and self._pending[0][0] <= self.tick_count:
            _, car_id, plan = self._pending.pop(0)
            self.cars[car_id] = Car(
                id=car_id,
                pos=plan["depot_out_pos"],
                target_depot_id=plan["depot_in_id"],
                path=list(plan["path"]),
            )

    def _move_cars(self):
        if not self.cars:
            return

        # Step 1 -- desired moves
        desired: List[Tuple[int, Pos, Pos, TileType]] = []
        for car_id in sorted(self.cars.keys()):
            car = self.cars[car_id]
            if not car.path:
                continue
            nxt = car.path[0]
            nxt_tile = self.grid[nxt[0]][nxt[1]]
            if nxt_tile in (TileType.ROAD, TileType.DEPOT_IN):
                desired.append((car_id, car.pos, nxt, nxt_tile))

        if not desired:
            return

        # Step 2 -- available capacity per road tile
        avail: Dict[Pos, int] = {}
        for pos, cap in self.road_capacity.items():
            avail[pos] = cap - self.road_occupancy.get(pos, 0)

        # Step 3 -- iterative approval
        approved: set = set()
        changed = True
        while changed:
            changed = False
            for car_id, cur, nxt, nxt_tile in desired:
                if car_id in approved:
                    continue

                can_move = False
                if nxt_tile == TileType.DEPOT_IN:
                    can_move = True
                elif nxt_tile == TileType.ROAD:
                    if avail.get(nxt, 0) > 0:
                        can_move = True

                if can_move:
                    approved.add(car_id)
                    changed = True
                    if nxt_tile == TileType.ROAD:
                        avail[nxt] -= 1
                    if self.grid[cur[0]][cur[1]] == TileType.ROAD:
                        avail[cur] = avail.get(cur, 0) + 1

        # Step 4 -- apply approved moves
        to_remove: List[int] = []
        for car_id, cur, nxt, nxt_tile in desired:
            if car_id not in approved:
                continue
            car = self.cars[car_id]
            car.path.pop(0)
            car.pos = nxt

            if self.grid[cur[0]][cur[1]] == TileType.ROAD:
                self.road_occupancy[cur] -= 1
            if nxt_tile == TileType.ROAD:
                self.road_occupancy[nxt] = self.road_occupancy.get(nxt, 0) + 1

            if nxt_tile == TileType.DEPOT_IN:
                depot_id = self.depot_at.get(nxt)
                if depot_id is not None:
                    dep = self.depots.get(depot_id)
                    if dep and dep.amount > 0:
                        dep.amount -= 1
                to_remove.append(car_id)

        for cid in to_remove:
            del self.cars[cid]

    def step(self):
        """Advance simulation by one tick."""
        self.tick_count += 1
        self._spawn_pending()
        self._move_cars()

    def total_supply(self) -> int:
        return sum(d.amount for d in self.depots.values() if d.kind == "out")

    def total_demand(self) -> int:
        return sum(d.amount for d in self.depots.values() if d.kind == "in")

    def cars_on_road(self) -> int:
        depot_in_pos = {d.pos for d in self.depots.values() if d.kind == "in"}
        return sum(1 for c in self.cars.values() if c.pos not in depot_in_pos)

    def is_done(self) -> bool:
        """Check if simulation is complete (no cars left and no pending spawns)."""
        return not self.cars and not self._pending

    def all_delivered(self) -> bool:
        """Check if all demand has been satisfied (all depot_in amounts are 0)."""
        return all(d.amount == 0 for d in self.depots.values() if d.kind == "in")

    def get_road_capacity(self, p: Pos) -> int:
        return self.road_capacity.get(p, 0)

    def get_road_occupancy(self, p: Pos) -> int:
        return self.road_occupancy.get(p, 0)

    def export_map(self, path: Optional[str] = None):
        path = path or self.export_path
        if path is None:
            raise ValueError("no export path specified")
        data = {
            "grid": self.grid,
            "depots": self.depots,
            "init_depots": self.init_depots,
            "depot_at": self.depot_at,
            "road_capacity": self.road_capacity,
            "road_terrain": self.road_terrain,
            "road_occupancy": {p: 0 for p in self.road_capacity},
            "car_plan": self.car_plan,
            "departures": self.saved_departures,
            "map_params": {
                "width": self.w,
                "height": self.h,
                "seed": self.seed,
                "initial_budget": self.initial_budget,
                "with_water": self.with_water,
                "with_mountain": self.with_mountain,
                "with_buildings": self.with_buildings,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

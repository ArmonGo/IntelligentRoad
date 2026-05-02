
from __future__ import annotations
import heapq
import sys
import os
from collections import deque, defaultdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from minisim import (
    TrafficSim,
    TileType,
    TERRAIN_MAX_CAPACITY,
    DIR4,
    Pos,
)

# Dijkstra over raw terrain  (used by build_network)
def _tile_build_cost(sim: TrafficSim, pos: Pos) -> float:
    """Uniform hop cost for path-finding in build_network.
    Every traversable tile costs exactly 1 hop so that _dijkstra finds the
    fewest-tile path regardless of terrain type.  Depot and existing road
    tiles are free (0) so the path does not count them as new hops.
    Buildings are impassable (inf).
    """
    tile = sim.grid[pos[0]][pos[1]]
    if tile in (TileType.DEPOT_OUT, TileType.DEPOT_IN, TileType.ROAD):
        return 0.0
    if tile == TileType.BUILDING:
        return float("inf")
    return 1.0  # grass, water, mountain — uniform hop cost


def _dijkstra(sim: TrafficSim, start: Pos, end: Pos) -> Optional[list[Pos]]:
    """Shortest-cost path from start to end over terrain.
    """
    dist: dict[Pos, float] = {start: 0.0}
    prev: dict[Pos, Optional[Pos]] = {start: None}
    # heap: (cost, counter, pos) — counter breaks ties without comparing Pos
    counter = 0
    heap = [(0.0, counter, start)]

    while heap:
        d, _, cur = heapq.heappop(heap)
        if d > dist.get(cur, float("inf")):
            continue
        if cur == end:
            break
        for nb in sim._neighbors4(cur):
            # Depots are terminal-only: cannot pass through an intermediate depot
            if nb != end:
                nb_tile = sim.grid[nb[0]][nb[1]]
                if nb_tile in (TileType.DEPOT_OUT, TileType.DEPOT_IN):
                    continue
            cost = _tile_build_cost(sim, nb)
            if cost == float("inf"):
                continue
            new_d = d + cost
            if new_d < dist.get(nb, float("inf")):
                dist[nb] = new_d
                prev[nb] = cur
                counter += 1
                heapq.heappush(heap, (new_d, counter, nb))

    if end not in dist:
        return None

    path: list[Pos] = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


# Network builder


def build_network(sim: TrafficSim) -> None:
    """Build a road network using the Dijkstra shortest-hop union.

    Steps:
    * For every (OUT, IN) depot pair find the fewest-tile terrain path
       (hop cost = 1 per non-depot tile, buildings impassable).
    * Union all paths → road tile set; count overlap per tile.
    * Assign each tile its maximum terrain capacity.
    * If total cost exceeds budget: repeatedly pick the tile with the
       smallest overlap (tie-break: highest cost) and decrease its capacity
       by 1; remove the tile when capacity reaches 0.
    * Apply all desired capacities via add_road in one pass.
    """
    depots_out = sorted(
        [d for d in sim.init_depots.values() if d.kind == "out"],
        key=lambda d: d.id,
    )
    depots_in = sorted(
        [d for d in sim.init_depots.values() if d.kind == "in"],
        key=lambda d: d.id,
    )
    if not depots_out or not depots_in:
        return

    overlap: dict[Pos, int] = defaultdict(int)  # non-depot road tiles

    for d_out in depots_out:
        for d_in in depots_in:
            path = _dijkstra(sim, d_out.pos, d_in.pos)
            if path is None:
                print(
                    f"[build_network] WARNING: no path between "
                    f"depot_out {d_out.id} {d_out.pos} and "
                    f"depot_in  {d_in.id} {d_in.pos}",
                    file=sys.stderr,
                )
                continue
            for pos in path:
                tile = sim.grid[pos[0]][pos[1]]
                if tile not in (TileType.DEPOT_OUT, TileType.DEPOT_IN):
                    overlap[pos] += 1

    if not overlap:
        return

    # Assign max terrain capacity to every tile
    desired_cap: dict[Pos, int] = {}
    for pos in overlap:
        terrain = sim._terrain_at(pos)
        desired_cap[pos] = TERRAIN_MAX_CAPACITY.get(terrain, 1)

    # Budget trim: reduce least-overlap (tie: most expensive) tiles first
    if sim.initial_budget is not None:
        total_cost = sum(sim.calculate_road_cost(p, c) for p, c in desired_cap.items())
        while total_cost > sim.initial_budget and desired_cap:
            # pick tile: min overlap, tie-break highest cost
            pos = min(
                desired_cap,
                key=lambda p: (overlap[p], -sim.calculate_road_cost(p, desired_cap[p])),
            )
            old_cost = sim.calculate_road_cost(pos, desired_cap[pos])
            desired_cap[pos] -= 1
            if desired_cap[pos] == 0:
                del desired_cap[pos]
                total_cost -= old_cost
            else:
                new_cost = sim.calculate_road_cost(pos, desired_cap[pos])
                total_cost -= old_cost - new_cost

    # Apply all in one pass
    for pos, cap in desired_cap.items():
        ok, _ = sim.add_road(pos, capacity=cap)
        if not ok:
            # Fallback: try cap=1 if max was rejected for any reason
            sim.add_road(pos, capacity=1)

def _all_shortest_road_paths(
    sim: TrafficSim, out_pos: Pos, in_pos: Pos
) -> list[list[Pos]]:
    # returns all shortest paths if there are more than one; empty list if no path exists
    if out_pos == in_pos:
        return []

    # dist[pos] = min hops from out_pos
    dist: dict[Pos, int] = {out_pos: 0}
    preds: dict[Pos, list[Pos]] = {out_pos: []}
    queue: deque[Pos] = deque([out_pos])
    target_dist: Optional[int] = None

    while queue:
        cur = queue.popleft()
        d = dist[cur]
        if target_dist is not None and d >= target_dist:
            continue

        x, y = cur
        for dx, dy in DIR4:
            nb = (x + dx, y + dy)
            if not sim._in_bounds(nb):
                continue
            nb_tile = sim.grid[nb[0]][nb[1]]
            # Can step on road tiles or the destination
            if nb != in_pos and nb_tile != TileType.ROAD:
                continue

            new_d = d + 1
            if nb not in dist:
                dist[nb] = new_d
                preds[nb] = [cur]
                queue.append(nb)
                if nb == in_pos:
                    target_dist = new_d
            elif dist[nb] == new_d:
                preds[nb].append(cur)

    if in_pos not in dist:
        return []

    # Reconstruct all shortest paths (out_pos excluded from result)
    def _reconstruct(pos: Pos) -> list[list[Pos]]:
        if pos == out_pos:
            return [[]]
        result = []
        for pred in preds.get(pos, []):
            for prefix in _reconstruct(pred):
                result.append(prefix + [pos])
        return result

    return _reconstruct(in_pos)


# Internal simulation step  
def _movement_step(
    active_cars: dict[int, dict],
    occupancy: dict[Pos, int],
    sim: TrafficSim,
) -> list[tuple[int, int]]:
    if not active_cars:
        return []

    # Collect desired moves: (car_id, cur_pos, nxt_pos, nxt_tile)
    desired: list[tuple[int, Pos, Pos, TileType]] = []
    for car_id in sorted(active_cars):
        car = active_cars[car_id]
        if not car["path"]:
            continue
        nxt = car["path"][0]
        nxt_tile = sim.grid[nxt[0]][nxt[1]]
        if nxt_tile in (TileType.ROAD, TileType.DEPOT_IN):
            desired.append((car_id, car["pos"], nxt, nxt_tile))
    if not desired:
        return []

    # Available capacity per road tile (accounting for current occupancy)
    avail: dict[Pos, int] = {
        p: sim.road_capacity[p] - occupancy.get(p, 0) for p in sim.road_capacity
    }

    # Iterative chain-resolution approval (ascending car_id = priority)
    approved: set[int] = set()
    changed = True
    while changed:
        changed = False
        for car_id, cur, nxt, nxt_tile in desired:
            if car_id in approved:
                continue
            if nxt_tile == TileType.DEPOT_IN:
                can_move = True
            elif nxt_tile == TileType.ROAD:
                can_move = avail.get(nxt, 0) > 0
            else:
                can_move = False
            if can_move:
                approved.add(car_id)
                changed = True
                if nxt_tile == TileType.ROAD:
                    avail[nxt] -= 1
                if sim.grid[cur[0]][cur[1]] == TileType.ROAD:
                    avail[cur] = avail.get(cur, 0) + 1

    # Apply approved moves and update occupancy
    arrived: list[tuple[int, int]] = []
    to_remove: list[int] = []
    for car_id, cur, nxt, nxt_tile in desired:
        if car_id not in approved:
            continue
        car = active_cars[car_id]
        car["path"].pop(0)
        car["pos"] = nxt

        if sim.grid[cur[0]][cur[1]] == TileType.ROAD:
            occupancy[cur] = max(0, occupancy.get(cur, 0) - 1)
        if nxt_tile == TileType.ROAD:
            occupancy[nxt] = occupancy.get(nxt, 0) + 1
        elif nxt_tile == TileType.DEPOT_IN:
            arrived.append((car_id, car["dest"]))
            to_remove.append(car_id)

    for car_id in to_remove:
        del active_cars[car_id]

    return arrived


# Route planner
def solve_routing(sim: TrafficSim) -> tuple[list[dict], dict[int, int], dict]:
    """
    Steps:
      For each tick t:
        Phase A - move existing cars (chain resolution, ascending car_id)
        Phase B - spawn decisions:
          for d_out (asc id):
            for unassigned car (asc id):
              for d_in (asc distance from d_out):
                if active_needs(d_in) > 0:
                  try each BFS path; spawn on first one whose first tile
                  has available capacity (after Phase A + this Phase B so far)
    Returns (car_plan, departures, meta).
    """
    depots_out = sorted(
        [d for d in sim.init_depots.values() if d.kind == "out"],
        key=lambda d: d.id,
    )
    depots_in = sorted(
        [d for d in sim.init_depots.values() if d.kind == "in"],
        key=lambda d: d.id,
    )
    if not depots_out or not depots_in:
        return (
            [],
            {},
            {
                "makespan": None,
                "road_cost": 0,
                "occ": None,
                "solve_status": "EMPTY",
                "all_delivered": False,
            },
        )

    # Pre-compute all shortest road paths for every OD pair
    path_cache: dict[tuple[Pos, Pos], list[list[Pos]]] = {}
    for d_out in depots_out:
        for d_in in depots_in:
            key = (d_out.pos, d_in.pos)
            path_cache[key] = _all_shortest_road_paths(sim, d_out.pos, d_in.pos)

    # Distance rank: for each d_out, sort depots_in by min BFS hop count
    dist_rank: dict[int, list] = {}
    for d_out in depots_out:
        ranked = []
        for d_in in depots_in:
            paths = path_cache.get((d_out.pos, d_in.pos), [])
            hop = len(paths[0]) if paths else float("inf")
            ranked.append((hop, d_in.id, d_in))
        ranked.sort()
        dist_rank[d_out.id] = [item[2] for item in ranked]

    # Assign global car IDs: ascending by d_out.id, then local car index
    # unassigned[d_out.id] = list of local indices (0 .. amount-1)
    unassigned: dict[int, list[int]] = {d.id: list(range(d.amount)) for d in depots_out}
    # Global car id = base_id[d_out.id] + local_idx
    base_id: dict[int, int] = {}
    gid = 1
    for d_out in depots_out:
        base_id[d_out.id] = gid
        gid += d_out.amount

    def global_car_id(d_out_id: int, local_idx: int) -> int:
        return base_id[d_out_id] + local_idx

    # Planning state
    need = {d.id: d.amount for d in depots_in}
    arrived = {d.id: 0 for d in depots_in}
    on_the_way = {d.id: 0 for d in depots_in}

    occupancy: dict[Pos, int] = {p: 0 for p in sim.road_capacity}
    active_cars: dict[int, dict] = {}

    car_plan: list[dict] = []
    departures: dict[int, int] = {}

    d_out_by_id = {d.id: d for d in depots_out}

    def active_needs(d_in_id: int) -> int:
        return need[d_in_id] - arrived[d_in_id] - on_the_way[d_in_id]

    def all_done() -> bool:
        return all(active_needs(d.id) <= 0 for d in depots_in) and not active_cars

    MAX_TICKS = 20_000

    for t in range(MAX_TICKS):
        if all_done():
            break

        # Phase A - move existing cars
        for car_id, dest_id in _movement_step(active_cars, occupancy, sim):
            arrived[dest_id] += 1
            on_the_way[dest_id] -= 1

        # Phase B - spawn decisions
        for d_out in depots_out:
            if not unassigned[d_out.id]:
                continue

            still_unassigned: list[int] = []
            for local_idx in unassigned[d_out.id]:
                cid = global_car_id(d_out.id, local_idx)
                assigned = False

                for d_in in dist_rank[d_out.id]:
                    if active_needs(d_in.id) == 0:
                        continue

                    for path in path_cache.get((d_out.pos, d_in.pos), []):
                        if not path:
                            continue
                        first = path[0]
                        first_tile = sim.grid[first[0]][first[1]]

                        # Capacity check on first tile
                        if first_tile == TileType.DEPOT_IN:
                            # Direct adjacency (no road tile): always allowed
                            can_spawn = True
                        else:
                            # occupancy is updated immediately on each spawn in
                            # this Phase B, so no separate reserve counter needed
                            can_spawn = occupancy.get(first, 0) < sim.road_capacity.get(
                                first, 0
                            )

                        if can_spawn:
                            idx = len(car_plan)
                            car_plan.append(
                                {
                                    "depot_out_id": d_out.id,
                                    "depot_out_pos": d_out.pos,
                                    "depot_in_id": d_in.id,
                                    "path": list(path),
                                }
                            )
                            departures[idx] = t
                            on_the_way[d_in.id] += 1

                            if first_tile == TileType.DEPOT_IN:
                                # Car starts at depot_out, hops directly to depot_in
                                active_cars[cid] = {
                                    "pos": d_out.pos,
                                    "path": list(path),
                                    "dest": d_in.id,
                                }
                            else:
                                # Car is immediately placed on the first road tile.
                                # Increment occupancy so subsequent Phase B cars
                                # see the updated capacity.
                                occupancy[first] = occupancy.get(first, 0) + 1
                                active_cars[cid] = {
                                    "pos": first,
                                    "path": list(
                                        path[1:]
                                    ),  # remaining: tile2..depot_in
                                    "dest": d_in.id,
                                }

                            assigned = True
                            break  # found a valid path for this d_in

                    if assigned:
                        break  # stop trying other d_in for this car

                if not assigned:
                    still_unassigned.append(local_idx)

            unassigned[d_out.id] = still_unassigned

    solved = all_done()
    if not solved:
        print(
            f"[solve_routing] WARNING: did not finish within {MAX_TICKS} ticks.",
            file=sys.stderr,
        )

    # Aggregate occupancy: {tick: {pos: count}} from individual plans
    occ_by_tick: dict = defaultdict(lambda: defaultdict(int))
    for idx, entry in enumerate(car_plan):
        d = departures.get(idx, 0)
        for j, pos in enumerate(entry["path"][:-1]):  # exclude depot_in
            occ_by_tick[d + j][tuple(pos)] += 1
    occ = {tick: dict(v) for tick, v in occ_by_tick.items()}

    road_cost = sum(
        sim.calculate_road_cost(pos, cap) for pos, cap in sim.road_capacity.items()
    )
    # t is the planning-loop tick when all_done() triggered (≈ sim makespan + 1
    # because the loop checks all_done() one tick after the last car arrived).
    meta = {
        "makespan": t if solved else None,
        "road_cost": road_cost,
        "occ": occ,
        "solve_status": "HEURISTIC",
        "all_delivered": solved,
    }
    return car_plan, departures, meta


# Joint solver: network + routing
def solve_joint(sim: TrafficSim) -> tuple[list[dict], dict[int, int], dict]:
    """Build road network then solve routing.
    """
    build_network(sim)
    return solve_routing(sim)

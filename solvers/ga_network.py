
from __future__ import annotations
import copy
import heapq
import random
import sys
import os
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from minisim import (
    TrafficSim,
    TileType,
    TERRAIN_ROAD_COST,
    TERRAIN_MAX_CAPACITY,
    DIR4,
    Pos,
)

Chromosome = dict  # dict[Pos, list[int, int]]

_INF_MAKESPAN = 10001  # sentinel for infeasible / timed-out networks

def _candidate_tiles(sim: TrafficSim) -> list[Pos]:
    """All tiles eligible to be built as roads (excludes depots and buildings)."""
    depot_pos = {d.pos for d in sim.init_depots.values()}
    tiles = []
    for x in range(sim.w):
        for y in range(sim.h):
            pos = (x, y)
            if pos in depot_pos:
                continue
            tile = sim.grid[x][y]
            if tile == TileType.BUILDING:
                continue
            terrain = sim._terrain_name(tile)
            if TERRAIN_ROAD_COST.get(terrain, float("inf")) < float("inf"):
                tiles.append(pos)
    return sorted(tiles)


def _dijkstra(
    sim: TrafficSim,
    start: Pos,
    end: Pos,
    blocked: Optional[set] = None,
) -> Optional[list[Pos]]:
    """Uniform hop-count Dijkstra from start to end, with optional blocked tiles.
    all tiles have uniform cost: existing roads and the destination are free, all else is 1.
    """
    if blocked is None:
        blocked = set()

    # All depots except start/end are impassable intermediate tiles
    other_depots = {d.pos for d in sim.init_depots.values()} - {start, end}

    dist: dict[Pos, float] = {start: 0.0}
    prev: dict[Pos, Optional[Pos]] = {start: None}
    ctr = 0
    heap = [(0.0, ctr, start)]

    while heap:
        d, _, cur = heapq.heappop(heap)
        if d > dist.get(cur, float("inf")):
            continue
        if cur == end:
            break
        x, y = cur
        for dx, dy in DIR4:
            nb = (x + dx, y + dy)
            if not sim._in_bounds(nb):
                continue
            if nb in blocked or nb in other_depots:
                continue
            nb_tile = sim.grid[nb[0]][nb[1]]
            if nb_tile == TileType.BUILDING:
                continue
            # Uniform hop cost: existing road or destination = free, else 1
            cost = 0.0 if (nb == end or nb_tile == TileType.ROAD) else 1.0
            new_d = d + cost
            if new_d < dist.get(nb, float("inf")):
                dist[nb] = new_d
                prev[nb] = cur
                ctr += 1
                heapq.heappush(heap, (new_d, ctr, nb))

    if end not in dist:
        return None

    path: list[Pos] = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _chrom_cost(chrom: Chromosome, sim: TrafficSim) -> int:
    """Total road build cost of a chromosome."""
    return sum(
        sim.calculate_road_cost(pos, genes[1])
        for pos, genes in chrom.items()
        if genes[0]
    )

def _copy_chrom(chrom: Chromosome) -> Chromosome:
    return {pos: list(v) for pos, v in chrom.items()}


# Chromosome initialisation
def seed_chromosome(sim: TrafficSim) -> Chromosome:
    """Chromosome seeded from rule_based.build_network (max-capacity + budget trim).
    First deepcopy sim, runs build_network on the copy, then reads the resulting
    road_capacity to populate the chromosome. Guaranteed feasible for any map
    that rule_based can connect.
    """
    from solvers.rule_based import build_network as _build_network

    sim_seed = copy.deepcopy(sim)
    _build_network(sim_seed)

    cand = _candidate_tiles(sim)
    chrom: Chromosome = {pos: [0, 1] for pos in cand}

    for pos, cap in sim_seed.road_capacity.items():
        if pos in chrom:
            chrom[pos] = [1, cap]

    return chrom


def variant_capacity_upgrade(
    seed: Chromosome,
    sim: TrafficSim,
    rng: random.Random,
) -> Chromosome:
    """Seed topology; randomly bump capacities on a third of road tiles."""
    chrom = _copy_chrom(seed)
    road_tiles = [pos for pos, (rb, _) in chrom.items() if rb == 1]
    if not road_tiles:
        return chrom
    n_upgrade = max(1, len(road_tiles) // 3)
    for pos in rng.sample(road_tiles, min(n_upgrade, len(road_tiles))):
        tile = sim.grid[pos[0]][pos[1]]
        terrain = sim._terrain_name(tile)
        max_cap = TERRAIN_MAX_CAPACITY.get(terrain, 1)
        cur_cap = chrom[pos][1]
        if cur_cap < max_cap:
            chrom[pos][1] = rng.randint(cur_cap + 1, max_cap)
    return chrom


def variant_path_alternative(
    seed: Chromosome,
    sim: TrafficSim,
    rng: random.Random,
) -> Chromosome:
    """Seed + an alternative terrain path for one random OD pair.
    """
    chrom = _copy_chrom(seed)
    depots_out = [d for d in sim.init_depots.values() if d.kind == "out"]
    depots_in = [d for d in sim.init_depots.values() if d.kind == "in"]
    if not depots_out or not depots_in:
        return chrom

    d_out = rng.choice(depots_out)
    d_in = rng.choice(depots_in)

    orig = _dijkstra(sim, d_out.pos, d_in.pos)
    if orig is None or len(orig) < 3:
        return chrom

    interior = orig[1:-1]
    lo = max(0, len(interior) // 3)
    hi = min(len(interior), 2 * len(interior) // 3 + 1)
    blocked = set(interior[lo:hi])

    alt = _dijkstra(sim, d_out.pos, d_in.pos, blocked=blocked)
    if alt is None:
        return chrom
    for pos in alt[1:-1]:
        if pos in chrom:
            chrom[pos][0] = 1

    return chrom


def variant_road_expansion(
    seed: Chromosome,
    sim: TrafficSim,
    rng: random.Random,
) -> Chromosome:
    """Seed + a random blob of tiles adjacent to the existing road network."""
    chrom = _copy_chrom(seed)
    frontier = [
        nb
        for pos, (rb, _) in chrom.items()
        if rb == 1
        for dx, dy in DIR4
        if (nb := (pos[0] + dx, pos[1] + dy)) in chrom and chrom[nb][0] == 0
    ]
    if not frontier:
        return chrom
    n_add = rng.randint(1, max(1, len(frontier) // 3))
    for pos in rng.sample(frontier, min(n_add, len(frontier))):
        chrom[pos][0] = 1
    return chrom


def _random_chromosome(
    cand: list[Pos],
    sim: TrafficSim,
    rng: random.Random,
    road_prob: float = 0.25,
) -> Chromosome:
    """Fully random chromosome, sparse (25% road), kept for diversity."""
    chrom: Chromosome = {}
    for pos in cand:
        rb = 1 if rng.random() < road_prob else 0
        tile = sim.grid[pos[0]][pos[1]]
        terrain = sim._terrain_name(tile)
        max_cap = TERRAIN_MAX_CAPACITY.get(terrain, 1)
        cap_g = rng.randint(1, max_cap)
        chrom[pos] = [rb, cap_g]
    return chrom


def initial_population(
    sim: TrafficSim,
    size: int,
    rng: random.Random,
) -> list[Chromosome]:
    """Build the initial GA population.
    Composition :
      1 exact seed
      20% capacity-upgrade variants
      20% path-alternative variants
      20% road-expansion variants
      10% fully random (diversity hedge)
      rest: lightly mutated seed copies
    """
    cand = _candidate_tiles(sim)
    seed = seed_chromosome(sim)
    pop: list[Chromosome] = [seed]

    n_upgrade = max(1, size * 20 // 100)
    n_alt = max(1, size * 20 // 100)
    n_expand = max(1, size * 20 // 100)
    n_random = max(1, size * 10 // 100)

    for _ in range(n_upgrade):
        pop.append(variant_capacity_upgrade(seed, sim, rng))
    for _ in range(n_alt):
        pop.append(variant_path_alternative(seed, sim, rng))
    for _ in range(n_expand):
        pop.append(variant_road_expansion(seed, sim, rng))
    for _ in range(n_random):
        pop.append(_random_chromosome(cand, sim, rng))

    # Fill remainder with lightly mutated seed copies
    while len(pop) < size:
        c = _copy_chrom(seed)
        c = mutate(c, sim, road_rate=0.05, cap_rate=0.10, cap_down_rate=0.10, rng=rng)
        pop.append(c)
    return [repair(c, sim) for c in pop[:size]]


# GA operators
def decode(sim: TrafficSim, chrom: Chromosome) -> TrafficSim:
    """Apply chromosome roads to a deep copy of sim. Returns the copy."""
    sim_copy = copy.deepcopy(sim)
    for pos, (rb, cap_g) in chrom.items():
        if rb:
            sim_copy.add_road(pos, cap_g)
    return sim_copy


def _is_connected(sim: TrafficSim) -> bool:
    # depots_in_pos = {d.pos for d in sim.init_depots.values() if d.kind == "in"}
    depots_out = [d for d in sim.init_depots.values() if d.kind == "out"]

    for d_out in depots_out:
        visited: set[Pos] = {d_out.pos}
        stack = [d_out.pos]
        found = False
        while stack:
            x, y = stack.pop()
            for dx, dy in DIR4:
                nb = (x + dx, y + dy)
                if not sim._in_bounds(nb) or nb in visited:
                    continue
                nb_tile = sim.grid[nb[0]][nb[1]]
                if nb_tile == TileType.DEPOT_IN:
                    found = True
                    break
                if nb_tile == TileType.ROAD:
                    visited.add(nb)
                    stack.append(nb)
            if found:
                break
        if not found:
            return False
    return True


_INF_FITNESS = (_INF_MAKESPAN, float("inf"))  # sentinel for infeasible networks

def fitness(
    sim: TrafficSim,
    chrom: Chromosome,
    route_solver: Optional[Callable] = None,
) -> tuple[int, float]:
    """Fitness = (makespan, road_cost), lower is better on both axes.

    Primary objective:  makespan (minimise ticks to deliver all cars).
    Secondary objective: road_cost (minimise build cost when makespan ties).

    Returns _INF_FITNESS if:
      - no roads built, or connectivity check fails (cheap BFS guard)
      - route_solver reports all_delivered=False or makespan=None
    stderr is suppressed to silence warnings from partial/disconnected networks.
    """
    import io as _io

    if route_solver is None:
        from solvers.rule_based import solve_routing as _default

        route_solver = _default

    sim_copy = decode(sim, chrom)

    # Cheap BFS connectivity guard before calling the full solver
    if not sim_copy.road_capacity or not _is_connected(sim_copy):
        return _INF_FITNESS

    _old_stderr = sys.stderr
    sys.stderr = _io.StringIO()
    try:
        solver_result = route_solver(sim_copy)
    finally:
        sys.stderr = _old_stderr

    meta = solver_result[2] if len(solver_result) >= 3 else {}
    makespan = meta.get("makespan")
    all_delivered = meta.get("all_delivered", False)

    if not all_delivered or makespan is None:
        return _INF_FITNESS

    road_cost = _chrom_cost(chrom, sim)
    return (makespan, road_cost)


def crossover(
    c1: Chromosome,
    c2: Chromosome,
    rng: random.Random,
) -> tuple[Chromosome, Chromosome]:
    """Uniform crossover: each tile inherits its genes from one parent."""
    child1: Chromosome = {}
    child2: Chromosome = {}
    for pos in c1:
        if rng.random() < 0.5:
            child1[pos] = list(c1[pos])
            child2[pos] = list(c2[pos])
        else:
            child1[pos] = list(c2[pos])
            child2[pos] = list(c1[pos])
    return child1, child2


def mutate(
    chrom: Chromosome,
    sim: TrafficSim,
    road_rate: float = 0.02,
    cap_rate: float = 0.05,
    cap_down_rate: float = 0.05,
    rng: Optional[random.Random] = None,
) -> Chromosome:
    if rng is None:
        rng = random.Random()
    for pos, genes in chrom.items():
        if rng.random() < road_rate:
            genes[0] = 1 - genes[0]
        if genes[0] == 1:
            tile = sim.grid[pos[0]][pos[1]]
            terrain = sim._terrain_name(tile)
            max_cap = TERRAIN_MAX_CAPACITY.get(terrain, 1)
            if rng.random() < cap_rate:
                genes[1] = max(1, min(max_cap, genes[1] + rng.choice([-1, 1])))
            elif rng.random() < cap_down_rate:
                # Capacity-decrease only: prune over-built roads
                if genes[1] > 1:
                    genes[1] -= 1
                else:
                    genes[0] = 0  # already at cap=1, remove the tile entirely
    return chrom


def repair(chrom: Chromosome, sim: TrafficSim) -> Chromosome:
    """Enforce budget: reduce capacity (then remove) highest-cost tiles until within budget.
    Same to rule_based.build_network's trim strategy
    """
    if sim.initial_budget is None:
        return chrom

    total = sum(
        sim.calculate_road_cost(pos, genes[1])
        for pos, genes in chrom.items()
        if genes[0]
    )

    while total > sim.initial_budget:
        candidates = [
            (sim.calculate_road_cost(pos, genes[1]), pos)
            for pos, genes in chrom.items()
            if genes[0]
        ]
        if not candidates:
            break
        old_cost, pos = max(candidates)  # trim most expensive first
        genes = chrom[pos]
        if genes[1] > 1:
            genes[1] -= 1
            new_cost = sim.calculate_road_cost(pos, genes[1])
            total -= old_cost - new_cost
        else:
            genes[0] = 0
            total -= old_cost

    return chrom


def _tournament_select(
    population: list[Chromosome],
    fitnesses: list[int],
    k: int,
    rng: random.Random,
) -> Chromosome:
    """Tournament selection: sample k candidates, return copy of the fittest."""
    idxs = rng.sample(range(len(population)), min(k, len(population)))
    best = min(idxs, key=lambda i: fitnesses[i])
    return _copy_chrom(population[best])


# Main GA loop
def run_ga(
    sim: TrafficSim,
    pop_size: int = 50,# change in run.py
    generations: int = 100, # change in run.py
    crossover_rate: float = 0.8,
    road_mutation_rate: float = 0.02,
    cap_mutation_rate: float = 0.05,
    cap_down_mutation_rate: float = 0.05,
    tournament_k: int = 3,
    elitism: int = 2,
    route_solver: Optional[Callable] = None,
    seed_rng: Optional[int] = None,
    verbose: bool = True,
) -> tuple[list[dict], dict[int, int], dict]:
    """Run the genetic algorithm and return the best plan found.
    Fitness = (makespan, road_cost), primary = makespan, secondary = road_cost.
    Modifies sim in-place by applying the best chromosome's roads.
    """
    if route_solver is None:
        from solvers.rule_based import solve_routing as _rs
        route_solver = _rs
    rng = random.Random(seed_rng)

    # Initialise
    pop = initial_population(sim, pop_size, rng)
    fits = [fitness(sim, c, route_solver) for c in pop]

    best_idx = min(range(len(pop)), key=lambda i: fits[i])
    best_chrom = _copy_chrom(pop[best_idx])
    best_fit = fits[best_idx]
    history = [best_fit[0]]  # track makespan per generation

    if verbose:
        n_feasible = sum(1 for f in fits if f[0] < _INF_MAKESPAN)
        feasible_spans = [f[0] for f in fits if f[0] < _INF_MAKESPAN]
        mean_ms = sum(feasible_spans) / max(1, n_feasible)
        print(
            f"[ga] gen=  0  best={best_fit[0]}(cost={best_fit[1]})  "
            f"mean={mean_ms:.1f}  feasible={n_feasible}/{pop_size}",
            file=sys.stderr,
        )

    # Evolution
    for gen in range(1, generations + 1):
        # Elitism: always keep top `elitism` individuals unchanged
        sorted_idx = sorted(range(len(pop)), key=lambda i: fits[i])
        new_pop = [_copy_chrom(pop[i]) for i in sorted_idx[:elitism]]
        new_fits = [fits[i] for i in sorted_idx[:elitism]]

        # Fill rest of population via selection → crossover → mutate → repair
        while len(new_pop) < pop_size:
            p1 = _tournament_select(pop, fits, tournament_k, rng)
            p2 = _tournament_select(pop, fits, tournament_k, rng)

            if rng.random() < crossover_rate:
                c1, c2 = crossover(p1, p2, rng)
            else:
                c1, c2 = p1, p2  # already copies from tournament

            c1 = mutate(
                c1,
                sim,
                road_mutation_rate,
                cap_mutation_rate,
                cap_down_mutation_rate,
                rng,
            )
            c2 = mutate(
                c2,
                sim,
                road_mutation_rate,
                cap_mutation_rate,
                cap_down_mutation_rate,
                rng,
            )
            c1 = repair(c1, sim)
            c2 = repair(c2, sim)

            f1 = fitness(sim, c1, route_solver)
            f2 = fitness(sim, c2, route_solver)
            new_pop.extend([c1, c2])
            new_fits.extend([f1, f2])

        pop = new_pop[:pop_size]
        fits = new_fits[:pop_size]

        gen_best_idx = min(range(len(pop)), key=lambda i: fits[i])
        gen_best_fit = fits[gen_best_idx]

        if gen_best_fit < best_fit:
            best_fit = gen_best_fit
            best_chrom = _copy_chrom(pop[gen_best_idx])

        history.append(best_fit[0])

        if verbose and (gen % 10 == 0 or gen == 1):
            n_feasible = sum(1 for f in fits if f[0] < _INF_MAKESPAN)
            feasible_spans = [f[0] for f in fits if f[0] < _INF_MAKESPAN]
            mean_ms = sum(feasible_spans) / max(1, n_feasible)
            print(
                f"[ga] gen={gen:3d}  best={best_fit[0]}(cost={best_fit[1]})  "
                f"mean={mean_ms:.1f}  feasible={n_feasible}/{pop_size}",
                file=sys.stderr,
            )

    if verbose:
        print(
            f"[ga] done  best_makespan={best_fit[0]}  best_cost={best_fit[1]}",
            file=sys.stderr,
        )

    # Apply best network to sim, run route_solver one final time
    for pos, (rb, cap_g) in best_chrom.items():
        if rb:
            sim.add_road(pos, cap_g)

    solver_result = route_solver(sim)
    car_plan = solver_result[0]
    departures = solver_result[1]
    route_meta = solver_result[2] if len(solver_result) >= 3 else {}

    road_cost = sum(sim.calculate_road_cost(p, c) for p, c in sim.road_capacity.items())

    meta = {
        # Prefer the final route_solver's makespan; fall back to GA best_fit
        "makespan": route_meta.get("makespan", best_fit[0]),
        "road_cost": road_cost,
        "occ": route_meta.get("occ"),
        "solve_status": "HEURISTIC",
        "all_delivered": route_meta.get("all_delivered", best_fit[0] < _INF_MAKESPAN),
        "fitness_history": history,
    }
    return car_plan, departures, meta


# Joint-solver alias
def solve_joint(
    sim: TrafficSim,
    **kwargs,
) -> tuple[list[dict], dict[int, int], dict]:
    return run_ga(sim, **kwargs)

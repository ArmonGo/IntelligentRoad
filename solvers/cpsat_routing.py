from __future__ import annotations
import sys
import os
from collections import defaultdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from minisim import TrafficSim, TileType, DIR4
from solvers.rule_based import _all_shortest_road_paths
from ortools.sat.python import cp_model

def _road_adj(sim: TrafficSim, depot_pos: Pos) -> list[Pos]:
    """Road tiles orthogonally adjacent to a depot tile."""
    x, y = depot_pos
    result = []
    for dx, dy in DIR4:
        nb = (x + dx, y + dy)
        if sim._in_bounds(nb) and sim.grid[nb[0]][nb[1]] == TileType.ROAD:
            result.append(nb)
    return result

# CP-SAT routing solver 
def solve_cpsat_routing(
    sim: TrafficSim,
    time_limit_s: float = 60.0,
    T_max: Optional[int] = None,
) -> tuple[list[dict], dict[int, int], dict]:
    """Time-expanded flow CP-SAT routing solver on a fixed road network.
    Models aggregate flow over road tiles only (no path enumeration).
    OD assignment, path routing, and departure timing are all free variables.
    sim          : TrafficSim with roads already built
    time_limit_s : CP-SAT wall-clock time limit
    T_max        : time horizon (auto-estimated via BFS if None)
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
            [],{},
            {
            "makespan": None,
            "road_cost": 0,
            "occ": None,
            "solve_status": "INFEASIBLE",
            "all_delivered": False,
            },
        )

    N = sum(d.amount for d in depots_out)
    U = N  # upper bound: at most N cars on any tile

    # Road tiles only — much smaller than the full grid
    road_cells = [
        (x, y)
        for x in range(sim.w)
        for y in range(sim.h)
        if sim.grid[x][y] == TileType.ROAD
    ]
    road_set = set(road_cells)
    tile_cap = {pos: sim.road_capacity[pos] for pos in road_cells}
    # Source / demand adjacency (road tiles adjacent to each depot)
    sources = {
        d.id: {"adj": _road_adj(sim, d.pos), "stock": d.amount} for d in depots_out
    }
    demands = {
        d.id: {"adj": _road_adj(sim, d.pos), "demand": d.amount} for d in depots_in
    }

    for d in depots_out + depots_in:
        adj = (sources if d.kind == "out" else demands)[d.id]["adj"]
        if not adj:
            print(
                f"[cpsat_routing] WARNING: depot {d.id} ({d.kind}) at {d.pos} "
                "has no adjacent road tile — problem may be infeasible",
                file=sys.stderr,
            )

    # T_max from BFS max path length
    if T_max is None:
        max_path_len = 1
        for d_out in depots_out:
            for d_in in depots_in:
                paths = _all_shortest_road_paths(sim, d_out.pos, d_in.pos)
                if paths:
                    max_path_len = max(max_path_len, len(paths[0]))
        T_max = N + max_path_len + 10
    T = T_max

    # Build model
    model = cp_model.CpModel()

    # Time-expanded variables (road tiles only)
    occ: dict = {}
    wait: dict = {}
    flow: dict = {}
    spawn: dict = {}
    arrive: dict = {}

    for t in range(T + 1):
        for pos in road_cells:
            occ[(pos, t)] = model.NewIntVar(0, U, f"o_{pos[0]}_{pos[1]}_{t}")

    for t in range(T):
        for pos in road_cells:
            wait[(pos, t)] = model.NewIntVar(0, U, f"w_{pos[0]}_{pos[1]}_{t}")
            x, y = pos
            for dx, dy in DIR4:
                nb = (x + dx, y + dy)
                if nb in road_set:
                    flow[(pos, nb, t)] = model.NewIntVar(
                        0, U, f"f_{pos[0]}_{pos[1]}_{nb[0]}_{nb[1]}_{t}"
                    )

    for s_id, sdata in sources.items():
        for u in sdata["adj"]:
            for t in range(T):
                spawn[(s_id, u, t)] = model.NewIntVar(
                    0, U, f"sp_{s_id}_{u[0]}_{u[1]}_{t}"
                )

    for d_id, ddata in demands.items():
        for u in ddata["adj"]:
            for t in range(T):
                arrive[(u, d_id, t)] = model.NewIntVar(
                    0, U, f"ar_{u[0]}_{u[1]}_{d_id}_{t}"
                )

    # Initial condition: road starts empty
    for pos in road_cells:
        model.Add(occ[(pos, 0)] == 0)

    # Tile capacity respected at every tick
    for t in range(T + 1):
        for pos in road_cells:
            model.Add(occ[(pos, t)] <= tile_cap[pos])

    # Supply / demand satisfaction
    for s_id, sdata in sources.items():
        model.Add(
            sum(spawn[(s_id, u, t)] for u in sdata["adj"] for t in range(T))
            == sdata["stock"]
        )
    for d_id, ddata in demands.items():
        model.Add(
            sum(arrive[(u, d_id, t)] for u in ddata["adj"] for t in range(T))
            == ddata["demand"]
        )

    # Adjacency sets for conservation equations
    demand_adj_set: dict = defaultdict(list)
    for d_id, ddata in demands.items():
        for u in ddata["adj"]:
            demand_adj_set[u].append(d_id)

    source_adj_set: dict = defaultdict(list)
    for s_id, sdata in sources.items():
        for u in sdata["adj"]:
            source_adj_set[u].append(s_id)

    # Occupancy decomposition: occ[u,t] = wait[u,t] + Σ outflows + Σ arrivals
    for t in range(T):
        for pos in road_cells:
            x, y = pos
            out_flows = [
                flow[(pos, (x + dx, y + dy), t)]
                for dx, dy in DIR4
                if (x + dx, y + dy) in road_set
            ]
            arr_here = [arrive[(pos, d_id, t)] for d_id in demand_adj_set.get(pos, [])]
            model.Add(wait[(pos, t)] + sum(out_flows) + sum(arr_here) == occ[(pos, t)])

    # State transition: occ[u,t+1] = wait[u,t] + Σ inflows + Σ spawns
    for t in range(T):
        for pos in road_cells:
            x, y = pos
            in_flows = [
                flow[((x + dx, y + dy), pos, t)]
                for dx, dy in DIR4
                if (x + dx, y + dy) in road_set and ((x + dx, y + dy), pos, t) in flow
            ]
            spawns_here = [
                spawn[(s_id, pos, t)] for s_id in source_adj_set.get(pos, [])
            ]
            model.Add(
                occ[(pos, t + 1)] == wait[(pos, t)] + sum(in_flows) + sum(spawns_here)
            )

    # End condition all cars out by tick T
    for pos in road_cells:
        model.Add(occ[(pos, T)] == 0)

    # Makespan latest tick with any arrival
    makespan_var = model.NewIntVar(0, T, "makespan")
    for t in range(T):
        has_arr_t = model.NewBoolVar(f"ha_{t}")
        total_arr = sum(
            arrive[(u, d_id, t)]
            for d_id, ddata in demands.items()
            for u in ddata["adj"]
        )
        model.Add(total_arr >= 1).OnlyEnforceIf(has_arr_t)
        model.Add(total_arr == 0).OnlyEnforceIf(has_arr_t.Not())
        model.Add(makespan_var >= t).OnlyEnforceIf(has_arr_t)

    model.Minimize(makespan_var)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.log_search_progress = False
    solver.parameters.num_search_workers = 8
    solver.parameters.linearization_level = 2
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(
            f"[cpsat_routing] No solution (status={solver.StatusName(status)}), "
            "falling back to rule_based",
            file=sys.stderr,
        )
        from solvers.rule_based import solve_routing as rb_routing

        return rb_routing(sim)  # already returns 3-tuple

    # Extract solution, and aggregate occ directly from solver variables
    flow_makespan = int(solver.Value(makespan_var))
    solve_status = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
    print(
        f"[cpsat_routing] {solve_status if status == cp_model.OPTIMAL else 'FEASIBLE (timeout)'} "
        f"flow_makespan={flow_makespan}  road_tiles={len(road_cells)}  "
        f"wall={solver.WallTime():.2f}s",
        file=sys.stderr,
    )

    occ_result: dict = {}
    for t in range(T + 1):
        tick_occ = {
            pos: solver.Value(occ[(pos, t)])
            for pos in road_cells
            if solver.Value(occ[(pos, t)]) > 0
        }
        if tick_occ:
            occ_result[t] = tick_occ

    road_cost = sum(sim.calculate_road_cost(p, c) for p, c in sim.road_capacity.items())
    meta = {
        "makespan": flow_makespan + 1,
        "road_cost": road_cost,
        "occ": occ_result,
        "solve_status": solve_status,
        "all_delivered": True,
    }
    return [], {}, meta
solve_routing = solve_cpsat_routing

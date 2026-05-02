"""
TrafficSim solver interface contract
=====================================

All solvers live in this package and share a common interface.

Routing solver (network already built in sim):
-----------------------------------------------
    def solve_routing(sim: TrafficSim, **kwargs) -> tuple[list[dict], dict[int, int]]:
        '''
        Args:
            sim: TrafficSim with roads already set.  Must NOT modify roads or depots.
        Returns:
            car_plan   – list of dicts (see schema below)
            departures – {car_plan_index: depart_tick}
        '''

Joint solver (builds network then routes):
------------------------------------------
    def solve_joint(sim: TrafficSim, **kwargs) -> tuple[list[dict], dict[int, int]]:
        '''
        Args:
            sim: TrafficSim with terrain + depots; solver adds roads to sim.
        Returns:
            car_plan, departures  (same schema as above)
        '''

car_plan entry schema:
----------------------
    {
        "depot_out_id":  int,          # source depot id
        "depot_out_pos": (int, int),   # source depot grid position
        "depot_in_id":   int,          # destination depot id
        "path":          list[(int,int)]  # moves from depot_out_pos (exclusive)
                                          # to depot_in_pos (inclusive)
    }

    path[0] must be a ROAD or DEPOT_IN tile adjacent to depot_out_pos.
    path[-1] == depot_in.pos.

departures schema:
------------------
    {idx: tick}  where idx is the 0-based index into car_plan and tick >= 0.
    The sim spawns car idx+1 at the first step() call where tick_count >= tick.

Harness usage:
--------------
    from solvers.utils import run_solver, evaluate
    from solvers.rule_based import solve_joint

    results, solved_sim = run_solver(solve_joint, sim)
    # results: {"makespan": int, "road_cost": int, "all_delivered": bool,
    #           "timed_out": bool, "solve_time_s": float}
"""

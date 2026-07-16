from __future__ import annotations
import argparse
import copy
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Callable


def _load_dotenv() -> None:
    """Parse the nearest .env file and inject missing keys into os.environ."""
    search = Path(__file__).resolve().parent
    for _ in range(4):  # walk up at most 4 levels
        env_file = search / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
            break
        search = search.parent


_load_dotenv()


MODEL_MAPPING: dict[str, str] = {
    "deepseek/deepseek-v4-pro": "deepseek",
    "openai/gpt-5.5": "gpt",
    "qwen/qwen3.7-max": "qwen",
    "google/gemini-3.5-flash": "gemini",
    "anthropic/claude-fable-5": "claude",
    "z-ai/glm-5.2": "glm"
}

def _model_short(model: str) -> str:
    """Map full model ID to a short display name.

    Uses MODEL_MAPPING first; falls back to the last path segment with
    dots/colons/spaces normalised to dashes/underscores.
    """
    if model in MODEL_MAPPING:
        return MODEL_MAPPING[model]
    short = model.split("/")[-1]
    return short.replace(".", "-").replace(":", "-").replace(" ", "_")


def _agent_key(method_type: str, model_name: str) -> str:
    """Convert a generic agent-type name to its model-specific registry key.

    Replaces the literal word "Agent" with the short model name:
        "Agent_Joint"     + "deepseek" → "deepseek_Joint"
        "Heuristic_Agent" + "qwen"     → "Heuristic_qwen"
        "Agent_Agent"     + "gpt"      → "gpt_gpt"
    """
    return method_type.replace("Agent", model_name)


def _make_tag(method_key: str, map_name: str) -> str:
    """Build the base filename tag: <method_key>_<map_name>.

    method_key already encodes solver type and model
    (e.g. deepseek_Joint, Heuristic_CPSAT) so no separate model segment
    is needed.
    """
    return f"{method_key}_{map_name}"


def _make_log_prefix(method_key: str, map_name: str) -> str:
    """Build the agent-log filename prefix.

    Format:  <method_key>_<map_name>
    Example: deepseek_Joint_small_sufficient

    The timestamp is implicit from the parent run directory
    (results/runs/<run_id>/agent_logs/).
    """
    return f"{method_key}_{map_name}"


sys.path.insert(0, os.path.dirname(__file__))

from minisim import TrafficSim, import_map
from solvers import rule_based, cpsat_routing, cpsat_joint, ga_network, utils

# Optional agent import
try:
    from solvers import agent as agent_solver

    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False


def load_maps(size, budget) -> TrafficSim:
    path = f"./maps/ori_map_{size}_{budget}.pkl"
    sim = import_map(path)
    return sim


MAP_FACTORIES: dict[str, Callable[[], TrafficSim]] = {
    "small_sufficient": lambda: load_maps("small", "sufficient"),
    "small_tight": lambda: load_maps("small", "tight"),
    "medium_sufficient": lambda: load_maps("medium", "sufficient"),
    "medium_tight": lambda: load_maps("medium", "tight"),
    "large_sufficient": lambda: load_maps("large", "sufficient"),
    "large_tight": lambda: load_maps("large", "tight"),
}


def _build_solver_registry(
    model: str,
    agent_calls: int,
    route_calls: int,
    minimize_cost: bool,
    ga_pop: int,
    ga_gens: int,
    cpsat_time: float,
    verbose: bool,
    log_dir=None,
) -> dict[str, Callable]:
    model_name = _model_short(model)

    #  Non-agent solvers 

    def heuristic_joint(sim):
        return utils.run_solver(rule_based.solve_joint, sim)

    def cpsat_joint_fn(sim):
        return utils.run_solver(
            lambda s: cpsat_joint.solve_joint(
                s, time_limit_s=cpsat_time, minimize_cost=minimize_cost
            ),
            sim,
        )

    def heuristic_cpsat(sim):
        def _fn(s):
            rule_based.build_network(s)
            return cpsat_routing.solve_routing(s)

        return utils.run_solver(_fn, sim)

    def ga_heuristic(sim):
        return utils.run_solver(
            lambda s: ga_network.solve_joint(
                s,
                pop_size=ga_pop,
                generations=ga_gens,
                route_solver=rule_based.solve_routing,
                seed_rng=0,
                verbose=verbose,
            ),
            sim,
        )

    def ga_cpsat(sim):
        return utils.run_solver(
            lambda s: ga_network.solve_joint(
                s,
                pop_size=ga_pop,
                generations=ga_gens,
                route_solver=cpsat_routing.solve_routing,
                seed_rng=0,
                verbose=verbose,
            ),
            sim,
        )

    registry: dict[str, Callable] = {
        "Heuristic_Joint": heuristic_joint,
        "CPSAT_Joint": cpsat_joint_fn,
        "Heuristic_CPSAT": heuristic_cpsat,
        "GA_Heuristic": ga_heuristic,
        "GA_CPSAT": ga_cpsat,
    }

    if AGENT_AVAILABLE:
        #  Agent solvers (keys include model name) 

        def agent_joint(sim, log_prefix="", session_id=None, user=None):
            s = copy.deepcopy(sim)
            meta = agent_solver.solve_joint(
                s,
                model=model,
                max_tool_calls=agent_calls,
                log_dir=log_dir,
                log_prefix=log_prefix,
                session_id=session_id,
                user=user,
                verbose=verbose,
            )
            return meta, s

        def heuristic_agent(sim, log_prefix="", session_id=None, user=None):
            s = copy.deepcopy(sim)
            rule_based.build_network(s)
            meta = agent_solver.solve_routing(
                s,
                model=model,
                max_tool_calls=route_calls,
                log_dir=log_dir,
                log_prefix=log_prefix,
                session_id=session_id,
                user=user,
                verbose=verbose,
            )
            return meta, s

        def agent_heuristic(sim, log_prefix="", session_id=None, user=None):
            s = copy.deepcopy(sim)
            meta = agent_solver.solve_network(
                s,
                model=model,
                route_solver=rule_based.solve_routing,
                max_tool_calls=agent_calls,
                log_dir=log_dir,
                log_prefix=log_prefix,
                session_id=session_id,
                user=user,
                verbose=verbose,
            )
            return meta, s

        def agent_cpsat(sim, log_prefix="", session_id=None, user=None):
            s = copy.deepcopy(sim)
            meta = agent_solver.solve_network(
                s,
                model=model,
                route_solver=cpsat_routing.solve_routing,
                max_tool_calls=agent_calls,
                log_dir=log_dir,
                log_prefix=log_prefix,
                session_id=session_id,
                user=user,
                verbose=verbose,
            )
            return meta, s

        def agent_dual(sim, log_prefix="", session_id=None, user=None):
            s = copy.deepcopy(sim)
            meta = agent_solver.run_dual_agent(
                s,
                network_model=model,
                route_model=model,
                network_max_calls=int(agent_calls / 2),
                route_max_calls=int(route_calls / 2),
                log_dir=log_dir,
                log_prefix=log_prefix,
                session_id=session_id,
                user=user,
                verbose=verbose,
            )
            return meta, s

        registry.update(
            {
                f"{model_name}_Joint": agent_joint,
                f"Heuristic_{model_name}": heuristic_agent,
                f"{model_name}_Heuristic": agent_heuristic,
                f"{model_name}_CPSAT": agent_cpsat,
                f"{model_name}_{model_name}": agent_dual,
            }
        )

    return registry


SOLVER_DESCRIPTIONS = {
    # Non-agent
    "Heuristic_Joint": "Rule-based joint (network + routing)",
    "CPSAT_Joint": "CP-SAT joint (time-expanded flow model)",
    "Heuristic_CPSAT": "Rule-based network + CP-SAT routing",
    "GA_Heuristic": "GA network + rule-based routing",
    "GA_CPSAT": "GA network + CP-SAT routing",
    # Agent (generic type names)
    "Agent_Joint": "LLM designs network + routes                [AGENT]",
    "Heuristic_Agent": "Rule-based network + LLM routing            [AGENT]",
    "Agent_Heuristic": "LLM network + rule-based routing            [AGENT]",
    "Agent_CPSAT": "LLM network + CP-SAT routing                [AGENT]",
    "Agent_Agent": "Dual-agent (LLM network + LLM routing)      [AGENT]",
}

NON_AGENT_SOLVERS = [
    "Heuristic_Joint",
    "CPSAT_Joint",
    "Heuristic_CPSAT",
    "GA_Heuristic",
    "GA_CPSAT",
]
AGENT_SOLVER_TYPES = [
    "Agent_Joint",
    "Heuristic_Agent",
    "Agent_Heuristic",
    "Agent_CPSAT",
    "Agent_Agent",
]
ALL_SOLVERS = NON_AGENT_SOLVERS + AGENT_SOLVER_TYPES

def _make_meta_serialisable(obj):
    """Recursively convert non-JSON-serialisable types.

    Handles tuple keys (e.g. (x,y) in occ dicts) and tuple values.
    """
    if isinstance(obj, dict):
        return {
            (str(k) if isinstance(k, tuple) else k): _make_meta_serialisable(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_make_meta_serialisable(i) for i in obj]
    return obj


def save_result(
    run_dir: Path, method_key: str, map_name: str, meta: dict, sim: TrafficSim
) -> None:
    tag = _make_tag(method_key, map_name)
    # Save meta as JSON
    meta_path = run_dir / f"{tag}.json"
    with open(meta_path, "w") as f:
        json.dump(_make_meta_serialisable(meta), f, indent=2, default=str)
    # Save sim state as pickle
    sim_path = run_dir / f"{tag}.pkl"
    with open(sim_path, "wb") as f:
        pickle.dump(sim, f)


def load_results(run_dir: str | Path) -> dict[str, dict]:
    run_dir = Path(run_dir)
    results = {}
    for json_path in sorted(run_dir.glob("*.json")):
        if json_path.stem in ("config", "summary"):
            continue
        tag = json_path.stem
        pkl_path = run_dir / f"{tag}.pkl"
        with open(json_path) as f:
            meta = json.load(f)
        sim = None
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                sim = pickle.load(f)
        results[tag] = {"meta": meta, "sim": sim}
    return results

def run_experiments(
    maps: list[str],
    solvers: list[str],
    output_dir: str = "results/runs",
    models: list[str] | None = None,
    agent_calls: int = 20,
    route_calls: int = 8,
    minimize_cost: bool = True,
    ga_pop: int = 30,
    ga_gens: int = 50,
    cpsat_time: float = 60.0,
    verbose: bool = True,
) -> Path:
    if models is None:
        models = ["deepseek/deepseek-v3.2"]

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id,
        "maps": maps,
        "solvers": solvers,
        "models": models,
        "agent_calls": agent_calls,
        "route_calls": route_calls,
        "minimize_cost": minimize_cost,
        "ga_pop": ga_pop,
        "ga_gens": ga_gens,
        "cpsat_time_s": cpsat_time,
        "timestamp": run_id,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    agent_log_dir = run_dir / "agent_logs"
    agent_log_dir.mkdir(exist_ok=True)

    # Build one non-agent registry (model param unused for these) and one
    # per-model agent registry.
    base_registry = _build_solver_registry(
        model=models[0],
        agent_calls=agent_calls,
        route_calls=route_calls,
        minimize_cost=minimize_cost,
        ga_pop=ga_pop,
        ga_gens=ga_gens,
        cpsat_time=cpsat_time,
        verbose=verbose,
        log_dir=None,
    )

    model_registries: dict[str, dict] = {}
    if AGENT_AVAILABLE:
        for m in models:
            if m not in model_registries:
                model_registries[m] = _build_solver_registry(
                    model=m,
                    agent_calls=agent_calls,
                    route_calls=route_calls,
                    minimize_cost=minimize_cost,
                    ga_pop=ga_pop,
                    ga_gens=ga_gens,
                    cpsat_time=cpsat_time,
                    verbose=verbose,
                    log_dir=str(agent_log_dir),
                )

    summary: list[dict] = []

    for map_name in maps:
        sim_factory = MAP_FACTORIES[map_name]
        for method in solvers:
            # Determine (registry, actual_key, model) triples to run.
            # Non-agent: single run with static key.
            # Agent: one run per model, with model-specific key.
            if method in NON_AGENT_SOLVERS:
                if method not in base_registry:
                    print(f"[run] skipping {method} (not in registry)")
                    continue
                run_triples = [(base_registry, method, None)]
            else:
                if not AGENT_AVAILABLE:
                    print(f"[run] skipping {method} (agent package missing)")
                    continue
                run_triples = [
                    (model_registries[m], _agent_key(method, _model_short(m)), m)
                    for m in models
                ]

            for registry, actual_key, model in run_triples:
                label = f"{actual_key}" if model else actual_key
                print(f"\n[run] ===== {label} on {map_name} =====")
                sim_orig = sim_factory()
                t0 = time.time()
                try:
                    if model:
                        log_prefix = _make_log_prefix(actual_key, map_name)
                        # session_id: per-session cost on openrouter.ai/activity
                        # user: run-level grouping across all sessions in this run
                        session_id = f"{run_id}/{actual_key}_{map_name}"
                        fn = registry[actual_key]
                        result = fn(
                            sim_orig,
                            log_prefix=log_prefix,
                            session_id=session_id,
                            user=run_id,
                        )
                    else:
                        fn = registry[actual_key]
                        result = fn(sim_orig)
                    if isinstance(result, tuple) and len(result) == 2:
                        meta, solved_sim = result
                    else:
                        meta, solved_sim = result, sim_orig
                except Exception as exc:
                    print(f"[run] ERROR in {label}/{map_name}: {exc}")
                    meta = {
                        "solve_status": "ERROR",
                        "error": str(exc),
                        "makespan": None,
                        "all_delivered": False,
                    }
                    solved_sim = sim_orig
                elapsed = time.time() - t0
                meta.setdefault("solve_time_s", round(elapsed, 2))
                if model:
                    meta["model"] = model

                save_result(run_dir, actual_key, map_name, meta, solved_sim)
                summary.append(
                    {
                        "method": actual_key,
                        "map": map_name,
                        "model": model,
                        "makespan": meta.get("makespan"),
                        "road_cost": meta.get("road_cost"),
                        "all_delivered": meta.get("all_delivered"),
                        "solve_status": meta.get("solve_status"),
                        "solve_time_s": meta.get("solve_time_s"),
                        "tool_calls_cnt": meta.get("tool_calls_cnt"),
                        "inner_route_runs": meta.get("inner_route_runs"),
                        "route_calls_per_run": meta.get("route_calls_per_run"),
                        "total_tool_calls": meta.get("total_tool_calls"),
                    }
                )
                calls_info = ""
                if meta.get("tool_calls_cnt") is not None:
                    calls_info = f"  tool_calls={meta['tool_calls_cnt']}"
                    if meta.get("inner_route_runs") is not None:
                        calls_info += f"(net)+{meta['inner_route_runs']}route_runs"
                print(
                    f"[run] done  makespan={meta.get('makespan')}  "
                    f"cost={meta.get('road_cost')}  "
                    f"delivered={meta.get('all_delivered')}  "
                    f"time={elapsed:.1f}s{calls_info}"
                )

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[run] results saved to {run_dir}")
    return run_dir

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TrafficSim solver benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--map",
        nargs="+",
        default=["small_sufficient"],
        choices=list(MAP_FACTORIES) + ["all"],
        help="Map(s) to run on (default: small_sufficient)",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=NON_AGENT_SOLVERS,
        help="Solver(s) to run, or 'all'/'non-agent'/'agent'. "
        "Use --list to see all names.",
    )
    parser.add_argument(
        "--output",
        default="results/runs",
        help="Output base directory (default: results/runs)",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=["deepseek/deepseek-v3.2"],
        help="LLM model ID(s) for agent solvers — agent solvers run once "
        "per model; non-agent solvers run once regardless. "
        "Agent_Agent always uses the same model for both agents. "
        "Example: --model deepseek/deepseek-v3.2 anthropic/claude-sonnet-4.6",
    )
    parser.add_argument(
        "--agent-calls",
        type=int,
        default=20,
        help="max_tool_calls for network/joint agent (default: 20)",
    )
    parser.add_argument(
        "--route-calls",
        type=int,
        default=10,
        help="max_tool_calls for routing agent (default: 10)",
    )
    parser.add_argument(
        "--no-minimize-cost",
        action="store_true",
        help="Disable secondary cost minimization in CP-SAT / agent",
    )
    parser.add_argument(
        "--ga-pop", type=int, default=30, help="GA population size (default: 30)"
    )
    parser.add_argument(
        "--ga-gens", type=int, default=50, help="GA generations (default: 50)"
    )
    parser.add_argument(
        "--cpsat-time",
        type=float,
        default=300.0,
        help="CP-SAT time limit in seconds (default: 300)",
    )  # 600 for large map
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-step verbose output"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available solvers and exit"
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable solvers:")
        for name, desc in SOLVER_DESCRIPTIONS.items():
            avail = (
                "" if (AGENT_AVAILABLE or "[AGENT]" not in desc) else "  [unavailable]"
            )
            print(f"  {name:<20} {desc}{avail}")
        print("\nModel mapping (for short names in filenames):")
        for full, short in MODEL_MAPPING.items():
            print(f"  {full}  →  {short}")
        return

    # Expand shorthand solver lists
    solver_list: list[str] = []
    for s in args.solvers:
        if s == "all":
            solver_list = ALL_SOLVERS
            break
        elif s == "non-agent":
            solver_list.extend(NON_AGENT_SOLVERS)
        elif s == "agent":
            solver_list.extend(AGENT_SOLVER_TYPES)
        elif s == "agent_dual":
            solver_list.append("Agent_Agent")
        else:
            solver_list.append(s)
    solver_list = list(dict.fromkeys(solver_list))  # deduplicate, preserve order

    # Expand map list
    map_list = list(MAP_FACTORIES) if "all" in args.map else args.map

    run_experiments(
        maps=map_list,
        solvers=solver_list,
        output_dir=args.output,
        models=args.model,
        agent_calls=args.agent_calls,
        route_calls=args.route_calls,
        minimize_cost=not args.no_minimize_cost,
        ga_pop=args.ga_pop,
        ga_gens=args.ga_gens,
        cpsat_time=args.cpsat_time,
        verbose=not args.quiet,
    )

if __name__ == "__main__":
    main()

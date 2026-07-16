from __future__ import annotations
import copy
import json
import os
import sys
import time
from collections import deque, defaultdict
from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np
import io as _io

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from minisim import TrafficSim, TileType, DIR4, Pos
from solvers.utils import world_to_json, evaluate, validate_plan

DEFAULT_MODEL = "openrouter/deepseek/deepseek-v3.2"
DEFAULT_MAX_CALLS = 15  # safety ceiling on tool calls per agent run
DEFAULT_MAX_TOKENS = 16384


def _build_client(api_key: Optional[str] = None):
    """Return an OpenAI-compatible client pointed at OpenRouter."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai package required: pip install openai") from exc

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY"),
    )


def _call_llm(
    client,
    model: str,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    retries: int = 3,
    session_id: Optional[str] = None,
    user: Optional[str] = None,
):
    """Call the LLM API with optional OpenRouter trace fields.

    session_id : groups all turns of one agent session - filter on
                 openrouter.ai/activity by session to see per-session cost.
    user       : broader grouping (e.g. run_id) - lets you see cost for a
                 whole experiment run across all its sessions.
    """
    delay = 2.0
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if user:
        kwargs["user"] = user
    if session_id:
        # session_id is OpenRouter-specific; pass via extra_body so the
        # OpenAI SDK forwards it without rejecting it as an unknown param.
        kwargs["extra_body"] = {"session_id": session_id}

    for attempt in range(retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            if "rate" in msg or "429" in msg:
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
            raise
    raise RuntimeError("LLM call failed after retries")


def _apply_roads(sim: TrafficSim, roads_list: list[dict]) -> list[str]:
    """Apply a list of {"pos": [x,y], "capacity": int} dicts to sim."""
    errors: list[str] = []
    if not isinstance(roads_list, list):
        return ["roads must be a list"]
    for i, r in enumerate(roads_list):
        try:
            pos = tuple(r["pos"])
            cap = int(r.get("capacity", 1))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"roads[{i}]: bad format {exc}, returned: {r}")
            continue
        ok, reason = sim.add_road(pos, cap)
        if not ok:
            errors.append(f"roads[{i}] {list(pos)} cap={cap}: {reason}")
    return errors


def _check_connectivity(sim: TrafficSim) -> list[str]:
    depots_out = [d for d in sim.init_depots.values() if d.kind == "out"]
    depots_in = [d for d in sim.init_depots.values() if d.kind == "in"]
    errors: list[str] = []

    for d_out in depots_out:
        visited: set[Pos] = set()
        queue: deque[Pos] = deque()
        for dx, dy in DIR4:
            nb = (d_out.pos[0] + dx, d_out.pos[1] + dy)
            if 0 <= nb[0] < sim.w and 0 <= nb[1] < sim.h and nb not in visited:
                tile = sim.grid[nb[0]][nb[1]]
                if tile in (TileType.ROAD, TileType.DEPOT_IN):
                    visited.add(nb)
                    queue.append(nb)
        while queue:
            cur = queue.popleft()
            if sim.grid[cur[0]][cur[1]] == TileType.DEPOT_IN:
                continue
            for dx, dy in DIR4:
                nb = (cur[0] + dx, cur[1] + dy)
                if nb in visited or not (0 <= nb[0] < sim.w and 0 <= nb[1] < sim.h):
                    continue
                t = sim.grid[nb[0]][nb[1]]
                if t in (TileType.ROAD, TileType.DEPOT_IN):
                    visited.add(nb)
                    queue.append(nb)
        for d_in in depots_in:
            reachable = d_in.pos in visited or any(
                (d_in.pos[0] + dx, d_in.pos[1] + dy) in visited for dx, dy in DIR4
            )
            if not reachable:
                errors.append(
                    f"depot_in {d_in.id} at {list(d_in.pos)} not reachable "
                    f"from depot_out {d_out.id} at {list(d_out.pos)}"
                )
    return errors


def _budget_status(sim: TrafficSim) -> dict:
    used = sum(sim.calculate_road_cost(p, c) for p, c in sim.road_capacity.items())
    limit = sim.initial_budget
    return {
        "used": used,
        "limit": limit,
        "ok": limit is None or used <= limit,
        "over_by": max(0, used - limit) if limit is not None else 0,
    }


# Car plan
def _parse_agent_car_plan(
    parsed: dict,
    sim: TrafficSim,
) -> tuple[list[dict], dict[int, int], list[str]]:
    """Convert agent car plan JSON to tuple (car_plan, departures, errors)."""
    errors: list[str] = []
    car_plan: list[dict] = []
    departures: dict[int, int] = {}
    cars = parsed.get("cars", [])
    if not isinstance(cars, list):
        return [], {}, ["'cars' must be a list"]

    depots_out = {d.id: d for d in sim.init_depots.values() if d.kind == "out"}
    depots_in = {d.id: d for d in sim.init_depots.values() if d.kind == "in"}
    assigned_out: dict[int, int] = defaultdict(int)
    assigned_in: dict[int, int] = defaultdict(int)

    for i, c in enumerate(cars):
        try:
            from_id = int(c["from"])
            to_id = int(c["to"])
            path = [tuple(p) for p in c["path"]]
            depart = int(c.get("depart", 0))
        # format error
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"cars[{i}]: bad format {exc}, returned: {c}")
            continue
        # wrong depot out id
        if from_id not in depots_out:
            errors.append(f"cars[{i}]: unknown depot_out id {from_id}")
            continue
        # not destination
        if to_id not in depots_in:
            errors.append(f"cars[{i}]: unknown depot_in id {to_id}")
            continue
        d_out = depots_out[from_id]
        d_in = depots_in[to_id]
        assigned_out[from_id] += 1
        assigned_in[to_id] += 1
        # no path
        if not path:
            errors.append(f"cars[{i}]: empty path")
            continue
        # not closed path
        if path[-1] != d_in.pos:
            errors.append(f"cars[{i}]: path ends at {path[-1]}, expected {d_in.pos}")
        for step, pos in enumerate(path[:-1]):
            if not (0 <= pos[0] < sim.w and 0 <= pos[1] < sim.h):
                errors.append(f"cars[{i}]: step {step} {pos} out of bounds")
                break
            if sim.grid[pos[0]][pos[1]] != TileType.ROAD:
                errors.append(
                    f"cars[{i}]: step {step} {pos} is not ROAD "
                    f"(is {sim.grid[pos[0]][pos[1]].name})"
                )
                break
        seq = [d_out.pos] + path
        for a, b in zip(seq, seq[1:]):
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
                errors.append(
                    f"cars[{i}]: non-adjacent step {a} -> {b} "
                    f"(no teleport; move one tile up/down/left/right)"
                )
                break
        idx = len(car_plan)
        car_plan.append(
            {
                "depot_out_id": from_id,
                "depot_out_pos": d_out.pos,
                "depot_in_id": to_id,
                "path": list(path),
            }
        )
        departures[idx] = depart

    total_supply = sum(d.amount for d in depots_out.values())
    if len(cars) != total_supply:
        errors.append(f"car count {len(cars)} != total supply {total_supply}")
    for d_id, d in depots_out.items():
        if assigned_out[d_id] != d.amount:
            errors.append(
                f"depot_out {d_id}: {assigned_out[d_id]} cars assigned but supply={d.amount}"
            )
    for d_id, d in depots_in.items():
        if assigned_in[d_id] != d.amount:
            errors.append(
                f"depot_in {d_id}: {assigned_in[d_id]} cars assigned but demand={d.amount}"
            )
    return car_plan, departures, errors


_TOOL_GET_WORLD: dict = {
    "type": "function",
    "function": {
        "name": "get_world_json",
        "description": (
            "Get the full map description: grid dimensions, terrain, depot positions "
            "and amounts, currently built roads, and terrain cost/capacity rules."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

# Specific the optimization phase. First minimise makespan then budget
_PHASE_PROP: dict = {
    "type": "string",
    "enum": ["minimize_makespan", "minimize_cost"],
    "description": (
        "Your current optimisation goal. Use 'minimize_makespan' until you "
        "believe makespan cannot improve further, then switch to 'minimize_cost' to "
        "find a cheaper network with the SAME or BETTER makespan."
    ),
}

_ROADS_PROP: dict = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "pos": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
            },
            "capacity": {"type": "integer", "minimum": 1},
        },
        "required": ["pos", "capacity"],
    },
    "description": "List of road tiles (location and capacity) to build.",
}

_CARS_PROP: dict = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "from": {"type": "integer", "description": "depot_out id"},
            "to": {"type": "integer", "description": "depot_in id"},
            "path": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
                "description": "Ordered [x,y] positions from first road tile to depot_in (inclusive).",
            },
            "depart": {"type": "integer", "description": "Departure tick (≥ 0)."},
        },
        "required": ["from", "to", "path", "depart"],
    },
    "description": "One entry per car (total = sum of all depot_out amounts).",
}

_TOOL_PROPOSE_NETWORK: dict = {
    "type": "function",
    "function": {
        "name": "propose_network",
        "description": (
            "Propose a road network. The system applies the roads, checks connectivity "
            "and budget, then runs a route solver to obtain makespan. "
            "Returns makespan, road_cost, and any errors."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "roads": _ROADS_PROP,
                "phase": _PHASE_PROP,
            },
            "required": ["roads"],
        },
    },
}

_TOOL_PROPOSE_PLAN: dict = {
    "type": "function",
    "function": {
        "name": "propose_plan",
        "description": (
            "Propose a complete road network AND car routes. The system applies the "
            "roads, validates the car paths, then runs a full simulation replay to "
            "obtain the actual makespan (no approximation). "
            "Returns makespan, road_cost, and any errors."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "roads": _ROADS_PROP,
                "cars": _CARS_PROP,
                "phase": _PHASE_PROP,
            },
            "required": ["roads", "cars"],
        },
    },
}

_TOOL_PROPOSE_ROUTES: dict = {
    "type": "function",
    "function": {
        "name": "propose_routes",
        "description": (
            "Propose car routes on the already-built road network. The system validates "
            "paths and runs a full simulation replay to obtain the actual makespan. "
            "Returns makespan and any validation errors."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cars": _CARS_PROP,
                "phase": _PHASE_PROP,
            },
            "required": ["cars"],
        },
    },
}

_TOOL_SUBMIT_NETWORK: dict = {
    "type": "function",
    "function": {
        "name": "submit_network",
        "description": (
            "Submit your final road network as the committed answer. "
            "Pass the exact roads you want to commit. The system evaluates them "
            "and records the result as your final answer. "
            "Use this instead of a separate propose_network call when you are done exploring."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "roads": _ROADS_PROP,
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why this solution was chosen.",
                },
            },
            "required": ["roads"],
        },
    },
}

_TOOL_SUBMIT_PLAN: dict = {
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": (
            "Submit your final road network AND car routes as the committed answer. "
            "Pass the exact roads and cars you want to commit — the system runs a full "
            "simulation replay and records the result as your final answer. "
            "Use this instead of a separate propose_plan call when you are done exploring."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "roads": _ROADS_PROP,
                "cars": _CARS_PROP,
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why this solution was chosen.",
                },
            },
            "required": ["roads", "cars"],
        },
    },
}

_TOOL_SUBMIT_ROUTES: dict = {
    "type": "function",
    "function": {
        "name": "submit_routes",
        "description": (
            "Submit your final car routes as the committed answer. "
            "Pass the exact car paths you want to commit — the system runs a full "
            "simulation replay and records the result as your final answer. "
            "Use this instead of a separate propose_routes call when you are done exploring."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cars": _CARS_PROP,
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why this solution was chosen.",
                },
            },
            "required": ["cars"],
        },
    },
}

# Names that trigger final-commit logic in the loop
_SUBMIT_NAMES = {"submit_network", "submit_plan", "submit_routes"}

# Tool sets per agent mode
_TOOLS_NETWORK = [_TOOL_GET_WORLD, _TOOL_PROPOSE_NETWORK, _TOOL_SUBMIT_NETWORK]
_TOOLS_JOINT = [_TOOL_GET_WORLD, _TOOL_PROPOSE_PLAN, _TOOL_SUBMIT_PLAN]
_TOOLS_ROUTING = [_TOOL_GET_WORLD, _TOOL_PROPOSE_ROUTES, _TOOL_SUBMIT_ROUTES]


# Tool executors


def _make_world_executor(world_json: dict) -> Callable:
    def _exec(_: dict) -> dict:
        return world_json

    return _exec


def _make_network_executor(sim: TrafficSim, route_solver_fn: Callable) -> Callable:
    """Returns a tool executor for propose_network."""

    def _exec(args: dict) -> dict:
        roads = args.get("roads", [])
        phase = args.get("phase", "minimize_makespan")

        sim_copy = copy.deepcopy(sim)  # copy sim forevery propose
        road_errors = _apply_roads(sim_copy, roads)
        connectivity_errors = _check_connectivity(sim_copy)
        budget = _budget_status(sim_copy)

        # initialization
        makespan = None
        all_delivered = False
        car_plan = []
        departures = {}
        occ = None

        if not connectivity_errors and budget["ok"]:
            _old_stderr = sys.stderr
            sys.stderr = _io.StringIO()
            try:
                result = route_solver_fn(sim_copy)
                car_plan = result[0]
                departures = result[1]
                meta = result[2] if len(result) >= 3 else {}
                makespan = meta.get("makespan")
                all_delivered = meta.get("all_delivered", False)
                occ = meta.get("occ")
            except Exception:
                pass
            finally:
                sys.stderr = _old_stderr

        llm_result = {
            "makespan": makespan,
            "road_cost": budget["used"],
            "all_delivered": all_delivered,
            "budget_ok": budget["ok"],
            "budget_limit": budget["limit"],
            "budget_over_by": budget["over_by"],
            "connectivity_errors": connectivity_errors[:5],
            "road_errors": road_errors[:5],
        }
        # Tracking agent propose
        llm_result["_internal"] = {
            "car_plan": car_plan,
            "departures": departures,
            "occ": occ,
            "roads": roads,
            "phase": phase,
        }
        return llm_result

    return _exec


def _make_plan_executor(sim: TrafficSim) -> Callable:
    """Returns a tool executor for propose_plan (joint: roads + car routes, replay)."""

    def _exec(args: dict) -> dict:
        roads = args.get("roads", [])
        cars_raw = args.get("cars", [])
        phase = args.get("phase", "minimize_makespan")

        sim_copy = copy.deepcopy(sim)
        road_errors = _apply_roads(sim_copy, roads)
        connectivity_errors = _check_connectivity(sim_copy)
        budget = _budget_status(sim_copy)
        car_plan, departures, val_errors = _parse_agent_car_plan(
            {"cars": cars_raw}, sim_copy
        )
        makespan = None
        all_delivered = False

        if (
            not road_errors
            and not connectivity_errors
            and budget["ok"]
            and not val_errors
        ):
            sim_copy.car_plan = car_plan
            sim_copy.saved_departures = departures
            ev = evaluate(sim_copy)
            makespan = ev.get("makespan")
            all_delivered = ev.get("all_delivered", False)

        llm_result = {
            "makespan": makespan,
            "road_cost": budget["used"],
            "all_delivered": all_delivered,
            "budget_ok": budget["ok"],
            "budget_limit": budget["limit"],
            "budget_over_by": budget["over_by"],
            "connectivity_errors": connectivity_errors[:5],
            "road_errors": road_errors[:5],
            "validation_errors": val_errors[:5],
        }
        llm_result["_internal"] = {
            "car_plan": car_plan,
            "departures": departures,
            "occ": None,  # replay doesn't produce aggregate occ
            "roads": roads,
            "phase": phase,
        }
        return llm_result

    return _exec


def _make_routes_executor(sim: TrafficSim) -> Callable:
    """Returns a tool executor for propose_routes (routing-only, replay)."""
    road_cost = sum(sim.calculate_road_cost(p, c) for p, c in sim.road_capacity.items())

    def _exec(args: dict) -> dict:
        cars_raw = args.get("cars", [])
        phase = args.get("phase", "minimize_makespan")
        car_plan, departures, val_errors = _parse_agent_car_plan(
            {"cars": cars_raw}, sim
        )
        makespan = None
        all_delivered = False
        if not val_errors:
            sim_copy = copy.deepcopy(sim)
            sim_copy.car_plan = car_plan
            sim_copy.saved_departures = departures
            ev = evaluate(sim_copy)
            makespan = ev.get("makespan")
            all_delivered = ev.get("all_delivered", False)
        llm_result = {
            "makespan": makespan,
            "road_cost": road_cost,
            "all_delivered": all_delivered,
            "validation_errors": val_errors[:5],
        }
        llm_result["_internal"] = {
            "car_plan": car_plan,
            "departures": departures,
            "occ": None,
            "roads": list(
                {"pos": list(p), "capacity": c} for p, c in sim.road_capacity.items()
            ),
            "phase": phase,
        }
        return llm_result

    return _exec


# Best-result tracker
def update_best(best: dict, llm_result: dict, verbose: bool) -> None:
    """Update best solution in-place if the new result is better."""
    makespan = llm_result.get("makespan")
    road_cost = llm_result.get("road_cost")
    all_delivered = llm_result.get("all_delivered", False)
    internal = llm_result.get("_internal", {})
    phase = internal.get("phase", "minimize_makespan")

    if not all_delivered or makespan is None:
        return
    is_better = (
        best["makespan"] is None
        or makespan < best["makespan"]
        or (
            phase == "minimize_cost"
            and makespan == best["makespan"]
            and road_cost is not None
            and (best["road_cost"] is None or road_cost < best["road_cost"])
        )
    )
    if is_better:
        best["makespan"] = makespan
        best["road_cost"] = road_cost
        best["all_delivered"] = True
        best["car_plan"] = internal.get("car_plan", [])
        best["departures"] = internal.get("departures", {})
        best["roads"] = internal.get("roads", [])
        best["occ"] = internal.get("occ")
        if verbose:
            cost_tag = f"  cost={road_cost}" if phase == "minimize_cost" else ""
            print(f"[agent] new best makespan={makespan}{cost_tag} phase={phase}")


def update_last_proposal(last_proposal: dict, llm_result: dict) -> None:
    """Update last_proposal with the most recent tool result, valid or not."""
    internal = llm_result.get("_internal", {})
    last_proposal["makespan"] = llm_result.get("makespan")
    last_proposal["road_cost"] = llm_result.get("road_cost")
    last_proposal["all_delivered"] = llm_result.get("all_delivered", False)
    last_proposal["car_plan"] = internal.get("car_plan", [])
    last_proposal["departures"] = internal.get("departures", {})
    last_proposal["roads"] = internal.get("roads", [])
    last_proposal["occ"] = internal.get("occ")


# Core agent loop

def _run_agent_loop(
    sim: TrafficSim,
    model: str,
    client,
    tools: list[dict],
    executors: dict[str, Callable],
    system_prompt: str,
    max_tool_calls: int,
    verbose: bool,
    log_dir: Optional[str],
    log_prefix: str = "",
    debug: bool = False,
    session_id: Optional[str] = None,
    user: Optional[str] = None,
) -> tuple[dict, list[dict]]:
    best: dict = {
        "makespan": None,
        "road_cost": None,
        "all_delivered": False,
        "car_plan": [],
        "departures": {},
        "roads": [],
        "occ": None,
    }
    last_proposal: dict = {
        "makespan": None,
        "road_cost": None,
        "all_delivered": False,
        "car_plan": [],
        "departures": {},
        "roads": [],
        "occ": None,
    }
    submitted: dict = {}  # filled when agent calls submit_solution
    logs: list[dict] = []

    # Open the log file immediately so events are written live
    _log_fh = None
    if log_dir:
        try:
            _log_path = Path(log_dir)
            _log_path.mkdir(parents=True, exist_ok=True)
            _log_fname = _log_path / f"{log_prefix}.jsonl"
            _log_fh = open(_log_fname, "a", encoding="utf-8")
            _log_fh.write(
                json.dumps(
                    {
                        "type": "session_start",
                        "model": model,
                        "prefix": log_prefix,
                        "max_tool_calls": max_tool_calls,
                        "ts": time.time(),
                    }
                )
                + "\n"
            )
            _log_fh.flush()
        except Exception:
            _log_fh = None  # logging failure must never break the solver

    def _log(entry: dict) -> None:
        logs.append(entry)
        if _log_fh:
            try:
                _log_fh.write(json.dumps(entry, default=str) + "\n")
                _log_fh.flush()
            except Exception:
                pass

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Start by calling get_world_json to inspect the map, then propose "
                "solutions using the available tools. Improve iteratively until "
                "satisfied, then stop calling tools.\n\n"
                "Map summary: "
                f"{sim.w}×{sim.h} grid, "
                f"{sum(d.amount for d in sim.init_depots.values() if d.kind == 'out')} cars total, "
                f"budget={'unlimited' if sim.initial_budget is None else sim.initial_budget}."
            ),
        },
    ]

    tool_calls_made = 0
    done = False  # set to True when agent calls a submit_* tool
    aborted = False  # set to True when the session is force-ended (e.g. bad submits)
    no_tool_streak = 0  # consecutive LLM turns with no tool calls
    _MAX_NO_TOOL = 5  # give up after this many consecutive reasoning-only turns
    submit_format_fails = 0  # count of malformed/empty submit_* attempts
    _MAX_SUBMIT_FAILS = 3  # give up after this many bad submits
    _final_warned = False  # True once the "last chance" warning has been injected
    world_inspected = False  # True once the agent has called get_world_json

    # Build per-tool format hints + required-field lists from the schemas passed
    # in, for format reminders and submit validation.
    _tool_hints: dict[str, str] = {}
    _tool_required: dict[str, list[str]] = {}
    for _t in tools:
        _fn = _t.get("function", {})
        _tname = _fn.get("name", "")
        _params = _fn.get("parameters", {})
        _required = _params.get("required", [])
        _tool_required[_tname] = list(_required)
        _props = _params.get("properties", {})
        _hints = []
        for _f in _required:
            _desc = _props.get(_f, {}).get("description", "")[:60]
            _hints.append(f'"{_f}" ({_desc})')
        _tool_hints[_tname] = ", ".join(_hints) if _hints else "(no required fields)"

    while tool_calls_made < max_tool_calls and not done and not aborted:
        # Inject a "last chance" warning while a few calls still remain, so the
        # agent has room to submit (and re-submit if a submit is rejected).
        remaining = max_tool_calls - tool_calls_made
        if not _final_warned and remaining <= 3:
            _final_warned = True
            has_proposal = last_proposal["makespan"] is not None
            if has_proposal:
                warn = (
                    f"You have only {remaining} tool call(s) left. "
                    f"Your best proposal so far: makespan={last_proposal['makespan']}  "
                    f"cost={last_proposal['road_cost']}  "
                    f"delivered={last_proposal['all_delivered']}. "
                    "Submit your best solution NOW via the applicable submit_* tool "
                    "(`submit_network` / `submit_plan` / `submit_routes`) — send the COMPLETE "
                    "roads/cars JSON and keep any 'reason' short so it isn't truncated."
                )
            else:
                warn = (
                    f"You have only {remaining} tool call(s) left and no valid solution yet. "
                    "Call a submit_* tool (`submit_network` / `submit_plan` / `submit_routes`) "
                    "NOW with your best solution — send the COMPLETE roads/cars JSON and keep "
                    "any 'reason' short so it isn't truncated."
                )
            messages.append({"role": "user", "content": warn})
            _log({"type": "final_warning", "content": warn, "remaining": remaining})
            if verbose or debug:
                print(f"[agent]  last-chance warning injected ({remaining} calls left)")

        response = _call_llm(
            client, model, messages, tools=tools, session_id=session_id, user=user
        )
        msg = response.choices[0].message

        reasoning: str | None = (
            getattr(msg, "reasoning", None)
            or getattr(msg, "reasoning_content", None)
            or (getattr(msg, "model_extra", None) or {}).get("reasoning")
            or (getattr(msg, "model_extra", None) or {}).get("reasoning_content")
        )

        if reasoning:
            _log({"type": "reasoning", "content": reasoning})
            if debug:
                print(f"\n[agent:raw] === reasoning (tool_calls={tool_calls_made}) ===")
                print("[agent:raw] reasoning:", reasoning)
            elif verbose:
                preview = reasoning[:120].replace("\n", " ")
                print(
                    f"[agent] reasoning: {preview}{'…' if len(reasoning) > 120 else ''}"
                )

        if msg.content:
            _log({"type": "content", "content": msg.content})
            if debug:
                print(f"\n[agent:raw] === content (tool_calls={tool_calls_made}) ===")
                print(msg.content)
            elif verbose and not reasoning:
                # Only echo content as reasoning preview when no dedicated reasoning field
                preview = msg.content[:120].replace("\n", " ")
                print(
                    f"[agent] reasoning: {preview}{'…' if len(msg.content) > 120 else ''}"
                )

        if not msg.tool_calls:
            no_tool_streak += 1
            remaining = max_tool_calls - tool_calls_made
            has_proposal = last_proposal["makespan"] is not None
            last_chance = no_tool_streak >= _MAX_NO_TOOL

            if has_proposal:
                push = (
                    f"You have a valid proposal (makespan={last_proposal['makespan']}, "
                    f"cost={last_proposal['road_cost']}, "
                    f"delivered={last_proposal['all_delivered']}) that is NOT yet submitted, "
                    f"and {remaining} tool call(s) left. Do NOT end the session — call a "
                    "submit_* tool (`submit_network` / `submit_plan` / `submit_routes`) NOW to "
                    "commit it (submitting is final). Keep exploring with propose_* only if you "
                    "intend to improve it first."
                )
            elif world_inspected:
                push = (
                    f"You have {remaining} tool call(s) left and have proposed nothing yet. "
                    "You have already inspected the map with get_world_json — now call a "
                    "propose_* tool to evaluate a solution; once it works, call a submit_* tool "
                    "to commit it."
                )
            else:
                push = (
                    f"You have {remaining} tool call(s) left and have proposed nothing yet. "
                    "Do NOT end the session. Call get_world_json to inspect the map, then a "
                    "propose_* tool to evaluate a solution; once it works, call a submit_* tool "
                    "to commit it."
                )
            # Escalate firmness the longer the agent refuses to act.
            if no_tool_streak >= 2:
                push += (
                    " Reasoning alone is NOT recorded and does NOT count — you MUST call a tool "
                    "this turn."
                )
            if last_chance:
                push += (
                    " This is your final reminder: call a tool now or the session ends and your "
                    "work is lost."
                )

            messages.append({"role": "user", "content": push})
            _log({"type": "nudge", "content": push, "streak": no_tool_streak,
                  "has_proposal": has_proposal, "last_chance": last_chance})
            if verbose or debug:
                print(f"[agent] no tool call (streak {no_tool_streak}/{_MAX_NO_TOOL}) "
                      f"has_proposal={has_proposal} — pushed to "
                      f"{'submit' if has_proposal else 'propose'}")

            # Stop only after delivering the final push, so the model always got it.
            if last_chance:
                _log({
                    "type": "forced_stop",
                    "reason": "no_tool_calls",
                    "streak": no_tool_streak,
                    "max_no_tool": _MAX_NO_TOOL,
                    "has_proposal": has_proposal,
                    "submitted": done,
                    "tool_calls_made": tool_calls_made,
                    "content": (
                        f"Agent reasoned {no_tool_streak} times in a row without calling a "
                        f"tool (limit {_MAX_NO_TOOL}); session forced to stop without submitting."
                    ),
                })
                if verbose or debug:
                    print(
                        f"[agent] WARNING: {no_tool_streak} consecutive reasoning-only turns "
                        f"without acting --- stopping."
                    )
                break
            continue

        # Agent called at least one tool - reset the no-tool streak
        no_tool_streak = 0

        # Append assistant message (with tool_calls) to history
        tc_dicts = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
        messages.append(
            {"role": "assistant", "content": msg.content, "tool_calls": tc_dicts}
        )

        for tc in msg.tool_calls:
            tool_calls_made += 1
            name = tc.function.name
            if name == "get_world_json":
                world_inspected = True
            args_malformed = False
            try:
                args = json.loads(tc.function.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
                args_malformed = True
            # Guard against models returning JSON null (parsed to Python None)
            if not isinstance(args, dict):
                args = {}
                args_malformed = True

            executor = executors.get(name)
            if executor is None:
                result = {"error": f"unknown tool: {name}"}
            else:
                try:
                    result = executor(args)
                except Exception as _exec_exc:
                    result = {"error": f"tool execution failed: {_exec_exc}"}
                    if verbose or debug:
                        print(f"[agent] tool {name} raised exception: {_exec_exc}")

            # Commit logic for submit_* tools but ONLY if the submission is well-formed. check the anwser format
            submit_rejected_why: Optional[str] = None
            if name in _SUBMIT_NAMES:
                required = _tool_required.get(name, [])
                if args_malformed:
                    submit_rejected_why = (
                        "the arguments were not valid JSON (likely truncated)"
                    )
                else:
                    missing = [f for f in required if not args.get(f)]
                    if missing:
                        submit_rejected_why = (
                            "missing or empty required field(s): " + ", ".join(missing)
                        )

                if submit_rejected_why is None:
                    # Well-formed submit
                    update_best(best, result, verbose)
                    update_last_proposal(last_proposal, result)
                    submitted.update(last_proposal)
                    done = True
                    reason = args.get("reason", "")
                    clean_result = {k: v for k, v in result.items() if k != "_internal"}
                    clean_result["submitted"] = True
                    clean_result["message"] = "Solution committed as final answer."
                    result = clean_result
                    if verbose or debug:
                        print(
                            f"[agent] submitted makespan={submitted['makespan']}  "
                            f"cost={submitted['road_cost']}  "
                            f"delivered={submitted['all_delivered']}"
                            + (f"reason={reason!r}" if reason else "")
                        )
                else:
                    # Malformed/empty submit 
                    submit_format_fails += 1
                    clean_result = {k: v for k, v in result.items() if k != "_internal"}
                    clean_result["submitted"] = False
                    clean_result["rejected"] = submit_rejected_why
                    result = clean_result
                    if verbose or debug:
                        print(
                            f"[agent] submit REJECTED "
                            f"({submit_format_fails}/{_MAX_SUBMIT_FAILS}): "
                            f"{submit_rejected_why}"
                        )
            else:
                update_best(best, result, verbose)
                update_last_proposal(last_proposal, result)

            # Log
            log_entry: dict = {
                "type": "tool_call",
                "tool_call_id": tc.id,
                "tool": name,
                "phase": args.get("phase", "minimize_makespan"),
                "roads_count": len(args.get("roads", [])),
                "cars_count": len(args.get("cars", [])),
                "result": {k: v for k, v in result.items() if k != "_internal"},
            }
            _log(log_entry)

            clean_result = {k: v for k, v in result.items() if k != "_internal"}
            if debug:
                print(f"\n[agent:raw] --- tool call #{tool_calls_made}: {name} ---")
                debug_args = {
                    k: (
                        f"[{len(v)} items]" if isinstance(v, list) and len(v) > 4 else v
                    )
                    for k, v in args.items()
                }
                print(f"[agent:raw] args: {json.dumps(debug_args, indent=2)}")
                print(f"[agent:raw] result: {json.dumps(clean_result, indent=2)}")
            elif verbose:
                print(
                    f"[agent] tool={name}  "
                    f"phase={args.get('phase','?')}  "
                    f"makespan={clean_result.get('makespan')}  "
                    f"cost={clean_result.get('road_cost')}  "
                    f"delivered={clean_result.get('all_delivered')}"
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(clean_result),
                }
            )

            if submit_rejected_why is not None:
                hint = _tool_hints.get(name, f"check the schema for {name}")
                last_try = submit_format_fails >= _MAX_SUBMIT_FAILS
                reminder = (
                    f"Your `{name}` was NOT accepted: {submit_rejected_why}. It was not "
                    f"committed. Re-send `{name}` with the COMPLETE arguments as one valid "
                    f"JSON object (required: {hint}); include every road and every car, and "
                    f"keep any 'reason' short so the JSON is not truncated."
                )
                messages.append({"role": "user", "content": reminder})
                _log(
                    {
                        "type": "submit_rejected",
                        "tool": name,
                        "reason": submit_rejected_why,
                        "attempt": submit_format_fails,
                        "max_attempts": _MAX_SUBMIT_FAILS,
                        "raw_args": tc.function.arguments,
                    }
                )
                if verbose or debug:
                    print(f"[agent] !!! submit rejected reminder injected for {name}")
                if last_try:
                    _log(
                        {
                            "type": "submit_format_failed",
                            "reason": "too_many_malformed_submits",
                            "attempts": submit_format_fails,
                            "max_attempts": _MAX_SUBMIT_FAILS,
                            "last_reason": submit_rejected_why,
                            "content": (
                                f"Agent failed to submit a well-formed solution "
                                f"{submit_format_fails} times; session ended without a "
                                f"committed answer."
                            ),
                        }
                    )
                    aborted = True
                    if verbose or debug:
                        print(
                            f"[agent] WARNING: {submit_format_fails} malformed submit "
                            f"attempts --- stopping."
                        )

            # If a non-submit tool had malformed args, inject the generic format
            elif args_malformed and name not in _SUBMIT_NAMES:
                hint = _tool_hints.get(name, f"check the schema for {name}")
                reminder = (
                    f"Your call to `{name}` had missing or malformed arguments "
                    f"(received: {tc.function.arguments!r}). "
                    f"Please retry with a valid JSON object. "
                    f"Required fields: {hint}."
                )
                messages.append({"role": "user", "content": reminder})
                _log(
                    {
                        "type": "format_reminder",
                        "tool": name,
                        "raw_args": tc.function.arguments,
                        "hint": hint,
                    }
                )
                if verbose or debug:
                    print(f"[agent] !!! format reminder injected for {name}")

            if done or aborted:
                break  # stop processing remaining tool calls in this batch

    if (verbose or debug) and not done:
        if tool_calls_made >= max_tool_calls:
            print(
                f"[agent] reached max_tool_calls={max_tool_calls} without submit solution"
            )

    if _log_fh:
        try:
            _log_fh.write(
                json.dumps(
                    {
                        "type": "session_end",
                        "tool_calls_made": tool_calls_made,
                        "done": done,
                        "ts": time.time(),
                    }
                )
                + "\n"
            )
            _log_fh.close()
        except Exception:
            pass

    def snapshot(d: dict) -> dict:
        return {
            "makespan": d["makespan"],
            "road_cost": d["road_cost"],
            "all_delivered": d.get("all_delivered", False),
            "car_plan": d["car_plan"],
            "departures": d["departures"],
            "roads": d["roads"],
            "occ": d.get("occ"),
        }

    _best_snap = snapshot(best)

    if submitted:
        _final_snap = snapshot(submitted)
        _final_snap["if_submit"] = "submitted"
    else:
        _final_snap = None  # agent never submitted; _best holds the best proposal

    meta_out = {
        # Convenience mirrors: None when agent never submitted
        "makespan": _final_snap["makespan"] if _final_snap else None,
        "road_cost": _final_snap["road_cost"] if _final_snap else None,
        "all_delivered": _final_snap["all_delivered"] if _final_snap else False,
        "occ": _final_snap["occ"] if _final_snap else None,
        "solve_status": (
            "AGENT_OPTIMAL"
            if (_final_snap and _final_snap["makespan"] is not None)
            else "AGENT_FAILED"
        ),
        "tool_calls_cnt": tool_calls_made,
        # _best is saved for diagnostics only and it never read by the result
        "_best": _best_snap,
        "_final": _final_snap,
    }
    return meta_out, logs


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_TERRAIN_RULES_TEXT = """\
Terrain build rules (cost = base_cost × capacity):
  grass    → base_cost=1,  max_capacity=5  (cheapest)
  water    → base_cost=5,  max_capacity=3
  mountain → base_cost=10, max_capacity=2  (most expensive)
  building → cannot build roads here"""

_ROAD_RULES_TEXT = """\
Road rules:
  • Cars travel only on ROAD tiles.
  • Each car moves ONE tile at a time (up/down/left/right). A path must be a
    continuous chain from DEPOT_OUT to DEPOT_IN with NO jumps/teleports — every
    consecutive pair of positions must be 4-adjacent (differ by exactly one tile).
  • Depots are endpoints only — cars cannot pass through them as intermediate tiles.
  • Road capacity = max cars simultaneously on that tile.
  • When tiles are over-capacity, cars queue and makespan increases.
  • DEPOT_OUT tiles are supply depots (cars start here).
  • DEPOT_IN tiles are demand depots (cars must arrive here)."""

_PHASE_INSTRUCTIONS = """\
Optimisation phases:
  1. Start with phase="minimize_makespan" — focus on reducing total delivery time.
  2. When you believe makespan cannot be improved further, switch to
     phase="minimize_cost" --- find a cheaper network that achieves the same makespan.
     Remove tiles not on critical paths, or reduce capacity on lightly-used segments.
  3. When satisfied, call submit_* (i.e. `submit_network` / `submit_plan` / `submit_routes`) to commit your chosen answer, then stop."""

_PHASE_INSTRUCTIONS_ROUTING = """\
Optimisation phases:
  1. Start with phase="minimize_makespan" — focus on reducing total delivery time.
  2. When satisfied, call submit_* (i.e. `submit_network` / `submit_plan` / `submit_routes`) to commit your chosen answer, then stop."""


_SUBMIT_INSTRUCTIONS = """\
Finalising:
  • When satisfied, call the submit_* tool (i.e. `submit_network` / `submit_plan` / `submit_routes`)
    with the exact solution you want to commit as your final answer.
  • The submit call evaluates the solution and records it — no separate propose_* call needed.
  • Use propose_* freely to explore; only the submit call is recorded as final.
  • You may submit at any time — the loop stops after the first submit."""


def _system_prompt_network() -> str:
    return f"""\
You are a road network designer for a grid-based traffic simulation.

Your task: design a road network so all cars can travel from DEPOT_OUT to DEPOT_IN
with minimum makespan.  A route solver will handle car routing for you — focus on
building a well-connected, well-capacitated network.

{_TERRAIN_RULES_TEXT}

{_ROAD_RULES_TEXT}

{_PHASE_INSTRUCTIONS}

Design tips:
  • Build direct paths between each OUT↔IN depot pair.
  • Higher capacity (2–5) on shared/bottleneck tiles reduces queuing.
  • Avoid expensive terrain (mountain/water) when cheaper alternatives exist.
  • Parallel routes let multiple depot pairs deliver simultaneously.

Road output format (used in propose_network):
  {{"roads": [{{"pos": [x, y], "capacity": int}}, ...]}}

{_SUBMIT_INSTRUCTIONS}
"""


def _system_prompt_dual_network() -> str:
    return f"""\
You are a road network designer in a two-agent pipeline for a grid-based traffic simulation.

Your task: design a road network so all cars can travel from DEPOT_OUT to DEPOT_IN
with minimum makespan.

How the pipeline works:
  • Each time you call propose_network, a separate routing AI agent will find the
    best possible car routes on your network and replay them in the simulator.
  • The makespan you receive back is exact — it is the actual simulation result.
  • If your network is invalid (disconnected depots, over budget), no routing is
    attempted and you will receive an error immediately.
  • Use the makespan and error feedback to iteratively improve your network design.

Important — routing results can vary between calls:
  • The routing agent is an AI and may produce different plans on repeated calls,
    even for the same network.
  • If you believe your current network is good but the makespan seems suboptimal,
    you can call propose_network again with the same roads — you may get a better
    routing result.
  • Only change your network design when you have a reason to (connectivity gaps,
    capacity bottlenecks, excessive road cost).

{_TERRAIN_RULES_TEXT}

{_ROAD_RULES_TEXT}

{_PHASE_INSTRUCTIONS}

Design tips:
  • Build direct paths between each OUT↔IN depot pair.
  • Higher capacity (2–5) on shared/bottleneck tiles reduces queuing.
  • Avoid expensive terrain (mountain/water) when cheaper alternatives exist.
  • Parallel routes let multiple depot pairs deliver simultaneously.

Road output format (used in propose_network):
  {{"roads": [{{"pos": [x, y], "capacity": int}}, ...]}}

{_SUBMIT_INSTRUCTIONS}
"""


def _system_prompt_dual_routing() -> str:
    return f"""\
You are a routing agent in a two-agent pipeline for a grid-based traffic simulation.

The road network has already been built by a separate network design agent.
Your task: find the best possible car routes on this network to minimise makespan.
The simulation will replay your plan exactly — makespan is the actual result.

This is a single evaluation run — do your best within the allowed tool calls,
then call submit_routes to commit your best answer.

{_ROAD_RULES_TEXT}

Routing tips:
  • path[0] must be a ROAD tile adjacent to the DEPOT_OUT.
  • path[-1] must be the DEPOT_IN position itself.
  • All intermediate path tiles must be ROAD.
  • Stagger departure ticks to avoid congestion on shared tiles.
  • The simulation replays your plan exactly — makespan is the actual result.

Car output format (used in propose_routes):
  {{"cars": [{{"from": depot_out_id, "to": depot_in_id,
              "path": [[x,y], ...], "depart": int}}, ...]}}

{_SUBMIT_INSTRUCTIONS}
"""


def _system_prompt_joint() -> str:
    return f"""\
You are a joint road network and routing designer for a grid-based traffic simulation.

Your task: design BOTH the road network AND the car routes so all cars are delivered
with minimum makespan.  The simulation will replay your plan exactly to measure
the actual makespan — there is no approximation.

{_TERRAIN_RULES_TEXT}

{_ROAD_RULES_TEXT}

{_PHASE_INSTRUCTIONS}

Design tips (network):
  • Build direct paths between each OUT↔IN depot pair.
  • Higher capacity on shared tiles prevents queuing.
  • Parallel branches let multiple flows advance simultaneously.

Design tips (routing):
  • path[0] must be a ROAD tile adjacent to the DEPOT_OUT.
  • path[-1] must be the DEPOT_IN position itself.
  • All intermediate path tiles must be ROAD.
  • Stagger departure ticks to avoid congestion on shared tiles.
  • One entry per car (total cars = sum of all depot_out amounts).

Car output format (used in propose_plan):
  {{"cars": [{{"from": depot_out_id, "to": depot_in_id,
              "path": [[x,y], ...], "depart": int}}, ...]}}

{_SUBMIT_INSTRUCTIONS}
"""


def _system_prompt_routing() -> str:
    return f"""\
You are a route planning agent for a grid-based traffic simulation.

The road network is already built.  Your task: assign each car a departure tick
and a path through the existing roads to minimise makespan.

{_ROAD_RULES_TEXT}

{_PHASE_INSTRUCTIONS_ROUTING}

Routing tips:
  • path[0] must be a ROAD tile adjacent to the DEPOT_OUT.
  • path[-1] must be the DEPOT_IN position itself.
  • All intermediate path tiles must be ROAD.
  • Stagger departures on shared tiles to avoid capacity conflicts.
  • The simulation replays your plan exactly — makespan is the actual result.

Car output format (used in propose_routes):
  {{"cars": [{{"from": depot_out_id, "to": depot_in_id,
              "path": [[x,y], ...], "depart": int}}, ...]}}

{_SUBMIT_INSTRUCTIONS}
"""


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def solve_joint(
    sim: TrafficSim,
    *,
    model: str = DEFAULT_MODEL,
    max_tool_calls: int = DEFAULT_MAX_CALLS,
    api_key: Optional[str] = None,
    log_dir: Optional[str] = None,
    log_prefix: str = "joint",
    session_id: Optional[str] = None,
    user: Optional[str] = None,
    verbose: bool = True,
    debug: bool = False,
) -> dict:
    """Single LLM designs both network and routes; simulation replay measures makespan.

    Returns meta with _final and _best snapshots.
    """
    client = _build_client(api_key)
    world_json = world_to_json(sim)
    plan_exec = _make_plan_executor(sim)
    executors = {
        "get_world_json": _make_world_executor(world_json),
        "propose_plan": plan_exec,
        "submit_plan": plan_exec,
    }
    meta, _ = _run_agent_loop(
        sim=sim,
        model=model,
        client=client,
        tools=_TOOLS_JOINT,
        executors=executors,
        system_prompt=_system_prompt_joint(),
        max_tool_calls=max_tool_calls,
        verbose=verbose,
        log_dir=log_dir,
        debug=debug,
        log_prefix=log_prefix,
        session_id=session_id,
        user=user,
    )
    return meta


def solve_network(
    sim: TrafficSim,
    *,
    model: str = DEFAULT_MODEL,
    route_solver: Optional[Callable] = None,
    max_tool_calls: int = DEFAULT_MAX_CALLS,
    api_key: Optional[str] = None,
    log_dir: Optional[str] = None,
    log_prefix: str = "network",
    session_id: Optional[str] = None,
    user: Optional[str] = None,
    verbose: bool = True,
    debug: bool = False,
) -> dict:
    
    if route_solver is None:
        from solvers.rule_based import solve_routing as route_solver  # type: ignore

    client = _build_client(api_key)
    world_json = world_to_json(sim)

    net_exec = _make_network_executor(sim, route_solver)
    executors = {
        "get_world_json": _make_world_executor(world_json),
        "propose_network": net_exec,
        "submit_network": net_exec,
    }
    meta, _ = _run_agent_loop(
        sim=sim,
        model=model,
        client=client,
        tools=_TOOLS_NETWORK,
        executors=executors,
        system_prompt=_system_prompt_network(),
        max_tool_calls=max_tool_calls,
        verbose=verbose,
        log_dir=log_dir,
        debug=debug,
        log_prefix=log_prefix,
        session_id=session_id,
        user=user,
    )
    return meta


def solve_routing(
    sim: TrafficSim,
    *,
    model: str = DEFAULT_MODEL,
    max_tool_calls: int = DEFAULT_MAX_CALLS,
    api_key: Optional[str] = None,
    log_dir: Optional[str] = None,
    log_prefix: str = "routing",
    session_id: Optional[str] = None,
    user: Optional[str] = None,
    verbose: bool = True,
    debug: bool = False,
) -> dict:

    client = _build_client(api_key)
    world_json = world_to_json(sim)
    routes_exec = _make_routes_executor(sim)
    executors = {
        "get_world_json": _make_world_executor(world_json),
        "propose_routes": routes_exec,
        "submit_routes": routes_exec,
    }
    meta, _ = _run_agent_loop(
        sim=sim,
        model=model,
        client=client,
        tools=_TOOLS_ROUTING,
        executors=executors,
        system_prompt=_system_prompt_routing(),
        max_tool_calls=max_tool_calls,
        verbose=verbose,
        log_dir=log_dir,
        debug=debug,
        log_prefix=log_prefix,
        session_id=session_id,
        user=user,
    )
    return meta


def run_dual_agent(
    sim: TrafficSim,
    *,
    network_model: str = DEFAULT_MODEL,
    route_model: str = DEFAULT_MODEL,
    network_max_calls: int = DEFAULT_MAX_CALLS,
    route_max_calls: int = DEFAULT_MAX_CALLS // 2,
    api_key: Optional[str] = None,
    log_dir: Optional[str] = None,
    log_prefix: str = "dual",
    session_id: Optional[str] = None,
    user: Optional[str] = None,
    verbose: bool = True,
    debug: bool = False,
) -> dict:
   
    client = _build_client(api_key)
    world_json = world_to_json(sim)

    # Counters shared across all _dual_network_exec calls
    inner_run: list[int] = [0]  # valid network proposals only
    net_proposal_cnt: list[int] = [0]  # all network proposals (valid + invalid)
    route_calls_log: list[int] = []  # tool calls per proposal (0 = invalid)

    def _dual_network_exec(args: dict) -> dict:
        """propose_network executor: validates roads then runs inner route agent."""
        roads = args.get("roads", [])
        phase = args.get("phase", "minimize_makespan")

        net_proposal_cnt[0] += 1
        sim_copy = copy.deepcopy(sim)
        road_errors = _apply_roads(sim_copy, roads)
        connectivity_errors = _check_connectivity(sim_copy)
        budget = _budget_status(sim_copy)

        # If network is invalid, skip routing and return errors directly
        if connectivity_errors or not budget["ok"]:
            route_calls_log.append(0)
            return {
                "makespan": None,
                "road_cost": budget["used"],
                "all_delivered": False,
                "budget_ok": budget["ok"],
                "budget_limit": budget["limit"],
                "budget_over_by": budget["over_by"],
                "connectivity_errors": connectivity_errors[:5],
                "road_errors": road_errors[:5],
                "route_agent_status": "SKIPPED_INVALID_NETWORK",
                "_internal": {
                    "phase": phase,
                    "roads": roads,
                    "car_plan": [],
                    "departures": {},
                    "occ": None,
                },
            }

        # Valid network -- run inner route agent
        inner_run[0] += 1
        prefix = f"{log_prefix}_route_{inner_run[0]}"
        if verbose or debug:
            print(
                f"[dual_agent] network valid — running route agent (run #{inner_run[0]}, "
                f"max_calls={route_max_calls})"
            )

        net_world_json = world_to_json(sim_copy)
        inner_routes_exec = _make_routes_executor(sim_copy)
        route_executors = {
            "get_world_json": _make_world_executor(net_world_json),
            "propose_routes": inner_routes_exec,
            "submit_routes": inner_routes_exec,
        }
        route_meta, _ = _run_agent_loop(
            sim=sim_copy,
            model=route_model,
            client=client,
            tools=_TOOLS_ROUTING,
            executors=route_executors,
            system_prompt=_system_prompt_dual_routing(),
            max_tool_calls=route_max_calls,
            verbose=verbose,
            log_dir=log_dir,
            debug=debug,
            log_prefix=prefix,
            session_id=session_id,
            user=user,
        )
        route_calls_log.append(route_meta.get("tool_calls_cnt", 0))

        if route_meta.get("_final", {}).get("if_submit") == "submitted":
            snap = route_meta.get("_final")
        else:
            snap = {}  # no submission
        car_plan = snap.get("car_plan", [])
        departures = snap.get("departures", {})
        makespan = snap.get("makespan")
        all_delivered = snap.get("all_delivered", False)
        road_cost = budget["used"]

        return {
            "makespan": makespan,
            "road_cost": road_cost,
            "all_delivered": all_delivered,
            "budget_ok": budget["ok"],
            "budget_limit": budget["limit"],
            "budget_over_by": 0,
            "connectivity_errors": [],
            "road_errors": road_errors[:5],
            "route_agent_status": route_meta.get("solve_status", "UNKNOWN"),
            "route_agent_calls": inner_run[0],
            "_internal": {
                "phase": phase,
                "roads": roads,
                "car_plan": car_plan,
                "departures": departures,
                "occ": route_meta.get("occ"),
            },
        }

    net_executors = {
        "get_world_json": _make_world_executor(world_json),
        "propose_network": _dual_network_exec,
        "submit_network": _dual_network_exec,
    }

    if verbose:
        print(
            f"[dual_agent] starting  network_max_calls={network_max_calls}  "
            f"route_max_calls={route_max_calls}"
        )

    net_meta, _ = _run_agent_loop(
        sim=sim,
        model=network_model,
        client=client,
        tools=_TOOLS_NETWORK,
        executors=net_executors,
        system_prompt=_system_prompt_dual_network(),
        max_tool_calls=network_max_calls,
        verbose=verbose,
        log_dir=log_dir,
        debug=debug,
        log_prefix=f"{log_prefix}_network",
        session_id=session_id,
        user=user,
    )

    # Only use the submitted result; None fields when agent never submitted
    final_snap = net_meta.get("_final", {})
    submitted = final_snap.get("if_submit") == "submitted"

    net_calls = net_meta.get("tool_calls_cnt", 0)
    total_route = sum(route_calls_log)
    meta = {
        # Convenience mirrors — None when no submission
        "makespan": final_snap.get("makespan") if submitted else None,
        "road_cost": final_snap.get("road_cost") if submitted else None,
        "all_delivered": final_snap.get("all_delivered") if submitted else False,
        "occ": final_snap.get("occ") if submitted else None,
        "solve_status": net_meta.get("solve_status", "AGENT_DUAL"),
        # Tool-call accounting
        "tool_calls_cnt": net_calls,
        "net_proposals": net_proposal_cnt[0],
        "inner_route_runs": inner_run[0],
        "route_calls_per_run": route_calls_log,
        "total_tool_calls": net_calls + total_route,
        # Full snapshots
        "_best": net_meta.get("_best"),
        "_final": net_meta.get("_final"),
    }
    return meta



#----------------Validation----------------------------------------------

def compare_solutions(meta: dict) -> dict:
    """Compare the best-during-iteration vs the agent's final proposal.
    Check if the submission is valid solution. (no teleporting, no over-capacity, all cars delivered, etc.)
    """
    snap_b = meta.get("_best")
    snap_f = meta.get("_final")  # None when agent never submitted
    if snap_f is None:
        snap_f = {}
        print(
            "[compare_solutions] no final submission found; comparing best vs empty final"
        )

    if snap_b is None and snap_f is None:
        return {"error": "no _best or _final snapshots found in meta"}

    snap_b = snap_b or {}
    snap_f = snap_f or {}

    ms_b = snap_b.get("makespan")
    ms_f = snap_f.get("makespan")
    rc_b = snap_b.get("road_cost")
    rc_f = snap_f.get("road_cost")

    # Road-tile comparison
    def _roads_to_dict(roads: list) -> dict:
        out = {}
        for r in roads:
            pos = r.get("pos", [])
            key = (
                (pos[0], pos[1])
                if isinstance(pos, list) and len(pos) == 2
                else tuple(pos)
            )
            out[key] = r.get("capacity", 1)
        return out

    rd_b = _roads_to_dict(snap_b.get("roads", []))
    rd_f = _roads_to_dict(snap_f.get("roads", []))

    common = set(rd_b) & set(rd_f)
    only_best = sorted(set(rd_b) - set(rd_f))
    only_final = sorted(set(rd_f) - set(rd_b))
    cap_changed = [
        {"pos": list(p), "cap_best": rd_b[p], "cap_final": rd_f[p]}
        for p in sorted(common)
        if rd_b[p] != rd_f[p]
    ]
    roads_identical = not only_best and not only_final and not cap_changed

    # Route comparison
    cp_b = snap_b.get("car_plan", [])
    cp_f = snap_f.get("car_plan", [])
    routes_identical = cp_b == cp_f

    solutions_identical = (
        ms_b == ms_f and rc_b == rc_f and roads_identical and routes_identical
    )

    return {
        "solutions_identical": solutions_identical,
        "final_all_delivered": snap_f.get("all_delivered", False),
        "makespan_best": ms_b,
        "makespan_final": ms_f,
        "makespan_delta": (
            (ms_f - ms_b) if (ms_b is not None and ms_f is not None) else None
        ),
        "road_cost_best": rc_b,
        "road_cost_final": rc_f,
        "road_cost_delta": (
            (rc_f - rc_b) if (rc_b is not None and rc_f is not None) else None
        ),
        "roads_only_in_best": [list(p) for p in only_best],
        "roads_only_in_final": [list(p) for p in only_final],
        "roads_capacity_changed": cap_changed,
        "roads_identical": roads_identical,
        "cars_count_best": len(cp_b),
        "cars_count_final": len(cp_f),
        "routes_identical": routes_identical,
    }


def validate_agent_result(
    sim: TrafficSim,
    meta: dict,
    which: str = "_final",
    makespan_tolerance: int = 0,
    max_ticks: int = 10_000,
) -> dict:
  
    result = {
        "snapshot": which,
        "has_solution": False,
        "network_connected": False,
        "connectivity_errors": [],
        "replayable": False,  # False for flow-based routing (no car-level plan)
        "plan_executable": None,  # None = not verifiable (no car_plan to replay)
        "plan_errors": [],
        "claimed_makespan": None,
        "actual_makespan": None,
        "makespan_match": None,  # None = not verifiable
        "actual_all_delivered": False,
        "timed_out": False,
        "road_cost": None,
        "valid": False,
    }

    snap = meta.get(which)
    if not snap:
        result["plan_errors"] = [
            f"meta has no '{which}' snapshot (agent produced no committed solution)"
        ]
        return result
    result["has_solution"] = True

    claimed = snap.get("makespan")
    result["claimed_makespan"] = claimed

    # 1. Rebuild the proposed network on a fresh copy of the original sim.
    #    Agent runs save the RAW grid (no roads) as their sim.pkl the road
    #    network lives only in the snapshot, so we reconstruct it here. This is
    #    idempotent for routing solvers whose sim already carries these roads.
    sim_copy = copy.deepcopy(sim)
    road_errors = _apply_roads(sim_copy, snap.get("roads", []))
    result["road_cost"] = sum(
        sim_copy.calculate_road_cost(p, c) for p, c in sim_copy.road_capacity.items()
    )

    # 2. Connectivity — depot-in reachable from depot-out over the road network.
    connectivity_errors = _check_connectivity(sim_copy)
    result["connectivity_errors"] = road_errors + connectivity_errors
    result["network_connected"] = not connectivity_errors and not road_errors

    # 3. Departure plan and structural validation, then full replay.
    #    Snapshots round-tripped through JSON have list positions/keys; the sim
    #    uses positions as dict keys during replay, so normalise to tuples/ints.
    car_plan = _normalize_car_plan(snap.get("car_plan") or [])
    departures = {int(k): int(v) for k, v in (snap.get("departures") or {}).items()}

    if car_plan:
        result["replayable"] = True
        ok_struct, plan_errors = validate_plan(sim_copy, car_plan, departures)
        sim_copy.car_plan = car_plan
        sim_copy.saved_departures = departures
        ev = evaluate(sim_copy, max_ticks=max_ticks)
        result["actual_makespan"] = ev.get("makespan")
        result["actual_all_delivered"] = ev.get("all_delivered", False)
        result["timed_out"] = ev.get("timed_out", False)
        result["plan_errors"] = plan_errors
        result["plan_executable"] = (
            ok_struct
            and result["actual_all_delivered"]
            and not result["timed_out"]
        )
        # 4. Makespan agreement agent's claim vs independent replay.
        actual = result["actual_makespan"]
        if claimed is not None and actual is not None:
            result["makespan_match"] = abs(claimed - actual) <= makespan_tolerance
    else:
        result["plan_errors"] = [
            "snapshot has empty car_plan (flow-based routing); "
            "makespan not independently replayable"
        ]

    # only connectivity gates validity.
    checks = [result["network_connected"]]
    if result["replayable"]:
        checks += [bool(result["plan_executable"]), bool(result["makespan_match"])]
    result["valid"] = all(checks)
    return result


def _normalize_car_plan(car_plan: list[dict]) -> list[dict]:
    """Return a copy of car_plan with positions coerced to tuples.
    """
    normalized: list[dict] = []
    for entry in car_plan:
        e = dict(entry)
        if e.get("path") is not None:
            e["path"] = [tuple(p) for p in e["path"]]
        if e.get("depot_out_pos") is not None:
            e["depot_out_pos"] = tuple(e["depot_out_pos"])
        normalized.append(e)
    return normalized

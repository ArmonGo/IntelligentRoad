from __future__ import annotations
import copy
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Optional
import io as _io

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from minisim import TrafficSim, TileType, DIR4, Pos
from solvers.utils import world_to_json, evaluate

DEFAULT_MODEL = "openrouter/deepseek/deepseek-v3.2" # change this setting in run.py
DEFAULT_MAX_CALLS = 15  # same change this in run.py

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
    max_tokens: int = 4096,
    retries: int = 3,
    session_id: Optional[str] = None,
    user: Optional[str] = None,
):
    """Call the LLM API with optional OpenRouter trace fields.

    session_id : groups all turns of one agent session — filter on
                 openrouter.ai/activity by session to see per-session cost.
    user       : broader grouping (e.g. run_id) — lets you see cost for a
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
    """BFS check: return error strings for depot_in tiles unreachable from any depot_out.
    Manually roll back"""
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

# Best-result tracker, but didnt use it at the end. use the submitted solution only
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
    """ The agent calls tools freely until it either:
      - Returns a message with no tool_calls (signals it is done), or
      - The safety ceiling max_tool_calls is reached.
    """
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
    no_tool_streak = 0  # consecutive LLM turns with no tool calls
    _MAX_NO_TOOL = 3  # give up after this many consecutive nudges
    _final_warned = False  # True once the "last chance" warning has been injected

    # Build per-tool format hints from the schemas passed in, for format reminders
    _tool_hints: dict[str, str] = {}
    for _t in tools:
        _fn = _t.get("function", {})
        _tname = _fn.get("name", "")
        _params = _fn.get("parameters", {})
        _required = _params.get("required", [])
        _props = _params.get("properties", {})
        _hints = []
        for _f in _required:
            _desc = _props.get(_f, {}).get("description", "")[:60]
            _hints.append(f'"{_f}" ({_desc})')
        _tool_hints[_tname] = ", ".join(_hints) if _hints else "(no required fields)"

    while tool_calls_made < max_tool_calls and not done:
        # Inject a "last chance" warning when only a few calls remain
        remaining = max_tool_calls - tool_calls_made
        if not _final_warned and remaining <= 1:
            _final_warned = True
            has_proposal = last_proposal["makespan"] is not None
            if has_proposal:
                warn = (
                    f"You have {remaining} tool call left. "
                    f"Your best proposal so far: makespan={last_proposal['makespan']}  "
                    f"cost={last_proposal['road_cost']}  "
                    f"delivered={last_proposal['all_delivered']}. "
                    "Use your last call to submit your best solution via the applicable "
                    "submit_* tool (i.e. `submit_network` / `submit_plan` / `submit_routes`) NOW with your best solution."
                )
            else:
                warn = (
                    f"You have {remaining} tool call left. Without any valid solutions so far. "
                    "Still, call a submit_* tool (i.e. `submit_network` / `submit_plan` / `submit_routes`) "
                    "NOW with your best solution."
                )
            messages.append({"role": "user", "content": warn})
            _log({"type": "final_warning", "content": warn, "remaining": remaining})
            if verbose or debug:
                print(f"[agent]  last-chance warning injected ({remaining} calls left)")

        response = _call_llm(
            client, model, messages, tools=tools, session_id=session_id, user=user
        )
        msg = response.choices[0].message
        # Some models surface chain-of-thought in a separate field rather than
        # msg.content (e.g. msg.reasoning, msg.reasoning_content, model_extra).
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

        # No tool calls in this turn
        if not msg.tool_calls:
            no_tool_streak += 1
            # Hard stop after too many consecutive no-tool turns
            if no_tool_streak >= _MAX_NO_TOOL:
                if verbose or debug:
                    print(
                        f"[agent] WARNING: agent produced no tool calls {no_tool_streak} times "
                        f"in a row without submitting --- stopping."
                    )
                break

            # Nudge the agent back to using tools
            has_proposal = last_proposal["makespan"] is not None
            if has_proposal:
                nudge = (
                    "You have not submitted your solution yet. "
                    "When satisfied, call a submit_* tool (i.e. `submit_network` / `submit_plan` / `submit_routes`) "
                    "with the exact solution you want to commit. "
                    "Otherwise keep using the propose_* tools to improve."
                )
            else:
                nudge = (
                    "No solution has been proposed yet. "
                    "Please use the available tools — start with get_world_json to inspect the map, "
                    "then call a propose_* tool to evaluate a solution."
                )

            if verbose or debug:
                print(f"[agent] no tool call (streak {no_tool_streak}/{_MAX_NO_TOOL}) ")
            messages.append({"role": "user", "content": nudge})
            _log({"type": "nudge", "content": nudge, "streak": no_tool_streak})
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

            # Dispatch all tools through executors (submit_* share the same
            # executor as their propose_* counterpart but also trigger done=True)
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

            # Commit logic for submit_* tools
            if name in _SUBMIT_NAMES:
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

            # Strip internal tracking data, LLM only sees the clean feedback,
            # and doesnt need to see its own plan
            # so for dual agents, network designer doesnt see the proposed car plan
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

            # If args were malformed, inject a format reminder so the model
            # knows what went wrong and can retry with the correct structure
            if args_malformed:
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

            if done:
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
            "AGENT_OPTIMAL" if best["makespan"] is not None else "AGENT_FAILED"
        ),
        "tool_calls_cnt": tool_calls_made,
        # Full snapshots for comparison / visualization
        "_best": _best_snap,
        "_final": _final_snap,
    }
    return meta_out, logs


# System prompts

_TERRAIN_RULES_TEXT = """\
Terrain build rules (cost = base_cost × capacity):
  grass    -> base_cost=1,  max_capacity=5  (cheapest)
  water    -> base_cost=5,  max_capacity=3
  mountain -> base_cost=10, max_capacity=2  (most expensive)
  building -> cannot build roads here"""

_ROAD_RULES_TEXT = """\
Road rules:
  • Cars travel only on ROAD tiles.
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
  • Higher capacity (2-5) on shared/bottleneck tiles reduces queuing.
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
  • The makespan you receive back is exact - it is the actual simulation result.
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
  • Higher capacity (2-5) on shared/bottleneck tiles reduces queuing.
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
The simulation will replay your plan exactly - makespan is the actual result.

This is a single evaluation run - do your best within the allowed tool calls,
then call submit_routes to commit your best answer.

{_ROAD_RULES_TEXT}

Routing tips:
  • path[0] must be a ROAD tile adjacent to the DEPOT_OUT.
  • path[-1] must be the DEPOT_IN position itself.
  • All intermediate path tiles must be ROAD.
  • Stagger departure ticks to avoid congestion on shared tiles.
  • The simulation replays your plan exactly - makespan is the actual result.

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


# Public entry points


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
    """Single LLM designs network only; route_solver evaluates routing.

    Returns meta with _final and _best snapshots.
    """
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
    """Route-planning agent on a sim that already has a road network.
    Uses simulation replay for makespan evaluation.
    Returns meta with _final and _best snapshots.
    """
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
    """Two-agent pipeline: network agent designs roads, route agent plans routes.
    For each propose_network call from the network agent:
      1. The proposed roads are validated (connectivity + budget).
      2. If valid, a fresh route agent loop runs on that network and finds the
         best routes via simulation replay, meanwhile no route_solver involved.
      3. The actual makespan (or validation error) is returned as feedback to
         the network agent.
    The network agent iterates until it calls submit_network.
    """
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

        # All fields come from the same snapshot so makespan/car_plan/delivered
        # are always consistent with each other.
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

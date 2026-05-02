# IntelligentRoad

This repository contains the implementation for the models included in the experimental comparison as presented in:

_An Evaluation of Autonomous Agent Approaches for Time-Optimal Infrastructure Optimization_

**Abstract:** Large Language Models (LLMs) and autonomous AI agents are increasingly being explored for operational tasks that traditionally rely on exact solvers, heuristics, and meta-heuristic methods. Although early studies have demonstrated promising applications of LLMs in scheduling, routing, planning, and optimization, few studies have conducted rigorous benchmark evaluations of LLM-based agents against classical optimization methods. Therefore, to provide a more complete understanding of the capabilities and feasibility of using agents for operational tasks, we focus on a specific spatial infrastructure design problem that requires jointly solving road network planning and vehicle routing to minimize the makespan (i.e., the total completion time for deliveries from source depots to destination depots under a predetermined schedule).

To this end, we propose a fully autonomous LLM-agent framework that iteratively generates candidate solutions, invokes evaluation tools, and refines decisions based on external feedback. To assess the strengths and limitations of agent-based optimization, we benchmark the proposed framework against representative classical baselines, including constraint programming (CP-SAT), genetic algorithms, and handcrafted heuristics. Our comparison shows that although current LLM agents cannot yet outperform classical solvers, they exhibit clear potential to solve the problem and generate feasible joint infrastructure-routing solutions when the search space remains relatively small.

"""Upper level: budget-constrained continuous Network Design Problem (BNDP).

    min_y   TSTT(y) = sum_a x_a(y) t_a(x_a(y), y_a)
    s.t.    sum_a k_a y_a <= B,     0 <= y_a <= y_max   (a in candidate links)
            x(y) solves the user-equilibrium problem   (lower level)

y_a is the capacity added to link a as a fraction of its current capacity and
k_a = length_a * capacity_a / 1000 is its unit cost (lane-km proxy).

Solution approaches implemented here:
    uniform   spread the budget evenly (naive baseline)
    ioa       iterative optimisation-assignment: fix flows, optimise y, re-assign
              (ignores the users' reaction -> Cournot-Nash, not Stackelberg)
    sndp      system-optimal planner: assumes users follow SO (single-level, convex)
    ga        genetic algorithm on the true bi-level problem
    descent   SLSQP on TSTT(x*(y)) with finite-difference gradients of the
              equilibrium response (used to refine the GA result)
    pattern   projected compass search (alternative derivative-free refinement)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .assignment import solve_assignment
from .network import Network

# Classic continuous NDP candidate links on Sioux Falls (Abdulaal & LeBlanc, 1979),
# 1-based link ids from the TNTP file.
SIOUX_FALLS_CANDIDATES = [16, 17, 19, 20, 25, 26, 29, 39, 48, 74]


@dataclass
class DesignResult:
    method: str
    y: np.ndarray
    tstt: float
    cost: float
    evaluations: int
    seconds: float
    history: list[float] = field(default_factory=list)


class BudgetNDP:
    def __init__(self, net: Network, candidates_1based, budget_share=0.3, y_max=1.0,
                 search_gap=1e-4, final_gap=1e-7):
        self.net = net
        self.cand = np.array(candidates_1based) - 1
        self.y_max = y_max
        self.unit_cost = net.length[self.cand] * net.capacity[self.cand] / 1000.0
        self.budget = budget_share * float(self.unit_cost.sum() * y_max)
        self.search_gap = search_gap
        self.final_gap = final_gap
        self.evals = 0

    # --- helpers --------------------------------------------------------------
    def full_y(self, yc):
        y = np.zeros(self.net.n_links)
        y[self.cand] = yc
        return y

    def cost(self, yc) -> float:
        return float(np.dot(self.unit_cost, yc))

    def repair(self, yc):
        """Project onto the box, then scale down until the budget holds."""
        yc = np.clip(yc, 0.0, self.y_max)
        c = self.cost(yc)
        if c > self.budget:
            yc = yc * (self.budget / c)
        return yc

    def tstt_ue(self, yc, gap=None) -> float:
        self.evals += 1
        r = solve_assignment(self.net, self.full_y(yc), "ue", gap_tol=gap or self.search_gap)
        return r.tstt

    def _finish(self, method, yc, t0, history=None):
        yc = self.repair(yc)
        tstt = self.tstt_ue(yc, self.final_gap)
        return DesignResult(method, yc, tstt, self.cost(yc), self.evals, time.time() - t0, history or [])

    # --- baselines ------------------------------------------------------------
    def uniform(self):
        t0, self.evals = time.time(), 0
        yc = np.full(len(self.cand), self.y_max)
        return self._finish("uniform", yc, t0)

    def _optimise_y_fixed_cost(self, objective, y0, eps=1.5e-8):
        cons = [{"type": "ineq", "fun": lambda yc: self.budget - self.cost(yc)}]
        res = minimize(objective, y0, method="SLSQP", bounds=[(0, self.y_max)] * len(y0),
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-10, "eps": eps})
        return res.x

    def ioa(self, max_rounds=30, tol=1e-4):
        """Iterative optimisation-assignment (Allsop 1974; Friesz & Harker 1985)."""
        t0, self.evals = time.time(), 0
        net, cand = self.net, self.cand
        yc = np.zeros(len(cand))
        history = []
        for _ in range(max_rounds):
            r = solve_assignment(net, self.full_y(yc), "ue", gap_tol=self.search_gap)
            self.evals += 1
            history.append(r.tstt)
            xc = r.flow[cand]

            def obj(z):  # TSTT on candidate links with flows held fixed
                c = net.capacity[cand] * (1 + z)
                return float(np.sum(xc * net.fftt[cand] * (1 + net.b[cand] * (xc / c) ** net.power[cand]))) / 1e4

            new = self._optimise_y_fixed_cost(obj, yc)
            if np.max(np.abs(new - yc)) < tol:
                yc = new
                break
            yc = new
        return self._finish("ioa", yc, t0, history)

    def sndp(self):
        """Planner assumes system-optimal routing (single-level)."""
        t0, self.evals = time.time(), 0

        def obj(yc):
            self.evals += 1
            r = solve_assignment(self.net, self.full_y(np.clip(yc, 0, self.y_max)), "so", gap_tol=1e-6)
            return r.tstt / 1e6

        yc = self._optimise_y_fixed_cost(obj, np.full(len(self.cand), 0.1), eps=1e-3)
        return self._finish("sndp", yc, t0)

    # --- bi-level heuristics ----------------------------------------------------
    def descent(self, y0, gap=1e-6, eps=1e-2, name="descent", restarts=4):
        """Bi-level descent: SLSQP on TSTT(x*(y)), gradient by finite differences
        of the equilibrium response (the UE link flows are unique, so the
        upper-level objective is a well-defined function of y)."""
        t0, self.evals = time.time(), 0
        history = []

        def obj(yc):
            f = self.tstt_ue(np.clip(yc, 0, self.y_max), gap)
            history.append(f)
            return f / 1e6

        yc, best = self.repair(y0), np.inf
        for _ in range(restarts):            # SLSQP restarts until no progress
            yc = self.repair(self._optimise_y_fixed_cost(obj, yc, eps=eps))
            f = self.tstt_ue(yc, gap)
            if f > best * (1 - 1e-5):
                break
            best = f
        return self._finish(name, yc, t0, history)

    def genetic_algorithm(self, pop_size=24, generations=40, elite=2, mutation=0.15, seed=0):
        t0, self.evals = time.time(), 0
        rng = np.random.default_rng(seed)
        n = len(self.cand)
        pop = np.array([self.repair(rng.uniform(0, self.y_max, n)) for _ in range(pop_size)])
        pop[0] = self.repair(np.full(n, self.y_max))           # seed: uniform design
        fit = np.array([self.tstt_ue(ind) for ind in pop])
        history = [float(fit.min())]

        def tournament():
            i, j = rng.integers(0, pop_size, 2)
            return pop[i] if fit[i] < fit[j] else pop[j]

        for _ in range(generations):
            order = np.argsort(fit)
            children = [pop[k].copy() for k in order[:elite]]
            child_fit = [fit[k] for k in order[:elite]]
            while len(children) < pop_size:
                p1, p2 = tournament(), tournament()
                w = rng.uniform(-0.25, 1.25, n)                  # blend (BLX) crossover
                c = w * p1 + (1 - w) * p2
                mask = rng.random(n) < mutation                   # gaussian mutation
                c[mask] += rng.normal(0, 0.2 * self.y_max, mask.sum())
                c = self.repair(c)
                children.append(c)
                child_fit.append(self.tstt_ue(c))
            pop, fit = np.array(children), np.array(child_fit)
            history.append(float(fit.min()))
        best = pop[np.argmin(fit)]
        return self._finish("ga", best, t0, history), best

    def pattern_search(self, y0, step=0.1, min_step=0.005, gap=1e-6):
        """Projected compass search (derivative-free local refinement)."""
        t0, self.evals = time.time(), 0
        y = self.repair(y0)
        f = self.tstt_ue(y, gap)
        history = [f]
        while step >= min_step:
            improved = False
            for i in range(len(y)):
                for s in (+step, -step):
                    trial = y.copy()
                    trial[i] += s
                    # keep the budget fully used: take the extra spend from the others
                    trial = self.repair(trial)
                    ft = self.tstt_ue(trial, gap)
                    if ft < f - 1e-6 * f:
                        y, f, improved = trial, ft, True
                        history.append(f)
                        break
            if not improved:
                step /= 2
        return self._finish("ga+local", y, t0, history)

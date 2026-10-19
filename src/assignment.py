"""Lower-level problem: static traffic assignment.

* User equilibrium (UE, Wardrop's first principle) minimises the Beckmann
  potential  sum_a int_0^{x_a} t_a(w) dw  subject to flow conservation.
* System optimum (SO, Wardrop's second principle) minimises total travel time
  sum_a x_a t_a(x_a); it is solved with the same algorithms using marginal
  link costs  t_a + x_a t_a'.

Two algorithms are provided:

* ``method="gp"``  path-based gradient projection (Jayakrishnan et al., 1994).
  Converges to very small gaps (1e-8 and below) and is the default.
* ``method="cfw"`` link-based conjugate Frank-Wolfe (Mitradjieva & Lindberg,
  2013). Simple and memory-light, but stalls around a 1e-5 gap.

Convergence is measured with the relative gap
    (sum_a x_a c_a - sum_od q_od * pi_od) / sum_a x_a c_a.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .network import Network


@dataclass
class AssignmentResult:
    flow: np.ndarray
    time: np.ndarray
    tstt: float
    gap: float
    iterations: int
    gap_history: list[float]


class _ShortestPaths:
    def __init__(self, net: Network):
        self.net = net
        self.link_index = {(int(u), int(v)): a for a, (u, v) in enumerate(zip(net.init, net.term))}
        self.origins = np.array(sorted(net.od.keys()))
        self.demand = np.zeros((len(self.origins), net.n_nodes))
        for i, o in enumerate(self.origins):
            dests, q = net.od[o]
            self.demand[i, dests] = q

    def trees(self, cost: np.ndarray):
        net = self.net
        graph = csr_matrix((np.maximum(cost, 1e-12), (net.init, net.term)),
                           shape=(net.n_nodes, net.n_nodes))
        return dijkstra(graph, directed=True, indices=self.origins, return_predecessors=True)

    def path(self, pred_row, o, d) -> tuple[int, ...]:
        links = []
        v = d
        while v != o:
            u = pred_row[v]
            links.append(self.link_index[(int(u), int(v))])
            v = u
        return tuple(reversed(links))

    def all_or_nothing(self, cost: np.ndarray) -> tuple[np.ndarray, float]:
        """AON link flows and total shortest-path cost sum_od q * pi_od."""
        dist, pred = self.trees(cost)
        flow = np.zeros(self.net.n_links)
        sp_cost = 0.0
        for i, o in enumerate(self.origins):
            node_flow = self.demand[i].copy()
            sp_cost += float(np.dot(node_flow, np.where(np.isfinite(dist[i]), dist[i], 0.0)))
            for v in np.argsort(-dist[i]):        # farthest nodes first
                if v == o or node_flow[v] == 0 or pred[i, v] < 0:
                    continue
                u = pred[i, v]
                flow[self.link_index[(u, v)]] += node_flow[v]
                node_flow[u] += node_flow[v]
        return flow, sp_cost


def _cost_functions(net: Network, y, objective: str):
    """Return (link cost, derivative of link cost) for UE or SO."""
    c = net.eff_capacity(y)
    p = net.power
    k = 1.0 if objective == "ue" else p + 1.0
    if objective not in ("ue", "so"):
        raise ValueError(objective)

    def cost(x):
        return net.fftt * (1.0 + net.b * k * (x / c) ** p)

    def deriv(x):
        return net.fftt * net.b * k * p * np.power(np.maximum(x, 0.0) / c, p - 1) / c

    return cost, deriv


def _relative_gap(sp: _ShortestPaths, x, cost_fn) -> float:
    cost = cost_fn(x)
    _, spc = sp.all_or_nothing(cost)
    total = float(np.dot(x, cost))
    return (total - spc) / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Path-based gradient projection
# ---------------------------------------------------------------------------
def _gradient_projection(net, sp, cost_fn, deriv_fn, gap_tol, max_iter):
    # initial all-or-nothing on free-flow costs
    cost = cost_fn(np.zeros(net.n_links))
    dist, pred = sp.trees(cost)
    paths: dict[tuple[int, int], dict[tuple[int, ...], float]] = {}
    x = np.zeros(net.n_links)
    for i, o in enumerate(sp.origins):
        dests, q = net.od[o]
        for d, dem in zip(dests, q):
            p = sp.path(pred[i], o, d)
            paths[(o, d)] = {p: dem}
            x[list(p)] += dem

    history = []
    gap = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        gap = _relative_gap(sp, x, cost_fn)
        history.append(gap)
        if gap < gap_tol:
            break
        for i, o in enumerate(sp.origins):
            dist, pred = sp.trees(cost_fn(x))
            dests, _ = net.od[o]
            for d in dests:
                pset = paths[(o, d)]
                best = sp.path(pred[i], o, d)
                pset.setdefault(best, 0.0)
                c = cost_fn(x)
                best_links = np.array(best)
                c_best = c[best_links].sum()
                for p in list(pset.keys()):
                    if p == best:
                        continue
                    h = pset[p]
                    pl = np.array(p)
                    excess = c[pl].sum() - c_best
                    if excess <= 0:
                        continue
                    diff = np.setxor1d(pl, best_links)
                    curv = deriv_fn(x)[diff].sum()
                    shift = h if curv <= 0 else min(h, excess / curv)
                    x[pl] -= shift
                    x[best_links] += shift
                    pset[best] += shift
                    pset[p] = h - shift
                    if pset[p] <= 1e-12:
                        del pset[p]
                    c = cost_fn(x)
                    c_best = c[best_links].sum()
        np.maximum(x, 0.0, out=x)
    return x, gap, it, history


# ---------------------------------------------------------------------------
# Conjugate Frank-Wolfe
# ---------------------------------------------------------------------------
def _line_search(deriv, lo=0.0, hi=1.0, iters=60) -> float:
    if deriv(hi) <= 0:
        return hi
    if deriv(lo) >= 0:
        return lo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if deriv(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _conjugate_frank_wolfe(net, sp, cost_fn, deriv_fn, gap_tol, max_iter, x0=None):
    x = sp.all_or_nothing(cost_fn(np.zeros(net.n_links)))[0] if x0 is None else x0.copy()
    s_prev = None
    history = []
    gap = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        cost = cost_fn(x)
        aon, spc = sp.all_or_nothing(cost)
        total = float(np.dot(x, cost))
        gap = (total - spc) / total if total > 0 else 0.0
        history.append(gap)
        if gap < gap_tol:
            break
        if s_prev is None:
            s = aon
        else:
            h = deriv_fn(x)
            num = np.dot((s_prev - x) * h, aon - x)
            den = np.dot((s_prev - x) * h, aon - s_prev)
            alpha = num / den if den != 0 else 0.0
            alpha = min(max(alpha, 0.0), 1.0 - 1e-6)
            s = alpha * s_prev + (1 - alpha) * aon
        d = s - x
        lam = _line_search(lambda t: float(np.dot(cost_fn(x + t * d), d)))
        x = x + lam * d
        s_prev = None if lam < 1e-6 else s
    return x, gap, it, history


def solve_assignment(net: Network, y=None, objective: str = "ue", method: str = "gp",
                     gap_tol: float = 1e-6, max_iter: int | None = None) -> AssignmentResult:
    """Solve UE (objective='ue') or SO (objective='so') for capacity expansion y."""
    sp = _ShortestPaths(net)
    cost_fn, deriv_fn = _cost_functions(net, y, objective)
    if method == "gp":
        x, gap, it, hist = _gradient_projection(net, sp, cost_fn, deriv_fn, gap_tol, max_iter or 200)
    elif method == "cfw":
        x, gap, it, hist = _conjugate_frank_wolfe(net, sp, cost_fn, deriv_fn, gap_tol, max_iter or 2000)
    else:
        raise ValueError(method)
    t = net.travel_time(x, y)
    return AssignmentResult(x, t, float(np.dot(x, t)), float(gap), it, hist)

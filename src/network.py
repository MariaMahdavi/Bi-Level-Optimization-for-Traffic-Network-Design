"""Network data structures and TNTP loaders.

A network is a set of directed links with BPR travel-time functions:

    t_a(x_a, y_a) = t0_a * (1 + b_a * (x_a / (c_a * (1 + y_a)))**p_a)

where y_a >= 0 is the capacity expansion of link a (a fraction of its
original capacity) chosen by the upper-level planner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Network:
    name: str
    n_nodes: int
    init: np.ndarray          # tail node of each link (0-based)
    term: np.ndarray          # head node of each link (0-based)
    capacity: np.ndarray
    length: np.ndarray
    fftt: np.ndarray          # free-flow travel time t0
    b: np.ndarray
    power: np.ndarray
    od: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    # od[origin] = (destinations, demands)

    @property
    def n_links(self) -> int:
        return len(self.init)

    @property
    def total_demand(self) -> float:
        return float(sum(d.sum() for _, d in self.od.values()))

    # --- BPR link performance functions -----------------------------------
    def eff_capacity(self, y=None):
        return self.capacity if y is None else self.capacity * (1.0 + y)

    def travel_time(self, x, y=None):
        c = self.eff_capacity(y)
        return self.fftt * (1.0 + self.b * (x / c) ** self.power)

    def marginal_cost(self, x, y=None):
        """d(x * t(x))/dx: the link cost users would face under system optimum."""
        c = self.eff_capacity(y)
        return self.fftt * (1.0 + self.b * (self.power + 1) * (x / c) ** self.power)

    def beckmann(self, x, y=None):
        """Beckmann potential sum_a int_0^{x_a} t_a(w) dw (UE objective)."""
        c = self.eff_capacity(y)
        return float(np.sum(self.fftt * (x + self.b * c / (self.power + 1) * (x / c) ** (self.power + 1))))

    def tstt(self, x, y=None):
        """Total system travel time sum_a x_a t_a(x_a)."""
        return float(np.dot(x, self.travel_time(x, y)))

    def scaled(self, factor: float) -> "Network":
        """Copy of the network with all demands multiplied by ``factor``."""
        od = {o: (d.copy(), q * factor) for o, (d, q) in self.od.items()}
        return Network(self.name, self.n_nodes, self.init, self.term, self.capacity,
                       self.length, self.fftt, self.b, self.power, od)


def _read_tntp_body(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if "<END OF METADATA>" in line:
            return lines[i + 1:]
    return lines


def load_tntp(net_file: str | Path, trips_file: str | Path, name: str = "") -> Network:
    net_file, trips_file = Path(net_file), Path(trips_file)
    rows = []
    for line in _read_tntp_body(net_file):
        line = line.strip()
        if not line or line.startswith("~"):
            continue
        parts = line.rstrip(";").split()
        rows.append([float(v) for v in parts[:7]])
    arr = np.array(rows)
    init = arr[:, 0].astype(int) - 1
    term = arr[:, 1].astype(int) - 1
    n_nodes = int(max(init.max(), term.max()) + 1)

    od: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    origin = None
    buf: dict[int, float] = {}
    for line in _read_tntp_body(trips_file):
        line = line.strip()
        if not line:
            continue
        if line.startswith("Origin"):
            if origin is not None and buf:
                od[origin] = (np.array(list(buf.keys())), np.array(list(buf.values())))
            origin = int(line.split()[1]) - 1
            buf = {}
            continue
        for item in line.split(";"):
            if ":" in item:
                dest, val = item.split(":")
                dest, val = int(dest) - 1, float(val)
                if val > 0 and dest != origin:
                    buf[dest] = val
    if origin is not None and buf:
        od[origin] = (np.array(list(buf.keys())), np.array(list(buf.values())))

    return Network(name or net_file.stem.replace("_net", ""), n_nodes, init, term,
                   arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5], arr[:, 6], od)


def braess_network(with_bridge: bool) -> Network:
    """Classic Braess network: 4 nodes, 4000 travellers from node 0 to node 3.

    Links 0->1 and 2->3 are congestible (t = x/100); links 0->2 and 1->3 have
    a constant 45 min. The optional bridge 1->2 is (almost) free.
    Expressed as BPR with power 1 so the same solver can be reused.
    """
    eps = 1e-9
    init = [0, 0, 1, 2]
    term = [1, 2, 3, 3]
    fftt = [eps, 45.0, 45.0, eps]
    # t = t0 (1 + b x / c): with t0=eps, choose b/c so that t ~= x/100
    b = [1.0 / (100 * eps), 0.0, 0.0, 1.0 / (100 * eps)]
    if with_bridge:
        init.append(1); term.append(2); fftt.append(1e-6); b.append(0.0)
    n = len(init)
    return Network(
        "Braess" + (" + bridge" if with_bridge else ""), 4,
        np.array(init), np.array(term), np.ones(n), np.ones(n),
        np.array(fftt), np.array(b), np.ones(n),
        od={0: (np.array([3]), np.array([4000.0]))},
    )

"""Bend graph: flanges are nodes, bends are edges.

No networkx dependency -- the traversal is twenty lines and keeping the
dependency list short matters more than the convenience.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .model import Bend, Flange, Part


class UnflattenableError(Exception):
    """Raised when the part cannot be developed into a flat pattern."""


@dataclass
class TraversalStep:
    parent: int
    child: int
    bend: int


class BendGraph:
    def __init__(self, flanges: dict[int, Flange], bends: dict[int, Bend]):
        self.flanges = flanges
        self.bends = bends
        self.adj: dict[int, list[tuple[int, int]]] = {fid: [] for fid in flanges}
        for b in bends.values():
            if b.flange_a not in self.adj or b.flange_b not in self.adj:
                continue
            self.adj[b.flange_a].append((b.flange_b, b.id))
            self.adj[b.flange_b].append((b.flange_a, b.id))

    # ------------------------------------------------------------------
    @classmethod
    def from_part(cls, part: Part) -> BendGraph:
        return cls(part.flanges, part.bends)

    def largest_flange(self) -> int:
        if not self.flanges:
            raise UnflattenableError("part has no flange faces")
        return max(self.flanges.values(), key=lambda f: f.area).id

    def components(self) -> list[set[int]]:
        seen: set[int] = set()
        out: list[set[int]] = []
        for start in self.flanges:
            if start in seen:
                continue
            comp: set[int] = set()
            dq = deque([start])
            seen.add(start)
            while dq:
                n = dq.popleft()
                comp.add(n)
                for nb, _ in self.adj[n]:
                    if nb not in seen:
                        seen.add(nb)
                        dq.append(nb)
            out.append(comp)
        return out

    # ------------------------------------------------------------------
    def bfs(self, root: int) -> list[TraversalStep]:
        """Breadth-first traversal from `root`, returning parent->child steps.

        Raises if the graph contains a cycle. A cycle means a closed section
        (a tube, a box with a welded or interlocking seam) and such a part has
        no developable flat pattern -- better to say so than to emit garbage.
        """
        if root not in self.flanges:
            raise UnflattenableError(f"base flange {root} not in graph")

        order: list[TraversalStep] = []
        visited = {root}
        used_bends: set[int] = set()
        dq = deque([root])

        while dq:
            node = dq.popleft()
            for nb, bid in self.adj[node]:
                if bid in used_bends:
                    continue
                if nb in visited:
                    raise UnflattenableError(
                        f"closed section detected: bend {bid} closes a loop between "
                        f"flange {node} and flange {nb}. A closed section cannot be "
                        f"developed into a flat pattern."
                    )
                used_bends.add(bid)
                visited.add(nb)
                order.append(TraversalStep(parent=node, child=nb, bend=bid))
                dq.append(nb)

        unreached = set(self.flanges) - visited
        if unreached:
            raise UnflattenableError(
                f"flanges {sorted(unreached)} are not connected to the base flange "
                f"by any detected bend. The part may be multi-body, or a bend was "
                f"missed during detection."
            )
        return order

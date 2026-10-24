"""Section 3: Spatial Structure (CD(Sigma)) — Typed Multicategory / Copy-Discard PROP & Proposition 3.3.

Implements:
- Sort(name, validator) for semantic sorts S with domain invariants D(s)
- Hyperedge(name, in_sorts, out_sorts, fn) supporting port-grafting (o_i), parallel tensor, copy (Delta), discard (!)
- Universal shift-register impossibility verifier on M_reg(add, mul) (Proposition 3.3(1))
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Sort:
    name: str
    domain_check: Callable[[Any], bool] = lambda x: True

    def validate(self, value: Any) -> bool:
        return bool(self.domain_check(value))


WEALTH_SORT = Sort("Wealth", lambda w: isinstance(w, (int, float)) and w > 0.0)
DRIFT_SORT = Sort("Drift", lambda f: isinstance(f, (int, float)))


def make_var_sort(sigma_min2: float = 1e-6) -> Sort:
    return Sort("Var", lambda v: isinstance(v, (int, float)) and v >= sigma_min2)


@dataclass(frozen=True)
class Hyperedge:
    """A typed morphism o : (s_1, ..., s_k) -> (t_1, ..., t_m) in CD(Sigma)."""
    name: str
    in_sorts: tuple[Sort, ...]
    out_sorts: tuple[Sort, ...]
    fn: Callable[..., tuple[Any, ...]]

    def eval(self, *inputs: Any) -> tuple[Any, ...]:
        if len(inputs) != len(self.in_sorts):
            raise ValueError(f"{self.name}: expected {len(self.in_sorts)} inputs, got {len(inputs)}")
        for idx, (s, val) in enumerate(zip(self.in_sorts, inputs)):
            if not s.validate(val):
                raise ValueError(f"{self.name}: input port {idx} ({s.name}) violated domain invariant with {val!r}")
        outputs = self.fn(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        if len(outputs) != len(self.out_sorts):
            raise ValueError(f"{self.name}: expected {len(self.out_sorts)} outputs, got {len(outputs)}")
        for idx, (s, val) in enumerate(zip(self.out_sorts, outputs)):
            if not s.validate(val):
                raise ValueError(f"{self.name}: output port {idx} ({s.name}) violated domain invariant with {val!r}")
        return outputs

    def graft(self, g: "Hyperedge", port_index: int) -> "Hyperedge":
        """Port-grafting composition f o_i g (Definition 3.2, 0-indexed port_index)."""
        if len(g.out_sorts) != 1:
            raise ValueError("Port-grafting requires g to have a single output sort")
        if not (0 <= port_index < len(self.in_sorts)):
            raise IndexError(f"port_index {port_index} out of range for {len(self.in_sorts)} inputs")
        if self.in_sorts[port_index].name != g.out_sorts[0].name:
            raise TypeError(
                f"Sort mismatch on port {port_index}: expected {self.in_sorts[port_index].name}, got {g.out_sorts[0].name}"
            )
        new_in = self.in_sorts[:port_index] + g.in_sorts + self.in_sorts[port_index + 1:]
        p = len(g.in_sorts)

        def composed(*args: Any) -> tuple[Any, ...]:
            prefix = args[:port_index]
            mid_in = args[port_index:port_index + p]
            suffix = args[port_index + p:]
            g_out = g.eval(*mid_in)
            return self.eval(*prefix, g_out[0], *suffix)

        return Hyperedge(f"({self.name} o_{port_index} {g.name})", new_in, self.out_sorts, composed)

    def tensor(self, other: "Hyperedge") -> "Hyperedge":
        """Monoidal product f (x) g in CD(Sigma)."""
        new_in = self.in_sorts + other.in_sorts
        new_out = self.out_sorts + other.out_sorts
        k = len(self.in_sorts)

        def tensored(*args: Any) -> tuple[Any, ...]:
            return self.eval(*args[:k]) + other.eval(*args[k:])

        return Hyperedge(f"({self.name} (x) {other.name})", new_in, new_out, tensored)

    def compose(self, earlier: "Hyperedge") -> "Hyperedge":
        """Sequential categorical composition self o earlier in CD(Sigma)."""
        if len(self.in_sorts) != len(earlier.out_sorts) or any(
            a.name != b.name for a, b in zip(self.in_sorts, earlier.out_sorts)
        ):
            raise TypeError("Sequential composition sort mismatch")

        def seq(*args: Any) -> tuple[Any, ...]:
            mid = earlier.eval(*args)
            return self.eval(*mid)

        return Hyperedge(f"({self.name} o {earlier.name})", earlier.in_sorts, self.out_sorts, seq)


def copy_wire(sorts: tuple[Sort, ...]) -> Hyperedge:
    """Diagonal comonoid morphism Delta_u : u -> u (x) u in CD(Sigma)."""
    return Hyperedge(
        f"Delta[{','.join(s.name for s in sorts)}]",
        sorts,
        sorts + sorts,
        lambda *xs: tuple(xs) + tuple(xs),
    )


def discard_wire(sorts: tuple[Sort, ...]) -> Hyperedge:
    """Counit comonoid morphism !_u : u -> I in CD(Sigma)."""
    return Hyperedge(f"![{','.join(s.name for s in sorts)}]", sorts, (), lambda *_: ())


def register_pipeline(x0: float, x1: float, order: str) -> dict[str, tuple[float, float]]:
    """2-register shift-overwrite pipeline F_b(x0, x1) = (b(x0, x1), x0) (Definition 3.2(2))."""
    f_add = lambda a, b: (a + b, a)
    f_mul = lambda a, b: (a * b, a)
    if order == "sum-then-prod":
        first = f_add(x0, x1)
        second = f_mul(first[0], first[1])
    elif order == "prod-then-sum":
        first = f_mul(x0, x1)
        second = f_add(first[0], first[1])
    else:
        raise ValueError(f"Unknown order: {order}")
    return {"after_first": first, "after_second": second}


def verify_shift_register_impossibility(max_length: int = 6) -> dict[str, Any]:
    """Proposition 3.3(1): Exhaustively checks all words W_K in M_reg(add, mul) up to length max_length
    on input (2, 3), verifying the inductive invariant u_K > v_K >= 2 for K >= 1 and W_K(2, 3) != (5, 6).
    """
    real_sort = Sort("R", lambda x: isinstance(x, (int, float)))
    add_edge = Hyperedge("add", (real_sort, real_sort), (real_sort,), lambda a, b: (a + b,))
    mul_edge = Hyperedge("mul", (real_sort, real_sort), (real_sort,), lambda a, b: (a * b,))
    fanout_cd = add_edge.tensor(mul_edge).compose(copy_wire((real_sort, real_sort)))
    cartesian_target = fanout_cd.eval(2.0, 3.0)

    f_add = lambda s: (s[0] + s[1], s[0])
    f_mul = lambda s: (s[0] * s[1], s[0])

    order_sum_prod = f_mul(f_add((2.0, 3.0)))
    order_prod_sum = f_add(f_mul((2.0, 3.0)))

    states = [(2.0, 3.0)]
    words_checked = 1
    invariant_holds = states[0] != cartesian_target

    for _k in range(1, max_length + 1):
        next_states = []
        for s in states:
            for op in (f_add, f_mul):
                u, v = op(s)
                if not (u > v >= 2.0) or (u, v) == cartesian_target:
                    invariant_holds = False
                next_states.append((u, v))
                words_checked += 1
        states = next_states

    return {
        "cartesian_target": cartesian_target,
        "order_sum_then_prod": order_sum_prod,
        "order_prod_then_sum": order_prod_sum,
        "words_checked": words_checked,
        "invariant_holds": invariant_holds,
    }

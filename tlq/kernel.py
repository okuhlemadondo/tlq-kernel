"""TLQ Kernel v2.2.2-RC — Obligation & Rewrite Spine.

Provides the generic transformation, obligation (S/L/D/V), ledger, quantitative slack accumulation,
and named rejection machinery used across vertical problem rewrites and certificate-guarded solvers.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Any, Callable, Iterable

S, L, D, V = "S", "L", "D", "V"
PROPS = ("sound", "complete", "full")


class Rejected(Exception):
    """A transformation or derivation refused by the kernel. `failures` lists (rule, obligation, reason)."""
    def __init__(self, failures):
        self.failures = list(failures)
        super().__init__("; ".join(f"{r}: [{o}] {why}" for r, o, why in self.failures))

    def names(self) -> set[str]:
        return {o for _, o, _ in self.failures}


class NotExecutable(Exception):
    pass


@dataclass
class Obligation:
    name: str
    kind: str                                   # S / L / D / V
    check: Callable[..., tuple] | None = None   # S/L: (problem)->(ok,why); V: (problem, solution)->(ok,why); D: (problem)->(contradicted,why)
    required: bool = True                       # failure rejects the transformation
    for_property: str | None = None             # if not required: failure drops this preservation property
    lemma: str = ""                             # for L: the theorem/lemma invoked


@dataclass
class Explicit:
    """An explicitly presented solution set produced by a terminal solver."""
    solutions: list
    note: str = ""


@dataclass
class Transformation:
    name: str
    preserves: frozenset
    applies: Callable[[Any], tuple]                         # (problem) -> (ok, why)
    obligations: Callable[[Any], list]                      # (problem) -> [Obligation]
    forward: Callable[[Any], Any]                           # problem -> problem | [problems] | Explicit
    lift: Callable[[Any, Any], Any] = lambda P, s: s        # (input problem, solution) -> lifted solution
    slack: Callable[[Any], float] | None = None             # optional per-step rewrite error bound eps_rho >= 0 (Def. 5.1)
    fanout: bool = False
    terminal: bool = False
    executable: bool = True
    cost: Callable | None = None


@dataclass
class StepRecord:
    rule: str
    problem_kind: str
    checked: list = field(default_factory=list)             # (obligation, kind, status, why)
    preserves: frozenset = frozenset()
    slack_eps: float = 0.0


@dataclass
class Result:
    solutions: list
    preserves: frozenset
    ledger: list
    trace: list
    total_slack: float = 0.0
    note: str = ""
    classes: list = field(default_factory=list)

    @property
    def solution(self):
        cls = self.classes or [[s] for s in self.solutions]
        if len(cls) != 1:
            raise ValueError(f"{len(cls)} inequivalent solutions: selection is not declared")
        return cls[0][0]


def partition(solutions, equivalent):
    classes = []
    for s in solutions:
        for c in classes:
            if equivalent is not None and equivalent(c[0], s):
                c.append(s)
                break
        else:
            classes.append([s])
    return classes


class Derivation:
    """Build with .apply(rule) / .fan(rule, plan) ; finish with .solve(claim=...)."""

    def __init__(self, problem, ledger=None):
        self.root = problem
        self.nodes = [problem]
        self.steps: list[tuple[Transformation, StepRecord, list]] = []
        self.ledger = list(ledger or [])
        self.branches = None
        self.terminal = None

    @property
    def current(self):
        return self.nodes[-1]

    def _check(self, rule: Transformation, P):
        ok, why = rule.applies(P)
        if not ok:
            raise Rejected([(rule.name, "applicability", why)])
        eps_rho = float(rule.slack(P)) if rule.slack is not None else 0.0
        if not math.isfinite(eps_rho) or eps_rho < 0.0:
            raise Rejected([
                (
                    rule.name,
                    "slack",
                    f"invalid rewrite slack {eps_rho}: Definition 5.1 requires a finite non-negative error bound eps_rho >= 0",
                )
            ])
        rec = StepRecord(rule.name, getattr(P, "kind", type(P).__name__), slack_eps=eps_rho)
        failures, deferred, props = [], [], set(rule.preserves)
        for ob in rule.obligations(P):
            if ob.kind in (S, L):
                good, why = ob.check(P)
                rec.checked.append((ob.name, ob.kind, "ok" if good else "FAILED", why))
                if not good:
                    if ob.required:
                        failures.append((rule.name, ob.name, why))
                    elif ob.for_property:
                        props.discard(ob.for_property)
            elif ob.kind == D:
                contradicted, why = ob.check(P) if ob.check else (False, "")
                if contradicted:
                    rec.checked.append((ob.name, D, "CONTRADICTED", why))
                    failures.append((rule.name, ob.name, "declared assumption contradicts the world model: " + why))
                else:
                    rec.checked.append((ob.name, D, "declared", why))
                    self.ledger.append((rule.name, ob.name))
            elif ob.kind == V:
                rec.checked.append((ob.name, V, "deferred", ""))
                deferred.append(ob)
        if failures:
            raise Rejected(failures)
        rec.preserves = frozenset(props)
        return rec, deferred

    def apply(self, rule: Transformation):
        if self.branches or self.terminal:
            raise Rejected([(rule.name, "structure", "derivation already branched or terminated")])
        P = self.current
        rec, deferred = self._check(rule, P)
        if rule.terminal:
            if not rule.executable:
                raise NotExecutable(rule.name)
            out = rule.forward(P)
            if not isinstance(out, Explicit):
                raise Rejected([(rule.name, "structure", "terminal rule did not return an explicit solution set")])
            self.terminal = (rule, rec, out, deferred)
            return self
        if rule.fanout:
            raise Rejected([(rule.name, "structure", "use .fan() for fan-out rules")])
        self.steps.append((rule, rec, deferred))
        self.nodes.append(rule.forward(P) if rule.executable else None)
        return self

    def fan(self, rule: Transformation, plan: Callable[[int, "Derivation"], "Derivation"]):
        if self.branches or self.terminal:
            raise Rejected([(rule.name, "structure", "derivation already branched or terminated")])
        if not rule.fanout:
            raise Rejected([(rule.name, "structure", "use .apply() for non-fan-out rules")])
        P = self.current
        rec, deferred = self._check(rule, P)
        subs = [plan(i, Derivation(Q, ledger=[])) for i, Q in enumerate(rule.forward(P))]
        self.branches = (rule, rec, subs, deferred)
        return self

    def _solve_here(self):
        if self.terminal:
            rule, rec, out, deferred = self.terminal
            sols = out.solutions
            Derivation._run_V(rule, rec, deferred, self.current, sols)
            return sols, frozenset(rec.preserves), [], [rec], rec.slack_eps
        if self.branches:
            rule, rec, subs, deferred = self.branches
            results = [s.solve() for s in subs]
            sols = rule.lift(self.current, [r.solutions for r in results])
            props = frozenset(rec.preserves).intersection(*[r.preserves for r in results])
            ledger = [x for r in results for x in r.ledger]
            trace = [rec] + [("branch", i, r.trace) for i, r in enumerate(results)]
            Derivation._run_V(rule, rec, deferred, self.current, sols)
            props = props & rec.preserves
            max_branch_slack = max((r.total_slack for r in results), default=0.0)
            return sols, props, ledger, trace, rec.slack_eps + max_branch_slack
        raise Rejected([("derivation", "structure", "no terminal solver: the derivation does not reach an explicit solution set")])

    @staticmethod
    def _run_V(rule, rec, deferred, P, sols):
        props = set(rec.preserves)
        for ob in deferred:
            bad = None
            for s in sols:
                good, why = ob.check(P, s)
                if not good:
                    bad = why
                    break
            if not sols:
                good, why = ob.check(P, None)
                bad = None if good else why
            rec.checked.append((ob.name, V, "ok" if bad is None else "FAILED", bad or ""))
            if bad is not None:
                if ob.required:
                    raise Rejected([(rule.name, ob.name, "a-posteriori check failed: " + bad)])
                if ob.for_property:
                    props.discard(ob.for_property)
        rec.preserves = frozenset(props)

    def solve(self, claim: Iterable[str] = ()) -> Result:
        if any(n is None for n in self.nodes):
            raise NotExecutable("derivation contains a non-executable transformation")
        tail = Derivation.__new__(Derivation)
        tail.__dict__.update(self.__dict__)
        tail.nodes = [self.nodes[-1]]
        sols, props, ledger, trace, total_slack = Derivation._solve_here(tail)
        for k in range(len(self.steps) - 1, -1, -1):
            rule, rec, deferred = self.steps[k]
            P = self.nodes[k]
            sols = [rule.lift(P, s) for s in sols]
            Derivation._run_V(rule, rec, deferred, P, sols)
            props = props & rec.preserves
            total_slack += rec.slack_eps
            trace = [rec] + trace
        claim = frozenset(claim)
        if not claim <= props:
            raise Rejected([("derivation", "claim", f"claimed {sorted(claim)} but the derivation certifies only {sorted(props)}")])
        eq = getattr(self.root, "equivalent", None)
        return Result(sols, props, self.ledger + ledger, trace, total_slack=total_slack, classes=partition(sols, eq))

    def clone(self):
        import copy
        d = Derivation.__new__(Derivation)
        d.root, d.nodes, d.ledger = self.root, list(self.nodes), list(self.ledger)
        d.steps = [(r, copy.deepcopy(rec), list(dv)) for r, rec, dv in self.steps]
        d.branches, d.terminal = self.branches, self.terminal
        return d

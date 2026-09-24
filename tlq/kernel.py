"""TLQ kernel prototype v0.1 — generic machinery only.

Nothing in this module knows about finance, teams, games or any forcing case.

Objects
-------
Problem            : any object with a `.kind` attribute (W, E, T live inside it).
Obligation         : a preservation obligation under stated assumptions, with a justification mode:
                       S  syntactic   checked on the problem specification now
                       L  lemma       a named theorem whose hypotheses are checked now
                       D  declared    an assumption about the world; accepted into the ledger
                                      unless the problem's own world model contradicts it
                       V  verified    checked a posteriori on the solution
                     An obligation is either `required` (failure => the transformation is rejected)
                     or attached to a single preservation property (failure => that property is dropped).
Transformation     : semantic part  = (forward map, lift, declared preservation, obligations)
                     executable part = `executable` flag (semantic legitimacy != executability).
Derivation         : a DAG: linear steps, fan-out into branches, terminal solvers.
                     Lifts compose backward; preservation composes by intersection;
                     the ledger is the union of declared obligations; V obligations run after solving.

Preservation vocabulary (for a transformation P => P' with lift l):
  'sound'    every l(s') with s' in Sol(P') is in Sol(P)
  'complete' Sol(P) nonempty  =>  Sol(P') nonempty           (existence-completeness)
  'full'     every s in Sol(P) equals l(s') for some s' in Sol(P')
These three compose under sequential composition by intersection. Quantitative slack and confluence are
deliberately NOT part of this vocabulary (confluence is a property of rule systems, tested by `confluence`).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

S, L, D, V = "S", "L", "D", "V"
PROPS = ("sound", "complete", "full")


class Rejected(Exception):
    """A transformation or derivation the kernel refuses. `failures` lists (rule, obligation, reason)."""
    def __init__(self, failures):
        self.failures = list(failures)
        super().__init__("; ".join(f"{r}: [{o}] {why}" for r, o, why in self.failures))

    def names(self):
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
    lemma: str = ""                             # for L: the theorem invoked


@dataclass
class Explicit:
    """An explicitly presented solution set (what a solver produces). `solutions` is a list."""
    solutions: list
    note: str = ""


@dataclass
class Transformation:
    name: str
    preserves: frozenset
    applies: Callable[[Any], tuple]                         # (problem) -> (ok, why)
    obligations: Callable[[Any], list]                      # (problem) -> [Obligation]
    forward: Callable[[Any], Any]                           # problem -> problem | [problems] (fan-out) | Explicit
    lift: Callable[[Any, Any], Any] = lambda P, s: s        # (input problem, solution(s) of output) -> solution of input
    fanout: bool = False
    terminal: bool = False
    executable: bool = True
    cost: Callable | None = None      # optional resource estimate (executable kernel only; not part of the semantics)


@dataclass
class StepRecord:
    rule: str
    problem_kind: str
    checked: list = field(default_factory=list)     # (obligation, kind, status, why)
    preserves: frozenset = frozenset()


@dataclass
class Result:
    solutions: list
    preserves: frozenset
    ledger: list
    trace: list
    note: str = ""
    classes: list = field(default_factory=list)   # solutions partitioned by the solution concept's own equivalence

    @property
    def solution(self):
        """A representative, if the solutions form a single equivalence class under the task's solution concept."""
        cls = self.classes or [[s] for s in self.solutions]
        if len(cls) != 1:
            raise ValueError(f"{len(cls)} inequivalent solutions: selection is not declared")
        return cls[0][0]


def partition(solutions, equivalent):
    """Partition solutions by an equivalence relation supplied by the solution concept (None = identity)."""
    classes = []
    for s in solutions:
        for c in classes:
            if equivalent is not None and equivalent(c[0], s):
                c.append(s); break
        else:
            classes.append([s])
    return classes


class Derivation:
    """Build with .apply(rule) / .fan(rule, plan) ; finish with .solve(claim=...)."""

    def __init__(self, problem, ledger=None):
        self.root = problem
        self.nodes = [problem]          # problems along the current linear spine
        self.steps: list[tuple[Transformation, StepRecord, list]] = []   # (rule, record, deferred V obligations)
        self.ledger = list(ledger or [])
        self.branches = None            # (rule, record, [Derivation]) after a fan-out
        self.terminal = None            # (rule, record, Explicit, deferred V)

    # -------------------------------------------------------------- one step
    @property
    def current(self):
        return self.nodes[-1]

    def _check(self, rule: Transformation, P):
        ok, why = rule.applies(P)
        if not ok:
            raise Rejected([(rule.name, "applicability", why)])
        rec = StepRecord(rule.name, getattr(P, "kind", type(P).__name__))
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
        """Fan-out: rule.forward returns a list of problems; `plan(i, subderivation)` derives branch i."""
        P = self.current
        rec, deferred = self._check(rule, P)
        subs = [plan(i, Derivation(Q, ledger=[])) for i, Q in enumerate(rule.forward(P))]
        self.branches = (rule, rec, subs, deferred)
        return self

    # -------------------------------------------------------------- solve and lift back
    def _solve_here(self):
        """Return (solutions at self.current, preservation of the tail, ledger of the tail, trace of the tail)."""
        if self.terminal:
            rule, rec, out, deferred = self.terminal
            sols = out.solutions
            Derivation._run_V(rule, rec, deferred, self.current, sols)
            return sols, frozenset(rec.preserves), [], [rec]
        if self.branches:
            rule, rec, subs, deferred = self.branches
            results = [s.solve() for s in subs]
            sols = rule.lift(self.current, [r.solutions for r in results])
            props = frozenset(rec.preserves).intersection(*[r.preserves for r in results])
            ledger = [x for r in results for x in r.ledger]
            trace = [rec] + [("branch", i, r.trace) for i, r in enumerate(results)]
            Derivation._run_V(rule, rec, deferred, self.current, sols)
            props = props & rec.preserves
            return sols, props, ledger, trace
        raise Rejected([("derivation", "structure", "no terminal solver: the derivation does not reach an explicit solution set")])

    @staticmethod
    def _run_V(rule, rec, deferred, P, sols):
        """Run a-posteriori obligations. Required ones reject; property-attached ones drop the property."""
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

    def solve(self, claim: Iterable[str] = ()):
        if any(n is None for n in self.nodes):
            raise NotExecutable("derivation contains a non-executable transformation")
        tail = Derivation.__new__(Derivation)
        tail.__dict__.update(self.__dict__)
        tail.nodes = [self.nodes[-1]]
        sols, props, ledger, trace = Derivation._solve_here(tail)
        # lift back through the linear spine, running V obligations at each level
        for k in range(len(self.steps) - 1, -1, -1):
            rule, rec, deferred = self.steps[k]
            P = self.nodes[k]
            sols = [rule.lift(P, s) for s in sols]
            Derivation._run_V(rule, rec, deferred, P, sols)
            props = props & rec.preserves
            trace = [rec] + trace
        claim = frozenset(claim)
        if not claim <= props:
            raise Rejected([("derivation", "claim", f"claimed {sorted(claim)} but the derivation certifies only {sorted(props)}")])
        eq = getattr(self.root, "equivalent", None)
        return Result(sols, props, self.ledger + ledger, trace, classes=partition(sols, eq))

    def clone(self):
        """Independent copy for search: problems are shared (rules never mutate them); step records are copied."""
        import copy
        d = Derivation.__new__(Derivation)
        d.root, d.nodes, d.ledger = self.root, list(self.nodes), list(self.ledger)
        d.steps = [(r, copy.deepcopy(rec), list(dv)) for r, rec, dv in self.steps]
        d.branches, d.terminal = self.branches, self.terminal
        return d


# ------------------------------------------------------------------ rule systems
def confluence(problem, instances: list, signature: Callable[[Any], Any], max_steps=50):
    """Explore every order of applying rule instances until none applies. Returns the set of normal-form signatures.
    A rule system is confluent on `problem` iff exactly one normal form is reached."""
    seen, normal = set(), set()

    def explore(P, depth):
        sig = signature(P)
        if (sig, depth) in seen:
            return
        seen.add((sig, depth))
        moved = False
        for rule in instances:
            if rule.applies(P)[0] and all(ob.check(P)[0] for ob in rule.obligations(P) if ob.kind in (S, L) and ob.required):
                moved = True
                explore(rule.forward(P), depth + 1)
        if not moved:
            normal.add(sig)
    explore(problem, 0)
    return normal

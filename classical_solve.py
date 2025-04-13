import uuid
from pathlib import Path

import pulp
from docplex.mp.model import Model


def docplex_to_pulp(mdl: Model):
    filename = f"{uuid.uuid4()}.mps"
    mdl.export_as_mps(filename)
    vars, prob = pulp.LpProblem.fromMPS(filename)
    Path(filename).unlink()
    return prob


def analyze(p: pulp.LpProblem) -> None:
    p.solve()

    print(f"Status: {pulp.LpStatus[p.status]}")
    print(f"{p.name} = {pulp.value(p.objective)}")

    print("Variables")
    print(f"{'Name':16} {'Value':>12} {'Reduced cost':>12} {'Coefficient':>12}")
    for v in p.variables():
        print(
            f"{v.name:16.16} {pulp.value(v):12.6} {v.dj:12.6}"
            f" {p.objective.get(v, 0):12}",
        )

    print("Constraints")
    print(f"{'Name':16} {'Value':>12} {'Shadow price':>12} {'Bound':>12}")
    for c in p.constraints.values():
        print(
            f"{c.name or '':16.16} {-c.constant - c.slack:12} {c.pi:12.6}"
            f" {-c.constant:12}",
        )


def verify(p: pulp.LpProblem, soln):
    for v in p.variables():
        v.varValue = soln[v.name]

    return p.valid()

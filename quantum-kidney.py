import numpy as np
import pyomo.core as pyo


def ilp(a: np.ndarray, b: np.ndarray, c: np.ndarray, bound: int) -> pyo.ConcreteModel:
    # model constraint: a*x <= b
    model = pyo.ConcreteModel()
    assert b.ndim == c.ndim == 1

    num_vars = len(c)
    num_constraints = len(b)

    assert a.shape == (num_constraints, num_vars)

    model.x = pyo.Var(
        # here we bound x to be from 0 to to a given bound
        range(num_vars),
        domain=pyo.NonNegativeIntegers,
        bounds=(0, bound),
    )

    @model.Constraint(range(num_constraints))
    def monotone_rule(model, idx):
        return a[idx, :] @ list(model.x.values()) <= float(b[idx])

    # model objective: max(c * x)
    model.cost = pyo.Objective(expr=c @ list(model.x.values()), sense=pyo.maximize)

    return model

A = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
b = np.array([1, 2, 3])
c = np.array([1, 2, 3])

# Instantiate the model
ilp_model = ilp(A, b, c, 3)

ilp_model.pprint()

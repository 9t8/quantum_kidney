import kepio
import matplotlib.pyplot as plt
from docplex.mp.model import Model

import classical_solve
import quantum_solve


def invert(adj):
    inv = {}
    for src in adj:
        for dst in adj[src]:
            if dst in inv:
                inv[dst].append(src)
            else:
                inv[dst] = [src]
    return inv


def kep_to_model(filename):
    adj, w = kepio.read_kep(filename)

    kill_list = ["dummy"]
    while kill_list != []:
        seen_recip_is = set()
        for donor_i in adj:
            adj[donor_i] = [*filter(lambda i: i in adj, adj[donor_i])]
            for recip_i in adj[donor_i]:
                seen_recip_is.add(recip_i)

        kill_list = [
            donor_i
            for donor_i in adj
            if adj[donor_i] == [] or donor_i not in seen_recip_is
        ]
        for donor_i in kill_list:
            adj.pop(donor_i)

    inv = invert(adj)

    mdl = Model()

    var_names = []
    matches = {}
    for donor_i in adj:
        for recip_i in adj[donor_i]:
            var_names.append(f"match_{donor_i}_{recip_i}")
            matches[(donor_i, recip_i)] = mdl.binary_var(f"match_{donor_i}_{recip_i}")

    mdl.maximize(mdl.sum(m for m in matches.values()))

    # No empty constraints because that breaks OpenQAOA!
    for donor_i in adj:  # Donor has 1 (spare) organ
        mdl.add_constraint(
            mdl.sum(matches[(donor_i, recip_i)] for recip_i in adj[donor_i]) <= 1,
            f"donor_{donor_i}",
        )
    for recip_i in inv:  # Recip needs 1 organ
        mdl.add_constraint(
            mdl.sum(matches[(donor_i, recip_i)] for donor_i in inv[recip_i]) <= 1,
            f"recip_{recip_i}",
        )
    for i in adj:  # Donors are not altruistic
        mdl.add_constraint(
            mdl.sum(matches[(donor_i, i)] for donor_i in inv[i])
            - mdl.sum(matches[(i, recip_i)] for recip_i in adj[i])
            >= 0,
            f"pair_{i}",
        )

    return mdl, var_names


def postselect(samples, var_names, prob):
    good_samples = {}
    for sample in samples:
        soln = {}
        for var_name, is_matched in zip(var_names, sample):
            soln[var_name] = is_matched == "1"

        if classical_solve.verify(prob, soln):
            good_samples[sample] = samples[sample]
    return good_samples


def analyze(dataset, shots, min_vars=1, max_vars=20):
    mdl, var_names = kep_to_model(f"build/small/{dataset}.input.gz")

    if not (min_vars <= mdl.number_of_variables <= max_vars):
        return mdl.number_of_variables

    samples = quantum_solve.quantum_solve(mdl, shots)
    prob = classical_solve.docplex_to_pulp(mdl)

    solns = postselect(samples, var_names, prob)

    classical_solve.analyze(prob)

    objective_counts = [0] * int(1.1 - prob.objective.value())
    for soln in solns:
        objective_counts[soln.count("1")] += solns[soln]
    
    # Graphing
    fig, ax = plt.subplots()
    ax.stairs(objective_counts, fill=True)
    ax.vlines(
        -prob.objective.value(),
        0,
        max(objective_counts),
        linestyle="--",
        color="red",
        label="optimal",
    )
    ax.legend()
    ax.set_ylabel("count")
    ax.set_xlabel("values")
    ax.set_title(f"Dataset {dataset}: vars = {mdl.number_of_variables}")
    fig.savefig(f"build/{dataset}.png")

    return mdl.number_of_variables

# Solves all solvable datasets
vars_dist = {}
for i in range(10, 40, 10):
    for j in range(1, 51):
        vars = analyze(f"{i}_{j:02}", 5000)
        if vars in vars_dist:
            vars_dist[vars] += 1
        else:
            vars_dist[vars] = 1

print("Variables distribution:", dict(sorted(vars_dist.items())))

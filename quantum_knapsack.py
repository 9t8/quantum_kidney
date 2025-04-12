import numpy as np
import matplotlib.pyplot as plt
from openqaoa.problems import FromDocplex2IsingModel
from docplex.mp.model import Model
from collections import defaultdict
import pennylane as qml
# -----------------------------   QAOA circuit ------------------------------------

def quantum_solve(mdl):
    shots = 1000  # Number of samples used
    dev = qml.device("default.qubit", shots=shots)
    n_items = mdl.number_of_binary_variables

    @qml.qnode(dev)
    def qaoa_circuit(gammas, betas, h, J, num_qubits):
        wmax = max(
            np.max(np.abs(list(h.values()))), np.max(np.abs(list(h.values())))
        )  # Normalizing the Hamiltonian is a good idea
        p = len(gammas)
        # Apply the initial layer of Hadamard gates to all qubits
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
        # repeat p layers the circuit shown in Fig. 1
        for layer in range(p):
            # ---------- COST HAMILTONIAN ----------
            for ki, v in h.items():  # single-qubit terms
                qml.RZ(2 * gammas[layer] * v / wmax, wires=ki[0])
            for kij, vij in J.items():  # two-qubit terms
                qml.CNOT(wires=[kij[0], kij[1]])
                qml.RZ(2 * gammas[layer] * vij / wmax, wires=kij[1])
                qml.CNOT(wires=[kij[0], kij[1]])
            # ---------- MIXER HAMILTONIAN ----------
            for i in range(num_qubits):
                qml.RX(-2 * betas[layer], wires=i)
        return qml.sample()


    def samples_dict(samples, n_items):
        """Just sorting the outputs in a dictionary"""
        results = defaultdict(int)
        for sample in samples:
            results["".join(str(i) for i in sample)[:n_items]] += 1
        return results


    # Annealing schedule for QAOA
    betas = np.linspace(0, 1, 10)[::-1]  # Parameters for the mixer Hamiltonian
    gammas = np.linspace(0, 1, 10)  # Parameters for the cost Hamiltonian (Our Knapsack problem)

    fig, ax = plt.subplots()
    ax.plot(betas, label=r"$\beta_i$", marker="o", markersize=8, markeredgecolor="black")
    ax.plot(gammas, label=r"$\gamma_i$", marker="o", markersize=8, markeredgecolor="black")
    ax.set_xlabel("i", fontsize=18)
    ax.legend()
    fig.savefig('build/fig1.png')


    lambda_1, lambda_2 = (
        0.96,
        0.0371,
    )
    ising_hamiltonian = FromDocplex2IsingModel(
        mdl,
        unbalanced_const=True,
        # strength_ineq=[lambda_1, lambda_2], 
    ).ising_model

    h_new = {
        tuple(i): w for i, w in zip(ising_hamiltonian.terms, ising_hamiltonian.weights) if len(i) == 1
    }
    J_new = {
        tuple(i): w for i, w in zip(ising_hamiltonian.terms, ising_hamiltonian.weights) if len(i) == 2
    }

    samples_unbalanced = samples_dict(
            qaoa_circuit(gammas, betas, h_new, J_new, num_qubits=n_items), n_items
        )
    return samples_unbalanced

'''
values_unbalanced = {
    sum_values(sample_i, values_list): count
    for sample_i, count in samples_unbalanced.items()
    if sum_weight(sample_i, weights_list) <= maximum_weight
}  # saving only the solutions that fulfill the constraint

print(
    f"The number of solutions using unbalanced penalization is {samples_unbalanced[opt_str]} out of {shots}"
)


fig, ax = plt.subplots()
ax.hist(
    values_unbalanced.keys(),
    weights=values_unbalanced.values(),
    bins=50,
    edgecolor="black",
    label="unbalanced",
    align="right",
)

ax.vlines(-min_cost, 0, 3000, linestyle="--", color="black", label="Optimal", linewidth=2)
ax.set_yscale("log")
ax.legend()
ax.set_ylabel("counts")
ax.set_xlabel("values")
fig.savefig('build/fig2.png')
'''
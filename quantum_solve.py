import numpy as np
from openqaoa.problems import FromDocplex2IsingModel
from collections import defaultdict
import pennylane as qml
import math


def quantum_solve(mdl, shots):
    dev = qml.device("default.qubit", shots=shots)
    n_items = mdl.number_of_variables
    p = int(math.log2(n_items)*10)
    @qml.qnode(dev)
    def qaoa_circuit(gammas, betas, h, J, num_qubits):
        wmax = max(
            np.max(np.abs(list(h.values()))), np.max(np.abs(list(h.values())))
        )
        # Apply the initial layer of Hadamard gates to all qubits
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
        # p Layers
        for layer in range(int(p)):
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
    
    # Dictionary of solutions and frequency
    def samples_dict(samples, n_items):
        results = defaultdict(int)
        for sample in samples:
            results["".join(str(i) for i in sample)[:n_items]] += 1
        return results

    # Annealing schedule for QAOA
    betas = np.linspace(np.pi/4, 0, p)  # Parameters for the mixer Hamiltonian
    gammas = np.linspace(0, math.pi/2, p)  # Parameters for the cost Hamiltonian

    # Unbalancede Penalization Coefficients
    lambda_1, lambda_2 = (
        0.85,
        0.85,
    )
    # Generate Hamiltonian
    ising_hamiltonian = FromDocplex2IsingModel(
        mdl,
        unbalanced_const=True,
        strength_ineq=[lambda_1, lambda_2],
    ).ising_model

    # Qubit terms
    h_new = {
        tuple(i): w
        for i, w in zip(ising_hamiltonian.terms, ising_hamiltonian.weights)
        if len(i) == 1
    }
    J_new = {
        tuple(i): w
        for i, w in zip(ising_hamiltonian.terms, ising_hamiltonian.weights)
        if len(i) == 2
    }

    # Execute Circuit
    samples_unbalanced = samples_dict(
        qaoa_circuit(gammas, betas, h_new, J_new, num_qubits=n_items), n_items
    )
    return samples_unbalanced

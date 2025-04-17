# QAOA for the kidney exchange problem

Bitcamp 2025 Advanced Quantum Track winner

https://devpost.com/software/kidney-donation-chain-integer-programming-with-qaoa

[Slideshow](https://docs.google.com/presentation/d/1QfcvnHXAY4uffTjImgYDVJnK4g6A-NWtu9WTt6nQ9Tk/edit?usp=sharing)

1. Run `kepio.unzip_data()`
1. Execute `quantum_kidney.py`

## Inspiration

We were looking into applications of linear programming and found kidney donation chain optimization in a textbook. There hasn't been any interest in using quantum computing for this problem, so we decided to look into it.

## What it does

We use a quantum algorithm, QAOA, to solve the kidney exchange problem. We think we achieved exponential speedup.

## How we built it

We downloaded datasets from the internet, modernized the provided parser, wrote our own pruner and problem generator, used OpenQAOA to convert the BIP to an Ising model, used PennyLane to simulate QAOA, used the CPLEX Python bindings, PuLP, and CBC to get classical solutions, used PuLP to filter the quantum output, and used Matplotlib to generate charts.

## Challenges we ran into

The SDK we plannede to use, classiq, requires manual verification of new accounts and was buggy, so we had to switch tools. OpenQAOA's DOcplex to Ising model converter threw obscure errors. Performance constraints of quantum simulation on classical computers prevented us from using datasets with more than 30 donor-recipient pairs, even with unbalanced penalization to reduce variable count. Additionally, pruning optimizations made predicting model size from dataset size challenging.

## Accomplishments that we're proud of

We implemented a working quantum algorithm that has an exponential speedup in many realistic cases.

## What we learned

We learned a lot about the applications, implementation, and analysis of quantum optimization algorithms.

## What's next

We could add support for success probabilities and altruistic donors. We could run further simulations to properly optimize the hamiltonian parameters for QAOA.

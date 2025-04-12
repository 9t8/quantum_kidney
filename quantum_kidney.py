import pulp
import kepio
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

    mdl = Model()

    matches = {}
    vars = []
    for donor_i in adj:
        for recip_i in adj[donor_i]:
            matches[(donor_i, recip_i)] = mdl.binary_var(f'match_{donor_i}_{recip_i}')
            vars.append((donor_i, recip_i))

    mdl.maximize(mdl.sum(m for m in matches.values()))
    
    # No empty constraints because that breaks OpenQAOA!
    for donor_i in adj:
        mdl.add_constraint(mdl.sum(matches[(donor_i, recip_i)] for recip_i in adj[donor_i]) <= 1, f'donor_{donor_i}')
    inv = invert(adj)
    for recip_i in inv:
        mdl.add_constraint(mdl.sum(matches[(donor_i, recip_i)] for donor_i in inv[recip_i]) <= 1, f'recip_{recip_i}')

    return mdl, (vars, adj, inv)

def is_satisfying(sample, verif_info):
    vars, adj, inv = verif_info
    matches = {}
    for pair, is_matched in zip(vars, sample):
        matches[pair] = (is_matched == '1')

    for donor_i in adj:
        if sum(matches[(donor_i, recip_i)] for recip_i in adj[donor_i]) > 1:
            return False
    for recip_i in inv:
        if sum(matches[(donor_i, recip_i)] for donor_i in inv[recip_i]) > 1:
            return False
    return True

def postselect(samples, verif_info):
    good_samples = {}
    for sample in samples:
        if is_satisfying(sample, verif_info):
            good_samples[sample] = samples[sample]
    return good_samples

mdl, verif_info = kep_to_model('build/small/10_03.input.gz')

prob = classical_solve.docplex_to_pulp(mdl)

classical_solve.analyze(prob)

quantum_out = quantum_solve.quantum_solve(mdl)
print(postselect(quantum_out, verif_info))

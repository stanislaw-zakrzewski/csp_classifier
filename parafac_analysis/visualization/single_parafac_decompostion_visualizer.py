import numpy as np
import matplotlib.pyplot as plt

from tkinter import filedialog as fd


def visualize_best_decomposition(factors, rank, replica, atoms, selected_channels, a_label, b_label):
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle(
        'PARAFAC decomposition of rank {}, replica: {})'.format(rank, replica))
    fig.set_figwidth(25)
    fig.set_figheight(10)

    for atom_index in atoms.keys():
        atom_metadata = atoms[atom_index]
        atom_relation = atom_metadata['relation']
        atom_pvalue = atom_metadata['pvalue']
        atom_weight = atom_metadata['weight']

        x1 = selected_channels
        y1 = factors.factors[1][:, atom_index]
        y2 = factors.factors[2][:, atom_index]
        x2 = range(1, len(y2) + 1)
        ax1.plot(x1, y1, label='Atom {}: {} {} than {} (p = {}, weight = {})'.format(atom_index, a_label, atom_relation,
                                                                                     b_label,
                                                                                     "{0:.3g}".format(atom_pvalue),
                                                                                     "{0:.3g}".format(atom_weight)))
        ax2.plot(x2, y2, label="Atom {}: {} {} than {} (p = {}, weight = {})".format(atom_index, a_label, atom_relation,
                                                                                     b_label,
                                                                                     "{0:.3g}".format(atom_pvalue),
                                                                                     "{0:.3g}".format(atom_weight)))
    ax1.legend()
    ax2.legend()
    plt.show()


filename = fd.askopenfilename(filetypes=[("NumPy data files", "*.npy")])

with open(filename, 'rb') as out_file:
    data_file = np.load(out_file, allow_pickle=True)
    all_decompositions = data_file.item().get("decompositions")
    selected_channels = data_file.item().get("metadata")['selected_channels']
    a_label = data_file.item().get("metadata")['a_label']
    b_label = data_file.item().get("metadata")['b_label']
    statistically_significant_decompositions = data_file.item().get("statistically_significant_decompositions")
    structured_statistically_significant_decompositions = {}
    for statistically_significant_decomposition in statistically_significant_decompositions:
        rank = statistically_significant_decomposition['rank']
        replica = statistically_significant_decomposition['replica']
        atom = statistically_significant_decomposition['atom']
        pvalue = statistically_significant_decomposition['pvalue']
        relation = statistically_significant_decomposition['relation']
        weight = statistically_significant_decomposition['weight']
        if rank not in structured_statistically_significant_decompositions:
            structured_statistically_significant_decompositions[rank] = {}
        sssd_rank = structured_statistically_significant_decompositions[rank]
        if replica not in sssd_rank:
            sssd_rank[replica] = {}
        sssd_rank_replica = sssd_rank[replica]
        sssd_rank_replica[atom] = {'pvalue': pvalue, 'relation': relation, 'weight': weight}

    print("Statistically significant PARAFAC ranks")
    for available_rank in structured_statistically_significant_decompositions:
        print(available_rank)
    parafac_rank = int(input("Select rank: "))

    print("Statistically significant PARAFAC replicas for rank {}".format(parafac_rank))
    for available_replica in structured_statistically_significant_decompositions[parafac_rank]:
        print('{}: {} atoms'.format(available_replica, len(
            structured_statistically_significant_decompositions[parafac_rank][available_replica])))
    parafac_replica = int(input("Select replica: "))
    parafac_atoms = structured_statistically_significant_decompositions[parafac_rank][parafac_replica]
    selected_decomposition_factors = all_decompositions[parafac_rank][parafac_replica].factors
    visualize_best_decomposition(selected_decomposition_factors, parafac_rank, parafac_replica, parafac_atoms,
                                 selected_channels, a_label, b_label)
    # print(selected_decomposition)

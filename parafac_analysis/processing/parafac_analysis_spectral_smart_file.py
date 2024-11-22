import numpy as np

from data_classes.subject import Subject
from tkinter import filedialog as fd
from parafac_analysis.processing.parafac_analysis_spectral_smart import adapt_selected_channels, process


def main():
    subject_path = fd.askopenfilename(filetypes=[("European Data Format files", "*.edf")])
    subject = Subject(subject_path)
    print("Fill PARAFAC decomposition parameters")
    starting_rank = int(input('Starting rank: '))
    end_rank = int(input('End rank: '))
    replicas = input('Replica count (default: 5): ')
    if replicas != '':
        replicas = int(replicas)
    else:
        replicas = 5
    iterations = input('Optimizer iterations >=10 (default: 10): ')
    if iterations != '':
        iterations = int(iterations)
        if iterations < 10:
            iterations = 10
    else:
        iterations = 10

    selected_frequency_band = input('Selected frequency band (default: "2,28"): ')
    if selected_frequency_band:
        selected_frequency_band = selected_frequency_band.split(',')
        selected_frequency_band = (int(selected_frequency_band[0]), int(selected_frequency_band[1]))
    else:
        selected_frequency_band = (2, 28)

    print('Available channels: {}'.format(', '.join(subject.electrode_names)))
    selected_channels = input('Select channels (default: "C5,C3,C1,Cz,C2,C4,C6"): ')
    if selected_channels:
        selected_channels = selected_channels.split(',')
    else:
        selected_channels = ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6']

    label_names = {}
    for label_name in subject.id_dict:
        label_names[subject.id_dict[label_name]] = label_name
    selected_channels = adapt_selected_channels(subject, selected_channels)

    print("Available labels:")
    unique_events, event_counts = np.unique(subject.events[:, 2], return_counts=True)
    cardinalities = dict(zip(unique_events, event_counts))
    for label_name in label_names:
        print('\t* {}: {} (cardinality: {})'.format(label_name, label_names[label_name], cardinalities[label_name]))
    selected_label_1 = int(input("Select first label:"))
    selected_label_2 = int(input("Select second label:"))
    selected_labels = {
        selected_label_1: label_names[selected_label_1],
        selected_label_2: label_names[selected_label_2]
    }

    process(subject, selected_frequency_band, selected_channels, selected_labels, starting_rank, end_rank, replicas,
            iterations)


main()

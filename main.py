import matplotlib.pyplot as plt
import numpy as np

from classifiers.flat import process


def get_individual_accuracy(predicions, correct):
    all = [0, 0, 0]
    cor = [0, 0, 0]
    for i in range(len(predicions)):
        all[correct[i]] += 1
        if predicions[i] == correct[i]:
            cor[predicions[i]] += 1
    results = [2, 2, 2]
    if all[0] > 0: results[0] = cor[0] / all[0]
    if all[1] > 0: results[1] = cor[1] / all[1]
    if all[2] > 0: results[2] = cor[2] / all[2]
    return results


def create_confusion_matrix(n_classes, predictions_sets, corrects_sets):
    confusion_matrix = np.zeros((n_classes, n_classes))
    for fold_index in range(len(predictions_sets)):
        predictions = predictions_sets[fold_index]
        corrects = corrects_sets[fold_index]
        for index in range(len(predictions)):
            confusion_matrix[predictions[index]][corrects[index]] += 1
    return confusion_matrix


def calculate_recall(predictions, corrects, hand):
    numerator = 0
    denominator = 0
    for i in range(len(predictions)):
        if hand == corrects[i]:  # Actual condition: Positive
            if predictions[i] == corrects[i]:  # Predicted condition: Positive
                numerator += 1  # TP
            denominator += 1  # TP + FN
    if denominator == 0: return 0, 0
    return numerator, denominator


def calculate_precision(predictions, corrects, hand):
    numerator = 0
    denominator = 0
    for i in range(len(predictions)):
        if hand == predictions[i]:  # Predicted condition: Positive
            if predictions[i] == corrects[i]:  # Actual condition: Positive
                numerator += 1  # TP
            denominator += 1  # TP + FP
    if denominator == 0: return 0, 0
    return numerator, denominator


def calculate_combined_recall(predictions, corrects):
    recall_data = []
    for i in range(3):
        recall_data.append(calculate_recall(predictions, corrects, i))
    numerator = 0
    denominator = 0
    for i in recall_data:
        numerator += i[0]
        denominator += i[1]
    if denominator == 0: return 0
    return numerator / denominator


def calculate_combined_precision(predictions, corrects):
    recall_data = []
    for i in range(3):
        recall_data.append(calculate_precision(predictions, corrects, i))
    numerator = 0
    denominator = 0
    for i in recall_data:
        numerator += i[0]
        denominator += i[1]
    if denominator == 0: return 0
    return numerator / denominator


def main(subjects_id, band, channels, randomness):
    chosen_bands = [band]
    precision_numerator = [0, 0]
    precision_denominator = [0, 0]
    recall_numerator = [0, 0]
    recall_denominator = [0, 0]

    for subject_index in subjects_id:
        print('Processing', subject_index, 'of', len(subjects_id), 'subjects')
        if subject_index < 10:
            subject_name = 's0{}'.format(subject_index)
        else:
            subject_name = 's{}'.format(subject_index)
        window_times, window_scores, csp_filters, epochs_info, predictions, corrects = process(subject_name,
                                                                                               chosen_bands, channels,
                                                                                               randomness)

        # update_predictions = []
        # for i in range(len(predictions)):
        #     update_predictions.append([])
        #     for j in range(len(predictions[i])):
        #         if predictions[i][j] == 0:
        #             update_predictions[i].append(1)
        #         else:
        #             update_predictions[i].append(0)
        # predictions = update_predictions

        for i in range(len(predictions)):
            for j in range(2):
                nprec, dprec = calculate_precision(predictions[i], corrects[i], j)
                nrec, drec = calculate_recall(predictions[i], corrects[i], j)
                precision_numerator[j] = precision_numerator[j] + nprec
                precision_denominator[j] = precision_denominator[j] + dprec
                recall_numerator[j] = recall_numerator[j] + nrec
                recall_denominator[j] = recall_denominator[j] + drec

        accuracy_nominator = 0
        accuracy_denumerator = 0
        for i in range(len(predictions)):
            for j in range(len(predictions[i])):
                accuracy_denumerator += 1
                if predictions[i][j] == corrects[i][j]:
                    accuracy_nominator += 1

        print('Accuracy: {}'.format(accuracy_nominator / accuracy_denumerator))

    final_precision = [0, 0]
    final_recall = [0, 0]
    for i in range(2):
        if precision_denominator[i] == 0:
            final_precision[i] = 0
        else:
            final_precision[i] = precision_numerator[i] / precision_denominator[i]

    for i in range(2):
        if recall_denominator[i] == 0:
            final_recall[i] = 0
        else:
            final_recall[i] = recall_numerator[i] / recall_denominator[i]

    print('Combined precision for movement class: {}'.format(final_precision[0]))
    print('Combined precision for rest class: {}'.format(final_precision[1]))

    print('Combined recall for movement class: {}'.format(final_recall[0]))
    print('Combined recall for rest class: {}'.format(final_recall[1]))

    return accuracy_nominator / accuracy_denumerator


def configuration_to_label(configuration):
    channels = configuration['channels']
    if len(channels) == 0:
        channels = 'all'
    else:
        channels = len(channels)
    return '{}Hz width {} channels {} randomness'.format(
        configuration['band_width'],
        channels,
        configuration['randomness'])


channels1 = ['C3', 'C4']
channels2 = ['C5', 'C3', 'C1', 'C2', 'C4', 'C6', 'FC3', 'CP3', 'FC4', 'CP4']
channels3 = []

start_frequency = 5
end_frequency = 30
configurations = [
    # {'channels': channels3, 'band_width': 1, 'randomness': 0},
    # {'channels': channels3, 'band_width': 2, 'randomness': 0},
    {'channels': channels3, 'band_width': 3, 'randomness': 0},
    {'channels': channels3, 'band_width': 4, 'randomness': 0},
    # {'channels': channels3, 'band_width': 5, 'randomness': 0}
]

labels = list(map(configuration_to_label, configurations))
processed_data = []
for i in range(len(configurations)):
    processed_data.append([])
bins = []
for frequency in range(start_frequency, end_frequency):
    for index, configuration in enumerate(configurations):
        processed_data[index].append(main(
            [1],
            (frequency, frequency + configuration['band_width']),
            configuration['channels'],
            configuration['randomness']
        ))
    bins.append('{}Hz'.format(frequency))

for index, label in enumerate(labels):
    plt.plot(bins, processed_data[index], label=label)
plt.legend()
plt.show()

from classifiers.flat import process


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


def train_classifier(bands, channels, randomness):
    precision_numerator = [0, 0]
    precision_denominator = [0, 0]
    recall_numerator = [0, 0]
    recall_denominator = [0, 0]

    window_times, window_scores, csp_filters, epochs_info, predictions, corrects, classifier, mne_info = process(bands,
                                                                                                                 channels,
                                                                                                                 randomness,
                                                                                                                 n_splits=1)

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

    return csp_filters, classifier, mne_info


def configuration_to_label(config):
    channels = config['channels']
    if len(channels) == 0:
        channels = 'all'
    else:
        channels = len(channels)
    return '{}Hz width {} channels {} randomness'.format(
        config['band_width'],
        channels,
        config['randomness'])

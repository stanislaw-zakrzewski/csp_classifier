import numpy as np

from config import use_common_average_reference


def preprocess_subject(signals):
    if use_common_average_reference:
        for signal in signals:
            seamless = signal['seamless']
            averages = [[] for _ in range(32)]
            for sample in range(seamless.shape[1]):
                # TODO fix this to make it faster
                # concatenate((averages_left, full(64, [average(data_left[:, i])])), axis=1)
                # concatenate((averages_right, full(64, [average(data_right[:, i])])), axis=1)
                sample_average = np.average(seamless[:, sample])
                for j in range(32):
                    averages[j].append(sample_average)
            seamless -= averages

    return

    averages_movement = []
    averages_rest = []
    for i in range(32):
        averages_movement.append([])
        averages_rest.append([])

    for i in range(movement_npy.shape[1]):
        # TODO fix this to make it faster
        # concatenate((averages_left, full(64, [average(data_left[:, i])])), axis=1)
        # concatenate((averages_right, full(64, [average(data_right[:, i])])), axis=1)
        average_movement = np.average(movement_npy[:, i])
        for j in range(32):
            averages_movement[j].append(average_movement)

    for i in range(rest_npy.shape[1]):
        # TODO fix this to make it faster
        # concatenate((averages_left, full(64, [average(data_left[:, i])])), axis=1)
        # concatenate((averages_right, full(64, [average(data_right[:, i])])), axis=1)
        average_rest = np.average(rest_npy[:, i])
        for j in range(32):
            averages_rest[j].append(average_rest)

    movement_npy -= averages_movement
    rest_npy -= averages_rest
    np.save('preprocessed_data/movement.npy',
            movement_npy)
    np.save('preprocessed_data/rest.npy',
           rest_npy)


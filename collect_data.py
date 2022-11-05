from matplotlib import pyplot as plt
import numpy as np

import pygds

print("Initializing")
d = pygds.GDS()
pygds.configure_demo(d)
d.SetConfiguration()

print("Acquisition started")

rest_signal = [[]]
movement_signal = [[]]
for _ in range(32):
    rest_signal[-1].append([])
    movement_signal[-1].append([])

total_iter = 0

rest = True
try_len = 8 + np.random.randint(4)


def processCallback(samples):
    try:
        global rest
        global try_len
        global total_iter
        global rest_signal
        global movement_signal
        if rest:
            print("Rest")
            for channel in range(32):
                rest_signal[-1][channel] = np.concatenate((rest_signal[-1][channel], list(samples[:, channel])))
        else:
            print("Movement")
            for channel in range(32):
                movement_signal[-1][channel] = np.concatenate((movement_signal[-1][channel], list(samples[:, channel])))

        # print(i)
        # if(i > 4):
        #     d.Close()
        #     return
        total_iter += 1

        try_len -= 1
        if try_len == 0:
            try_len = 8 + np.random.randint(4)
            if rest:
                rest = False
                rest_signal.append([])
                for _ in range(32):
                    rest_signal[-1].append([])
            else:
                rest = True
                movement_signal.append([])
                for _ in range(32):
                    movement_signal[-1].append([])

        if total_iter > 240 * 2:
            return False
        return True
    except Exception as e:
        print(e)


# d.GetData(d.SamplingRate//2, scope) # to standardowy use-case
print(d.SamplingRate)
d.GetData(d.SamplingRate // 2, processCallback)  # tu uzywamy wlasnej funkcji, zeby wybrac kanały
# del scope

d.Close()
del d

movement_signal_npy = np.array(movement_signal, dtype=object)
rest_signal_npy = np.array(rest_signal, dtype=object)

with open('data/movement.npy', 'wb') as f:
    np.save(f, movement_signal_npy)
with open('data/rest.npy', 'wb') as f:
    np.save(f, rest_signal_npy)

# tak wczytujemy potem te eksperymenty
# loaded =  np.load('movement.npy', allow_pickle=True)
print('EXPERIMENT FINISHED')



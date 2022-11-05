import numpy as np
from mne.io import RawArray

import pygds
from config import configurations


def live_demo(csp, lda, mne_info):
    print("Inicjalizacja trochę trwa...")
    d = pygds.GDS()
    pygds.configure_demo(d)  # Tu sie trzeba przyjrzec blizej - co i jak tam jest ustawiane
    d.SetConfiguration()
    i = 0
    band0, band1 = configurations.experiment_frequency_range

    def processCallback(samples):
        global i
        i += 1
        try:
            ret = []
            for _ in range(32):
                ret.append([])
            for sample in samples:
                for index, channel in enumerate(sample):
                    if index < 32:
                        ret[index].append(channel)
            raw = RawArray(ret, mne_info, verbose='CRITICAL')
            raw.filter(band0, band1, l_trans_bandwidth=2, h_trans_bandwidth=2, filter_length=500, fir_design='firwin',
                       skip_by_annotation='edge', verbose='CRITICAL')
            flt = raw.get_data()
            res = lda.predict(csp.transform(np.array([flt])))
            if res[0] == 0:
                print('Movement')
            else:
                print('Rest')
        except Exception as e:
            print(e)

        if i < 10: return True
        return False

    a = d.GetData(d.SamplingRate * 2, processCallback)

import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataSplitter:
    def __init__(self):
        pass
    def equi_width_bins(lst, num_bins):
        hist, bins = np.histogram(lst, bins=num_bins)
        print("Bin Edges:", bins)
        print("Histogram Counts:", hist)
        return bins


    def binned_mean(lst, num_bins):
        result = scipy.stats.binned_statistic(lst, lst, bins=num_bins, statistic='mean')
        #'sum for statistics another option
        bin_edges = result.bin_edges
        bin_means = result.statistic

        # Print the result
        print("Bin Edges:", bin_edges)
        print("Binned Mean:", bin_means)
        return bin_edges


    def equalObs(x, nbin):
        nlen = len(x)
        return np.interp(np.linspace(0, nlen, nbin + 1),
                         np.arange(nlen),
                         np.sort(x))

    def equi_freq_bins(lst, num_bins):
        n, bins, patches = plt.hist(lst, DataSplitter.equalObs(lst, num_bins), edgecolor='black')
        print("Bin Edges:", bins)
        print("Frequency:", n)

        return bins

    def percentile_bins(lst, percentile_lst=[50, 80, 95]): #[50,80,95] autowlm plot [50, 90, 99]
        return np.percentile(lst, percentile_lst)

    def count_qs_bins(lst, thresholds):
        def get_bin(val, threshold):
            i = 1
            while not (threshold[i-1] < val and val < threshold[i]):
                i +=1
                if int(val) == 900:
                    temp = threshold[len(threshold)-2:]
                    return (temp[0], temp[1])
            return (threshold[i-1] , threshold[i])

        dct = {}

        for v in lst:
            try:
                b = get_bin(v, thresholds)
            except:
                print(v, thresholds)
                exit()
            try:
                dct[b] += 1
            except:
                dct[b] = 1

        return dct, thresholds

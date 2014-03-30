#!/usr/bin/env python2
import glob
import os
import gzip
import cPickle
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats


def load(path = './bt_trace.pickle'):
    f = gzip.GzipFile(path, 'rb')
    data = cPickle.load(f)
    f.close()
    return data

working_dir = os.path.dirname(os.path.realpath(__file__))
total_trials = 0
d_acc = dict.fromkeys(('avg_deg','avg_cl','avg_bt','agg_deg','agg_cl','agg_bt','t_deg','t_cl','t_bt'),0)


for name in glob.glob(os.path.join(working_dir,"results/tmg_4_*")):
    print "Processing file %s"%(name)
    try:
       (tmg,data) = load(name)
    except:
        continue
    # for each graph statistic, we want to see if it can be used to correctly identify the highest mobility merchant
    idx_mobility = -1
    idx_data = dict.fromkeys(('avg_deg','avg_cl','avg_bt','agg_deg','agg_cl','agg_bt','t_deg','t_cl','t_bt'),-1)
    max_mobility = -1
    max_data = dict.fromkeys(('avg_deg','avg_cl','avg_bt','agg_deg','agg_cl','agg_bt','t_deg','t_cl','t_bt'),-1)
    # for each merchant
    for i in xrange(tmg.eta):
        m = 'm_'+str(i)
        for key in data.keys():
            if max_data[key] <= data[key][m]:
                max_data[key] = data[key][m]
                idx_data[key] = i
        if max_mobility <= tmg.prob_mobility[i]:
            max_mobility = tmg.prob_mobility[i]
            idx_mobility = i
    print (idx_mobility,idx_data)
    # check our results
    for key in data.keys():
        if idx_mobility == idx_data[key]:
            d_acc[key] += 1.0
    total_trials += 1.0

for key in d_acc.keys():
    d_acc[key] = d_acc[key]/total_trials

print "Detection accuracy"
print "|\t Type   \t|\tDegree  \t|\tCloseness  \t|\tBetweenness\t|"
print "|\tTemporal  \t|\t%f\t|\t%f\t|\t%f\t|"%(d_acc['t_deg'],d_acc['t_cl'],d_acc['t_bt'])
print "|\tAggregated\t|\t%f\t|\t%f\t|\t%f\t|"%(d_acc['agg_deg'],d_acc['agg_cl'],d_acc['agg_bt'])
print "|\tAverage  \t|\t%f\t|\t%f\t|\t%f\t|"%(d_acc['avg_deg'],d_acc['avg_cl'],d_acc['avg_bt'])

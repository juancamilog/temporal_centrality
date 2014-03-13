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

res = None
mer = None
n=0.0
std_from_data = False


mobilities = []

pearson_r = dict.fromkeys(('avg_deg','avg_cl','avg_bt','agg_deg','agg_cl','agg_bt','t_deg','t_cl','t_bt'),None)
m_data = dict.fromkeys(('avg_deg','avg_cl','avg_bt','agg_deg','agg_cl','agg_bt','t_deg','t_cl','t_bt'),None)
for key in m_data.keys():
    m_data[key] = []

for name in glob.glob(os.path.join(working_dir,"results/tmg_1_*")):
    print "Processing file %s"%(name)
    (tmg,data) = load(name)
    # for each merchant
    for i in xrange(tmg.eta):
        m = 'm_'+str(i)
        for key in data.keys():
            m_data[key].append(data[key][m])
        mobilities.append(tmg.prob_mobility[i])

mobilities = np.array(mobilities)
for key in m_data.keys():
    pearson_r[key] = scipy.stats.pearsonr(np.array(m_data[key]),mobilities)
print pearson_r

print "Pearson correlation coefficient with the merchant mobilities"
print "|\t Type   \t|\tDegree  \t|\tCloseness  \t|\tBetweenness\t|"
print "|\tTemporal  \t|\t%f\t|\t%f\t|\t%f\t|"%(pearson_r['t_deg'][0],pearson_r['t_cl'][0],pearson_r['t_bt'][0])
print "|\tAggregated\t|\t%f\t|\t%f\t|\t%f\t|"%(pearson_r['agg_deg'][0],pearson_r['agg_cl'][0],pearson_r['agg_bt'][0])
print "|\tAverage  \t|\t%f\t|\t%f\t|\t%f\t|"%(pearson_r['avg_deg'][0],pearson_r['avg_cl'][0],pearson_r['avg_bt'][0])

print "2 tailed p values of the correlation coefficients"
print "|\t Type   \t|\tDegree  \t|\tCloseness  \t|\tBetweenness\t|"
print "|\tTemporal  \t|\t%f\t|\t%f\t|\t%f\t|"%(pearson_r['t_deg'][1],pearson_r['t_cl'][1],pearson_r['t_bt'][1])
print "|\tAggregated\t|\t%f\t|\t%f\t|\t%f\t|"%(pearson_r['agg_deg'][1],pearson_r['agg_cl'][1],pearson_r['agg_bt'][1])
print "|\tAverage  \t|\t%f\t|\t%f\t|\t%f\t|"%(pearson_r['avg_deg'][1],pearson_r['avg_cl'][1],pearson_r['avg_bt'][1])

#!/usr/bin/env python2
import glob
import os
import gzip
import cPickle
from matplotlib import pyplot as plt
import numpy as np

import matplotlib

font = {'style' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

def load(path = './bt_trace.pickle'):
    f = gzip.GzipFile(path, 'rb')
    data = cPickle.load(f)
    f.close()
    return data

working_dir = os.path.dirname(os.path.realpath(__file__))

res = None
mer = None
n=0.0
mer_std_from_data = False
res_std_from_data = True

for name in glob.glob(os.path.join(working_dir,"results/tmg_1_*")):
    (tmg,data) = load(name)
    if res is None:
        res = {}
        mer = {}
        res['mean'] = dict.fromkeys(data.keys(),0)
        mer['mean'] = dict.fromkeys(data.keys(),0)
        mer['std'] = dict.fromkeys(data.keys(),0)
        res['std'] = dict.fromkeys(data.keys(),0)
    n+=1.0

    for key in data.keys():
        print "avg resident %s = %f"%(key,data[key]['avg_r'])
        print "avg merchant %s = %f"%(key,data[key]['avg_m'])
        delta = data[key]['avg_r'] - res['mean'][key]
        res['mean'][key] += delta/n
        if not res_std_from_data:
            res['std'][key] += delta*(data[key]['avg_r'] - res['mean'][key])
        else:
            res['std'][key] += data[key]['var_r']*30

        delta = data[key]['avg_m'] - mer['mean'][key]
        mer['mean'][key] += delta/n
        if not mer_std_from_data:
            mer['std'][key] += delta*(data[key]['avg_m'] - mer['mean'][key])
        else:
            mer['std'][key] += data[key]['var_m']*1

for key in data.keys():
    print "res std %s: %f"%(key,res['std'][key])
    print "mer std %s: %f"%(key,mer['std'][key])
    if not res_std_from_data:
        res['std'][key] = np.sqrt(res['std'][key]/(n-1))
    else:
        res['std'][key] = np.sqrt(res['std'][key]/(30*n))
    if not mer_std_from_data:
        mer['std'][key] = np.sqrt(mer['std'][key]/(n-1))
    else:
        mer['std'][key] = np.sqrt(mer['std'][key]/(1*n))
print "Resident: "
print res
print "Merchant: "
print mer

c = np.array(range(3))
w = 0.35
labels = ['Aggregated','Average','Temporal']
error_config = {'ecolor': '0.5'}
#####
plt.figure()
plt.title("Degree")

r_values = [res['mean']['agg_deg'],res['mean']['avg_deg'],res['mean']['t_deg']]
r_std = [res['std']['agg_deg'],res['std']['avg_deg'],res['std']['t_deg']]
plt.bar(c,r_values,w,color='w', label='Resident',yerr=r_std,error_kw=error_config)

m_values = [mer['mean']['agg_deg'],mer['mean']['avg_deg'],mer['mean']['t_deg']]
m_std = [mer['std']['agg_deg'],mer['std']['avg_deg'],mer['std']['t_deg']]
plt.bar(c+w, m_values,w,color='k', label='Merchant',yerr=r_std,error_kw=error_config)

plt.xticks(c+w,labels)
plt.legend()
plt.tight_layout()
plt.ylim([0,1])
plt.savefig(os.path.join(working_dir,"figures/tmg_bar_degree.png"))
#####
plt.figure()
plt.title("Closeness")

r_values = [res['mean']['agg_cl'],res['mean']['avg_cl'],res['mean']['t_cl']]
r_std = [res['std']['agg_cl'],res['std']['avg_cl'],res['std']['t_cl']]
plt.bar(c,r_values,w,color='w', label='Resident',yerr=r_std,error_kw=error_config)

m_values = [mer['mean']['agg_cl'],mer['mean']['avg_cl'],mer['mean']['t_cl']]
m_std = [mer['std']['agg_cl'],mer['std']['avg_cl'],mer['std']['t_cl']]
plt.bar(c+w, m_values,w,color='k', label='Merchant',yerr=r_std,error_kw=error_config)

plt.xticks(c+w,labels)
plt.legend()
plt.tight_layout()
plt.ylim([0,1])
plt.savefig(os.path.join(working_dir,"figures/tmg_bar_closeness.png"))
#####
plt.figure()
plt.title("Betweenness")

r_values = [res['mean']['agg_bt'],res['mean']['avg_bt'],res['mean']['t_bt']]
r_std = [res['std']['agg_bt'],res['std']['avg_bt'],res['std']['t_bt']]
plt.bar(c,r_values,w,color='w', label='Resident',yerr=r_std,error_kw=error_config)

m_values = [mer['mean']['agg_bt'],mer['mean']['avg_bt'],mer['mean']['t_bt']]
m_std = [mer['std']['agg_bt'],mer['std']['avg_bt'],mer['std']['t_bt']]
plt.bar(c+w, m_values,w,color='k', label='Merchant',yerr=r_std,error_kw=error_config)

plt.xticks(c+w,labels)
plt.legend()
plt.tight_layout()
plt.ylim([0,1])
plt.savefig(os.path.join(working_dir,"figures/tmg_bar_betweenness.png"))

plt.show()

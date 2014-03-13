#!/usr/bin/env python2
import os
import gzip
import cPickle
from matplotlib import pyplot as plt
from traveling_merchant_graph import TMG

def save(data,path = './bt_trace.pickle', protocol=-1):
    f = gzip.GzipFile(path, 'wb')
    cPickle.dump(data, f, protocol)
    f.close()

working_dir = os.path.dirname(os.path.realpath(__file__))

if __name__=='__main__':
    param_keys = ['runs','simulation_timesteps','merchants','villages','residents_per_village','prob_village_edge','prob_add_edge','prob_rm_edge']
    parameter_sets = [dict(zip(param_keys,[100,100,1,5,6,0.4,0.1,0.1])),
                      dict(zip(param_keys,[100,100,4,5,6,0.4,0.1,0.1]))]

    for params in parameter_sets:
        print params
        for i in xrange(params["runs"]):
            print "Run #%d:"%(i)
            tmg = TMG(params['merchants'],params['villages'],params['residents_per_village'],params['prob_village_edge'],params['prob_add_edge'],params['prob_rm_edge'])
            tmg.init_gnp()
            tmg.simulate(params['simulation_timesteps'])
            # compute temporal statistics
            results = tmg.get_statistics()
            # add placeholder nodes for the averages across residents and merchants
            for k in results.keys():
                results[k]['avg_r'] = 0
                results[k]['var_r'] = 0
                results[k]['avg_m'] = 0
                results[k]['var_m'] = 0
    
            n_res = 0
            n_merch = 0
            for v in results['t_deg'].keys():
                # accumulate averages and standard deviation
                if str(v).startswith("m_"):
                    n_merch +=1
                    for k in results.keys():
                        delta =results[k][v] - results[k]['avg_m']
                        results[k]['avg_m'] += delta/n_merch
                        results[k]['var_m'] += delta*(results[k][v] - results[k]['avg_m'])
                else:
                    n_res +=1
                    for k in results.keys():
                        n_res +=1
                        delta =results[k][v] - results[k]['avg_r']
                        results[k]['avg_r'] += delta/n_res
                        results[k]['var_r'] += delta*(results[k][v] - results[k]['avg_r'])
            
            for k in results.keys():
                results[k]['var_r']= results[k]['var_r']/(max(n_res-1,1))
                results[k]['var_m']= results[k]['var_m']/(max(n_merch-1,1))
            # save run results to file
            filename = "results/tmg_"+str(tmg.eta)+"_"+str(tmg.nu)+"_"+str(tmg.gamma)+"_"+str(tmg.p)+"_"+str(tmg.b)+"_"+str(tmg.d)+"_run_"+str(i)+".pickle"

            filename = os.path.join(working_dir,filename)
            save((tmg,results),filename)

#!/usr/bin/env python2

import multiprocessing as mp
import parse_dataset
import os
import gc

def process_bt_trace_helper((bt_trace,start_date,end_date,resolution)):
    print "Process %d"%(os.getpid())
    print (start_date,end_date,resolution)
    res =  parse_dataset.stats_from_bt_trace(bt_trace,start_date,end_date,resolution)
    print "Computing stats done"
    gc.collect()
    return res


def results_cb(r):
    print "Process %d finished"%(os.getpid())
    gc.collect()


if __name__=='__main__':
    pool = mp.Pool(processes=5)

    # load bt_trace
    working_dir = os.path.dirname(os.path.realpath(__file__))
    (bt_trace,macs) = parse_dataset.load(os.path.join(working_dir,"./mit_bt_trace.pickle"))

    #some date ranges
    short_ranges = [('9/9/2004','9/10/2004'),('9/9/2004','9/11/2004'),('9/9/2004','9/12/2004'),('9/9/2004','9/13/2004'),('9/9/2004','9/14/2004'),('9/9/2004','9/15/2004')]
    short_ranges_2 = [('9/10/2004','9/11/2004'),('9/11/2004','9/13/2004'),('9/12/2004','9/15/2004'),('9/13/2004','9/17/2004'),('9/14/2004','9/19/2004'),('9/15/2004','9/21/2004')]
    long_ranges = [('9/1/2004','10/1/2004'),('10/1/2004','11/1/2004')]
    time_resolutions = [60,30,10,5]
    print "Starting workers"
    results = []
    for r in short_ranges:
        for t in time_resolutions:
            pool.apply_async(process_bt_trace_helper,args = ((bt_trace,r[0],r[1],t),), callback=results_cb)


    for r in short_ranges_2:
        for t in time_resolutions:
            pool.apply_async(process_bt_trace_helper,args = ((bt_trace,r[0],r[1],t),), callback=results_cb)

    for r in long_ranges:
        for t in time_resolutions:
            pool.apply_async(process_bt_trace_helper,args = ((bt_trace,r[0],r[1],t),), callback=results_cb)

    gc.collect()
    pool.close()
    print "Waiting until pool is done"
    pool.join()
    print "Done!"

#!/usr/bin/env python2
import scipy.io
import numpy as np
import gc
import os
import time
import gzip
import cPickle
from datetime import datetime, date, timedelta
from temporal_graph import *
from matplotlib import pyplot as plt
from sets import Set

def to_datetime(date_in):
    if isinstance(date_in,str):
        return datetime.strptime(date_in, "%m/%d/%Y")
    elif date_in.dtype.type is np.string_ or date_in.dtype.type is np.unicode_:
        return datetime.strptime(date_in[()], "%m/%d/%Y")
    else:
        if date_in > 0:
            return datetime.fromordinal(int(date_in)) + timedelta(days=date_in%1) - timedelta(days = 366)
        else:
            return datetime.fromordinal(1)

def date_range(start,end):
    for i in xrange((end - start).days):
        yield start + timedelta(i)

def mac_to_hash(dataset):
    subjects = dataset['s']
    valid_subjects = []
    mac_to_hash_dict = {}
    s_mac = 0
    s_hash = 0
    for s in subjects:
        if s['my_mac']>0:
            s_mac = int(s['my_mac'][()],16)
        elif s['mac']>0:
            s_mac = s['mac'].astype(int)[()]
        if s['my_hashedNumber'][()] > 0:
            s_hash = s['my_hashedNumber'][()]
            mac_to_hash_dict[s_mac] = s_hash
        else:
            s_hash = s_hash + 1
            mac_to_hash_dict[s_mac] = s_hash
    return mac_to_hash_dict

def active_users_per_day(dataset,start_,end_):
    start_date = datetime.strptime(start_, "%m/%d/%Y")
    end_date = datetime.strptime(end_, "%m/%d/%Y")
    mac_to_hash_dict = mac_to_hash(dataset)
    subjects = dataset['s']

    active_per_day = dict.fromkeys(date_range(start_date,end_date),0)
    for mac in mac_to_hash_dict:
        hsh = mac_to_hash_dict[mac]-1
        my_startdate = to_datetime(subjects[hsh]['my_startdate'])
        my_enddate = to_datetime(subjects[hsh]['my_enddate'])

        if my_startdate >= start_date and my_enddate > my_startdate:
            s = max(start_date,my_startdate)
            e = min(end_date,my_enddate)
            s=s.replace(hour=0,minute=0,second=0,microsecond=0)
            e=e.replace(hour=0,minute=0,second=0,microsecond=0)
            for d in date_range(s,e):
                active_per_day[d] += 1
    return active_per_day

def plot_active_users_per_day(dataset,start_,end_):
    active_users_dict = active_users_per_day(dataset,start_,end_)
    order = np.argsort(active_users_dict.keys())
    plt.plot(np.array(active_users_dict.keys())[order],np.array(active_users_dict.values())[order])
    ax = plt.gca()
    ax.grid(True)
    plt.title("Active subjects per day (from reported participation dates)")
    plt.ylabel("Number of active subjects")
    plt.xlabel("Date")
    plt.show()


def extract_bluetooth_data(dataset):
    print "Extracting bluetooth data..."
    mac_to_hash_dict = mac_to_hash(dataset)
    subjects = dataset['s']
    bt_data = {}
    for mac in mac_to_hash_dict:
        hsh = mac_to_hash_dict[mac]-1
        scan_dates = subjects[hsh]['device_date']
        scan_macs =  subjects[hsh]['device_macs']
        bt_data[mac] = {}
        bt_data[mac]['scan_dates'] = scan_dates
        bt_data[mac]['scan_macs'] = scan_macs
    print "Done!"
    return (bt_data,mac_to_hash_dict)

def process_bt_trace(bt_data,start_='8/1/2004',end_='7/14/2005',resolution_mins=10):
    print "Processing bluetooth trace between %s and %s..."%(start_,end_)
    start_date = datetime.strptime(start_, "%m/%d/%Y")
    end_date = datetime.strptime(end_, "%m/%d/%Y")
    # for the given start date, end date and time resolution, we preallocate a
    # data structure that will have (end_minutes)/resolution_mins elements
    # where end_minutes is the number of minutes from start_date to end_date
    # Every bluetooth scan will be put into the bin that starts at time
    resolution = 60.0*float(resolution_mins)
    bt_activity = {}
    active_per_day = dict.fromkeys(date_range(start_date,end_date),0)
    bt_events = 0

    for mac in bt_data.keys():
        scan_idx = 0
        scan_dates = bt_data[mac]['scan_dates']
        scan_macs = bt_data[mac]['scan_macs']
        active_days = {}
        for bt_scan_date_f in scan_dates:
            bt_scan_date = to_datetime(bt_scan_date_f)
            if bt_scan_date >= start_date and bt_scan_date <= end_date:
                tdelta = bt_scan_date-start_date
                # get the bin number
                td = int(np.ceil(float(tdelta.total_seconds())/resolution)-1)
                if td not in bt_activity:
                    bt_activity[td] = Set()
                # get the links from the list of devices that were found during the scan
                if scan_macs[scan_idx].ndim == 0:
                    scan_macs[scan_idx] = np.array([scan_macs[scan_idx]])

                for bt_scan in scan_macs[scan_idx].tolist():
                    bt_activity[td].add((mac,int(bt_scan)))
                    bt_events +=1

                bt_scan_date = bt_scan_date.replace(hour=0,minute=0,second=0,microsecond=0)
                active_days[bt_scan_date]=1
            scan_idx += 1
        # accumulate count of active users per day
        for d in active_days:
            active_per_day[d] += 1
    print "Done! Found %d bluetooth proximity events"%(bt_events)
    return (bt_activity,active_per_day)

def create_temporal_graph(bt_activity_dict, macs, include_perifery=False):

    end_time = max(bt_activity_dict.keys())
    G = temporal_graph(end_time)
    G.add_vertices(macs)
    edges = []
    for t in bt_activity_dict.keys():
        for pair in bt_activity_dict[t]:
            if include_perifery:
                if pair[0] not in G.vertices:
                    G.add_vertices(pair[0])
                if pair[1] not in G.vertices:
                    G.add_vertices(pair[1])
                edges.append(((pair[0].pair[1]),(t,t)))
            elif pair[0] in G.vertices and pair[1] in G.vertices:
                edges.append(((pair[0],pair[1]),(t,t)))
    G.add_temporal_edges(edges)
    return G

def load_mat(path = "../datasets/RealityMining/realitymining.mat"):
    print "Loading MIT Reality Mining dataset (it will take a while, be patient)..."
    dataset = scipy.io.loadmat(path, chars_as_strings=True, squeeze_me=True)
    gc.collect()
    print "Done!"
    return dataset

def save(data,path = './bt_trace.pickle', protocol=-1):
    print "Saving data to %s"%(path)
    f = gzip.GzipFile(path, 'wb')
    cPickle.dump(data, f, protocol)
    f.close()

def load(path = './bt_trace.pickle'):
    print "Loading data from %s"%(path)
    f = gzip.GzipFile(path, 'rb')
    data = cPickle.load(f)
    f.close()
    return data

def get_graph_from_dataset(path="./reality_mining_992004_to_9102004_r60.pickle"):
    (bt,macs,activity) = load(path)
    G = create_temporal_graph(bt,macs)
    et = max(bt.keys())
    return (G,et)

def compute_statistics(G,start_time=0, end_time=1):
    print "-> temporal degree"
    tdeg = compute_temporal_degree(G,start_time,end_time)
    print "-> temporal closeness"
    tcl = compute_temporal_closeness(G,start_time,end_time)
    print "-> temporal betweenness"
    tbt = compute_temporal_betweenness(G,start_time,end_time)
    print "-> static statistics"
    static_stats = compute_static_graph_statistics(G,start_time,end_time)
    print "-> Done!"
    return (tdeg,tcl,tbt,static_stats)

def stats_from_bt_trace(bt_trace,start_date,end_date,resolution):
    working_dir = os.path.dirname(os.path.realpath(__file__))
    results_dir = os.path.join(working_dir,"mit_dataset_results/")
    filename = "mit_bt_trace_"+start_date+"_"+end_date+"_r"+str(resolution)
    filename = filename.replace("/","_")
    # parse bt activity data
    (bt_activity,active_per_day) = process_bt_trace(bt_trace,start_date,end_date,resolution)
    # save it to disk for later analysis
    save((bt_activity,active_per_day),os.path.join(results_dir,filename+".pickle"))
    # create graph
    g = create_temporal_graph(bt_activity,bt_trace.keys())
    et = max(bt_activity.keys())
    # compute statistics
    print "Computing statistics from graph"
    stats = compute_statistics(g,0,et)
    # create a dictionary from the results
    results = {}
    results['t_deg']= stats[0]
    results['t_cl']= stats[1]
    results['t_bt']= stats[2]
    results['agg_deg']= stats[3][0][0]
    results['agg_cl']= stats[3][0][1]
    results['agg_bt']= stats[3][0][2]
    results['avg_deg']= stats[3][1][0]
    results['avg_cl']= stats[3][1][1]
    results['avg_bt']= stats[3][1][2]
    # save the stats
    save(results,os.path.join(results_dir,filename+"_stats.pickle"))

    return True


#!/usr/bin/env python2
import parse_dataset
import os

working_dir = os.path.dirname(os.path.realpath(__file__))

filename = "../datasets/RealityMining/realitymining.mat"

path = os.path.join(working_dir,filename)

dataset = parse_dataset.load_mat(path)

(bt_trace,macs) = parse_dataset.extract_bluetooth_data(dataset)

parse_dataset.save((bt_trace,macs),os.path.join(working_dir,"./mit_bt_trace.pickle"))
print "Done!"

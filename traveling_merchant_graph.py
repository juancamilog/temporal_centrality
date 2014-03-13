#!/usr/bin/env python2
import networkx as nx
import numpy as np
from temporal_graph import *

class TMG:
    def __init__(self,n_merchants,n_villages,n_residents,prob_village_edge,prob_add_edge,prob_rm_edge):
        self.eta = n_merchants
        self.nu = n_villages
        self.gamma = n_residents
        self.p = prob_village_edge
        self.b = prob_add_edge
        self.d = prob_rm_edge

        self.prob_mobility = []

        self.graph = nx.Graph()

    def init_gnp(self):
        # generate the village graphs
        for i in xrange(0,self.nu):
            G_i = nx.fast_gnp_random_graph(self.gamma,self.p)
            
            #mapping = lambda x: x + i*self.nu
            new_labels = [v + i*self.gamma for v in G_i.nodes_iter()]
            mapping = dict(zip(G_i.nodes_iter(),new_labels))

            G_i = nx.relabel_nodes(G_i,mapping,copy=True)
            self.graph.add_nodes_from(G_i.nodes())
            self.graph.add_edges_from(G_i.edges())

        # connect the merchants to a random node in a random graph
        
        for j in xrange(0,self.eta):
            merchant_label = 'm_'+str(j)
            self.graph.add_node(merchant_label)

            # pick a random graph
            i = np.random.randint(self.nu)

            # pick a random resident
            r = np.random.randint(self.gamma)

            # add the edge from the merchant to the corresponding resident
            self.graph.add_edge(merchant_label,r+i*self.gamma)

            # set the probability of mobility for this merchant to a number between 0.5 and 1
            self.prob_mobility.append(np.random.uniform(0.5,1))
        
        # initialize temporal graph
        self.temporal_graph = temporal_graph(0)
        # add vertices
        self.temporal_graph.add_vertices(self.graph.nodes())
        #add current edges to the temporal graph
        self.temporal_graph.append_snapshot(self.graph)
        self.current_timestep = 0

    def simulate(self, timesteps):
        for time in xrange(timesteps):
            # interal movement
            # for each village
            for i in xrange(0,self.nu):
                n_0 = i*self.gamma
                n_l = (i+1)*self.gamma
                #print "Internal movement for village %d (nodes %d to %d)"%(i,n_0,n_l-1)
                # we go though all the possible edges
                for n_i in xrange(n_0,n_l):
                    for n_j in xrange(n_0,n_l):
                        if n_j != n_i:
                            if self.graph.has_edge(n_i,n_j):
                                # remove this edge with probability self.d
                                if np.random.binomial(1,self.d):
                                    self.graph.remove_edge(n_i,n_j)
                            else:
                                # add this edge with probability self.b
                                if np.random.binomial(1,self.b):
                                    self.graph.add_edge(n_i,n_j)


            # external movement
            # for every merchant
            for j in xrange(0,self.eta):
                merchant_label = 'm_'+str(j)
                # decide if the merchant will move somewhere else
                #
                if np.random.binomial(1,self.prob_mobility[j]):
                    #print "External movement for %s"%(merchant_label)
                    # get the edge that connects this merchant
                    edges = self.graph.edges(merchant_label)

                    # get the id of the village
                    old_village = -1
                    old_node = -1
                    #print edges
                    if edges[0][0] == merchant_label:
                        old_node = edges[0][1]
                        old_village = int(edges[0][1]/self.gamma)
                    else:
                        old_node = edges[0][0]
                        old_village = int(edges[0][0]/self.gamma)

                    # drop it
                    self.graph.remove_edges_from(edges)

                    # select a random new village
                    new_village = old_village
                    while new_village == old_village:
                        new_village = np.random.randint(self.nu)

                    # pick a random resident
                    r = np.random.randint(self.gamma)

                    # add the edge from the merchant to the corresponding resident
                    new_node = r+new_village*self.gamma
                    self.graph.add_edge(merchant_label,new_node)
                    #print "Moved from village %d to %d (edge to %d replaced with edge to %d)"%(old_village,new_village,old_node,new_node)
                    #print self.graph.degree(merchant_label)

            #add current edges to the temporal graph
            self.temporal_graph.append_snapshot(self.graph)
            self.current_timestep += 1


    def compute_statistics(self, start_time=0, end_time=1):
        end_time=self.current_timestep
        tdeg = compute_temporal_degree(self.temporal_graph,start_time,end_time)
        tcl = compute_temporal_closeness(self.temporal_graph,start_time,end_time)
        tbt = compute_temporal_betweenness(self.temporal_graph,start_time,end_time)
        static_stats = compute_static_graph_statistics(self.temporal_graph,start_time,end_time)
        return (tdeg,tcl,tbt,static_stats)

    def get_statistics(self,start_time=0,end_time=1):
        end_time=self.current_timestep
        stats = self.compute_statistics(start_time,end_time)
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

        return results

#!/usr/bin/env python2
from temporal_graph import *
from matplotlib import pyplot as plt

if __name__ == "__main__":
    et = 3
    G = temporal_graph(et)
    G.add_vertices(['A','B','C','D'])
    #G.add_temporal_edges([(('A','C'),(1,1)),(('C','B'),(2,2)),(('A','D'),(3,3)),(('C','D'),(3,3)),(('C','A'),(4,4)),(('B','D'),(4,4))])
    #G.add_temporal_edges([(('A','C'),(1,1)),(('A','D'),(2,2)),(('B','D'),(3,3)),(('C','B'),(3,3))])
    G.add_temporal_edges([(('A','C'),(1,1)),(('A','D'),(2,2)),(('B','D'),(2,3)),(('C','D'),(3,3))])
    #G.add_temporal_edges([(('A','C'),(1,1)),(('A','B'),(2,2)),(('A','D'),(2,2)),(('B','C'),(2,3)),(('B','D'),(2,3)),(('C','D'),(3,3))])
    temp_degree =  compute_temporal_degree(G,0,et)
    temp_closeness = compute_temporal_closeness(G,0,et)
    temp_betweenness = compute_temporal_betweenness(G,0,et)

    static_stats = compute_static_graph_statistics(G,0,et)
    print "\tNode\t|\t Type   \t|\tDegree  \t|\tCloseness  \t|\tBetweenness\t|"
    for v in G.vertices:
        print "\t%s\t|\tTemporal  \t|\t%f\t|\t%f\t|\t%f\t|"%(v,temp_degree[v],temp_closeness[v],temp_betweenness[v])
        print "\t%s\t|\tAggregated\t|\t%f\t|\t%f\t|\t%f\t|"%(v,static_stats[0][0][v],static_stats[0][1][v],static_stats[0][2][v])
        print "\t%s\t|\tAverage  \t|\t%f\t|\t%f\t|\t%f\t|"%(v,static_stats[1][0][v],static_stats[1][1][v],static_stats[1][2][v])

    #t = 0
    #for G_t in G.snapshots:


    G.draw_time_ordered_graph()
    plt.show()

#!/usr/bin/env python2
import numpy as np
import multiprocessing as mp
import itertools

def temporal_shortest_path_single_source((G,verts,t,D,S,labels)):
    return temporal_shortest_path_ss(G,verts,t,D,S,labels)

def temporal_shortest_path_ss(G,verts,t,D,S,labels):
    for v in verts:
        vi = labels[v]
        for k_t in G.time_ordered_graph.successors_iter((v,t)): # at most n
            ki = labels[k_t[0]]
            D[t][vi,ki] = 1
            S[t][vi,ki] = 1

        for u in G.vertices: # n_vertices
            ui = labels[u]
            # if u is not reachable in one time step
            if D[t][vi,ui] > 1:
               for k_t in G.time_ordered_graph.successors_iter((v,t)): # at most n
                     ki = labels[k_t[0]]
                     d = D[t+1][ki,ui] + 1
                     if d < D[t][vi,ui]:
                         # there is a shortest path through k!
                         D[t][vi,ui] = d
                         S[t][vi,ui] = S[t+1][ki,ui]
                     elif d == D[t][vi,ui]:
                         # we accumulate the number of shortest paths
                         S[t][vi,ui] += S[t+1][ki,ui]
    return (D,S,t)

def temporal_shortest_path_reduce((D_1,S_1,t_1),(D_2,S_2,t_2)):
    if t_1 < t_2:
        return D_1,S_1
    if t_2 < t_1:
        return D_2,S_2
    D_1[t_1] = np.minimum(D_1[t_1],D_2[t_1])
    S_1[t_1] = S_1[t_1]+S_2[t_1]
    return (D_1,S_1,t_1)

def get_chunks(verts,chunksize):
    vert_iter = iter(verts)
    while True:
        x = tuple(itertools.islice(vert_iter,chunksize))
        if not x:
            return
        else:
            yield x

# compute the temporal closeness score
# TODO: this runs in O(m|V|^3) time, but it should be possible to compute in O(m|V|^2)
def compute_temporal_closeness_mp(G,start_time,end_time):
    # we start at the end time t = G.t_end, and go backwards in time
    n = len(G.vertices)
    m = end_time - start_time
    labels = {}
    # compute integer labels
    idx = 0
    for v in G.vertices:
        labels[v] = idx
        idx = idx + 1
    
    # this matrix stores the distances at time t+1
    D = {}
    S = {}

    # this stores the cumulative closeness score of G
    closeness = np.zeros(n)

    # our process pool
    n_processes = mp.cpu_count()
    mp_pool = mp.Pool(processes = n_processes)

    for t in xrange(end_time,start_time-1,-1): # m time steps
        # this matrix stores the distances at time t
        D[t] = np.ones((n,n))*np.inf
        # this stores the number of shortest paths between two nodes
        S[t] = np.eye(n)

        #for v in G.vertices: # n vertices
        #    D,S = temporal_shortest_path_single_source((G,(v,t),D,S,labels))
        # split the work into chunks
        n_chunks = n_processes*4
        if n_chunks > n:
            n_chunks = 1
        chunks = get_chunks(G.vertices,int(n/n_chunks))

        print "n_chunks = %d"%(n_chunks)
        print "chunk size = %d"%(int(n/n_chunks))

        # send the work to the process pool
        result_iterator = mp_pool.imap(temporal_shortest_path_single_source,zip([G]*n,chunks,[t]*n_chunks,[D]*n_chunks,[S]*n_chunks,[labels]*n_chunks))
        D,S,t = reduce(temporal_shortest_path_reduce,result_iterator)

        if t < end_time:
            # closeness is the sum of inverse shortest path distances for all v and u (with v!= u)
            closeness += (1/D[t] - np.eye(n)).sum(1)

    mp_pool.close()
    # at the end, we normalize closeness by (|V| - 1)*m
    closeness = closeness/((n-1)*m)
    return dict(zip(G.vertices, closeness))

# compute the temporal betweenness score
def compute_temporal_betweenness_mp(G,start_time,end_time):
    # we start at the end time t = G.t_end, and go backwards in time
    n = len(G.vertices)
    m = end_time - start_time
    labels = {}
    # compute integer labels
    idx = 0
    for v in G.vertices:
        labels[v] = idx
        idx = idx + 1
    
    # this dictionary stores the distance matrices D_t
    D = {}
    # this dictionary stores the  D_t
    S = {}
    # this stores the cumulative closeness score of G
    betweenness = np.zeros(n)
    # our process pool
    n_processes = mp.cpu_count()
    mp_pool = mp.Pool(processes = n_processes)

    for t in xrange(end_time,start_time-1,-1): # m time steps
        # this matrix stores the distances at time t
        D[t] = np.ones((n,n))*np.inf
        # this stores the number of shortest paths between two nodes
        S[t] = np.eye(n)

        #for v in G.vertices: # n vertices
        #    D,S,t = temporal_shortest_path_single_source((G,(v,t),D,S,labels))
        # split the work into chunks
        n_chunks = n_processes*4
        if n_chunks > n:
            n_chunks = 1
        chunks = get_chunks(G.vertices,int(n/n_chunks))

        # send the work to the process pool
        result_iterator = mp_pool.imap(temporal_shortest_path_single_source,zip([G]*n,chunks,[t]*n_chunks,[D]*n_chunks,[S]*n_chunks,[labels]*n_chunks))
        D,S,t = reduce(temporal_shortest_path_reduce,result_iterator)

        for s in G.vertices: # n vertices
            si = labels[s]
            for d in G.vertices: # n vertices
                di = labels[d]
                if si is not di and S[t][si,di] > 0:
                    for v in G.vertices: # n vertices
                        vi = labels[v]
                        if vi is not si and vi is not di: # m timesteps
                            for k in xrange(t+1,end_time):
                                if S[t][si,vi] > 0 and S[k][vi,di] > 0 and D[t][si,vi] == k-t:
                                    d_tk = D[t][si,vi]
                                    d_kj = D[k][vi,di]
                                    if D[t][si,di] == d_tk + d_kj:
                                        betweenness[vi] += S[t][si,vi]*S[k][vi,di]/S[t][si,di]
    mp_pool.close()
    betweenness = betweenness/((n-1)*m)
    return dict(zip(G.vertices,betweenness))

def compute_static_graph_statistics_mp(G,start_time,end_time):
    n = len(G.vertices)
    m = end_time - start_time
    agg_statistics = [dict.fromkeys(G.vertices,0),dict.fromkeys(G.vertices,0),dict.fromkeys(G.vertices,0)]*3
    avg_statistics = [dict.fromkeys(G.vertices,0),dict.fromkeys(G.vertices,0),dict.fromkeys(G.vertices,0)]*3

    aggregated_graph = nx.Graph()
    aggregated_graph.add_nodes_from(G.vertices)
    for t in xrange(start_time,end_time+1):
        aggregated_graph.add_edges_from(G.snapshots[t].edges_iter())
         
        dc = nx.degree_centrality(G.snapshots[t])
        cc = nx.closeness_centrality(G.snapshots[t])
        bc = nx.betweenness_centrality(G.snapshots[t])
        for v in G.vertices:
            avg_statistics[0][v] += dc[v]/m
            avg_statistics[1][v] += cc[v]/m
            avg_statistics[2][v] += bc[v]/m

    
    dc = nx.degree_centrality(aggregated_graph)
    cc = nx.closeness_centrality(aggregated_graph)
    bc = nx.betweenness_centrality(aggregated_graph)
    for v in G.vertices:
        agg_statistics[0][v] = dc[v]
        agg_statistics[1][v] = cc[v]
        agg_statistics[2][v] = bc[v]
    return (agg_statistics, avg_statistics)

#!/usr/bin/env python2
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sets import Set

class temporal_graph:
    # construct a temporal network that has snapshots from time 0 until time t_end
    def __init__(self, t_end):
        self.t_end = t_end
        # this list will contain the graphs snapshots
        self.snapshots =[]
        for t in xrange(0,t_end+1):
            self.snapshots.append(nx.Graph())

        # we will use this list of vertices to for all of the snapshots
        self.vertices = []
        # this list will contain all the temporal edges
        self.edges = []
        # this structure will keep our time ordered graph
        self.time_ordered_graph = nx.DiGraph()

    # add new vertices to all the snapshots
    def add_vertices(self,verts):
        self.vertices.extend(verts)
        n = len(verts)
        t=0
        for G_t in self.snapshots:
            # add nodes to the snapshot at  time t
            G_t.add_nodes_from(verts)
            # create time stamped copies of the vertices
            time_stamped_verts = zip(verts,[t]*n)
            # add them to the time ordered graph 
            self.time_ordered_graph.add_nodes_from(time_stamped_verts)
            # add edges from vertex (v,t-1) to (v,t)
            if t > 0:
               new_edges = [((v,t-1),(v,t)) for v in verts]
               self.time_ordered_graph.add_edges_from(new_edges)
            t = t + 1

    # add an edge to the subset of snapshots G_{start_time} to G_{end_time}
    # the edges should come in a tuple ((v_1,v_2), (start_time, end_time))
    def add_temporal_edges(self,edges):
        # only consider the start and end times that fall within 0 and self.t_end
        for e in edges:
            start_time = e[1][0]
            end_time = e[1][1]
            # check if the time frame is valid
            if end_time < start_time or start_time > self.t_end or end_time < 0:
                #ignore this edge if it has an invalid time interval
                continue

            # clip the time interval to the bounds given by self.t_start and self.t_end
            if start_time < 0:
                start_time = self.t_start
            if end_time > self.t_end:
                end_time = self.t_end

            # add edges to the snapshots
            for t in xrange(start_time, end_time+1):
                self.snapshots[t].add_edge(e[0][0],e[0][1])
                if t > 0:
                    # add edges to the time ordered graph ( from t-1 to t)
                    new_edges = [((e[0][0], t-1), (e[0][1],t)), ((e[0][1],t-1),(e[0][0],t))]
                    self.time_ordered_graph.add_edges_from(new_edges)
           
            # add edges to our global edge list
            self.edges.append((e[0],(start_time,end_time)))

    # add a graph snapshot to the end of the current snapshot list.
    def append_snapshot(self,G_t):
        self.t_end = self.t_end+1
        # TODO check that we are passing a valid snapshot (same vertex set)
        self.snapshots.append(nx.Graph(G_t))
        edgeset = G_t.edges()
        # get timestamped edges from current snapshot
        new_edges = zip(edgeset,[(self.t_end,self.t_end)]*len(edgeset))
        # append self edges from previous timestep to the current timestep
        for v in self.vertices:
            new_edges.append(((v,v),(self.t_end-1,self.t_end)))

        # add list of new edges to the temporal graph
        self.add_temporal_edges(new_edges)

    # draw the snapshot of the network at time t
    def draw_snapshot (self,t,labels =None):
        plt.figure()
        if labels is not None:
            npos = nx.graphviz_layout(self.snapshots[t], prog="fdp")
            nx.draw(self.snapshots[t], pos = npos, node_color='w', node_size=500, with_labels=False)
            nx.draw_networkx_labels(self.snapshots[t], npos, labels)
        else:
            nx.draw(self.snapshots[t], node_color='w', node_size=10, with_labels=True)

        plt.draw()
    
    # draw the time ordered graph derived from G_{i,j}
    def draw_time_ordered_graph(self,v_scale=1.0,h_scale=1.0):
        plt.figure()
        # compute the layout, the horizontal axis represents time and the vertical axeis corresponds to node label
        v_pos = {}
        
        #first, for each vertex we select a value for the vertical coordinate
        y = len(self.vertices)*v_scale
        for v in self.vertices:
            v_pos[v] = y
            y = y - v_scale

        npos = {}
        labels = {}
        # now, for every time step we select a horizontal coordinate and populate the layout dictionary
        for v in self.time_ordered_graph.nodes():
            npos[v] = np.array((h_scale*v[1],v_pos[v[0]]))
            labels[v] = "$%s_{%d}$"%(v[0],v[1])

        nx.draw(self.time_ordered_graph, pos = npos, with_labels=False, node_color='w', node_size=1000)
        nx.draw_networkx_labels(self.time_ordered_graph, npos, labels)
        plt.draw()
 
# computes the temporal degree
def compute_temporal_degree(G,start_time,end_time):
    verts = G.vertices
    n = len(verts)
    m = end_time - start_time
    
    # for every v in V, we sum the indegree and outdegree at every timestep, excluding the edges
    # from v_t to v_{t+1} (there are m such edges)
    vi = 0
    timestamps = range(start_time,end_time+1)
    # this stores the average degree of G
    degree = np.zeros(n)
    for v in verts:
        v_t = zip([v]*(m+1),timestamps)
        degree[vi] += sum(G.time_ordered_graph.degree(v_t).values()) - 2*m
        vi = vi + 1
    degree = degree/(2*(n-1)*m)
    return dict(zip(verts, degree))

# compute the temporal closeness score
# TODO: this runs in O(m|V|^3) time, but it should be possible to compute in O(m|V|^2)
def compute_temporal_closeness(G,start_time,end_time):
    # we start at the end time t = G.t_end, and go backwards in time
    verts = G.vertices
    n = len(verts)
    m = end_time - start_time
    labels = {}
    # compute integer labels
    idx = 0
    for v in verts:
        labels[v] = idx
        idx = idx + 1
    
    # this matrix stores the distances at time t+1
    D_tplus1 = np.eye(n)

    # this stores the cumulative closeness score of G
    closeness = np.zeros(n)
    for t in xrange(end_time,start_time-1,-1): # m time steps
        # this matrix stores the distances at time t
        D_t = np.ones((n,n))*np.inf

        for v in verts: # n vertices
            vi = labels[v]
            # k is reachable in one step from v
            for k_t in G.time_ordered_graph.successors_iter((v,t)): # at most n
                ki = labels[k_t[0]]
                D_t[vi,ki] = 1

            if t < end_time:
                for u in verts: # n_vertices
                    ui = labels[u]
                    # if u is not reachable in one time step
                    if D_t[vi,ui] > 1:
                       for k_t in G.time_ordered_graph.successors_iter((v,t)): # at most n
                             ki = labels[k_t[0]]
                             d = D_tplus1[ki,ui] + 1
                             if d < D_t[vi,ui]:
                                 D_t[vi,ui] = d
        D_tplus1 = D_t
        if t < end_time:
            # closeness is the sum of inverse shortest path distances for all v and u (with v!= u)
            closeness += (1/D_t - np.eye(n)).sum(1)
        #print "closeness computation done for time %d"%(t)

    # at the end, we normalize closeness by (|V| - 1)*m
    closeness = closeness/((n-1)*m)
    return dict(zip(verts, closeness))

# compute the temporal betweenness score
def compute_temporal_betweenness_old(G,start_time,end_time):
    # we start at the end time t = G.t_end, and go backwards in time
    verts = G.vertices
    n = len(verts)
    m = end_time - start_time
    labels = {}
    # compute integer labels
    idx = 0
    for v in verts:
        labels[v] = idx
        idx = idx + 1
    
    # this dictionary stores the distance matrices D_t
    D = {}
    # this dictionary stores the  D_t
    S = {}
    # this stores the cumulative closeness score of G
    betweenness = np.zeros(n)
    for t in xrange(end_time,start_time-1,-1): # m time steps
        # this matrix stores the distances at time t
        D[t] = np.ones((n,n))*np.inf
        # this stores the number of shortest paths between two nodes
        S[t] = np.eye(n)

        for v in verts: # n vertices
            vi = labels[v]
            if t < end_time:
                for k_t in G.time_ordered_graph.successors_iter((v,t)): # at most n
                    ki = labels[k_t[0]]
                    D[t][vi,ki] = 1
                    S[t][vi,ki] = 1

                for u in verts: # n_vertices
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
        print "First part done"
        for s in verts: # n vertices
            si = labels[s]
            for d in verts: # n vertices
                di = labels[d]
                if si is not di and S[t][si,di] > 0:
                    for v in verts: # n vertices
                        vi = labels[v]
                        if vi is not si and vi is not di: # m timesteps
                            if S[t][si,vi] > 0:
                                for k in xrange(t+1,end_time):
                                    if S[k][vi,di] > 0 and D[t][si,vi] == k-t:
                                        d_tk = D[t][si,vi]
                                        d_kj = D[k][vi,di]
                                        if D[t][si,di] == d_tk + d_kj:
                                            betweenness[vi] += S[t][si,vi]*S[k][vi,di]/S[t][si,di]
        print "Second part done"
        print S[t]
    print "Done overall"
    betweenness = betweenness/(0.5*(n-1)*(n-2)*m)
    return dict(zip(verts,betweenness))

# compute the temporal betweenness score
def compute_temporal_betweenness(G,start_time,end_time):
    import sys
    sys.setcheckinterval(2000)
    # we start at the end time t = G.t_end, and go backwards in time
    verts = G.vertices
    n = len(verts)
    m = end_time - start_time
    labels = {}
    # sets for computing the correct normalization values
    V_s = [None]*n
    V_d = [None]*n
    # compute integer labels
    idx = 0
    for v in verts:
        labels[v] = idx
        V_s[idx] = Set()
        V_s[idx].add(idx)
        V_d[idx] = Set()
        V_d[idx].add(idx)
        idx = idx + 1
    
    # this dictionary stores the distance matrices D_t
    D = {}
    # this dictionary stores the  D_t
    S = {}
    # this stores the cumulative closeness score of G
    betweenness = np.zeros(n)

    for t in xrange(end_time,start_time-1,-1): # m time steps
        # this matrix stores the distances at time t
        D[t] = np.ones((n,n))*np.inf
        # this stores the number of shortest paths between two nodes
        S[t] = np.eye(n)

        for v in verts: # n vertices
            vi = labels[v]
            if t < end_time:
                for k_t in G.time_ordered_graph.successors_iter((v,t)): # at most n
                    ki = labels[k_t[0]]
                    D[t][vi,ki] = 1
                    S[t][vi,ki] = 1

                for u in verts: # n_vertices
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
        #print "shortest temporal path computation done for time %d"%(t)
        vert_indices = labels.values()
        total_its = 0
        for si in vert_indices: # n vertices
            for di in vert_indices: # n vertices
                # s and d should be different
                if si == di:
                    continue
                # there should be a shortest path between s and d  
                if S[t][si,di] <= 0:
                    continue
                norm_const = 1.0/S[t][si,di]
                for vi in vert_indices: # n vertices
                    # v should be different from s and d
                    if vi == si or vi == di: 
                        continue
                    # there should exist a path between s and v
                    if S[t][si,vi] <= 0:
                        continue
                    k = D[t][si,vi] + t
                    total_its +=1
                    if S[k][vi,di] > 0:
                        d_tk = D[t][si,vi]
                        d_kj = D[k][vi,di]
                        V_s[vi].add(si)
                        V_d[vi].add(di)
                        if D[t][si,di] == d_tk + d_kj:
                            if np.isnan(S[t][si,vi]*S[k][vi,di]/S[t][si,di]):
                                print "Whoops! %d %d %d"%(si,vi,di)
                                print (S[t][si,vi],S[k][vi,di],S[t][si,di])
                                raw_input()
                            betweenness[vi] += S[t][si,vi]*S[k][vi,di]/S[t][si,di]
                            if np.isnan(betweenness[vi]):
                                print "Whoops! bt is nan, %d %d %d"%(si,vi,di)
                                print (S[t][si,vi],S[k][vi,di],S[t][si,di])
                                raw_input()
                # end for vi
            # end for di
        # end for si
        #print "betweenness computation done for time %d"%(t)
        #print "number of vertex triplets visited for betweennness %d"%(total_its)
    #betweenness = betweenness/(0.5*(n-1)*(n-2)*m)
    print V_s
    print V_d
    norm_ct = np.array([(0.5*m*len(V_s[vi])*len(V_d[vi])) for vi in xrange(n)])
    print norm_ct
    betweenness = np.divide(betweenness,0.5*np.power(norm_ct,1))

    return dict(zip(verts,betweenness))

def compute_static_graph_statistics(G,start_time,end_time):
    verts = G.vertices
    n = len(verts)
    m = end_time - start_time
    agg_statistics = [dict.fromkeys(verts,0),dict.fromkeys(verts,0),dict.fromkeys(verts,0)]*3
    avg_statistics = [dict.fromkeys(verts,0),dict.fromkeys(verts,0),dict.fromkeys(verts,0)]*3

    aggregated_graph = nx.Graph()
    aggregated_graph.add_nodes_from(verts)
    for t in xrange(start_time,end_time+1):
        aggregated_graph.add_edges_from(G.snapshots[t].edges_iter())
         
        dc = nx.degree_centrality(G.snapshots[t])
        cc = nx.closeness_centrality(G.snapshots[t])
        bc = nx.betweenness_centrality(G.snapshots[t])
        for v in verts:
            avg_statistics[0][v] += dc[v]/m
            avg_statistics[1][v] += cc[v]/m
            avg_statistics[2][v] += bc[v]/m

    
    dc = nx.degree_centrality(aggregated_graph)
    cc = nx.closeness_centrality(aggregated_graph)
    bc = nx.betweenness_centrality(aggregated_graph)
    for v in verts:
        agg_statistics[0][v] = dc[v]
        agg_statistics[1][v] = cc[v]
        agg_statistics[2][v] = bc[v]
    return (agg_statistics, avg_statistics)

from collections import defaultdict, namedtuple, Counter
import networkx as nx
import osmnx as ox
import math
from heapq import heappush, heappop
from itertools import count




# G=nx.read_shp('test.shp')



"""
TODO: Generate a class of cyclists that calculates stress and decides routes the same way and put them on the graph at the same time. 
Then we can make different classes of cyclists of different sizes to reflect a bell-curved population and overlay the results.

Eventually we might want to figure out where the bottlenecks are: 

Can a cyclist edit an edge's attribute dictionary in real time? If yes (and we run groups of identical cyclists over the graph)
    we can have the first explorer calculate stress and have everyone else read directly from the edge


Also: maybe have the cyclists capture points on their route where they decided against a certain edge/node (and why?)
"""



class Cyclist(object):
    def __init__(self, acceptable_stress, graph, location, calculate_stress, probabilistic=True):
        self.acceptable_stress = acceptable_stress
        self.graph = graph
        self.location = location
        self.probabilistic = probabilistic
        self.calculate_stress = calculate_stress

    def decide_weight(self, edgeDict):
        if edgeDict.get("stress", None):
            stress = edgeDict["stress"]

        else:
            stress = self.calculate_stress(edgeDict)

        if stress > self.acceptable_stress:
            if not (self.probabilistic and math.random()>0.9):
            # cyclist decides to skip the stress
                return 10000000                
        return edgeDict.get("length",0) # return edge length

    def astar_path_with_calculation(self, G, source, target, heuristic=None):
        """Return a list of nodes in a shortest path between source and target
        using the A* ("A-star") algorithm.

        There may be more than one shortest path.  This returns only one.

        Parameters
        ----------
        G : NetworkX graph

        source : node
           Starting node for path

        target : node
           Ending node for path

        heuristic : function
           A function to evaluate the estimate of the distance
           from the a node to the target.  The function takes
           two nodes arguments and must return a number.

        decide_weight: function
           A function to evaluate an edge dictionary and decide 
           whether to return the actual weight or an arbitrarily 
           large value (in the case of high stress). Relies on 
           self.calculate_weight.

        Raises
        ------
        NetworkXNoPath
            If no path exists between source and target.

        Examples
        --------
        >>> G = nx.path_graph(5)
        >>> print(nx.astar_path(G, 0, 4))
        [0, 1, 2, 3, 4]
        >>> G = nx.grid_graph(dim=[3, 3])  # nodes are two-tuples (x,y)
        >>> nx.set_edge_attributes(G, {e: e[1][0]*2 for e in G.edges()}, 'cost')
        >>> def dist(a, b):
        ...    (x1, y1) = a
        ...    (x2, y2) = b
        ...    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        >>> print(nx.astar_path(G, (0, 0), (2, 2), heuristic=dist, weight='cost'))
        [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]


        See Also
        --------
        shortest_path, dijkstra_path

        """
        if source not in G or target not in G:
            msg = 'Either source {} or target {} is not in G'
            raise nx.NodeNotFound(msg.format(source, target))

        if heuristic is None:
            # The default heuristic is h=0 - same as Dijkstra's algorithm
            def heuristic(u, v):
                return 0

        push = heappush
        pop = heappop

        # The queue stores priority, node, cost to reach, and parent.
        # Uses Python heapq to keep in priority order.
        # Add a counter to the queue to prevent the underlying heap from
        # attempting to compare the nodes themselves. The hash breaks ties in the
        # priority and is guaranteed unique for all nodes in the graph.
        c = count()
        queue = [(0, next(c), source, 0, None)]

        # Maps enqueued nodes to distance of discovered paths and the
        # computed heuristics to target. We avoid computing the heuristics
        # more than once and inserting the node into the queue too many times.
        enqueued = {}
        # Maps explored nodes to parent closest to the source.
        explored = {}

        while queue:
            # Pop the smallest item from queue.
            _, __, curnode, dist, parent = pop(queue)

            if curnode == target:
                path = [curnode]
                node = parent
                while node is not None:
                    path.append(node)
                    node = explored[node]
                path.reverse()
                return path

            if curnode in explored:
                continue

            explored[curnode] = parent

            for neighbor, w in G[curnode].items(): # w is the edge dict
                if neighbor in explored:
                    continue
                ncost = dist + self.decide_weight(edgeDict = w) 
                # This is what we changed. weight is a cyclist-owned function that takes in an edge attribute dictionary and calculates a weight.
                if neighbor in enqueued:
                    qcost, h = enqueued[neighbor]
                    # if qcost <= ncost, a less costly path from the
                    # neighbor to the source was already determined.
                    # Therefore, we won't attempt to push this neighbor
                    # to the queue
                    if qcost <= ncost:
                        continue
                else:
                    h = heuristic(neighbor, target)
                enqueued[neighbor] = ncost, h
                push(queue, (ncost + h, next(c), neighbor, ncost, curnode))

        raise nx.NetworkXNoPath("Node %s not reachable from %s" % (source, target))

    def heuristic(self, node1, node2):
        """
        Estimates the displacement between node1 (current location) and node2 (eventual destination).
        This ensures the algorithm starts off by trying the paths that lead us in the direction of the destination.
        """
        return self.graph.nodes()[node1]['displacement']

    def decide(self, goal):
        return self.astar_path_with_calculation(self.graph, self.location, goal, heuristic = self.heuristic)



def create_test_graph(stress_already_calculated = False):
    G = nx.DiGraph()

    if stress_already_calculated:
        edge12 = (1,2,{"length": 3, "stress":1, "roadType":"edge"})
        edge23 = (2,3,{"length": 4, "stress":1, "roadType":"edge"})
        edge34 = (3,4,{"length": 3, "stress":1, "roadType":"edge"})
        edge15 = (1,5,{"length": 3, "stress":4, "roadType":"edge"})
        edge54 = (5,4,{"length": 3, "stress":1, "roadType":"edge"})

    else:
        edge12 = (1,2,{"length": 3, "signalized":1, "separated":1, "traffic":1, "roadType":"edge"})
        edge23 = (2,3,{"length": 4, "signalized":1, "separated":1, "traffic":1, "roadType":"edge"})
        edge34 = (3,4,{"length": 3, "signalized":1, "separated":1, "traffic":1, "roadType":"edge"})
        edge15 = (1,5,{"length": 3, "signalized":0, "separated":0, "traffic":4, "roadType":"edge"})
        edge54 = (5,4,{"length": 3, "signalized":1, "separated":1, "traffic":1, "roadType":"edge"})

    
    nodeList = [(1, {"stress":1, 'displacement':10}),
                (2, {"stress":1, 'displacement':6}),
                (3, {"stress":1, 'displacement':3}),
                (4, {"stress":1, 'displacement':0}),
                (5, {"stress":4, 'displacement':7})]
    edgeList = [edge12,edge23,edge34,edge15,edge54]
    

    G.add_nodes_from(nodeList)
    G.add_edges_from(edgeList)
        
    return G


def calculate_stress(edgeDict, signal_weight=1, separation_weight=1, traffic_weight=1):
    stress = 1
    signalized = edgeDict.get("signalized",0) # each of these might be a function that reads more attributes from the edgeDict
    separated = edgeDict.get("separated",0)
    traffic = edgeDict.get("traffic",4)

    if not signalized:
        stress += signal_weight
    if not separated:
        stress += separation_weight

    stress += (traffic - 1)/2.0 * traffic_weight

    return min(stress, 4)





def test_cyclist_decision():

    G = create_test_graph()
    c1 = Cyclist(acceptable_stress = 2, graph = G, location=1, calculate_stress = calculate_stress, probabilistic = False)
    c2 = Cyclist(acceptable_stress = 4, graph = G, location=1, calculate_stress = calculate_stress, probabilistic = False)

    print("Unskilled cyclist:", c1.decide(goal=4))
    print("Skilled cyclist:", c2.decide(goal=4))


test_cyclist_decision()
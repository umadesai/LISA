from collections import defaultdict, namedtuple, Counter
import networkx as nx
import osmnx as ox
import math
from heapq import heappush, heappop
from itertools import count



"""
Can a cyclist edit an edge's attribute dictionary in real time? If yes (and we run groups of identical cyclists over the graph)
    we can have the first explorer calculate stress and have everyone else read directly from the edge

Also: maybe have the cyclists capture points on their route where they decided against a certain edge/node (and why?)
"""





class Cyclist(object):
    """
    An object representing a cyclist.

    decision_weights is a dictionary describing how much each cyclist cares about each road attribute category.
    calculate_stress is a function that uses the cyclist's feelings on each attribute to calculate the stress of an edge.

    """
    def __init__(self, acceptable_stress, graph, location, calculate_stress, decision_weights, probabilistic=True):
        self.acceptable_stress = acceptable_stress
        self.graph = graph
        self.location = location
        self.probabilistic = probabilistic
        self.calculate_stress = calculate_stress

        self.signal_weight = decision_weights.get("signal_weight",1)
        self.separation_weight = decision_weights.get("separation_weight",1)
        self.traffic_weight = decision_weights.get("traffic_weight",1)
        self.decision_model = decision_weights.get("name","Unknown decision")

    def decide_weight(self, edgeDict, currentNode):
        """
        Decides whether an edge is worth taking based on stress. High weights are bad.
        A node whose stress exeeds self.acceptable_stress is often (always if self.probabilistic = False) given an arbitrarily high weight.
        """
        if edgeDict.get("stress", None):
            stress = edgeDict["stress"]

        else:
            stress = self.calculate_stress(edgeDict, self.signal_weight, self.separation_weight, self.traffic_weight)
            # print("current node: ", currentNode, "edges: ", edgeDict.get("from",0), edgeDict.get("to",0), "stress: ", stress)

        if stress > self.acceptable_stress:
            if not (self.probabilistic and math.random()>0.9):
                # cyclist decides to skip the stressful edge
                print("skipping node: ", currentNode, "edges: ", edgeDict.get("from",0), edgeDict.get("to",0))
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
                ncost = dist + self.decide_weight(edgeDict = w, currentNode = neighbor) 
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

            # if curnode == target:
            #     path = [curnode]
            #     node = parent
            #     while node is not None:
            #         path.append(node)
            #         node = explored[node]
            #     path.reverse()
            #     return path

            # explored[curnode] = parent

        raise nx.NetworkXNoPath("Node %s not reachable from %s" % (source, target))

    def heuristic(self, node1, node2):
        """
        Estimates the displacement between node1 (current location) and node2 (eventual destination).
        This ensures the algorithm starts off by trying the paths that lead us in the direction of the destination.

        Eventually we'll be calculating as-the-crow-flies distance.
        """
        return self.graph.nodes()[node1]['displacement']

    def decide(self, goal):
        """
        Wrapper for the astar search.
        """
        return self.astar_path_with_calculation(self.graph, self.location, goal, heuristic = self.heuristic)


def calculate_stress(edgeDict, signal_weight=1, separation_weight=1, traffic_weight=1):
    """
    Takes in an edge attribute dictionary and calculates a stress level.
    """
    stress = 1
    signalized = edgeDict.get("signalized",0) # each of these might be a function that reads more attributes from the edgeDict
    separated = edgeDict.get("separated",0)
    traffic = edgeDict.get("traffic",4)

    if not signalized:
        stress += signal_weight
    if not separated:
        stress += separation_weight

    stress += (traffic - 1)/2.0 * traffic_weight # handwavey math so traffic doesn't just cap everything at 4

    return min(stress, 4)



def create_test_graph():
    G = nx.DiGraph()


    edge12 = (1,2,{"from":1, "to":2, "length": 3, "signalized":0, "separated":0, "traffic":1}) # only consistent good feature is traffic
    edge23 = (2,3,{"from":2, "to":3, "length": 4, "signalized":1, "separated":0, "traffic":1})
    edge34 = (3,4,{"from":3, "to":4, "length": 3, "signalized":0, "separated":1, "traffic":1})

    edge16 = (1,6,{"from":1, "to":6, "length": 3, "signalized":1, "separated":0, "traffic":2}) # only consistent good feature is signalization
    edge67 = (6,7,{"from":6, "to":7, "length": 4, "signalized":1, "separated":1, "traffic":4})
    edge74 = (7,4,{"from":7, "to":4, "length": 3, "signalized":1, "separated":0, "traffic":3})

    edge18 = (1,8,{"from":1, "to":8, "length": 3, "signalized":0, "separated":1, "traffic":1}) # only consistent good feature is separation
    edge89 = (8,9,{"from":8, "to":9, "length": 4, "signalized":1, "separated":1, "traffic":1})
    edge94 = (9,4,{"from":9, "to":4, "length": 3, "signalized":0, "separated":1, "traffic":1})

    edge15 = (1,5,{"from":1, "to":5, "length": 3, "signalized":0, "separated":0, "traffic":4}) # everything is bad here
    edge54 = (5,4,{"from":5, "to":4, "length": 3, "signalized":0, "separated":0, "traffic":4}) 


    
    nodeList = [(1, {'displacement':10}),
                (2, {'displacement':6}),
                (3, {'displacement':3}),
                (4, {'displacement':0}),
                (5, {'displacement':7}),
                (6, {'displacement':6}),
                (7, {'displacement':3}),
                (8, {'displacement':6}),
                (9, {'displacement':3})]
    edgeList = [edge12,edge23,edge34,edge15,edge54,edge16,edge67,edge74,edge18,edge89,edge94]
    

    G.add_nodes_from(nodeList)
    G.add_edges_from(edgeList)
        
    return G

def test_cyclist_decision():

    likes_low_traffic = {"name":"likes_low_traffic", "signal_weight":1, "separation_weight":1, "traffic_weight":2}
    likes_separation = {"name":"likes_separation", "signal_weight":1, "separation_weight":2, "traffic_weight":1}
    likes_signals = {"name":"likes_signals", "signal_weight":2, "separation_weight":1, "traffic_weight":1}

    decision_models = [likes_low_traffic, likes_separation, likes_signals]

    G = create_test_graph()

    unskilled_cyclists = []
    skilled_cyclists = []
    for i in range(3):
        unskilled_cyclists.append(Cyclist(acceptable_stress=3, graph=G, location=1, calculate_stress=calculate_stress, decision_weights=decision_models[i], probabilistic=False))
        skilled_cyclists.append(Cyclist(acceptable_stress=4, graph=G, location=1, calculate_stress=calculate_stress, decision_weights=decision_models[i], probabilistic=False))


    for u_cyclist in unskilled_cyclists:
        print("Unskilled cyclist - ", u_cyclist.decision_model, u_cyclist.decide(goal=4))

    for s_cyclist in skilled_cyclists:
        print("Skilled cyclist - ", s_cyclist.decision_model, s_cyclist.decide(goal=4))

    # c1 = Cyclist(acceptable_stress = 2, graph = G, location=1, calculate_stress = calculate_stress, decision_weights =  likes_low_traffic, probabilistic = False)
    # c2 = Cyclist(acceptable_stress = 4, graph = G, location=1, calculate_stress = calculate_stress, decision_weights =  likes_low_traffic, probabilistic = False)

    # print("Unskilled cyclist:", c1.decide(goal=4))
    # print("Skilled cyclist:", c2.decide(goal=4))


test_cyclist_decision()
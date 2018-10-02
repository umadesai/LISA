from collections import defaultdict, namedtuple, Counter
import networkx as nx


#TODO: Think about directionality in more detail, as well as multiple edges between the same two nodes

class Cyclist(object):
    def __init__(self, acceptableStress, graph, location):
        self.acceptableStress = acceptableStress
        self.graph = graph
        self.location = location

    def reconstruct_path(self, cameFrom, current):
        total_path = [current]
        while current in cameFrom:
            current = cameFrom[current]
            if current:
                total_path.append(current) 
        return total_path[::-1]


    #Copied off wikipedia's A* search algorithm page. 
    # Minimizes f(n) = g(n) + h(n) where g(n) is cost to get from start to n and h(n) is some heuristic for how far n is from the goal
    # In our case, h(n) could be the absolute (x/y) distance
    def aStarSearch(self, g, acceptableStress, start, goal): 
        # start and goal are node uids instead of the actual node objects, for convenience.
        # Graph[start] will give us the actual node object with uid = start.

        evaluated = set() # set of all the nodes we've evaluated
        known = {start} # set of all the nodes we know exist

        cameFrom = {start:None} # node : preceding_node_in_shortest_path

        gScore = defaultdict(lambda:2**31) # distance from start
        gScore[start] = 0
        fScore = defaultdict(lambda:2**31) # adjusted goodness (distance from goal
        fScore[start] = g[start].get_distance_from_goal(goal)

        while known:
            current = sorted([uid for uid in known], key = lambda x: fScore[x])[0] # we sort by the uids by fScore[uid] and take the first index
            # print("current:", current)
            if current == goal:
                return self.reconstruct_path(cameFrom, current)

            known.remove(current)
            evaluated.add(current)
            if not g[current].children:
                continue

            for child in g[current].children:
                # print("child:", child)
                if child in evaluated or g[child].stress > acceptableStress:
                    continue
                else:
                    tentative_gScore = gScore[current] + g[child].length

                if child not in known:
                    known.add(child)

                elif tentative_gScore > gScore[child]: # we found a worse path.
                    continue

                # This path is the best path so far.
                cameFrom[child] = current
                gScore[child] = tentative_gScore
                fScore[child] = tentative_gScore + g[child].get_distance_from_goal(goal)
                # print("child fScore:", fScore[child])

    def decide(self, goal):
        return self.aStarSearch(self.graph.nodes, self.acceptableStress, self.location, goal)


class Node(object):
    def __init__(self, uid, stress, length, distance_from_goal = None, children = None): # we use children because of directionality.
        self.uid = uid
        self.stress = stress
        self.length = length

        self.children = None
        self.children = self.add_children(children)

        self.distance_from_goal = distance_from_goal # in the future we might want an x/y coordinate for Nodes to calculate distance from goal.
        
    def add_children(self, listOfChildren):
        if self.children == None:
            self.children = listOfChildren
        else:
            self.children.extend(listOfChildren)

    def get_distance_from_goal(self, goal):
        return self.distance_from_goal
        # in the future, this might take in the goal's x/y coords and do some math.


class Graph(object):
    def __init__(self, nodeList):
        self.nodes = self.create_dict_from_nodeList(nodeList)

    def create_dict_from_nodeList(self, nodeList):
        nodes = {}
        for node in nodeList:
            nodes[node.uid] = node
        return nodes

    def __str__(self):
        return str(self.nodes)


def create_test_graph():
    node1 = Node(uid=1, stress=2, length=1, distance_from_goal=5)
    node2 = Node(uid=2, stress=2, length=10, distance_from_goal=4)
    node3 = Node(uid=3, stress=2, length=2, distance_from_goal=2)
    node4 = Node(uid=4, stress=2, length=1, distance_from_goal=0)
    node5 = Node(uid=5, stress=4, length=2, distance_from_goal=2)

    node1.add_children([5, 2])
    node2.add_children([3])
    node3.add_children([4])
    node5.add_children([4])


    nodeList = [node1, node2, node3, node4, node5]

    test_graph = Graph(nodeList)

    return test_graph


def test_cyclist_decision():

    G = create_test_graph()
    c1 = Cyclist(2, graph = G, location=1)
    c2 = Cyclist(4, graph = G, location=1)

    print(c1.decide(goal=4))
    print(c2.decide(goal=4))


test_cyclist_decision()








########################
# NetworkX stuff below #
########################    

# class World(object):
#     def __init__(self, graph=None):
#         if not graph:
#             self.graph = self.generate_graph()
#         self.cyclists = {} # [key]:[value] pairs of [skill]:[array of cyclists]?
#     def generate_graph(self):
#         G = nx.DiGraph()

#         node1, node1_Data = 1, {"stress":2, "length":"1", "roadType":"node", "roadId":1}
#         node2, node2_Data = 2, {"stress":2, "length":"2", "roadType":"node", "roadId":2}
#         node3, node3_Data = 3, {"stress":2, "length":"2", "roadType":"node", "roadId":3}
#         node4, node4_Data = 4, {"stress":2, "length":"1", "roadType":"node", "roadId":4}
#         node5, node5_Data = 5, {"stress":4, "length":"4", "roadType":"edge", "roadId":5}

       
#         nodeList = [(node1, node1_Data),(node2, node2_Data),(node3, node3_Data),(node4, node4_Data), (node5, node5_Data)]
        
#         # edge12 = (node1,node2,{"length": 3, "stress":1, "roadType":"edge"})
#         # edge23 = (node2,node3,{"length": 4, "stress":1, "roadType":"edge"})
#         # edge34 = (node3,node4,{"length": 3, "stress":1, "roadType":"edge"})
#         # edge14 = (node1,node4,{"length": 7, "stress":3, "roadType":"edge"})

#         edge12 = (node1,node2)
#         edge23 = (node2,node3)
#         edge34 = (node3,node4)
#         edge15 = (node1,node5)
#         edge54 = (node5,node4)
        
#         edgeList = [edge12,edge23,edge34,edge15, edge54]

#         G.add_nodes_from(nodeList)
#         G.add_edges_from(edgeList)
        
#         return G




# w = World()
# currentLocation = 1

# def bfsDistance(graph,start):
#     #works on connected graph. On a non-connected graph, looking up an unconnected node in the dictionary will return none.
#     stressDict = nx.get_node_attributes(graph, "stress")
#     lengthDict = nx.get_node_attributes(graph, "length")
#     distances = {}
#     depth = 0
#     nextlevel = {start}
#     while nextlevel:
#         thislevel = nextlevel
#         nextlevel = set()
#         for node in thislevel:
#             if node not in distances:
#                 distances[node] = stressDict[node]
#                 nextlevel.update([key for key in graph.adj[node]]) #node.children is a list/set of other nodes.
#         depth += 1
#     return distances

# connections = w.graph.adj[1]

# print(bfsDistance(w.graph, 1))

# print([connections[k] for k in connections])


# print(w.graph[1][2])

# c = Cyclist(skill=2,graph = w.graph, location=1)

# c.decide()

# w.graph.add_edges_from(edgeList)

# print([unpack(tup) for tup in w.graph.edges])
# print([unpack(node) for node in w.graph.nodes])
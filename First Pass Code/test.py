import networkx as nx


["separation", "signals", "visibility", "traffic"]


try:
    G=nx.read_shp("Shapefile/2017_LTS_Trails_FINAL.shp")
except:
    print("shapefile read failed")

# G = nx.DiGraph(day="Friday")
# G.add_nodes_from([(1,{"nodeAttr":1}), (2,{"nodeAttr":2})])


# def decide_stress():
#     return edgeDict.get("part1",0) + edgeDict.get("part2",0)


# edgeDict = {"part1":5, "part2": 10, "decide_stress": decide_stress}

# edge1 = (1,2,edgeDict)

# edgeList = [edge1]


print(len(G.nodes))
print(len(G.edges))

# G.add_edges_from(edgeList)

# G.add_edges_from([(1,2,{"edgeAttr":3})])

# print(G[1][2]["edgeAttr"])
# print([tup for tup in G[1].items()])
# print([k.get("decide_stress") for _ ,k in G[1].items()])

# def create_test_graph():
#     G = nx.DiGraph()

#     edge12 = (1,2,{"weight": 3, "stress":1, "roadType":"edge"}) # weight is length
#     edge23 = (2,3,{"weight": 4, "stress":1, "roadType":"edge"})
#     edge34 = (3,4,{"weight": 3, "stress":1, "roadType":"edge"})
#     edge15 = (1,5,{"weight": 7, "stress":4, "roadType":"edge"})
#     edge54 = (5,4,{"weight": 7, "stress":1, "roadType":"edge"})
#     edge44 = (4,4,{"weight": 7, "stress":1, "roadType":"edge"})

    
#     nodeList = [1,2,3,4,5]
#     edgeList = [edge12,edge23,edge34,edge15,edge54, edge44]
    

#     G.add_nodes_from(nodeList)
#     G.add_edges_from(edgeList)
        
#     return G


# G = create_test_graph()
# print(G[1][2])


# def heu(node, goal):
#     print(node1, node2)
#     if G[node1][node2]["stress"] > 3: #this gets the stress of the edge between them
#         if math.random()>0.1:
#             return 1000000
#     else:
#         return 0
#     return 

# print(nx.astar_path(G,1,4,heu))




# # print(list(G.edges([1,2])))
# print(G.edges.data())
# print(G.nodes.data())
# # # G.nodes[1]
# print(G.nodes())
# # print(G.number_of_nodes())
# print(G.edges())

# # import networkx as nx
# # G = nx.complete_graph(5)
# for node in G.nodes:
#     print(node)

# a = (1,2)
# b = list(a)
# c = [1,2]
# d = tuple(c)
# print(d)
import networkx as nx

G = nx.Graph(day="Friday")
G.add_nodes_from([(1,{"nodeAttr":1}), (2,{"nodeAttr":2})])

edge1 = (1,2,{"edgeAttr":3})

edgeList = [edge1]




G.add_edges_from(edgeList)

# G.add_edges_from([(1,2,{"edgeAttr":3})])

# # print(list(G.edges([1,2])))
print(G.edges.data())
print(G.nodes.data())
# # G.nodes[1]
print(G.nodes())
# print(G.number_of_nodes())
print(G.edges())

# import networkx as nx
# G = nx.complete_graph(5)
for node in G.nodes:
    print(node)

# a = (1,2)
# b = list(a)
# c = [1,2]
# d = tuple(c)
# print(d)
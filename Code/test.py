import networkx as nx
from Simulation_NetworkX import Cyclist
from Simulation_NetworkX import calculate_stress


a = [1,2,3]
b = [1,2,3]
# print(a==b)
# if b[4]:
#     print("we found something")


def create_test_graph():
    G = nx.DiGraph()

    edgeList = [

    (1,2,{"from":1, "to":2, "length": 3, "signalized":1, "separated":0, "traffic":1}), # only consistent good feature is traffic
    (2,3,{"from":2, "to":3, "length": 4, "signalized":1, "separated":0, "traffic":1}),
    (3,4,{"from":3, "to":4, "length": 3, "signalized":0, "separated":1, "traffic":1}),

    (1,6,{"from":1, "to":6, "length": 3, "signalized":1, "separated":0, "traffic":2}), # only consistent good feature is signalization
    (6,7,{"from":6, "to":7, "length": 4, "signalized":1, "separated":1, "traffic":4}),
    (7,4,{"from":7, "to":4, "length": 3, "signalized":1, "separated":0, "traffic":3}),

    (1,8,{"from":1, "to":8, "length": 3, "signalized":0, "separated":1, "traffic":1}), # only consistent good feature is separation
    (8,9,{"from":8, "to":9, "length": 4, "signalized":1, "separated":1, "traffic":1}),
    (9,4,{"from":9, "to":4, "length": 3, "signalized":0, "separated":1, "traffic":1}),

    (1,5,{"from":1, "to":5, "length": 3, "signalized":0, "separated":0, "traffic":4}), # everything is bad here
    (5,4,{"from":5, "to":4, "length": 3, "signalized":0, "separated":0, "traffic":4}),


    (3,8,{"from":3, "to":8, "length": 3, "signalized":1, "separated":1, "traffic":4}),
    (9,3,{"from":9, "to":3, "length": 3, "signalized":1, "separated":1, "traffic":4})

    ]

    
    nodeList = [(1, {'displacement':10}),
                (2, {'displacement':6}),
                (3, {'displacement':3}),
                (4, {'displacement':0}),
                (5, {'displacement':7}),
                (6, {'displacement':6}),
                (7, {'displacement':3}),
                (8, {'displacement':6}),
                (9, {'displacement':3})]
    # edgeList = [edge12,edge23,edge34,edge15,edge54,edge16,edge67,edge74,edge18,edge89,edge94, edge38, edge93]
    

    G.add_nodes_from(nodeList)
    G.add_edges_from(edgeList)
    # print(G.nodes(data=True))
    return G




def count_deviations(route_data, source, destination, optimal_routing_function):
    # deviance = 0
    # print("data: ", route_data, source, destination)
    if source==destination:
        return 0
    optimal_route = optimal_routing_function(source,destination)
    if route_data == optimal_route:
        return 0
    print("real life: ", route_data, "optimal: ", optimal_route)
    for i in range(len(route_data)):
        if optimal_route[i]!=route_data[i]:
            return count_deviations(route_data[i:], route_data[i], route_data[-1], optimal_routing_function) + 1
    return 0



likes_low_traffic = {"name":"likes_low_traffic", "signal_weight":1, "separation_weight":1, "traffic_weight":2}
likes_separation = {"name":"likes_separation", "signal_weight":1, "separation_weight":2, "traffic_weight":1}
likes_signals = {"name":"likes_signals", "signal_weight":2, "separation_weight":1, "traffic_weight":1}

decision_models = [likes_low_traffic, likes_separation, likes_signals]

G = create_test_graph()

def calculate_ideal_route(start,end):
    c = Cyclist(acceptable_stress=3, graph=G, location=start, calculate_stress=calculate_stress, decision_weights=decision_models[0], probabilistic=False)
    return c.decide(end)

def deviations_per_length(route, pathfinding_function):
    return count_deviations(route, route[0], route[-1], pathfinding_function)/(len(route)-1)

# print(deviations_per_length([1,2,3,8,9,3,4],calculate_ideal_route))




# print(foo(2,6))

# print(calculate_deviance_per_length([1,4,3,2],1,2,foo))
# print(calculate_deviance_per_length([1,3,2],1,2,foo))







def create_path_attribute_distribution(graph, path, attributes):
    """
    Takes in a graph, a path representing nodes through the graph, and a list of attributes of interest and returns a dictionary with ATTRIBUTE_NAME:DISTRIBUTION_COUNTER key:value pairs. This tells us how many traversed edges (no order involved) have a certain value of a certain attribute.
    """
    import collections
    attribute_counters = []
    for attribute in attributes:
        attribute_counters.append(collections.Counter())
    for i in range(len(path)-1):
        # print(path[i],path[i+1])
        edge = graph[path[i]][path[i+1]]
        # print(edge)
        for j in range(len(attributes)):
             attribute_counters[j].update([edge[attributes[j]]]) #possibly inefficient
    return {attributes[k]:attribute_counters[k] for k in range(len(attributes))}


def sum_of_squared_differences(distribution_dict_1, distribution_dict_2, path1_length, path2_length):
    """
    distribution_dict_1 and 2 are dictionaries with ATTRIBUTE_NAME:DISTRIBUTION_COUNTER key:value pairs representing two paths.

    For each attribute, we sum the difference between each value in the distribution counters for each path. 
    """

    differences = {}

    for attribute in distribution_dict_1:
        counter1 = distribution_dict_1[attribute]
        counter2 = distribution_dict_2[attribute]
        # print(attribute, counter1, counter2)
        seen = set()
        result = 0
        for key in counter1:
            # print("c1 key", key)
            seen.add(key)
            result += (counter1[key]/path1_length - counter2[key]/path2_length)**2
            # print("adding to result counter1: ", (counter1[key]/path_length - counter2[key]/path_length)**2)
        for key in counter2:
            # print("c2 key", key)
            if key not in seen:
                seen.add(key)
                result += (counter1[key]/path1_length - counter2[key]/path2_length)**2
                # print("adding to result counter2: ", (counter1[key]/path_length - counter2[key]/path_length)**2)
        # print(result)
        differences[attribute] = result

    return (sum([differences[m] for m in differences]), differences)


def calculate_attribute_differences(path1):
    path2 = calculate_ideal_route(path1[0],path1[-1])
    attributes = ["length", "signalized", "separated", "traffic"]

    path1_attribute_distribution = create_path_attribute_distribution(G,path1,attributes)
    path2_attribute_distribution = create_path_attribute_distribution(G,path2,attributes)
    # print(path1_attribute_distribution, path2_attribute_distribution)


    res = sum_of_squared_differences(path1_attribute_distribution, path2_attribute_distribution, path1_length = len(path1), path2_length = len(path2))

    return res

print(calculate_attribute_differences([1,2,3,8,9,3,4]))



def bucketer(raw_dict, weight_dict): 
    """
    Bundles a bunch of raw attributes into larger buckets, which cyclists have general opinions about.
    Some buckets are positive (signalized, separation) and others are likely negative (traffic)
    
    Some of the raw attributes might be big numbers (like traffic volume/speed limit)

    I'm not sure if we want a binary "contains one from each category" or some weighted average of all attributes.

    Current implementation is weighted average.


    raw_dict is the edge attribute dictionary that the graph is initialized with. It has a bunch of attributes in it which we can group into larger categories.

    Each attribute contributes a different weight to the category score, and these attribute:weight pairs are stored in weight_dict.
    """ 

    return {bucket_name:sum([raw_dict.get(x,0)*weight_dict[bucket_name].get(x,0) for x in weight_dict[bucket_name]]) for bucket_name in weight_dict}



raw_dict = {"stop_sign":1, "traffic_light":0, "bike_lane":0, "separate_path":0, "crosswalk":1, "traffic_volume": 4, "speed_limit":35}


weight_dict = {
    "signalized": {"stop_sign":0.5, "traffic_light":1},
    "separated" : {"bike_lane":0.5, "separate_path":1, "crosswalk":0.25}, # is crosswalk signal or separation? will it even show up on our graph?    
    "traffic"   : {"traffic_volume":10, "speed_limit":1},
    "misc"      : {}
}

print(bucketer(raw_dict, weight_dict))


def calculate_LTS(processed_dict, more_weights):
    """
    For the buckets, signalized and separated make LTS lower and traffic makes LTS higher
    """
    weighted_LTS_components = {bucket_name:sum([processed_dict.get(x,0)*more_weights[bucket_name].get(x,0) for x in more_weights[bucket_name]]) for bucket_name in more_weights}

    weighted_list = [weighted_LTS_components[x] for x in weighted_LTS_components]

    return round(sum(weighted_list)/(len(weighted_list)*1.0))









# ["separation", "signals", "visibility", "traffic"]

# if not 0:
#     print("stuff")
# else:
#     print("0 is false-y")


# try:
#     G=nx.read_shp("Shapefile/2017_LTS_Trails_FINAL.shp")
# except:
#     print("shapefile read failed")

# print(len(G.nodes))
# print(len(G.edges))



# G = nx.DiGraph(day="Friday")
# G.add_nodes_from([(1,{"nodeAttr":1}), (2,{"nodeAttr":2})])


# def decide_stress():
#     return edgeDict.get("part1",0) + edgeDict.get("part2",0)


# edgeDict = {"part1":5, "part2": 10, "decide_stress": 2}

# edge12 = (1,2,edgeDict)
# edge21 = (2,1, edgeDict)


# edgeList = [edge12, edge21]




# G.add_edges_from(edgeList)

# # print(G.adj[1])
# print(G.edges(1))
# print(G.in_edges(1))


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
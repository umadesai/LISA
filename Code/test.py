import networkx as nx
from Simulation_NetworkX import Cyclist
from Simulation_NetworkX import calculate_stress


def create_test_graph():
    G = nx.DiGraph()

    edgeList = [

        # only consistent good feature is traffic
        (1, 2, {"from": 1, "to": 2, "length": 3,
                "signalized": 1, "separated": 0, "traffic": 1}),
        (2, 3, {"from": 2, "to": 3, "length": 4,
                "signalized": 1, "separated": 0, "traffic": 1}),
        (3, 4, {"from": 3, "to": 4, "length": 3,
                "signalized": 0, "separated": 1, "traffic": 1}),

        # only consistent good feature is signalization
        (1, 6, {"from": 1, "to": 6, "length": 3,
                "signalized": 1, "separated": 0, "traffic": 2}),
        (6, 7, {"from": 6, "to": 7, "length": 4,
                "signalized": 1, "separated": 1, "traffic": 4}),
        (7, 4, {"from": 7, "to": 4, "length": 3,
                "signalized": 1, "separated": 0, "traffic": 3}),

        # only consistent good feature is separation
        (1, 8, {"from": 1, "to": 8, "length": 3,
                "signalized": 0, "separated": 1, "traffic": 1}),
        (8, 9, {"from": 8, "to": 9, "length": 4,
                "signalized": 1, "separated": 1, "traffic": 1}),
        (9, 4, {"from": 9, "to": 4, "length": 3,
                "signalized": 0, "separated": 1, "traffic": 1}),

        (1, 5, {"from": 1, "to": 5, "length": 3, "signalized": 0,
                "separated": 0, "traffic": 4}),  # everything is high stress
        (5, 4, {"from": 5, "to": 4, "length": 3,
                "signalized": 0, "separated": 0, "traffic": 4}),


        (3, 8, {"from": 3, "to": 8, "length": 3,
                "signalized": 1, "separated": 1, "traffic": 4}),
        (9, 3, {"from": 9, "to": 3, "length": 3,
                "signalized": 1, "separated": 1, "traffic": 4})

    ]

    nodeList = [(1, {'displacement': 10}),
                (2, {'displacement': 6}),
                (3, {'displacement': 3}),
                (4, {'displacement': 0}),
                (5, {'displacement': 7}),
                (6, {'displacement': 6}),
                (7, {'displacement': 3}),
                (8, {'displacement': 6}),
                (9, {'displacement': 3})]

    G.add_nodes_from(nodeList)
    G.add_edges_from(edgeList)
    return G


def count_deviations(route_data, source, destination,
                     optimal_routing_function):
    if source == destination:
        return 0
    optimal_route = optimal_routing_function(source, destination)
    if route_data == optimal_route:
        return 0
    print("real life: ", route_data, "optimal: ", optimal_route)
    for i in range(len(route_data)):
        if optimal_route[i] != route_data[i]:
            return 1 + count_deviations(route_data[i:],
                                        route_data[i],
                                        route_data[-1],
                                        optimal_routing_function)
    return 0


likes_low_traffic = {"name": "likes_low_traffic", "signal_weight": 1,
                     "separation_weight": 1, "traffic_weight": 2}
likes_separation = {"name": "likes_separation", "signal_weight": 1,
                    "separation_weight": 2, "traffic_weight": 1}
likes_signals = {"name": "likes_signals", "signal_weight": 2,
                 "separation_weight": 1, "traffic_weight": 1}

decision_models = [likes_low_traffic, likes_separation, likes_signals]

G = create_test_graph()


def calculate_ideal_route(start, end):
    c = Cyclist(acceptable_stress=3, graph=G, location=start,
                calculate_stress=calculate_stress,
                decision_weights=decision_models[0], probabilistic=False)
    return c.decide(end)


def deviations_per_length(route, pathfinding_function):
    return count_deviations(route, route[0], route[-1],
                            pathfinding_function)/(len(route)-1)


def create_path_attribute_distribution(graph, path, attributes):
    """
    Takes in a graph, a path representing nodes through the graph, and a
    list of attributes of interest and returns a dictionary with
    ATTRIBUTE_NAME:DISTRIBUTION_COUNTER key:value pairs. This tells us how
    many traversed edges (no order involved) have a certain value of a
    certain attribute.
    """
    import collections
    attribute_counters = []
    for attribute in attributes:
        attribute_counters.append(collections.Counter())
    for i in range(len(path)-1):
        edge = graph[path[i]][path[i+1]]
        for j in range(len(attributes)):
            attribute_counters[j].update([edge[attributes[j]]])
            # possibly inefficient
    return {attributes[k]: attribute_counters[k] for k in
            range(len(attributes))}


def sum_of_squared_differences(distribution_dict_1, distribution_dict_2,
                               path1_length, path2_length):
    """
    distribution_dict_1 and 2 are dictionaries with
    ATTRIBUTE_NAME:DISTRIBUTION_sCOUNTER key:value pairs
    representing two paths. For each attribute, we sum the difference between
    each value in the distribution counters for each path.
    """

    differences = {}

    for attribute in distribution_dict_1:
        counter1 = distribution_dict_1[attribute]
        counter2 = distribution_dict_2[attribute]
        seen = set()
        result = 0
        for key in counter1:
            seen.add(key)
            result += (counter1[key]/path1_length -
                       counter2[key]/path2_length)**2
        for key in counter2:
            if key not in seen:
                seen.add(key)
                result += (counter1[key]/path1_length -
                           counter2[key]/path2_length)**2
        differences[attribute] = result

    return (sum([differences[m] for m in differences]), differences)


def calculate_attribute_differences(path1):
    path2 = calculate_ideal_route(path1[0], path1[-1])
    attributes = ["length", "signalized", "separated", "traffic"]

    path1_attribute_distribution = create_path_attribute_distribution
    (G, path1, attributes)
    path2_attribute_distribution = create_path_attribute_distribution
    (G, path2, attributes)

    res = sum_of_squared_differences(
        path1_attribute_distribution, path2_attribute_distribution,
        path1_length=len(path1), path2_length=len(path2))

    return res


if __name__ == "__main__":
    calculate_attribute_differences([1, 2, 3, 8, 9, 3, 4])

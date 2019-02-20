from ProportionalAttributes_Cyclist import *

start = 1
end = 4

G = create_test_graph()
preference_low_traffic      = {"name": "likes_low_traffic", "signal_weight": 1, "separation_weight": 1, "traffic_weight": 2}
preference_likes_separation = {"name": "likes_separation",  "signal_weight": 1, "separation_weight": 2, "traffic_weight": 1}
preference_likes_signals    = {"name": "likes_signals",     "signal_weight": 2, "separation_weight": 1, "traffic_weight": 1}
preference_none             = {"name": "no_preference", "signal_weight": 1, "separation_weight": 1, "traffic_weight": 1}


cyclist_low_traffic         = Cyclist(acceptable_stress=3, graph=G, location=start, calculate_stress=calculate_stress, probabilistic=False, 
                                    decision_weights=preference_low_traffic)
cyclist_likes_separation    = Cyclist(acceptable_stress=3, graph=G, location=start, calculate_stress=calculate_stress, probabilistic=False, 
                                    decision_weights=preference_likes_separation)
cyclist_likes_signals       = Cyclist(acceptable_stress=3, graph=G, location=start, calculate_stress=calculate_stress, probabilistic=False, 
                                    decision_weights=preference_likes_signals)

route_lowtraffic            = cyclist_low_traffic.decide(goal=end)
route_separation            = cyclist_likes_separation.decide(goal=end)
route_signals               = cyclist_likes_signals.decide(goal=end)


# first route likes low traffic, second route likes separation, third route likes signals
realworld_routes = [route_lowtraffic, route_separation, route_signals]

def find_deviations(realworld_route):
    """
    """
    print("realworld_route: ", realworld_route)
    start = realworld_route[0]
    end = realworld_route[-1]

    predicted_cyclist = Cyclist(acceptable_stress=4, graph=G, location=start, calculate_stress=calculate_stress, 
                                probabilistic=False, decision_weights=preference_none)
    predicted_route = predicted_cyclist.decide(goal=end)
    print("predicted_route: ", predicted_route)
    deviations = []
    
    for node_index in range(1, len(realworld_route)):
        currentNode = realworld_route[node_index]

        predicted_route = predicted_cyclist.decide(goal=4)
        
        # check if predicted route's node_index node is different
        if (currentNode != end):
            # we KNOW that the currentNode will be the same as the predicted_route 
            predictedNode = predicted_route[1]
            new_start = None
            if currentNode == predictedNode:
                new_start = realworld_route[node_index+1]
            else:
                deviations.append({"origin": predicted_route[node_index-1], "predicted": predictedNode, "actual": currentNode})
                new_start = currentNode
            predicted_cyclist = Cyclist(acceptable_stress=4, graph=G, location=new_start, calculate_stress=calculate_stress, 
                                            probabilistic=False, decision_weights=preference_none)
    return deviations

def find_attribute_differences(deviations):
    """
    """
    attribute_differences = {}
    
    for deviation in deviations:
        origin_node     = deviation['origin']
        predicted_node  = deviation['predicted']
        actual_node     = deviation['actual']

        predicted_edge_attributes = G.get_edge_data(origin_node, predicted_node)
        actual_edge_attributes    = G.get_edge_data(origin_node, actual_node)

        for original_key in predicted_edge_attributes.keys():
            if original_key not in ["to", "from"]:
                current_attribute_difference = actual_edge_attributes[original_key] - predicted_edge_attributes[original_key]

                if (current_attribute_difference != 0):
                    attribute_change = actual_edge_attributes[original_key] - predicted_edge_attributes[original_key]
                    
                    if (attribute_change > 0):
                        revised_key = original_key + "+"
                    else:
                        revised_key = original_key + "-"

                    new_magnitude_change = abs(current_attribute_difference)
                    if (revised_key not in attribute_differences.keys()):
                        attribute_differences[revised_key] = new_magnitude_change
                    else:
                        original_magnitude_change = abs(attribute_differences[revised_key])
                        if (new_magnitude_change < original_magnitude_change):
                            attribute_differences[revised_key] = new_magnitude_change
    return attribute_differences


if __name__ == "__main__":
    for realworld_route in realworld_routes:
        # Just iterating through all the preferential routes
        deviations = find_deviations(realworld_route)
        print("\ndeviations: ", deviations)
        print("attribute differences: ", find_attribute_differences(deviations), "\n")

############################
# Processing existing data #
############################


def get_tagged_graph_from_geographical_data(geographical_data):
    """
    Generates tagged graph of nodes and edges from existing geographical data.

    This calls one of several functions that extract nodes, edges and node/edge
    attributes from geographical data that exists in different formats
    (OSM, linear-referenced etc.)
    """
    graph = graph_extraction_function(geographical_data)
    return graph


def translate_biker_data_into_graph_format(set_of_biker_data):
    """
    Processes biker data into some graph-compatible format.

    This calls one of several functions that convert biker data (route trace,
    average daily traffic etc.) into graph-compatible format.
    """
    set_of_processed_biker_data = data_conversion_function(set_of_biker_data)
    return set_of_processed_biker_data


######################################
# Generating new data for comparison #
######################################

def generate_similar_ideal_data(biker_data, graph):
    """
    Given biker_data, generates ideal_data that has similar initial and end
    conditions for comparison. This calls one of several functions that look at
    graph properties and generate "ideal cyclist behavior". Ideal cyclist
    behavior can then be compared to real cyclist behavior.

    """
    ideal_data = ideal_data_generator_function(biker_data, graph)
    return ideal_data


def get_difference_between(biker_data, ideal_data):
    """
    Given biker_data and ideal_data, calculates some numerical value that
    represents how different they are.

    This calls one of several functions that compare two sets of behavior
    (e.g. two routes, or two graphs labelled with average daily traffic).
    """
    difference = comparison_function(biker_data, ideal_data)
    return difference


##############
# Validation #
##############

def validate(set_of_biker_data, geographical_data,
             generate_similar_ideal_data):
    """
    set_of_biker_data is a collection/iterable of real data.
        in LTS: real_data is probably an integer array that represents nodes
        traversed.

    graph is a networkX graph tagged with some attributes.
        attributes can be general like "stress" or "goodness" or specific like
        "bike_lane_width".
    """
    difference = None
    graph = get_tagged_graph_from_geographical_data(geographical_data)
    set_of_processed_biker_data = translate_biker_data_into_graph_format
    (set_of_biker_data)
    for biker_data in set_of_processed_biker_data:
        ideal_data = generate_similar_ideal_data(biker_data, graph)
        difference += get_difference_between(biker_data, ideal_data)
    return difference

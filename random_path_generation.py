import numpy as numpy
import networkx as nx
import random

def getRandomPath(start: (int, int), len: int):
    path = [start] # list of nodes)
    while (len > 1):
        neighbors = getNeighbors(start)
        neighbor =random.choice(neighbors)
        path.append(neighbor)
        start = neighbor
        len -= 1
    return path

def getNeighbors(nodeID: (int, int)):
    '''
        Dummy function that returns a list of dummy adjacent coordinates
    '''
    return [(random.randint(1, 100), random.randint(1, 100)), 
                (random.randint(1, 100), random.randint(1, 100)), 
                (random.randint(1, 100), random.randint(1, 100))]

# def getEdge(startID: int, neighborID: int):
#     '''
#         Returns the edge representation
#     '''
#     pass
if __name__ == "__main__":
    print(getRandomPath((0,0), 2))
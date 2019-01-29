import shapefile as shp
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
"""
Methods to read a shapefile's contents and make a graph out of it
"""


def make_graph(shape_base, shape_dir="shapefiles/"):
    # got code from here:
    # https: // chrishavlin.wordpress.com/2016/11/16/shapefiles-tutorial/
    G = nx.read_shp(shape_dir + shape_base + ".shp")
    # print(list(nx.connected_components(G)))
    nx.draw(G)
    plt.show()


def plot_single_shape(shape_index, shape_base, shape_dir="shapefiles/"):
    # got code from here:
    # https: // chrishavlin.wordpress.com/2016/11/16/shapefiles-tutorial/
    shapefile = shp.Reader(shape_dir+shape_base)

    plt.figure()
    ax = plt.axes()
    ax.set_aspect('equal')
    shape_ex = shapefile.shape(shape_index)
    x_lon = np.zeros((len(shape_ex.points), 1))
    y_lat = np.zeros((len(shape_ex.points), 1))
    for ip in range(len(shape_ex.points)):
        x_lon[ip] = shape_ex.points[ip][0]
        y_lat[ip] = shape_ex.points[ip][1]

    plt.plot(x_lon, y_lat, 'k')
    plt.show()


# Use sparingly! Plots ALL the shapes in the shapefile
def plot_all_shapes(shape_base, shape_dir="shapefiles/"):
    shapefile = shp.Reader(shape_dir+shape_base)

    # plotting
    plt.figure()
    ax = plt.axes()
    ax.set_aspect('equal')

    for shape in list(shapefile.iterShapes()):
        npoints = len(shape.points)  # total points
        nparts = len(shape.parts)  # total parts

        if nparts == 1:
            x_lon = np.zeros((len(shape.points), 1))
            y_lat = np.zeros((len(shape.points), 1))
            for ip in range(len(shape.points)):
                x_lon[ip] = shape.points[ip][0]
                y_lat[ip] = shape.points[ip][1]
            plt.plot(x_lon, y_lat)

        else:  # loop over parts of each shape, plot separately
            for ip in range(nparts):  # loop over parts, plot separately
                i0 = shape.parts[ip]
                if (ip < nparts-1):
                    i1 = shape.parts[ip+1]-1
                else:
                    i1 = npoints

                seg = shape.points[i0:i1+1]
                x_lon = np.zeros((len(seg), 1))
                y_lat = np.zeros((len(seg), 1))
                for ip in range(len(seg)):
                    x_lon[ip] = seg[ip][0]
                    y_lat[ip] = seg[ip][1]

                plt.plot(x_lon, y_lat)
    plt.show()


if __name__ == "__main__":
    make_graph("Signalized_Intersections_ACISA")

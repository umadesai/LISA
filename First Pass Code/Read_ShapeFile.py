
import shapefile as shp
import numpy as np
import matplotlib.pyplot as plt


# got code from here: https://chrishavlin.wordpress.com/2016/11/16/shapefiles-tutorial/ 

shp_file_base="2017_LTS_Trails_FINAL"
shp_dir = "shapefiles/"
print("directory: ", shp_dir+shp_file_base)
shapefile = shp.Reader(shp_dir+shp_file_base)



print('number of shapes imported:',len(shapefile.shapes()))
print(' ')
print('geometry attributes in each shape:')
for name in dir(shapefile.shape()):
    if not name.startswith('__'):
       print(name)


"""
       PLOTTING
"""

""" PLOTS A SINGLE SHAPE """
# plt.figure()
# ax = plt.axes()
# ax.set_aspect('equal')
# shape_ex = shapefile.shape(5) # getting the fifth shape
# print("shape_ex: ", shape_ex)
# x_lon = np.zeros((len(shape_ex.points),1))
# y_lat = np.zeros((len(shape_ex.points),1))
# for ip in range(len(shape_ex.points)):
#     print("x_lon: ", shape_ex.points[ip][0])
#     print("y_lon: ", shape_ex.points[ip][1])
#     x_lon[ip] = shape_ex.points[ip][0]
#     y_lat[ip] = shape_ex.points[ip][1]

# plt.plot(x_lon,y_lat,'k') 

# # # use bbox (bounding box) to set plot limits
# plt.xlim(shape_ex.bbox[0],shape_ex.bbox[2])

# """ PLOTS ALL SHAPES """
# plt.figure()
# ax = plt.axes()
# ax.set_aspect('equal')
# for shape in list(shapefile.iterShapes()):
#     x_lon = np.zeros((len(shape.points),1))
#     y_lat = np.zeros((len(shape.points),1))
#     for ip in range(len(shape.points)):
#         x_lon[ip] = shape.points[ip][0]
#         y_lat[ip] = shape.points[ip][1]
    
#     plt.plot(x_lon,y_lat) 



""" PLOTS ALL SHAPES AND PARTS """
plt.figure()
ax = plt.axes() # add the axes
ax.set_aspect('equal')

for shape in list(shapefile.iterShapes()):
    npoints=len(shape.points) # total points
    nparts = len(shape.parts) # total parts

    if nparts == 1:
        x_lon = np.zeros((len(shape.points),1))
        y_lat = np.zeros((len(shape.points),1))
        for ip in range(len(shape.points)):
            x_lon[ip] = shape.points[ip][0]
            y_lat[ip] = shape.points[ip][1]
        plt.plot(x_lon,y_lat) 

    else: # loop over parts of each shape, plot separately
        for ip in range(nparts): # loop over parts, plot separately
            i0=shape.parts[ip]
            if ip < nparts-1:
               i1 = shape.parts[ip+1]-1
            else:
               i1 = npoints
            
            seg=shape.points[i0:i1+1]
            x_lon = np.zeros((len(seg),1))
            y_lat = np.zeros((len(seg),1))
            for ip in range(len(seg)):
                x_lon[ip] = seg[ip][0]
                y_lat[ip] = seg[ip][1]
            
            plt.plot(x_lon,y_lat) 

plt.show()


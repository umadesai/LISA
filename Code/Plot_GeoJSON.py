from Graph_Wrapper import Graph, Name
from Signalized_Intersections import *
from shapely.geometry import shape, Point, Polygon
import geopandas as gpd
import geojsonio

routes = gpd.read_file('FILENAME.geojson')
jsonversion = routes.to_json()


geojsonio.display(routes)
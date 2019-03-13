import geopandas as gpd
import fiona


if __name__ == "__main__":

    layerlist = fiona.listlayers('scratch_022819.gdb')
    gdf = gpd.read_file('scratch_022819.gdb', layer='Block')
    multilinestrings = gdf.geometry

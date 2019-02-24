import shapefile


def extract_routes(sf, property=None):
    """
    Return list of routes with optional property information
    """
    geo_data = sf.__geo_interface__['features']
    if property:
        return [(g['properties'][property], list(g['geometry']['coordinates']))
                for g in geo_data]
    return [list(g['geometry']['coordinates']) for g in geo_data]


if __name__ == "__main__":

    bicycle_lanes = extract_routes(
        shapefile.Reader("Bicycle_Lanes.shp"), 'BIKELANELE')

    signed_bike_routes = extract_routes(
        shapefile.Reader("Signed_Bike_Routes.shp"))

    bike_trails = extract_routes(
        shapefile.Reader("Bike_Trails.shp"), 'LENGTH')

    traffic_barriers = extract_routes(
        shapefile.Reader("Traffic_Barriers.shp"), 'FEATURECOD')

    traffic_volumes = extract_routes(
        shapefile.Reader("2016_Traffic_Volume.shp"), 'AADT')

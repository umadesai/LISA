from osgeo import ogr

driverName = "OpenFileGDB"
driver = ogr.GetDriverByName(driverName)
ds = driver.Open("scratch.gdb")
print(ds)
print()

# print(ds.GetLayerCount())

if True:
    for i in range(ds.GetLayerCount()):
        layer = ds.GetLayerByIndex(i)
        srs = layer.GetSpatialRef()
        print("%d. \tLayer: %s" % (i, layer.GetName()))
        print("\tFeature count: %s" % layer.GetFeatureCount())
        print("\tFIDColumn: ", layer.GetFIDColumn())
        print("\LayerDefn: ", layer.GetLayerDefn().GetName())
        # iterate over features
        feat = layer.GetNextFeature()
        while feat is not None:
            feat = layer.GetNextFeature()
        feat = None

block_layer = ds.GetLayerByIndex(0)

name = block_layer.GetName()
feature_count = block_layer.GetFeatureCount()
print(name, feature_count)

# features
# f1 = block_layer.GetNextFeature()
# print("feature 1: ", f1.GetFID(), f1)
# print("field count: ", f1.GetFieldCount())
# print("geom field count: ", f1.GetGeomFieldCount())
# print("geom ref: ", f1.GetGeometryRef())
# print("native: ", f1.GetNativeData())
# for i in range(f1.GetFieldCount()):
#     print("\t", f1.GetFieldAsString(i))
#     print("\t\tType: ", f1.GetFieldType(i))
#     # print("\t", f1.GetFieldDefnRef(i))
#     # print("\t", f1.GetFieldName(i))

f205 = block_layer.GetFeature(205)
print("feature 205: ", f205.GetFID(), f205)



#  layer = ds.GetLayer()
#  capabilities = [
#      ogr.OLCRandomRead,
#      ogr.OLCSequentialWrite,
#      ogr.OLCRandomWrite,
#      ogr.OLCFastSpatialFilter,
#      ogr.OLCFastFeatureCount,
#      ogr.OLCFastGetExtent,
#      ogr.OLCCreateField,
#      ogr.OLCDeleteField,
#      ogr.OLCReorderFields,
#      ogr.OLCAlterFieldDefn,
#      ogr.OLCTransactions,
#      ogr.OLCDeleteFeature,
#      ogr.OLCFastSetNextByIndex,
#      ogr.OLCStringsAsUTF8,
#      ogr.OLCIgnoreFields
#  ]
#
#  print("Layer Capabilities:")
#  for cap in capabilities:
#      print("  %s = %s" % (cap, layer.TestCapability(cap)))

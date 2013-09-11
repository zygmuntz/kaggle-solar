"""
Usage: gefs2csv.py <input file>.nc <output file>.csv <lat> <lon>
get data (time x (var x hour)) for a point given by lat and lon
dimensions: (u'time', u'ens', u'fhour', u'lat', u'lon')
median value from ensembles
"""

import sys
import numpy as np
import netCDF4 as nc

input_file = sys.argv[1]
output_file = sys.argv[2]
lat = float( sys.argv[3] )
lon = float( sys.argv[4] )

d = nc.Dataset( input_file )

latitudes = d.variables['lat'][:]
lat_i = np.where( latitudes == lat )[0][0]

longitudes = d.variables['lon'][:]
lon_i = np.where( longitudes == lon )[0][0]

latitudes = map( int, list( latitudes ))
longitudes = map( int, list( longitudes ))

print "\nlat: %s...%s, lon: %s...%s\n" % ( min( latitudes ), max( latitudes ), min( longitudes ), max( longitudes ))

# d.variables is an ordered dict
var_name, var = d.variables.popitem()
print "%s:\n%s\n" % ( var_name, var.long_name )

# select the point
var = var[:,:,:,lat_i,lon_i]

# median of ensemble values
# var = ( time x hour )
var = np.median( var, axis = 1 )
print "data shape: " + str( var.shape )

np.savetxt( output_file, var, delimiter = ",", fmt = "%.06f" )


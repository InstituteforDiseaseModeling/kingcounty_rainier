""" mapping.py

Functions to facilitate plotting with geopandas. """
import sys

## For data manipulation
import numpy as np
import pandas as pd

## For shapes
import geopandas

## For plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def PlotBorders(fig,axes,gdf,**kwargs):

	""" Plot the borders for all shapes in the shape layer provided. Plotting happens
	    on the axes provided.
	    Note: Key word arguements are passed directly to axes.plot(). """

	## Loop over shapes and plot individually. Multipolygons
	## are handled by looping over individual sections.
	for shape in gdf["geometry"]:

		## Check the type
		geo_type = shape.type
		if geo_type not in {"Polygon","MultiPolygon"}:
			raise NotImplementedError("Plotting for "+geo_type+" is not implemented!")

		## Polygon plotting (polygons have one level of
		## patches).
		if geo_type == "Polygon":
			x,y = shape.exterior.xy
			axes.plot(x,y,**kwargs)

		## Multipolygon plotting (multipolygons have more than
		## one polygon, each with a potential set of patches).
		if geo_type == "MultiPolygon":
			for polygon in shape:
				x,y = polygon.exterior.xy
				axes.plot(x,y,**kwargs)

	return 

def PlotDataOnGDF(fig,axes,series,gdf,
				  shape_name="dot_name",
				  clim=None,colorbar=False,**kwargs):

	""" Plot the data contained in series onto the shape layer (a fiona object).
	    Data is retrieved from the series via
				data = series.loc[shape_name(shape)]
	    where shape_name the name of the column in gdf to be matched with the index of
	    the series.
	    Note: **kwargs are passed directly to PatchCollection(). """

	## Set up storage for polygons and colors
	patches = []
	colors = []
	for name, data in series.iteritems():

		## Get the data associated with the shape
		shapes = gdf.loc[gdf[shape_name] == name,"geometry"].values
		if len(shapes) > 1:
			raise ValueError("Multiple shapes in gdf match with {}".format(name))
		elif len(shapes) == 0:
			raise ValueError("No shapes in gdf match with {}".format(name))
		shape = shapes[0]

		## Create polygons from parts of the shape. The
		## approach depends on shape type.
		geo_type = shape.type
		if geo_type not in {"Polygon","MultiPolygon"}:
			raise NotImplementedError("Plotting for "+geo_type+" is not implemented!")

		## For polygon type
		if geo_type == "Polygon":
			patch = np.array(shape.exterior.xy).T
			p = Polygon(patch)
			patches.append(p)
			colors.append(data)

		## For multipolygon type
		elif geo_type == "MultiPolygon":
			for polygon in shape:
				patch = np.array(polygon.exterior.xy).T
				p = Polygon(patch)
				patches.append(p)
				colors.append(data)

	## Create a patch collection
	collection = PatchCollection(patches,**kwargs)
	collection.set_array(np.array(colors))
	collection.set_clim(clim)

	## Add the collection to the axes
	axes.add_collection(collection)
	if colorbar:
		fig.colorbar(collection,ax=axes,fraction=0.035)

	## Set the limits via matplotlib autoscale
	axes.autoscale_view()

	return fig, axes
#!/usr/bin/env python
# coding: utf-8

import os
from os import listdir
from os.path import isfile, join


import numpy as np

import rasterio
import geopandas as gpd

from shapely.geometry import shape



#%%
#######################
## MANEJO DE RASTERS ##
#######################
# lectura, escritura y propiedades de rasters 
#%%
def guardar_GTiff(fn, crs, transform, mat, meta=None, nodata=None, bandnames=[], verbose = True):
    if len(mat.shape)==2:
        count=1
    else:
        count=mat.shape[0]
    if verbose: print(f'Guardando GeoTIFF de {count} bandas: {os.path.split(fn)[1]}.')


    if not meta:
        meta = {}

    meta['driver'] = 'GTiff'
    meta['height'] = mat.shape[-2]
    meta['width'] = mat.shape[-1]
    meta['count'] = count
    meta['crs'] = crs
    meta['transform'] = transform

    if 'dtype' not in meta: #if no datatype is specified, use float32
        meta['dtype'] = np.float32
    

    if nodata==None:
        pass
    else:
        meta['nodata'] = nodata

    with rasterio.open(fn, 'w', **meta) as dst:
        if count==1: #es una matriz bidimensional, la guardo
            dst.write(mat.astype(meta['dtype']), 1)
            if bandnames:
                dst.set_band_description(1, bandnames[0])
        else: #es una matriz tridimensional, guardo cada banda
            for b in range(count):
                dst.write(mat[b].astype(meta['dtype']), b+1)
            for b,bandname in enumerate(bandnames):
                dst.set_band_description(b+1, bandname)#   
                
#%%             
# Compute Minimum Bounding Box
def compute_mbb(fn, snap_to_grid = True, grid_step = 10):
    """dado archivo vectorial (shp, geojson, etc) 
    calcula el mínimo rectángulo que contenga 
    su primer objeto, usando vértices 
    en una grilla de paso dado."""
    

    gdf = gpd.read_file(fn)
    first_geom = gdf.iloc[0]['geometry'] #miro solo la primer geometría del archivo

    mX, mY, MX, MY = first_geom.bounds
    if snap_to_grid:
        mX = grid_step*(np.floor(mX/grid_step))
        MX = grid_step*(np.ceil(MX/grid_step))
        mY = grid_step*(np.floor(mY/grid_step))
        MY = grid_step*(np.ceil(MY/grid_step))

    mbb = shape({'type': 'Polygon',
          'coordinates': [((mX, MY),
                           (MX, MY),
                           (MX, mY),
                           (mX, mY),
                           (mX, MY))]})
    return [mbb]
#%%
# def guardar_GTiff_as_src(fn, src_fn, mat, verbose=True):
#     '''guarda un geoTiff copiando la georeferenciación de otro'''
#     with rasterio.open(src_fn) as src:
#         transform = src.transform
#         crs=src.crs #recuerdo el sistema de referencia para poder grabar
#     if len(mat.shape)==2:
#         count=1
#     else:
#         count=mat.shape[0]
#     if verbose: print(f'Guardando GeoTIFF de {count} bandas.')
#     with rasterio.open(
#     fn,
#     'w',
#     driver='GTiff',
#     height=mat.shape[-2],
#     width=mat.shape[-1],
#     count=count,
#     dtype=np.float32,
#     crs=crs,
#     transform=transform) as dst:
#         if len(mat.shape)==2:
#             dst.write(mat.astype(np.float32), 1)
#         else:
#             for b in range(0,count):
#                 dst.write(mat[b].astype(np.float32), b+1)

    

#%%
def list_dir(dir_in, prefix = '', postfix = '', verbose = True):
    '''lista todos los archivos de dir_in 
    que comienzan con prefix
    y terminan con postfix
    ordenados'''
    
    files = [f for f in listdir(dir_in) if isfile(join(dir_in, f))]
    files_with_prefix = [f for f in files if f.lower().startswith(prefix.lower())]
    files_with_bothfix = [f for f in files_with_prefix if f.lower().endswith(postfix.lower())]
    files_with_bothfix.sort()
    return files_with_bothfix

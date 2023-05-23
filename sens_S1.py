#!/usr/bin/env python
# coding: utf-8
import pysip as ps

import os
from os import listdir
from os.path import isfile, join

import datetime
import re


import numpy as np
import matplotlib.pyplot as plt

import rasterio
import rasterio.mask


from rasterio.plot import show
from rasterio.mask import mask
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

import geopandas as gpd
from shapely.geometry import mapping
from shapely.geometry.polygon import Polygon

   
import affine
from zipfile import ZipFile

#from rasterio.crs import CRS
#from rasterio.enums import Resampling
from rasterio import shutil as rio_shutil
#from rasterio.vrt import WarpedVRT
   

#%%
##########################
## Sensores específicos ##
##########################
# A. Imágenes sueltas
# B. Directorios con series de imágenes

#%%
#A. Imágenes sueltas
#%%
##################
## Sentinel - 1 ##
##################
# Procesar Sentinel1
def extract_Sentinel1_IW_grd_ZIP(zipfilename, mbb=None, verbose = True):
    """dado un zip de una Sentinel 1
    extrae las bandas
    Si se le pasa un polígono mbb en formato GJSON lo usa para recortar 
    la imagen, sino extrae la imagen completa.
    Ojo: el mbb debe usar el CRS de la imagen original.
    
    Devuelve la matriz con los datos extraídos, el crs y 
    la geotransformacion correspodientes"""
    

    ## vsizip bugfix
    os.environ['CPL_ZIP_ENCODING'] = 'UTF-8'

    ## look for 10m resolution bands: 02, 03, 04 and 08
    tileREXP = re.compile(r'.*.tiff$')
    if verbose: print(f'Leyendo ZIP {zipfilename}')

    bands = []
    with ZipFile(zipfilename,'r') as zfile:
        bandfns = [x for x in zfile.namelist() if re.match(tileREXP,x)]
        bandfns.sort()
        for bandfn in bandfns:
            fn = f'/vsizip/{zipfilename}/{bandfn}'
            if verbose: print(f'Leyendo {os.path.basename(fn)}.')
            with rasterio.open(fn) as src:
                gcps = src.get_gcps() 
                gt = rasterio.transform.from_gcps(gcps[0])
                crs = gcps[1]
                with WarpedVRT(src, src_crs = crs, src_transform = gt) as vrt:
                    if mbb: #si hay mbb hago un clip
                        array, out_transform = mask(vrt, mbb, crop=True)
                    else: #si no, uso la imagen entera
                        array = vrt.read()
                        out_transform = vrt.transform
            bands.append(np.true_divide(array[0], 10000, dtype=np.float32))
    return np.stack(bands), crs, out_transform
#%%
# Procesar Sentinel1 con crs=32721 (UTM21S)
# Borrar esta función, la siguiente la incluye como caso particular
def extract_Sentinel1_IW_grd_ZIP_UTM(zipfilename, mbb, verbose = True):
    """dado un zip de una Sentinel 1
    extrae las bandas dentro de
    un polígono mbb en formato GJSON (recorta) 
    
    Devuelve la matriz con los datos extraídos, el crs y 
    la geotransformacion correspodientes.
    
    Es rígida en cuanto usa CRS 32721
    devuelve el raster con 10m de resol usando snap=20m"""

    
    geom0_GJSON = mbb[0]
    dst_crs = CRS.from_epsg(32721)
    snap_step = 20
    grid_step = 10
    
    Cx=[c[0] for c in geom0_GJSON['coordinates'][0]]
    Cy=[c[1] for c in geom0_GJSON['coordinates'][0]]
    
    mX = snap_step*(np.floor(min(Cx)/snap_step))
    MX = snap_step*(np.ceil(max(Cx)/snap_step))
    mY = snap_step*(np.floor(min(Cy)/snap_step))
    MY = snap_step*(np.ceil(max(Cy)/snap_step))
    
    #defino un geotransform adecuado
    xres = grid_step
    yres = grid_step
    # Output image dimensions
    dst_height = int((MY-mY)/yres) 
    dst_width  = int((MX-mX)/xres)
    
    # Output image transform
    dst_transform = affine.Affine(xres, 0.0, mX,
                                  0.0, -yres, MY)
    
    vrt_options = {
        'resampling': Resampling.nearest,
        'crs': dst_crs,
        'transform': dst_transform,
        'height': dst_height,
        'width': dst_width,
    }
    
    
    ## vsizip bugfix
    os.environ['CPL_ZIP_ENCODING'] = 'UTF-8'

    ## look for 10m resolution bands: 02, 03, 04 and 08
    tileREXP = re.compile(r'.*.tiff$')
    if verbose: print(f'Leyendo ZIP {zipfilename}')

    bands = []
    with ZipFile(zipfilename,'r') as zfile:
        bandfns = [x for x in zfile.namelist() if re.match(tileREXP,x)]
        bandfns.sort()
        for bandfn in bandfns:
            fn = f'/vsizip/{zipfilename}/{bandfn}'
            if verbose: print(f'Leyendo {os.path.basename(fn)}.')
            with rasterio.open(fn) as src:
                #primero le pongo el CRS correcto
                gcps = src.get_gcps() 
                gt = rasterio.transform.from_gcps(gcps[0])
                crs = gcps[1]
                with WarpedVRT(src, src_crs = crs, src_transform = gt) as vrt_src:
                    #ahora lo cambio al nuevo
                    with WarpedVRT(vrt_src, **vrt_options) as vrt:
                        array = vrt.read()
                        out_transform = vrt.transform
            bands.append(np.true_divide(array[0], 10000, dtype=np.float32))
    return np.stack(bands), dst_crs, out_transform
#%%
# Procesar Sentinel1 con crs dado por el vectorial
def extract_Sentinel1_IW_grd_ZIP_CRS(zipfilename, ROI_fn, grid_step = 10, snap_step = 20, verbose = True):
    """dado un zip de una Sentinel 1
    extrae las bandas dentro de una zona dada en archivo vectorial.
    
    La salida está en el CRS del vectorial.
    
    Devuelve la matriz con los datos extraídos, el crs y 
    la geotransformacion correspodientes.

    La resolución espacial (10m) y snap (20m) se pueden especificar."""
    
   
    # Leo el vectorial (SHP o GeoJSON, etc)
    shapefile = gpd.read_file(ROI_fn)
    # Destination CRS from shapefile
    dst_crs = CRS.from_epsg(shapefile.crs.to_epsg())

    # Calculo mbb de su primer polígono
    geoms_sh = shapefile.geometry.values
    while type(geoms_sh)!=Polygon: #miro solo el primer polígono del archivo
        geoms_sh = geoms_sh[0]
    
    geom0_GJSON = mapping(geoms_sh)
    
    Cx=[c[0] for c in geom0_GJSON['coordinates'][0]]
    Cy=[c[1] for c in geom0_GJSON['coordinates'][0]]
    
    mX = snap_step*(np.floor(min(Cx)/snap_step))
    MX = snap_step*(np.ceil(max(Cx)/snap_step))
    mY = snap_step*(np.floor(min(Cy)/snap_step))
    MY = snap_step*(np.ceil(max(Cy)/snap_step))
    
    #defino un geotransform adecuado
    xres = grid_step
    yres = grid_step
    # Output image dimensions
    dst_height = int((MY-mY)/yres) 
    dst_width  = int((MX-mX)/xres)
    
    # Output image transform
    dst_transform = affine.Affine(xres, 0.0, mX,
                                  0.0, -yres, MY)
    
    vrt_options = {
        'resampling': Resampling.nearest,
        'crs': dst_crs,
        'transform': dst_transform,
        'height': dst_height,
        'width': dst_width,
    }
    
    
    ## vsizip bugfix
    os.environ['CPL_ZIP_ENCODING'] = 'UTF-8'

    ## look for 10m resolution bands: 02, 03, 04 and 08
    tileREXP = re.compile(r'.*.tiff$')
    if verbose: print(f'Leyendo ZIP {zipfilename}')

    bands = []
    with ZipFile(zipfilename,'r') as zfile:
        bandfns = [x for x in zfile.namelist() if re.match(tileREXP,x)]
        bandfns.sort()
        for bandfn in bandfns:
            fn = f'/vsizip/{zipfilename}/{bandfn}'
            if verbose: print(f'Leyendo {os.path.basename(fn)}.')
            with rasterio.open(fn) as src:
                #primero le pongo el CRS correcto
                gcps = src.get_gcps() 
                gt = rasterio.transform.from_gcps(gcps[0])
                crs = gcps[1]
                with WarpedVRT(src, src_crs = crs, src_transform = gt) as vrt_src:
                    #ahora lo cambio al nuevo
                    with WarpedVRT(vrt_src, **vrt_options) as vrt:
                        array = vrt.read()
                        out_transform = vrt.transform
            bands.append(np.true_divide(array[0], 10000, dtype=np.float32))
    return np.stack(bands), dst_crs, out_transform
#%%
def extract_and_reproject(in_fn, in_shp,out_fn, grid_step = 10):
    """Esto es algo de SENTINEL 1
    dado un poligono en un shp, y un GeoTIFF
    calcula el mínimo rectángulo que lo contenga
    con vértices en una grilla de paso dado"""

  
    #in_fn = 'S1/S1_1.tif'
    #out_fn = 'S1/S1_1_UTM.tif'
    #in_shp = 'shapes/zona_dique_UTM21S.shp'
    
    
    #leo el SHP 
    shapefile = gpd.read_file(in_shp)
    # Destination CRS from shapefile
    dst_crs = CRS.from_epsg(shapefile.crs.to_epsg())
    #calculo mbb de su primer polígono
    #10m como las Sentinel 2
    geoms_sh = shapefile.geometry.values
    while type(geoms_sh)!=Polygon: #miro solo el primer polígono del archivo
        geoms_sh = geoms_sh[0]
        
    
    geom0_GJSON = mapping(geoms_sh)
    
    Cx=[c[0] for c in geom0_GJSON['coordinates'][0]]
    Cy=[c[1] for c in geom0_GJSON['coordinates'][0]]
    
    mX = grid_step*(np.floor(min(Cx)/grid_step))
    MX = grid_step*(np.ceil(max(Cx)/grid_step))
    mY = grid_step*(np.floor(min(Cy)/grid_step))
    MY = grid_step*(np.ceil(max(Cy)/grid_step))
    
    #defino un geotransform adecuado
    xres = grid_step
    yres = grid_step
    # Output image dimensions
    dst_height = int((MY-mY)/yres) 
    dst_width  = int((MX-mX)/xres)
    
    # Output image transform
    dst_transform = affine.Affine(xres, 0.0, mX,
                                  0.0, -yres, MY)
    
    vrt_options = {
        'resampling': Resampling.nearest,
        'crs': dst_crs,
        'transform': dst_transform,
        'height': dst_height,
        'width': dst_width,
    }
    
    
    with rasterio.open(in_fn) as src:
        with WarpedVRT(src, **vrt_options) as vrt:
    
            # At this point 'vrt' is a full dataset with dimensions,
            # CRS, and spatial extent matching 'vrt_options'.
    
            # Read all data into memory.
            data = vrt.read()
    
            # Process the dataset in chunks.  Likely not very efficient.
            for _, window in vrt.block_windows():
                data = vrt.read(window=window)
    
            # Dump the aligned data into a new file. A VRT representing
            # this transformation can also be produced by switching
            # to the VRT driver.
            rio_shutil.copy(vrt, out_fn, driver='GTiff')

#%%
##########################
## Procesar Directorios ##
##########################
# Procesar directorios enteros de imágenes


#%%
##################
## Sentinel - 1 ##
##################
# Procesar directorios de Sentinel 1

#%%
def get_fechas_flist_S1(dir_in,verbose = True):
    '''extrae listas ordenada de fechas y archivos de 
    un directorio dir_in con Sentinel-1 zipeadas
    El resultado son dos listas: una con objetos tipo datetime y 
    otra los nombres de archivos correspondientes.'''
    ext ='.zip'
    files = [f for f in listdir(dir_in) if isfile(join(dir_in, f))]
    files_with_ext = [f for f in files if f.lower().endswith(ext.lower())]
    sfechas = [(re.search('_20\d\d\d\d\d\dT\d\d\d\d\d\d',fn).group(),fn) for fn in files_with_ext]
    sfechas.sort()
    fechas = [datetime.datetime(int(f[1:5]),   int(f[5:7]),   int(f[7:9]), 
                                int(f[10:12]), int(f[12:14]), int(f[14:16]))
                                for (f,fn) in sfechas]
    flist = [fn for (f,fn) in sfechas]
    
    if verbose: print(f'Recolecté {len(fechas)} filenames con fechas entre {fechas[0]} y {fechas[-1]}.')
    return fechas, flist
#%%
def clip_all_S1_from_dir(dir_in, dir_out, zona_vec_fn, compute_PI = False, 
                                verbose = True):
    '''para cada S2 en dir_in arma un clip con las bandas correspondientes 
    y lo guarda en dir_out. Blue_Green_Red_Nir_Swir_ndVi_mndWi'''
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    #busco los archivos en el directorio dir_in
    fechas, flist = get_fechas_flist_S1(dir_in,verbose = verbose)
    postfix = f'DC{"I" if compute_PI else ""}'
    for fecha,fn in zip(fechas,flist):
        if verbose: print('-'*25)
        S1_img, crs, out_transform = extract_Sentinel1_IW_grd_ZIP_CRS(os.path.join(dir_in,fn), zona_vec_fn, 
                                            verbose = verbose)
        out_fn = os.path.join(dir_out,
            f'S1_{fecha.year:4d}{fecha.month:02d}{fecha.day:02d}_{postfix}.tif')
        if compute_PI:
            S1_img = np.stack([S1_img[0],S1_img[1],S1_img[1]/S1_img[0]])
        ps.raster.guardar_GTiff(out_fn, crs, out_transform, S1_img, verbose=verbose)
    return fechas


#%%
def clip_all_S1_from_dir_origCRS(dir_in, dir_out, zona_vec_fn, compute_PI = False, 
                                verbose = True):
    '''para cada S2 en dir_in arma un clip con las bandas correspondientes 
    y lo guarda en dir_out. Blue_Green_Red_Nir_Swir_ndVi_mndWi'''
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    mbb_zona = ps.raster.compute_mbb(zona_vec_fn, snap_to_grid = False)
    #busco los archivos en el directorio dir_in
    fechas, flist = get_fechas_flist_S1(dir_in,verbose = verbose)
    postfix = f'DC{"I" if compute_PI else ""}'
    for fecha,fn in zip(fechas,flist):
        if verbose: print('-'*25)
        S1_img, crs, out_transform = extract_Sentinel1_IW_grd_ZIP(os.path.join(dir_in,fn), 
                                                                  mbb=mbb_zona, verbose = verbose)
        
        out_fn = os.path.join(dir_out,
            f'S1_{fecha.year:4d}{fecha.month:02d}{fecha.day:02d}_{postfix}.tif')
        if compute_PI:
            S1_img = np.stack([S1_img[0],S1_img[1],S1_img[1]/S1_img[0]])
        ps.raster.guardar_GTiff(out_fn, crs, out_transform, S1_img, verbose=verbose)
    return fechas

#%%
def create_png_from_all_S1_from_dir_origCRS(dir_in, dir_out, zona_vec_fn, 
                                verbose = True):
    '''para cada S2 en dir_in arma un clip con las bandas correspondientes 
    y lo guarda en dir_out. Blue_Green_Red_Nir_Swir_ndVi_mndWi'''
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    mbb_zona = ps.raster.compute_mbb(zona_vec_fn, snap_to_grid = False)
    #busco los archivos en el directorio dir_in
    fechas, flist = get_fechas_flist_S1(dir_in,verbose = verbose)
    postfix = 'DCI'
    for fecha,fn in zip(fechas,flist):
        if verbose: print('-'*25)
        S1_img, crs, out_transform = extract_Sentinel1_IW_grd_ZIP(os.path.join(dir_in,fn), 
                                                                  mbb=mbb_zona, verbose = verbose)
        
        out_fn = os.path.join(dir_out,
            f'S1_{fecha.year:4d}{fecha.month:02d}{fecha.day:02d}_{postfix}.tif')
        S1_img = np.stack([S1_img[0],S1_img[1],S1_img[1]/S1_img[0]])
        out_fn = os.path.join(dir_out,
            f'S1_{fecha.year:4d}{fecha.month:02d}{fecha.day:02d}_{postfix}.png')
        img_str = f'S1 {fecha.year:4d}{fecha.month:02d}{fecha.day:02d} VV-VH-IP'
        ps.visual.save_rgb(S1_img,[0,1,2], out_fn, p=5, title=img_str)
        #plt.savefig(out_fn, bbox_inches = 'tight',dpi=300)
        #plt.close()
    return fechas


#%%
def create_png_from_all_S1_ZIP_from_dir(dir_in, dir_out, mbb=None, 
                                verbose = True):
    '''ESTO NO ANDA AUN
    para cada S2 en dir_in arma un clip espacial con las tres bandas seleccioandas
    y lo guarda como PNG en dir_out. 
    Las bandas RBG de la salida son
    1) Red
    2) Green o ndVi
    3) Blue o mndWi.
    El nombre del archivo indica qué se guardó.'''
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    if compute_mndwi:
        compute_band11 = True
    else:
        compute_band11 = False
    #busco los archivos en el directorio dir_in
    fechas, flist = get_fechas_flist_S2(dir_in,verbose = verbose)
    postfix = f'R{"V" if compute_ndvi else "G"}{"W" if compute_mndwi else "B"}'
    for fecha,fn in zip(fechas,flist):
        if verbose: print('-'*25)
        S2_img, crs, out_transform = extract_10m_bands_Sentinel2_ZIP(os.path.join(dir_in,fn), mbb=mbb, 
                                            compute_ndvi = compute_ndvi, 
                                            compute_band11 = compute_band11, compute_mndwi = compute_mndwi, 
                                            verbose = verbose)
        img_str = f'S2_{fecha.year:4d}{fecha.month:02d}{fecha.day:02d}_{postfix}.png'
        out_fn = os.path.join(dir_out, img_str)
        i0 = 2
        i1 = (5 if compute_mndwi else 4) if compute_ndvi else 1
        i2 = (6 if compute_ndvi else 5) if compute_mndwi else 0
        png_img = S2_img[[i0,i1,i2]]
        for i,b in enumerate([i0,i1,i2]): #reescalo
            if b<3: #reescalo entre 0.05 y 0.2 para reflectancia
                png_img[i]-=0.05 
                png_img[i]/=0.15
            else: #reescalo entre -0.5 y 0.8 para índices
                png_img[i]+=0.5 
                png_img[i]/=1.3
            png_img[i][png_img[i]>1]=1
            png_img[i][png_img[i]<0]=0
        py=png_img[0].shape[0]/300+1
        px=png_img[0].shape[1]/300+1
        fig, ax = plt.subplots(1, 1, figsize = (px,py))
        show(png_img,ax=ax, interpolation='none', title=img_str, transform=out_transform)
        plt.savefig(out_fn, bbox_inches = 'tight',dpi=300)
        
    return fechas


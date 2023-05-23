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
from zipfile import ZipFile
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
## Sentinel - 2 ##
##################

# Procesar Sentinel2
def extract_10m_bands_Sentinel2(img_data_dir, mbb=None, compute_ndvi = True, verbose = True):
    """dado un directorio con las bandas de una Sentinel 2
    extrae las 4 bandas de 10m de resolucion (2, 3, 4 y 8) y computa el NDVI.
    Si se le pasa un polígono mbb en formato GJSON lo usa para recortar 
    la imagen, sino extrae la imagen completa.
    
    Devuelve la matriz con los datos extraídos, el crs y 
    la geotransformacion correspodientes"""
    
    ls = os.listdir(img_data_dir)
    band_names = ['B02.','B03.', 'B04.', 'B08.'] 
    bands = []
    for b in band_names:
        try:
            fn = [fn for fn in ls if b in fn][0]
        except:
            print(f"Banda {b} no encontrada en {img_data_dir}.")
        if verbose: print(f"Leyendo {fn}.")
        
        fn = os.path.join(img_data_dir,fn)
        with rasterio.open(fn) as src:
            crs=src.crs #recuerdo el sistema de referencia para poder grabar
            if mbb: #si hay mbb hago un clip
                array, out_transform = mask(src, mbb, crop=True)
            else: #si no, uso la imagen entera
                fn
                out_transform = src.transform
        bands.append(np.true_divide(array[0], 10000, dtype=np.float32))
    if compute_ndvi:
        if verbose: print('Computando NDVI.')
        bands.append((bands[3]-bands[2])/(bands[3]+bands[2]))
    bands[0], bands[2] = bands[2], bands[0] # para tener RGB y no BGR

    return np.stack(bands), crs, out_transform
    

#%%
def dup(array):
    '''duplica la resolución de un array.
    Lo uso para pasar la banda 11 de 20m a 10m de resolución.'''
    from scipy import ndimage
    return ndimage.zoom(array,2,order=0)

def dip(d):
    '''divide al medio la resolución de un array.
    Lo uso para pasar una de 10m a 20m de resolución.'''

    s = np.zeros([i//2 for i in d.shape], dtype = type(d[0,0]))
    s += d[0::2,0::2]
    s += d[0::2,1::2]
    s += d[1::2,0::2] 
    s += d[1::2,1::2] 
    return s/4

# Procesar Sentinel2 ZIP
def extract_10m_bands_Sentinel2_ZIP(zipfilename, mbb=None, compute_ndvi = True, 
                                    compute_band11 = False, compute_mndwi = False, 
                                    verbose = True):
    """dado un zip de una Sentinel 2
    extrae las 4 bandas de 10m de resolucion (2, 3, 4 y 8) y computa el NDVI.
    Si se le pasa un polígono mbb en formato GJSON lo usa para recortar 
    la imagen, sino extrae la imagen completa.
    
    Devuelve la matriz con los datos extraídos, el crs y 
    la geotransformacion correspodientes.
    
    Si se le pide, calcula el NDVI.
    Si se le pide MNDWI, computa banda 11, le duplica resolucion y calcula MNDWI también."""
    

    ## vsizip bugfix
    os.environ['CPL_ZIP_ENCODING'] = 'UTF-8'

    ## look for 10m resolution bands: 02, 03, 04 and 08
    B02REXP = re.compile(r'.*_B02.*.jp2$')
    tileREXP = re.compile(r'(.*_B(02|03|04|08).jp2$|.*_B(02|03|04|08)_10m.jp2$)')
    if compute_mndwi:
        compute_band11 = True
    if compute_band11:
        tileREXP = re.compile(r'(.*_B(02|03|04|08|11).jp2$|.*_B(02|03|04|08)_10m.jp2$|.*_B11_20m.jp2$)')
        B11REXP = re.compile(r'.*_(B11|B11_20m).jp2$')
        
    if verbose: print(f'Leyendo ZIP {zipfilename}')

    bands = []
    with ZipFile(zipfilename,'r') as zfile:
        bandfns = [x for x in zfile.namelist() if (re.match(tileREXP,x) and "MSK" not in x)]
        bandfns.sort()
        bandfns[0], bandfns[2] = bandfns[2], bandfns[0] #BGR -> RGB
        for bandfn in bandfns:
            fn = f'/vsizip/{zipfilename}/{bandfn}'
            if verbose: print(f'Leyendo {os.path.basename(fn)}.')
            with rasterio.open(fn) as src:
                crs=src.crs #recuerdo el sistema de referencia para poder grabar
                if mbb: #si hay mbb hago un clip
                    array, tmp_transform = mask(src, mbb, crop=True)
                else: #si no, uso la imagen entera
                    array = src.read()
                    tmp_transform = src.transform
                if compute_band11 and re.match(B11REXP, bandfn): #si es la banda 11
                    BSWIR=array[0]/10000 #lo guardo para calcular el mndwi en baja resol
                    array = [dup(BSWIR)]
            if re.match(B02REXP, bandfn): #si es la banda 02
                out_transform= tmp_transform #guardo la geotransformación
            bands.append(np.true_divide(array[0], 10000, dtype=np.float32))


    if compute_ndvi:
        if verbose: print('Computando NDVI.')
        bands.append((bands[3]-bands[0])/(bands[3]+bands[0]))
    if compute_mndwi:
        if verbose: print('Computando MNDWI.')
        BG=dip(bands[1]) #bajo la resolucion al Green
        bands.append(dup((BG-BSWIR)/(BG+BSWIR)))
    return np.stack(bands), crs, out_transform

#%%

# Procesar Sentinel2 ZIP
def extract_10m_bands_Sentinel2_ZIP_int(zipfilename, mbb=None, compute_ndvi = True, 
                                    compute_band11 = False, compute_mndwi = False, 
                                    verbose = True):
    """dado un zip de una Sentinel 2
    extrae las 4 bandas de 10m de resolucion (2, 3, 4 y 8) y computa el NDVI.
    Si se le pasa un polígono mbb en formato GJSON lo usa para recortar 
    la imagen, sino extrae la imagen completa.
    
    Devuelve la matriz con los datos extraídos, el crs y 
    la geotransformacion correspodientes.
    
    Si se le pide, calcula el NDVI.
    Si se le pide MNDWI, computa banda 11, le duplica resolucion y calcula MNDWI también."""
    

    ## vsizip bugfix
    os.environ['CPL_ZIP_ENCODING'] = 'UTF-8'

    ## look for 10m resolution bands: 02, 03, 04 and 08
    B02REXP = re.compile(r'.*_B02.*.jp2$')
    tileREXP = re.compile(r'(.*_B(02|03|04|08).jp2$|.*_B(02|03|04|08)_10m.jp2$)')
    if compute_mndwi:
        compute_band11 = True
    if compute_band11:
        tileREXP = re.compile(r'(.*_B(02|03|04|08|11)*.jp2$|.*_B(02|03|04|08)_10m.jp2$|.*_B11_20m.jp2$)')
        #B11REXP = re.compile(r'.*_B11*.jp2$')
        
    if verbose: print(f'Leyendo ZIP {zipfilename}')

    bands = []
    with ZipFile(zipfilename,'r') as zfile:
        bandfns = [x for x in zfile.namelist() if re.match(tileREXP,x)]
        bandfns.sort()
        for bandfn in bandfns:
            fn = f'/vsizip/{zipfilename}/{bandfn}'
            if verbose: print(f'Leyendo {os.path.basename(fn)}.')
            with rasterio.open(fn) as src:
                crs=src.crs #recuerdo el sistema de referencia para poder grabar
                if mbb: #si hay mbb hago un clip
                    array, tmp_transform = mask(src, mbb, crop=True)
                else: #si no, uso la imagen entera
                    array = src.read()
                    tmp_transform = src.transform
                #if compute_band11 and re.match(B11REXP, bandfn): #si es la banda 11
                if compute_band11 and bandfn.endswith('_B11_20m.jp2'): #si es la banda 11
                    BSWIR=array[0]#/10000 #lo guardo para calcular el mndwi en baja resol
                    array = [dup(BSWIR)]
            if re.match(B02REXP, bandfn): #si es la banda 02
                out_transform= tmp_transform #guardo la geotransformación
            #bands.append(np.true_divide(array[0], 10000, dtype=np.float32))


    if compute_ndvi:
        if verbose: print('Computando NDVI.')
        bands.append(np.array(((bands[3]-bands[2])/(bands[3]+bands[2]))*10000,dtype=np.int16))
    if compute_mndwi:
        if verbose: print('Computando MNDWI.')
        BG=dip(bands[1]) #bajo la resolucion al Green
        bands.append(np.array(bands.append(dup((BG-BSWIR)/(BG+BSWIR)))*10000,dtype=np.int16))
    return np.stack(bands), crs, out_transform

#%%

# Procesar Sentinel2 ZIP
def extract_20m_bands_Sentinel2_ZIP(zipfilename, mbb=None, verbose = True):
    """dado un zip de una Sentinel 2
    extrae las 4 bandas de 10m de resolucion (2, 3, 4 y 8) y computa el NDVI.
    Si se le pasa un polígono mbb en formato GJSON lo usa para recortar 
    la imagen, sino extrae la imagen completa.
    
    Devuelve la matriz con los datos extraídos, el crs y 
    la geotransformacion correspodientes.
    
    Si se le pide, calcula el NDVI.
    Si se le pide MNDWI, computa banda 11, le duplica resolucion y calcula MNDWI también."""
    

    ## vsizip bugfix
    os.environ['CPL_ZIP_ENCODING'] = 'UTF-8'

    ## look for 20m resolution bands
    #B02REXP = re.compile(r'.*_B02.*.jp2$')
    tileREXP = re.compile(r'.*_B(02|03|04|05|06|07|8A|11|12)_20m.jp2$')
    # if compute_mndwi:
    #     compute_band11 = True
    # if compute_band11:
    #     tileREXP = re.compile(r'(.*_B(02|03|04|08|11)*.jp2$|.*_B(02|03|04|08)_10m.jp2$|.*_B11_20m.jp2$)')
    #     B11REXP = re.compile(r'.*_B11*.jp2$')
        
    if verbose: print(f'Leyendo ZIP {zipfilename}')

    bands = []
    with ZipFile(zipfilename,'r') as zfile:
        bandfns = [x for x in zfile.namelist() if re.match(tileREXP,x)]
        bandfns.sort()
        for bandfn in bandfns:
            fn = f'/vsizip/{zipfilename}/{bandfn}'
            if verbose: print(f'Leyendo {os.path.basename(fn)}.')
            with rasterio.open(fn) as src:
                crs=src.crs #recuerdo el sistema de referencia para poder grabar
                if mbb: #si hay mbb hago un clip
                    array, tmp_transform = mask(src, mbb, crop=True)
                else: #si no, uso la imagen entera
                    array = src.read()
                    tmp_transform = src.transform
                #if compute_band11 and re.match(B11REXP, bandfn): #si es la banda 11
            #     if compute_band11 and bandfn.endswith('_B11_20m.jp2'): #si es la banda 11
            #         BSWIR=array[0]/10000 #lo guardo para calcular el mndwi en baja resol
            #         array = [dup(BSWIR)]
            #if re.match(B02REXP, bandfn): #si es la banda 02
            #     out_transform= tmp_transform #guardo la geotransformación
            #bands.append(np.true_divide(array[0], 10000, dtype=np.float32))
            bands.append(array[0])

    out_transform= tmp_transform #guardo la geotransformación
    # if compute_ndvi:
    #     if verbose: print('Computando NDVI.')
    #     bands.append((bands[3]-bands[2])/(bands[3]+bands[2]))
    # if compute_mndwi:
    #     if verbose: print('Computando MNDWI.')
    #     BG=dip(bands[1]) #bajo la resolucion al Green
    #     bands.append(dup((BG-BSWIR)/(BG+BSWIR)))
    
    errdic = np.geterr()
    np.seterr(divide='ignore')
    np.seterr(invalid='ignore')

    if verbose: print('Computando NDVI.')
    bands.append((bands[8]-bands[2])/(bands[8]+bands[2])) #(NIR-RED)/(NIR+RED)
    bands[0], bands[2] = bands[2], bands[0] # para tener RGB y no BGR
    if verbose: print('Computando NDWI1.')
    bands.append((bands[1]-bands[6])/(bands[1]+bands[6])) #(Green and SWIR1)
    if verbose: print('Computando NDWI2.')
    bands.append((bands[1]-bands[7])/(bands[1]+bands[7])) #(Green and SWIR2)

    np.seterr(divide = errdic['divide'])
    np.seterr(invalid= errdic['invalid'])
    
    return np.stack(bands), crs, out_transform

#%%

#%%
##########################
## Procesar Directorios ##
##########################
# Procesar directorios enteros de imágenes

def stack_dir(dir_in, fn_out = None, ext = '', verbose = True):
    '''crea un stack con todas las imágenes (de una banda) 
    en un directorio. Las ordena alfabéticamente por nombre de archivo.
    Si se le pasa un nombre de archivo fn_out graba el stack ahí.'''
    files = [f for f in listdir(dir_in) if isfile(join(dir_in, f))]
    files_with_ext = [f for f in files if f.lower().endswith(ext.lower())]
    files_with_ext.sort()
    list_of_arrays = []
    for fn in files_with_ext:
        if verbose: print(fn)
        with rasterio.open(join(dir_in,fn)) as src:
            transform = src.transform
            crs=src.crs #recuerdo el sistema de referencia para poder grabar
            array = src.read()
        list_of_arrays.append(array)
    stack = np.vstack(list_of_arrays)
    if verbose: print(f'Recolecté {len(list_of_arrays)} rasters.')
    if fn_out:
        if verbose: print(f'Guardando el stack en {fn}.')
        ps.raster.guardar_GTiff(fn_out, crs, transform, stack, verbose=verbose)
    
    return stack


#%%
##################
## Sentinel - 2 ##
##################
# Procesar directorios de Sentinel2

#%%
def get_fechas_S2(dir_in,verbose = True):
    '''extrae la lista ordenada de fechas de los archivos Sentinel-2
    de en una carpeta dada por dir_in.
    El resultado es una lista de objetos tipo datetime.'''
    ext ='.zip'
    files = [f for f in listdir(dir_in) if isfile(join(dir_in, f))]
    files_with_ext = [f for f in files if f.lower().endswith(ext.lower())]
    sfechas = [re.search('_20\d\d\d\d\d\dT\d\d\d\d\d\d',f).group() for f in files_with_ext]
    fechas = [datetime.datetime(int(f[1:5]),   int(f[5:7]),   int(f[7:9]), 
                                int(f[10:12]), int(f[12:14]), int(f[14:16]))
                                for f in sfechas]
    fechas.sort()
    if verbose: print(f'Recolecté {len(fechas)} fechas entre {fechas[0]} y {fechas[-1]}.')
    return fechas

#%%
def get_fechas_flist_S2(dir_in,verbose = True):
    '''extrae listas ordenada de fechas y archivos de 
    un directorio dir_in con Sentinel-2 zipeadas
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
def stack_S2_10m_bands_from_dir(dir_in, mbb=None, compute_ndvi = False, 
                                compute_band11 = False, compute_mndwi = False, 
                                verbose = True):
    '''crea un stack con todas las sentinel-2 de un directorio dado'''
    fechas, flist = get_fechas_flist_S2(dir_in,verbose = verbose)
    stack = []
    for fn in flist:
        S2_4bands, crs, out_transform = extract_10m_bands_Sentinel2_ZIP(os.path.join(dir_in,fn), mbb=mbb, 
                                            compute_ndvi = compute_ndvi, 
                                            compute_band11 = compute_band11, compute_mndwi = compute_mndwi, 
                                            verbose = verbose)
        stack.append(S2_4bands)
    return stack, crs, out_transform, fechas
#%%
def clip_all_S2_10m_bands_from_dir(dir_in, dir_out, mbb=None, compute_ndvi = False, 
                                compute_band11 = False, compute_mndwi = False, 
                                verbose = True):
    '''para cada S2 en dir_in arma un clip con las bandas correspondientes 
    y lo guarda en dir_out. Blue_Green_Red_Nir_Swir_ndVi_mndWi'''
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    if compute_mndwi:
        compute_band11 = True
    #busco los archivos en el directorio dir_in
    fechas, flist = get_fechas_flist_S2(dir_in,verbose = verbose)
    postfix = f'RGBN{"S" if compute_band11 else ""}{"V" if compute_ndvi else ""}{"W" if compute_mndwi else ""}'
    bandnames = ['Red','Green','Blue','NIR']
    if compute_band11: bandnames.append('SWIR11')
    if compute_ndvi: bandnames.append('NDVI')
    if compute_mndwi: bandnames.append('MNDWI')
    
    for fecha,fn in zip(fechas,flist):
        if verbose: print('-'*25)
        S2_img, crs, out_transform = extract_10m_bands_Sentinel2_ZIP(os.path.join(dir_in,fn), mbb=mbb, 
                                            compute_ndvi = compute_ndvi, 
                                            compute_band11 = compute_band11, compute_mndwi = compute_mndwi, 
                                            verbose = verbose)
        out_fn = os.path.join(dir_out,
            f'S2_{fecha.year:4d}{fecha.month:02d}{fecha.day:02d}_{postfix}.tif')
        
        ps.raster.guardar_GTiff(out_fn, crs, out_transform, S2_img, verbose=verbose, bandnames = bandnames)
    return fechas

#%%
def create_png_from_all_S2_from_dir(dir_in, dir_out, mbb=None, compute_ndvi = False, 
                                compute_mndwi = False, 
                                verbose = True):
    '''para cada S2 en dir_in arma un clip espacial con las tres bandas seleccioandas
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


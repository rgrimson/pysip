#!/usr/bin/env python
# coding: utf-8
#%%

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

#%%
##########################
## Sensores específicos ##
##########################
# A. Imágenes sueltas
# B. Directorios con series de imágenes


import tarfile

def extract_bands_Landsat_TAR(tar_fn, dir_temp, mbb=None, compute_ndvi = True, compute_ndwi = True, 
                                compute_mndwi1 = True, compute_mndwi2 = True, verbose = True):
    def Landsat_SR_Bands_from_tar(tar_open_file,bands_dict):
        bs = '|'.join([str(b) for b in bands_dict.values()])
        tileREXP = re.compile(rf'(.*SR_B({bs}).TIF$)')
        for tarinfo in tar_open_file:
            name = tarinfo.name.upper()
            #print(name,end=":\t\t")
            if re.match(tileREXP,name): 
                yield tarinfo
    #%

    # Descomprimir 
    if verbose: print(f'Leyendo {os.path.split(tar_fn)[1]}', end='. ')
    cod = os.path.split(tar_fn)[1][:4]
    if cod == 'LC08':
        sensor = 'L08'
        bands_dict = {'B':2, 'G':3, 'R':4, 'NIR':5, 'SWIR1':6, 'SWIR2':7}
    elif cod == 'LT05':
        sensor = 'L05'
        bands_dict = {'B':1, 'G':2, 'R':3, 'NIR':4, 'SWIR1':5, 'SWIR2':7}
    else:
        raise TypeError(f"No reconozco el sensor: {cod}.")
    if verbose: print(sensor)
        

        
    
    tar_open_file = tarfile.open(tar_fn)
    SR_Bands = list(Landsat_SR_Bands_from_tar(tar_open_file,bands_dict))
    tar_open_file.extractall(members=SR_Bands, path=dir_temp)
    tar_open_file.close()
    SR_Bands_fns = [os.path.join(dir_temp, member.name) for member in SR_Bands]
    SR_Bands_fns.sort()
    
    #%
    # Hacer algo
    bands = []
    
    for fn in SR_Bands_fns:
        if verbose: print(f'Leyendo {os.path.basename(fn)}.')
        with rasterio.open(fn) as src:
            crs=src.crs #recuerdo el sistema de referencia para poder grabar
            if mbb: #si hay mbb hago un clip
                array, out_transform = mask(src, mbb, crop=True)
            else: #si no, uso la imagen entera
                array = src.read()
                out_transform = src.transform
        #bands.append(np.true_divide(array[0], 10000, dtype=np.float32))
        bands.append(np.multiply(array[0], 0.0000275, dtype=np.float32)-0.2)
        	
    
    errdic = np.geterr()
    np.seterr(divide='ignore')
    np.seterr(invalid='ignore')
    
    #blue = 0
    green = 1
    red = 2
    nir = 3
    swir1 = 4
    swir2 = 5
    if compute_ndvi:
        if verbose: print('Computando NDVI.')
        bands.append((bands[nir]-bands[red])/(bands[nir]+bands[red]))
    if compute_ndwi:
        if verbose: print('Computando NDWI.')
        bands.append((bands[green]-bands[nir])/(bands[green]+bands[nir]))
    if compute_mndwi1:
        if verbose: print('Computando MNDWI1.')
        bands.append((bands[green]-bands[swir1])/(bands[green]+bands[swir1]))
    if compute_mndwi2:
        if verbose: print('Computando MNDWI2.')
        bands.append((bands[green]-bands[swir2])/(bands[green]+bands[swir2]))

    np.seterr(divide = errdic['divide'])
    np.seterr(invalid= errdic['invalid'])
    #%
    # Borrar
    for file in SR_Bands_fns:
        os.remove(file)
        #print("%s has been removed successfully" %file)
    bands[0], bands[2] = bands[2], bands[0] # para tener RGB y no BGR
    return np.stack(bands), crs, out_transform

#%%
def get_fechas_and_flist_Landsat(dir_in,verbose = True):
    '''extrae listas ordenada de fechas y archivos de 
    un directorio dir_in con Landsar comprimidas (TAR)
    El resultado son dos listas: una con objetos tipo datetime y 
    otra los nombres de archivos correspondientes.'''
    ext ='.tar'
    files = [f for f in listdir(dir_in) if isfile(join(dir_in, f))]
    files_with_ext = [f for f in files if f.lower().endswith(ext.lower())]
    sfechas = [(re.search('_\d\d\d\d\d\d\d\d_20',fn).group(),fn) for fn in files_with_ext]
    sfechas.sort()
    fechas = [datetime.date(int(f[1:5]),   int(f[5:7]),   int(f[7:9])) for (f,fn) in sfechas]
    flist = [fn for (f,fn) in sfechas]
    
    if verbose and fechas: print(f'Recolecté {len(fechas)} filenames con fechas entre {fechas[0]} y {fechas[-1]}.')
    return fechas, flist

#%%
def clip_all_Landsat_from_dir(dir_in, dir_out, dir_temp, mbb=None, compute_ndvi = True, compute_ndwi = True, 
                                compute_mndwi1 = True, compute_mndwi2 = True, verbose = True):
    '''para cada LT en dir_in arma un clip con las bandas correspondientes 
    y lo guarda en dir_out. Blue_Green_Red_Nir_Swir1_swIr2_ndVi_ndWi_Mndw1i_Mndw2i'''
    if not os.path.exists(dir_out):
        if not os.path.exists(os.path.dirname(dir_out)):
            if not os.path.exists(os.path.dirname(os.path.dirname(dir_out))):
                os.mkdir(os.path.dirname(os.path.dirname(dir_out)))
        if not os.path.exists(os.path.dirname(dir_out)):
            os.mkdir(os.path.dirname(dir_out))
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    #busco los archivos en el directorio dir_in
    fechas, flist = get_fechas_and_flist_Landsat(dir_in,verbose = verbose)
    postfix = f'RGBNSI{"V" if compute_ndvi else ""}{"W" if compute_ndwi else ""}{"M1" if compute_mndwi1 else ""}{"M2" if compute_mndwi2 else ""}'
    bandnames = ['Red','Green','Blue','NIR','SWIR1','SWIR2']
    if compute_ndvi: bandnames.append('NDVI')
    if compute_ndwi: bandnames.append('NDWI')
    if compute_mndwi1: bandnames.append('MNDWI1')
    if compute_mndwi2: bandnames.append('MNDWI2')
    
    for fecha,fn in zip(fechas,flist):
        if verbose>1: print('-'*25)
        try:
            LT_img, crs, out_transform = extract_bands_Landsat_TAR(os.path.join(dir_in,fn), 
                                                dir_temp, mbb=mbb, 
                                                compute_ndvi = compute_ndvi, 
                                                compute_ndwi = compute_ndwi, 
                                                compute_mndwi1 = compute_mndwi1, 
                                                compute_mndwi2 = compute_mndwi2, 
                                                verbose = max(0,verbose-1))
            out_fn = os.path.join(dir_out,
                f'{fn[:4]}_{fecha.year:4d}{fecha.month:02d}{fecha.day:02d}_{postfix}.tif')
            
            ps.raster.guardar_GTiff(out_fn, crs, out_transform, LT_img, verbose=verbose, bandnames = bandnames)
        except Exception as e: print(e)
    return fechas

#%%
def create_png_from_all_Landsat_from_dir(dir_in, dir_out, dir_temp, mbb=None, compute_ndvi = False, compute_ndwi = False, 
                                compute_mndwi1 = False, compute_mndwi2 = False, verbose = True): 
    '''para cada LT en dir_in arma un clip espacial con las tres bandas seleccioandas
    y lo guarda como PNG en dir_out. 
    Las bandas RBG de la salida son
    1) Red
    2) Green o ndVi
    3) Blue o ndWi o mndW1i o mndW2i
    El nombre del archivo indica qué se guardó.'''
    if not os.path.exists(dir_out):
        if not os.path.exists(os.path.dirname(dir_out)):
            if not os.path.exists(os.path.dirname(os.path.dirname(dir_out))):
                os.mkdir(os.path.dirname(os.path.dirname(dir_out)))
        if not os.path.exists(os.path.dirname(dir_out)):
            os.mkdir(os.path.dirname(dir_out))
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    #busco los archivos en el directorio dir_in
    fechas, flist = get_fechas_and_flist_Landsat(dir_in,verbose = verbose)
    postfix = f'R{"V" if compute_ndvi else "G"}{"M2" if compute_mndwi2 else ("M1" if compute_mndwi1 else ("W" if compute_ndwi else "B"))}'
    for fecha,fn in zip(fechas,flist):
        if verbose>1: print('-'*25)
        try:
            LT_img, crs, out_transform = extract_bands_Landsat_TAR(os.path.join(dir_in,fn), 
                                            dir_temp, mbb=mbb, 
                                            compute_ndvi = compute_ndvi, 
                                            compute_ndwi = compute_ndwi, 
                                            compute_mndwi1 = compute_mndwi1, 
                                            compute_mndwi2 = compute_mndwi2, 
                                            verbose = max(0,verbose-1))
        
            img_str = f'{fn[:4]}_{fecha.year:4d}{fecha.month:02d}{fecha.day:02d}_{postfix}.png'
            out_fn = os.path.join(dir_out, img_str)
            i0 = 0
            i1 = 6 if compute_ndvi else 1
            i2 = (7 if compute_ndvi else 6) if compute_mndwi2 else 2
            png_img = LT_img[[i0,i1,i2]]
            for i,b in enumerate([i0,i1,i2]): #reescalo
                if b<3: #reescalo entre 0.00 y 0.2 para reflectancia
                    png_img[i]-=0.00 
                    png_img[i]/=0.20
                else: #reescalo entre -0.5 y 0.8 para índices
                    png_img[i]+=0.5 
                    png_img[i]/=1.3
                png_img[i][png_img[i]>1]=1
                png_img[i][png_img[i]<0]=0
            py=png_img[0].shape[0]/300+1
            px=png_img[0].shape[1]/300+1
            fig, ax = plt.subplots(1, 1, figsize = (px,py))
            show(png_img,ax=ax, interpolation='none', title=img_str, transform=out_transform)
            if verbose: print(f'Guardando PNG: {img_str}')
            plt.savefig(out_fn, bbox_inches = 'tight',dpi=300)        
            plt.close()
        except Exception as e: print(e)
    return fechas

#%%
def compute_threshold_freq_from_series(dir_in,starts_with='', ends_with='.tif', band = 1, threshold=0, verbose = True):
    '''levanta las banda "band" de la serie en "dir_in" 
    que comienza con "starts_with" y termina con "ends_with"
    y calcula la cantidad de veces que supera el umbral threshold.'''

    flist = [s for s in os.listdir(dir_in) if (s.startswith(starts_with) and s.endswith(ends_with))]
    lfl = len(flist)

    #----
    f = flist[0]
    raster_fn = os.path.join(dir_in, f)
    with rasterio.open(raster_fn) as src:
        img = src.read(band)
    freq = np.zeros(img.shape,dtype=np.float32)
    #----
        
    for i,f in enumerate(flist):
        if verbose: print(f'************************************************\nLeyendo {i+1:2d}/{lfl}:{f}')
        raster_fn = os.path.join(dir_in, f)
        with rasterio.open(raster_fn) as src:
            freq += src.read(band)>threshold
    return freq / lfl


#%%
#A. Imágenes sueltas
#%%
##################
## Sentinel - 2 ##
##################

# Procesar Sentinel2
#def extract_10m_bands_Sentinel2(img_data_dir, mbb=None, compute_ndvi = True, verbose = True):
    """dado un directorio con las bandas de una Sentinel 2
    extrae las 4 bandas de 10m de resolucion (2, 3, 4 y 8) y computa el NDVI.
    Si se le pasa un polígono mbb en formato GJSON lo usa para recortar 
    la imagen, sino extrae la imagen completa.
    
    Devuelve la matriz con los datos extraídos, el crs y 
    la geotransformacion correspodientes"""

# Procesar Sentinel2 ZIP
#def extract_10m_bands_Sentinel2_ZIP(zipfilename, mbb=None, compute_ndvi = True, 
#                                    compute_band11 = False, compute_mndwi = False, 
#                                    verbose = True):
    """dado un zip de una Sentinel 2
    extrae las 4 bandas de 10m de resolucion (2, 3, 4 y 8) y computa el NDVI.
    Si se le pasa un polígono mbb en formato GJSON lo usa para recortar 
    la imagen, sino extrae la imagen completa.
    
    Devuelve la matriz con los datos extraídos, el crs y 
    la geotransformacion correspodientes.
    
    Si se le pide, calcula el NDVI.
    Si se le pide MNDWI, computa banda 11, le duplica resolucion y calcula MNDWI también."""
    
#%%
##########################
## Procesar Directorios ##
##########################
# Procesar directorios enteros de imágenes
#%%
##################
## Sentinel - 2 ##
##################
# Procesar directorios de Sentinel2

#%%
#def get_fechas_S2(dir_in,verbose = True):
    '''extrae la lista ordenada de fechas de los archivos Sentinel-2
    de en una carpeta dada por dir_in.
    El resultado es una lista de objetos tipo datetime.'''
#%%
#def get_fechas_flist_S2(dir_in,verbose = True):
    '''extrae listas ordenada de fechas y archivos de 
    un directorio dir_in con Sentinel-2 zipeadas
    El resultado son dos listas: una con objetos tipo datetime y 
    otra los nombres de archivos correspondientes.'''
#%%
#def stack_S2_10m_bands_from_dir(dir_in, mbb=None, compute_ndvi = False, 
                                #compute_band11 = False, compute_mndwi = False, 
                                #verbose = True):
    '''crea un stack con todas las sentinel-2 de un directorio dado'''
#%%
#def clip_all_S2_10m_bands_from_dir(dir_in, dir_out, mbb=None, compute_ndvi = False, 
                                #compute_band11 = False, compute_mndwi = False, 
                                #verbose = True):
    '''para cada S2 en dir_in arma un clip con las bandas correspondientes 
    y lo guarda en dir_out. Blue_Green_Red_Nir_Swir_ndVi_mndWi'''
#%%
#def create_png_from_all_S2_from_dir(dir_in, dir_out, mbb=None, compute_ndvi = False, 
                                #compute_mndwi = False, 
                                #verbose = True):
    '''para cada S2 en dir_in arma un clip espacial con las tres bandas seleccioandas
    y lo guarda como PNG en dir_out. 
    Las bandas RBG de la salida son
    1) Red
    2) Green o ndVi
    3) Blue o mndWi.
    El nombre del archivo indica qué se guardó.'''

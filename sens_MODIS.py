#!/usr/bin/env python
# coding: utf-8

from os import listdir
from os.path import isfile, join

import datetime

#%%
##########################
## Sensores específicos ##
##########################
# A. Imágenes sueltas
# B. Directorios con series de imágenes

#%%
#A. Imágenes sueltas
#%%
###########
## MODIS ##
###########

#%%
##########################
## Procesar Directorios ##
##########################
# Procesar directorios enteros de imágenes

#%%
def get_fechas_MODIS(dir_in,verbose = True):
    '''extrae la lista ordenada de fechas de los archivos MODIS 
    de en una carpeta dada por dir_in.
    El resultado es una lista de objetos tipo datetime.'''
    ext ='tif'
    files = [f for f in listdir(dir_in) if isfile(join(dir_in, f))]
    files_with_ext = [f for f in files if f.lower().endswith(ext.lower())]
    files_with_ext.sort()
    sfechas = [f.split('doy')[1].split('_')[0] for f in files_with_ext]
    
    años = [int(f[:4])for f in sfechas]
    dias = [int(f[4:])for f in sfechas]
    #convierto el dia juliano en fecha
    fechas = [datetime.datetime(y, 1, 1) + datetime.timedelta(d - 1) for (y,d) in zip (años, dias)]
    if verbose: print(f'Recolecté {len(fechas)} fechas entre {fechas[0]} y {fechas[-1]}.')
    return fechas

#%%

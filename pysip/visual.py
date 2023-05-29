#!/usr/bin/env python
# coding: utf-8



import numpy as np
import matplotlib.pyplot as plt
#%%
##############################
## Visualización de Rasters ##
##############################

#%%
# Visualizaciones básicas
def nequalize(array,p=5,nodata=None):
    """
    normalize and equalize a single band image
    """
    
    if len(array.shape)==2:
        vmin=np.percentile(array[array!=nodata],p)
        vmax=np.percentile(array[array!=nodata],100-p)
        eq_array = (array-vmin)/(vmax-vmin)
        eq_array[eq_array>1]=1
        eq_array[eq_array<0]=0
    elif len(array.shape)==3:
        eq_array = np.empty_like(array, dtype=float)
        for i in range(array.shape[0]):
            eq_array[i]=nequalize(array[i], p=p, nodata=nodata)
    return eq_array

#%%
def mostrar_indice(array, nodata=None, p = 5, vmin= None, vmax = None, title = ""):
    '''muestra un raster con un indice y su escala entre vmin y vmax. 
    Si no se están definidos vmin y vmax los computa con el percentil p.
    Se pueden evitar los valores nodata en el cálculo del percentil.'''
    if not vmin:
        vmin = np.percentile(array[array!=nodata],p)
    if not vmax:
        vmax = np.percentile(array[array!=nodata],100-p)    
    plt.imshow(array, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tight_layout()
    plt.title(title)
    plt.show()

#%%
def plot_rgb(array, band_list , p = 0, nodata = None, figsize = (12,6), title = None):
    '''
    Esta función toma como parámetros de entrada la matriz a ser ploteada, 
    una lista de índices correspondientes a las bandas que queremos usar, 
    en el orden que deben estar (ej: [1,2,3]), y un parámetro p que es opcional 
    que es el percentil de equalización.
    
    Por defecto tambien asigna un tamaño de figura en (12,6), que también puede ser modificado.
    
    Devuelve solamente un ploteo, no modifica el arreglo original.
    Nota: array debe ser una matriz con estas dimensiones de entrada: [bandas, filas, columnas]
    '''
    from rasterio.plot import show

    if not title:
        title = f'Combinación {band_list} \n (percentil {p}%)'
        
    img = nequalize(array[band_list], p=p, nodata=nodata)
    plt.figure(figsize = figsize)
    plt.title(title, size = 20)
    show(img)
    #plt.show()   
#%%
def save_rgb(array, band_list , out_fn, p = 0, nodata = None, figsize = (12,6), title = None, verbose = True):
    '''
    Esta función toma como parámetros de entrada la matriz a ser ploteada, 
    una lista de índices correspondientes a las bandas que queremos usar, 
    en el orden que deben estar (ej: [1,2,3]), y un parámetro p que es opcional 
    que es el percentil de equalización.
    
    Por defecto tambien asigna un tamaño de figura en (12,6), que también puede ser modificado.
    
    Devuelve solamente un ploteo, no modifica el arreglo original.
    Nota: array debe ser una matriz con estas dimensiones de entrada: [bandas, filas, columnas]
    '''
    from rasterio.plot import show

    if not title:
        title = f'Combinación {band_list} \n (percentil {p}%)'
        
    img = nequalize(array[band_list], p=p, nodata=nodata)
    plt.figure(figsize = figsize)
    #plt.title(title , size = 20)
    fig, ax = plt.subplots(1, 1, figsize = figsize)
    show(img,ax=ax, interpolation='none', title=title)#, transform=out_transform)
    if verbose: print(f'Guardando PNG: {title}')
    plt.savefig(out_fn, bbox_inches = 'tight',dpi=300)        
    plt.close()

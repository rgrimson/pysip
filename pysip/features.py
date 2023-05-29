#!/usr/bin/env python
# coding: utf-8


import os
from os import listdir
from os.path import isfile, join


import numpy as np

import rasterio
import rasterio.mask


from rasterio.mask import mask

import geopandas as gpd
from shapely.geometry import mapping
#%%
#%%
####################
## ROIs de Raster ##
####################

#READ ROIS
#%%
def read_rois(vec_fname, clase='clase'):
    '''lee los rois definidos en un archivo vectorial a un geopandas (gpd)
    agrega una columna 'class_id' numerando las clases que aparecen.
    Devuelve el gpd y la lista ordenada de nombre de clases para referencia.'''
    rois_gpd = gpd.read_file(vec_fname)
    clases=list(set(rois_gpd[clase]))
    clases.sort() #numero las clases de los ROIs alfabéticamente
    clase_dict = {clase:i for i, clase in enumerate(clases)}
    rois_gpd['class_id'] = [clase_dict[c] for c in rois_gpd[clase].values]
    return rois_gpd, clases
#%%
def read_numeric_rois(vec_fname, clase='clase'):
    '''lee los rois definidos en un archivo vectorial a un geopandas (gpd)
    agrega una columna 'class_id' numerando las clases que aparecen.
    Devuelve el gpd y la lista ordenada de nombre de clases para referencia.'''
    rois_gpd = gpd.read_file(vec_fname)
    rois_gpd['class_id'] = rois_gpd[clase]
    return rois_gpd
#%%
def read_pts(vec_fname, clase='clase', clases = None):
    '''lee los puntos definidos en un archivo vectorial a un geopandas (gpd)
    agrega una columna 'class_id' numerando las clases que aparecen.
    Devuelve el gpd y la lista ordenada de nombre de clases para referencia.
    Se puede forzar un listado de clases, si en un vectorial no aparecen todas las clases de interés.'''
    rois_gpd = gpd.read_file(vec_fname)
    if clases==None:
        clases=list(set(rois_gpd[clase]))
    clases.sort() #numero las clases de los ROIs alfabéticamente
    clase_dict = {clase:i for i, clase in enumerate(clases)}
    rois_gpd['class_id'] = [clase_dict[c] for c in rois_gpd[clase].values]
    return rois_gpd, clases


#%%
#Levanto en X, Y los datos etiquetados de todo un raster
def extract_ROIs_features_from_raster(raster_fn, rois_gpd):
    '''Dado un raster y ROIs en un gpd extrae los 
    valores espectrales de los pixels (X) y sus etiquetas (Y).
    Devuelve X, Y.'''
    #Leo los ROIS
    with rasterio.open(raster_fn) as src:
        d=src.count #cantidad de atributos = cantidad de bandas en el raster
                             
    
    nodata=-255.255 #elijo un valor raro para nodata
    
    #Preparo colección de atributos etiquetados. Comienza con 0 datos
    X = np.zeros([0,d],dtype = np.float32) #array con todos los atributos
    Y = np.zeros([0],dtype=int)            #array con sus etiquetas
    
    
    with rasterio.open(raster_fn) as src:
        for index, row in rois_gpd.iterrows():
            geom_sh = row['geometry']
            clase = row['class_id']
            geom_GJ = [mapping(geom_sh)]
            try:
                clip, _transform = mask(src, geom_GJ, crop=True,nodata=nodata)
                d,x,y = clip.shape
                D = list(clip.reshape([d,x*y]).T)
                D = [p for p in D if (not (p==nodata).prod())]
                DX = np.array(D)        
                DY = np.repeat(clase,len(D))
                if len(D):
                    X = np.concatenate((X,DX))
                    Y = np.concatenate((Y,DY))
            except Exception as e:
                print(f'Error! {e}')
                print(f"    en la fila {index}:\n  {row}\n")
                
    return X, Y
 
#%%
#Levanto en X, Y los datos etiquetados de todo un directorio
def read_ROIs_features_from_dir(dir_features, dir_vec, zona, verbose = True, ext = '.tif'):
    '''extrae los valores de los pixels en los ROIs de 
    todos los archivos en el directorio dir_features.
    Los ROIS deben venir de un geojson en el directorio dir_vec
    y tener el nombre '{zona}_ROIs'.'''
    
    vec_fname = os.path.join(dir_vec,f'{zona}_ROIs.geojson')
    rois_gpd,clases = read_rois(vec_fname)
    

    inicio = True
    #flist = os.listdir(dir_features)
    flist = [f for f in listdir(dir_features) if (
             isfile(join(dir_features, f)) and
             f.lower().endswith(ext.lower()))]
    flist.sort()
    feature_names = []
    for fn in flist:
        raster_fn = os.path.join(dir_features,fn)
        X_, Y_ = extract_ROIs_features_from_raster(raster_fn, rois_gpd)
        if inicio:
            X = X_
            Y = Y_
            inicio = False
        else:
            X = np.concatenate((X,X_),axis=1)
            if ((Y==Y_).all()):
                if verbose: print('.', end='')
            else:
                raise RuntimeError('No me coinciden las etiquetas')
        #feature names
        n_features = X_.shape[1]
        new_features = [fn.split('.')[0][:-n_features]+fn.split('.')[0][i-n_features] for i in range(n_features)]
        feature_names.extend(new_features)
        
                    
    return X, Y, feature_names
#%%
#Levanto en X, Y los datos etiquetados de todo un directorio
def extract_ROIs_features_from_dir(dir_features, vec_fname, verbose = True, ext = '.tif',clase='clase'):
    '''extrae los valores de los pixels en los ROIs de 
    todos los archivos en el directorio dir_features.
    '''

    rois_gpd,clases = read_rois(vec_fname,clase)

    inicio = True
    #flist = os.listdir(dir_features)
    flist = [f for f in listdir(dir_features) if (
             isfile(join(dir_features, f)) and
             f.lower().endswith(ext.lower()))]
    flist.sort()
    feature_names = []
    for fn in flist:
        raster_fn = os.path.join(dir_features,fn)
        X_, Y_ = extract_ROIs_features_from_raster(raster_fn, rois_gpd)
        if inicio:
            X = X_
            Y = Y_
            inicio = False
        else:
            X = np.concatenate((X,X_),axis=1)
            if ((Y==Y_).all()):
                if verbose: print('.', end='')
            else:
                raise RuntimeError('No me coinciden las etiquetas')
        #feature names
        n_features = X_.shape[1]
        new_features = [fn.split('.')[0][:-n_features]+fn.split('.')[0][i-n_features] for i in range(n_features)]
        feature_names.extend(new_features)
        
                    
    return X, Y, feature_names
#%%
from shapely.geometry import Point

def Random_Points_in_Polygon(polygon, number):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < number:
        pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append(pnt)
    return points

def Random_Points_in_MPolygon(mpoly, number):
    points = []
    NP=len(mpoly.geoms)
    marea = mpoly.area
    for i in range(NP):
        poly = mpoly.geoms[i]
        area = poly.area
        n = int(np.ceil(number*area/marea)) #selecciono proporcional al area, redondeando hacia arriba
        points = points + Random_Points_in_Polygon(poly, n)
    
    poi = np.array(points)
    crds = np.random.choice(range(len(points)),number) #descarto aleatoriamente algunos, si seleccioné de más
    pts = list(poi[crds])
    return pts 


#%%#Levanto en X, Y los datos etiquetados de pts al azar (con dist minima) de todo un directorio
def extract_ROIs_point_features_from_dir(dir_features, vec_fname, N=100, verbose = True, ext = '.tif',clase='clase'):
    '''extrae los valores de los pixels en 
    puntos al azar en cada polígono, N por clase, y leyendo
    todos los archivos en el directorio dir_features.
    '''

    #genero los puntos en cada clase
    #genero los multipolígonos por clase
    rois_gpd,clases = read_rois(vec_fname,clase)
    drois = rois_gpd.dissolve(by=clase)
    
    #genero numeros para las clases
    clases=list(drois.index)
    from functools import reduce 
    l = [[i]*N for i in range(len(clases))]
    Y = np.array(reduce(lambda a, b: a+b, l))
    


    #genero los puntos en cada clase
    if verbose: print('Seleccionando puntos en los ROIs.')
    pts = []
    for i in range(len(clases)):
        print(i)
        mpoly = drois.iloc[i]['geometry']
        pts.extend(Random_Points_in_MPolygon(mpoly, N))
    
    
    
    inicio = True
    #flist = os.listdir(dir_features)
    flist = [f for f in listdir(dir_features) if (
             isfile(join(dir_features, f)) and
             f.lower().endswith(ext.lower()))]
    flist.sort()
    if verbose: print(f'Leyendo datos espectrales de las {len(flist):d} imágenes.')
    feature_names = []
    for fn in flist:
        raster_fn = os.path.join(dir_features,fn)
        #X_, Y_ = extract_ROIs_features_from_raster(raster_fn, rois_gpd)
        #open raster file
        raster = rasterio.open(raster_fn)
        img = raster.read()

        #extract point value from raster
        X_ = []#np.empty([0,7],dtype=np.float32)
        for point in pts:
            x = point.xy[0][0]
            y = point.xy[1][0]
            row, col = raster.index(x,y)
            s = img[:,row,col]
            #X_ = np.concatenate((X_,s),axis=1)
            X_.append(s)
    
        X_ = np.array(X_)
    
        if inicio:
            X = X_
            inicio = False
            if verbose: print('.', end='')
        else:
            X = np.concatenate((X,X_),axis=1)
            if verbose: print('.', end='')
        #feature names
        n_features = X_.shape[1]
        new_features = [fn.split('.')[0][:-n_features]+fn.split('.')[0][i-n_features] for i in range(n_features)]
        feature_names.extend(new_features)
        
                    
    return X, Y, feature_names

#%%
#################################
## Levantar features puntuales ##
#################################

#%%#Levanto en X, Y los datos etiquetados de pts dados de un vectorial
def extract_point_features_from_raster(raster_fn, pts_gpd):
    '''Dado un raster y ROIs en un gpd extrae los 
    valores espectrales de los pixels (X) y sus etiquetas (Y).
    Devuelve X, Y.'''
    #Leo los ROIS
    with rasterio.open(raster_fn) as src:
        d=src.count #cantidad de atributos = cantidad de bandas en el raster
        img = src.read()
                             
    
    #Preparo colección de atributos etiquetados. Comienza con 0 datos
    N=len(pts_gpd)
    X = np.zeros([N,d],dtype = np.float32) #array con todos los atributos
    Y = np.zeros([N],dtype=int)            #array con sus etiquetas
    
    
    #with rasterio.open(raster_fn) as src:
    for i, point in pts_gpd.iterrows():
        x = point['geometry'].xy[0][0]
        y = point['geometry'].xy[1][0]
        row, col = src.index(x,y)
        X[i]=img[:,row,col]
        Y[i]=point['class_id']
                
    return X, Y
#%%#Gero puntos en ROIs, N en cada clase
def generate_points_from_ROIs(vec_fname, N=100, verbose = True, clase='clase'):
    '''genera puntos 
    al azar en cada polígono, N por clase.
    '''

    #genero los puntos en cada clase
    #genero los multipolígonos por clase
    rois_gpd,clases = read_rois(vec_fname,clase)
    drois = rois_gpd.dissolve(by=clase)
    
    #genero numeros para las clases
    clases=list(drois.index)
    from functools import reduce 
    l = [[i]*N for i in range(len(clases))]
    Y = np.array(reduce(lambda a, b: a+b, l))
    


    #genero los puntos en cada clase
    if verbose: print('Seleccionando puntos en los ROIs.')
    pts = []
    for i in range(len(clases)):
        print(i)
        mpoly = drois.iloc[i]['geometry']
        pts.extend(Random_Points_in_MPolygon(mpoly, N))
        
    return pts, Y

#%%#Levanto en X, Y los datos etiquetados de pts dados de todo un directorio
def extract_point_features_from_dir(dir_features, points, verbose = True, ext = '.tif'):
    '''extrae los valores de los pixels en 
    puntos al azar en cada polígono, N por clase, y leyendo
    todos los archivos en el directorio dir_features.
    '''
   
    
    
    inicio = True
    #flist = os.listdir(dir_features)
    flist = [f for f in listdir(dir_features) if (
             isfile(join(dir_features, f)) and
             f.lower().endswith(ext.lower()))]
    flist.sort()
    if verbose: print(f'Leyendo datos espectrales de las {len(flist):d} imágenes.')
    feature_names = []
    for fn in flist:
        raster_fn = os.path.join(dir_features,fn)
        #X_, Y_ = extract_ROIs_features_from_raster(raster_fn, rois_gpd)
        #open raster file
        raster = rasterio.open(raster_fn)
        img = raster.read()

        #extract point value from raster
        X_ = []#np.empty([0,7],dtype=np.float32)
        for point in points:
            x = point.xy[0][0]
            y = point.xy[1][0]
            row, col = raster.index(x,y)
            s = img[:,row,col]
            #X_ = np.concatenate((X_,s),axis=1)
            X_.append(s)
    
        X_ = np.array(X_)
    
        if inicio:
            X = X_
            inicio = False
            if verbose: print('.', end='')
        else:
            X = np.concatenate((X,X_),axis=1)
            if verbose: print('.', end='')
        #feature names
        n_features = X_.shape[1]
        new_features = [fn.split('.')[0][:-n_features]+fn.split('.')[0][i-n_features] for i in range(n_features)]
        feature_names.extend(new_features)
        
                    
    return X, feature_names
#%%
#Levanto en X, Y los datos etiquetados de todo un raster
def extract_features_from_raster(raster_fn):
    '''Dado un raster, extrae X, los 
    valores espectrales de sus pixels.
    Devuelve el tamaño del raster para poder reconstruirlo'''
    #Leo los ROIS
    with rasterio.open(raster_fn) as src:
        array = src.read()
        sh = array.shape
        
    if len(sh)==2:
        array = array.reshape([array.shape[0] * array.shape[1], 7])
    elif len(sh)==3:
        array = array.reshape([array.shape[0],array.shape[1]* array.shape[2]]).T
    return array, sh[-2:]
#%%
#Levanto en X, Y los datos etiquetados de todo un raster
def extract_features_from_dir(dir_features, ext = '.tif',verbose = True):
    '''Dado un directorio, extrae X, los 
    valores espectrales de losd pixels de todos
    los rasters que terminan con ext. 
    Devuelve el tamaño de un raster para poder reconstruirlo.
    Todo deben tener las mismas dimensiones'''
    #Leo los ROIS
    inicio = True
    flist = [f for f in listdir(dir_features) if (
             isfile(join(dir_features, f)) and
             f.lower().endswith(ext.lower()))]
    flist.sort()
    for fn in flist:
        raster_fn = os.path.join(dir_features,fn)
        X_, sh_ = extract_features_from_raster(raster_fn)
        if inicio:
            sh = sh_
            X = X_
            inicio = False
            if verbose: print('.', end='')
        else:
            X = np.concatenate((X,X_),axis=1)
            if (sh==sh_):
                if verbose: print('.', end='')
            else:
                raise RuntimeError('No me coinciden las dimensiones de los rasters')
        #feature names
    return X, sh


#%%
##########################
## Procesar Directorios ##
##########################
# Procesar directorios enteros de imágenes

#%%
#Levanto en X, Y los datos etiquetados de todo un directorio
def read_IMGs_features_from_dir(dir_features, verbose = True, ext = '.tif'):
    '''extrae los valores de los pixels en las imágenes de 
    todos los archivos en el directorio dir_features.
    Devuelve el tamaño (común) de los rasters para poder reconstruirlo'''
    
    inicio = True
    flist = [f for f in listdir(dir_features) if (
             isfile(join(dir_features, f)) and
             f.lower().endswith(ext.lower()))]
    flist.sort()
    #feature_names = []
    for fn in flist:
        raster_fn = os.path.join(dir_features,fn)
        X_, sh = extract_features_from_raster(raster_fn)
        if inicio:
            X = X_
            inicio = False
        else:
            X = np.concatenate((X,X_),axis=1)
            if verbose: print('.', end='')
        #feature names
                    
    return X, sh
#%%

#%%
########################
## Features Genéricos ##
########################

#%%
#PARSE MNDWI

def read_serie_as_uint8(dir_in,starts_with='', ends_with='.tif', band = 1, ci=-1, cs=1, verbose = True):
    '''levanta las banda "band" de la serie en "dir_in" 
    que comienza con "starts_with" y termina con "ends_with"
    en un array uint8 (ajustando ci~0 y cs~255 como cotas inferior y superior).'''

    dc=cs-ci
    escala = 256.0/dc


    flist = [s for s in os.listdir(dir_in) if (s.startswith(starts_with) and s.endswith(ends_with))]
    lfl = len(flist)
    M=[]
    for i,f in enumerate(flist):
        if verbose: print(f'************************************************\nLeyendo {i+1:2d}/{lfl}:{f}')
        raster_fn = os.path.join(dir_in, f)
        with rasterio.open(raster_fn) as src:
            img = src.read(band)
        img=(img-ci)*escala
        img[img<0]=0
        img[img>255]=255
        M.append(img.astype(np.uint8)) #'u1'
    return np.array(M)
    
#%%
def summary_serie_as_rgb(M, perc_r = 20, perc_g = 50, perc_b = 80):
    '''M es la serie, los otros los percentiles a usar
    devuelve una imágen RGB'''
    r=np.percentile(M, perc_r, axis=0)
    g=np.percentile(M, perc_g, axis=0)
    b=np.percentile(M, perc_b, axis=0)
    return np.stack([r,g,b],axis=0)/256



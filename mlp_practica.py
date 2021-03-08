# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 00:28:15 2021

@author: José Lopez
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import os


#Cargo los datos
os.chdir('E:/Udesa/Personal/Datos')
data=pd.read_csv('smarket.csv')

#Algoritmo de clasificacion que se usa para predecir la probabilidad 
#de una variable dependiente categorica. 

#Vamos a usar datos del S&P Stock Market. 
#Esta base contiene los retornos porcentuales del S&P 500 stock index por 1250 dÃ­as, 
#desde inicios de 2001 hasta el final de 2005. Para cada fecha, tenemos:
    # Lag1, Lag2,..., Lag5: retornos porcentuales de cada uno de los di­as anteriores.
    # Volume: volumen de acciones negociadas (nro de acciones diarias negociadas en miles de millones de dolares)
    # Today: retorno porcentual de hoy
    # Direction: variable binaria que toma valores "Down" y "Up" indicando si el mercado tuvo un retorno positivo o negativo.
    
print(data.head())

#Primero voy a limpiar los datos, es decir, ver promediosy esas cosas
data = data.drop(['Unnamed: 0'], axis=1)
print(np.mean(data))

#Veo si hay nans
print(data.isnull().values.any())
    #ok, no tiene valores nulos

#Cambio los str a num
#1 si el retorno fue positivo, 0 si fue negativo
   
data.loc[data['Direction'] == 'Up', 'Direction'] = 1
data.loc[data['Direction'] == 'Down', 'Direction'] = 0

#Divido en datos de entrada y salida

entrada = data.copy()
entrada = entrada.drop(['Year'], axis = 1)
entrada = np.array(entrada.drop(['Direction'], axis = 1))
salida = np.array(data.Direction).reshape(data.shape[0],1).astype('float64')


#Voy a programar una red neuronal con una sola capa oculta, para tratar de predecir si 
#el retorno del mercado fue positivo o negativo


#Defino algunos parametros iniciales
P = entrada.shape[0]       #numero de datos (filas)
N = entrada.shape[1]       #numero de unidades de entrada en la primera capa de entrada
H = N+1                     #numero de unidades de entrada en la capa oculta
M = salida.shape[1]         #numero de unidades de la capa de salida

#Paso los datos de matrices a tensores
x = torch.from_numpy(entrada)
z = torch.from_numpy(salida)

#Empiezo a crear
w1 = torch.randn( N+1, H, requires_grad=True)
w2 = torch.randn( H+1, M, requires_grad=True)

bias = torch.ones( P, 1)

lr = 1e-2
t, e = 0, 1.
while e>0.01 and t<9999:
    h = torch.cat( (x,bias), dim=1).mm(w1).sigmoid()
    y = torch.cat( (h,bias), dim=1).mm(w2).sigmoid()
    error = (y-z).pow(2).sum()
    error.backward()
    with torch.no_grad():
        w1 -= lr*w1.grad
        w2 -= lr*w2.grad
        w1.grad.zero_()
        w2.grad.zero_()
    e = error.item()/P
    t += 1
    if t%1000==0:
        print(t,e)





























#%%
class mlp( torch.nn.Module):                    # Module para hacer modelos propios.
    def __init__( self, N, H, M):      # Uso _ en lugar de self.
        super().__init__()
        self.l1 = torch.nn.Linear( N, H)   # Params de modelo.
        self.l2 = torch.nn.Linear( H, M)

    def forward( self, x):
        h = torch.sigmoid( self.l1( x))               # Grafo de computo.
        y = torch.sigmoid( self.l2( h))
        return y
    
model = mlp( N, H, M)
optim = torch.optim.SGD( model.parameters(), lr=0.01)
costf = torch.nn.MSELoss()

t, E = 0, 1.
while E>=0.01 and t<9999:
    y = model( x)
    optim.zero_grad()                       # Optim sabe que resetear.
    error = costf( y, z)
    error.backward()
    optim.step()                            # step aplica los gradientes.
    E = error.item()
    t += 1
    if t%1000==0:
        print( t, E)
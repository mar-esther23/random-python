#!/usr/bin/env python
# -*- coding: utf-8 -*- 

# Sean celdas que representan micro-ecosistemas
# Las celdas de bosque tienen un pequeño ecosistema
# Las celdas milpa son campos lindos
# Las celdas estensivo son campos feos
# Los agentes tratan de migrar a traves de las celdas

# poblacion = [tiempo] [x][y] [especieA][especieB][...]

# Basado en articulo Chaos in low-dimensional Lotka–Volterra models of competition. JA Vano, ... JC Sprott. 2006

import sys, copy, random
import numpy as np
from itertools import combinations, product
from scipy.integrate import odeint
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
from matplotlib import colorbar as cbar
from matplotlib import colors as colors

"""
CORRIDAS
"""
def correr_2D(poblacion_0, tipo, t_total, n_especies, f, r, a, m_milpa, m_intensivo, D, vecinos):
    #corre simulacion en 2D
    poblacion = [poblacion_0] #inicializa array poblacion
    for t in range(t_total):
        temp = np.zeros_like(poblacion[-1])
        #interacciones ecologicas y muerte
        for i in range(x_celdas): #para todo x y
            for j in range(y_celdas):
                if tipo[i][j] == 'b': #interacciones ecologicas
                    temp[i][j] = odeint(f, poblacion[-1][i][j], [0,1], args=(r,a))[-1]
                elif tipo[i][j] == 'm': #milpa
                    temp[i][j] = muerte(poblacion[-1][i][j], m_milpa)
                elif tipo[i][j] == 'i': #intensivo
                    temp[i][j] = muerte(poblacion[-1][i][j], m_intensivo)
        #migracion
        for i in range(n_especies):
            #si quieres variar la taza de migracion por especie aqui es donde debes de variarla
            temp[:,:,i] = migracion(temp[:,:,i], D, vecinos)
        poblacion.append(temp)
    return np.array(poblacion)


"""
VALORES INICIALES
"""

def genera_tipo_matriz_agroecologica(x_celdas, y_celdas, n_bosque=0, posicion_bosque=[], n_milpa=0, posicion_milpa=[]): #Distribucion de matriz agroecologica
    #recibe una lista de tuples o un comando
    #ej: [(2,3),(4,1),(2,2)]
    tipo = [['i' for i in range(y_celdas)] for j in range(x_celdas)] #inicializa todo con intensivo
    
    if posicion_bosque == "extremos":
        posicion_bosque = [(0,0),(x_celdas-1, y_celdas-1)]
    if posicion_milpa == "stepping": #method for square matrix
        posicion_milpa = [(i,i) for i in range(n_milpa,x_celdas-1,n_milpa)]
    
    if posicion_bosque == "random" or posicion_milpa == "random":
        #genera todas las posibles coordenadas
        pairs = [(x,y) for x in [i for i in range(x_celdas)] for y in [j for j in range(y_celdas)]]
        random.shuffle(pairs) #randomisa
        #quita los bosque y milpas ya declarados
        if type(posicion_bosque) == list:
            for p in posicion_bosque:
                try: pairs.remove(p)
                except: pass
        if type(posicion_milpa) == list:
            for p in posicion_milpa:
                try: pairs.remove(p)
                except: pass
        #selecciona n_bosques y n_milpas
        if posicion_bosque == "random":
            posicion_bosque = random.sample(pairs,n_bosque)
            for p in posicion_bosque: #quita para no confundir a random milpa
                try: pairs.remove(p)
                except: pass
        if posicion_milpa == "random":
            posicion_milpa = random.sample(pairs,n_milpa)
    
    for p in posicion_bosque: #escribe bosques en matriz
        tipo[p[0]][p[1]] = 'b'
    for p in posicion_milpa: #escribe bosques en matriz
        tipo[p[0]][p[1]] = 'm'
    return tipo

def genera_poblacion_inicial(tipo_matriz_agroecologica, n_especies, p0_bosque=0, p0_milpa=0, p0_intensivo=0): #Poblacion inicial
    if type(p0_bosque)==float: #all same value
        p0_bosque = [p0_bosque for i in range(n_especies)]
    elif type(p0_bosque)==int: #users
        p0_bosque = [p0_bosque for i in range(n_especies)]
    elif p0_bosque=="eq_caos":p0_bosque = [ 0.3013,  0.4586,  0.1307,  0.3557]
    
    if type(p0_milpa)==float: #all same value
        p0_milpa = [p0_milpa for i in range(n_especies)]
    elif type(p0_milpa)==int: #users
        p0_milpa = [p0_milpa for i in range(n_especies)]
        
    if type(p0_intensivo)==float: #all same value
        p0_intensivo = [p0_intensivo for i in range(n_especies)]
    elif type(p0_intensivo)==int: #users
        p0_intensivo = [p0_intensivo for i in range(n_especies)]
    
    poblacion_0 = copy.deepcopy(tipo)
    for x in range(len(tipo)): #inicializar poblaciones
        for y in range(len(tipo[0])):
            if tipo[x][y] == 'b': 
                if p0_bosque=="random": poblacion_0[x][y] = [random.random() for i in range(n_especies)]
                else: poblacion_0[x][y] = p0_bosque
            if tipo[x][y] == 'm': 
                if p0_milpa=="random": poblacion_0[x][y] = [random.random() for i in range(n_especies)]
                else: poblacion_0[x][y] = p0_milpa
            if tipo[x][y] == 'i': 
                if p0_intensivo=="random": poblacion_0[x][y] = [random.random() for i in range(n_especies)]
                else: poblacion_0[x][y] = p0_intensivo
    
    return np.array(poblacion_0)



"""
FUNCIONES
"""
def d_competencia(x,t,r,a): # interacciones ecologicas: competencia
   # recibe x = poblacion np.array
   #    r = taza de crecimiento cte o np.array
   #    a = matriz competencia entre especies np.array
   # regresa dx/dt np.array
   # Basado en articulo Chaos in low-dimensional Lotka–Volterra models of competition. JA Vano, ... JC Sprott. 2006
   dx = (x*r)*(1-(x*a).sum(axis=1))
   return dx

def migracion(x, D, tipo='linea',limite='fill'):
    # recibe matriz de celdas con poblaciones y coeficiente difusion
    # regresa nueva matriz de celdas con poblaciones  np.array
    # D puede ser un int que es el % total de migracion
    #             o un array determinando las direcciones si especial
    # tipo determina el numero de vecinos
    #     linea, vecinos4, vecinos8, y especial
    # limite es cerrado o circular
    #     fill o wrap
    
    if tipo == 'linea': # Migracion linea 1D
        if limite == 'fill': #cerrado
            #calcular perdida
            loss = np.array([1.] + [2. for i in range(len(x[0])-2)] + [1.])
            loss = (D/2.0) * loss * x[0]
            #calcular ganancia
            D = np.array([D/2.0,0,D/2.0])
            gain = np.convolve(x[0], D, 'same')
        if limite == 'wrap': #circular
            #calcular perdida
            loss = D * x[0]
            #calcular ganancia
            D = np.array([D/2.0,0,D/2.0])
            gain = np.convolve(x, D, 'full') #np.convolve(x, D, 'full')
            gain[1]+=gain[-1] #wrap
            gain[-2]+=gain[0] #wrap
            gain = gain[1:-1] #reduce
        return  [x[0] + gain - loss]
   
    # Migracion matriz 2D
    elif tipo == 'vecinos4':
        #calcular perdida
        if limite == 'fill': #cerrado
            loss = np.array([[2] + [3 for i in range(len(x[0])-2)] + [2]] + [[3] + [4 for i in range(len(x[0])-2)] + [3] for j in range(len(x)-2)] + [[2] + [3 for i in range(len(x[0])-2)] + [2]])
            loss = (D/4.0) * loss * x
        if limite == 'wrap': #circular
            loss = D * x
        print loss
        print sum(sum(loss))
        #calcular ganancia
        D = np.array([[0,D/4.0,0],[D/4.0,0,D/4.0],[0,D/4.0,0],]) #matriz dispercion
        gain = convolve2d(x, D, mode='same', boundary=limite)
        print gain
        print sum(sum(gain))
        #calcula total
        return  x + gain - loss
   
    elif tipo == 'vecinos8':
        #calcular perdida
        if limite == 'fill': #cerrado
            loss = np.array([[3] + [5 for i in range(len(x[0])-2)] + [3]] + [[5] + [8 for i in range(len(x[0])-2)] + [5] for j in range(len(x)-2)] + [[3] + [5 for i in range(len(x[0])-2)] + [3]])
            loss = (D/8.0) * loss * x
        if limite == 'wrap': #circular
            loss = D * x
        #calcular ganancia
        D = np.array([[D/8.0,D/8.0,D/8.0],[D/8.0,0,D/8.0],[D/8.0,D/8.0,D/8.0],]) #matriz dispercion
        gain = convolve2d(x, D, mode='same', boundary=limite)
        #calcula total
        return x + gain - loss
   
    # Si se pasa directamente un array de dispercion
    elif tipo == 'especial':
        #calcular perdida
        loss = sum(sum(D)) * x
        #calcular ganancia
        gain = convolve2d(x,D,mode='same', boundary='wrap')
        #calcula total
        return x + gain - loss

def muerte(x, m):
   # recibe x = poblacion   
   #     m = taza muerte cte o np.array
   # regresa x = poblacion superviviente  np.array
   x = x - x*m
   return x



"""
GRAFICAS
"""
def plot_xy(name, x, y, title='',labels=[], x_label='', y_label=''): #grafica x contra y
    #recibe x como numpy.array
    #Ejemplo imprimir poblacion vs tiempo
    #plot_xy(name, x, t, title,labels, x_label='Tiempo', y_label='Poblacion')
    plt.clf()
    try:
        for i in range(len(x[0])): #grafica cada x contra y
        #print i, labels[i]
            plt.plot(y, x[:,i], label=labels[i])
    except: plt.plot(y, x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc=0)
    plt.plot()
    #save plot
    f_format = name.split('.')[-1]
    name = name.split('.')[0]
    plt.savefig(name+'.'+f_format, format=f_format, bbox_inches='tight')
    #plt.show()

def plotHeatmap(name, data , x_label='' , y_label='', x_tick_labels=[], y_tick_labels=[]):
    #data is a 2x2 array normalized [0,1]
    plt.clf()
    fig, ax = plt.subplots()
    #delete top and right axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ## put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)
    ## want a more natural, table-like display
    ##ax.invert_yaxis()
    ##ax.xaxis.tick_top()
    ax.set_xticklabels(x_tick_labels, rotation=90, minor=False)
    ax.set_yticklabels(y_tick_labels, minor=False)
    #set colorbar
    cdict = {'red':   [(0.0,  1.0, 1.0),(0.01,  0.5, 0.5),(0.5,  0.0, 0.0),(1.0,  0.0, 0.0)],
        'green': [(0.0,  1.0, 1.0),(0.1, 1.0, 1.0),(1.0,  0.0, 0.0)],
        'blue':  [(0.0,  1.0, 1.0),(0.5,  1.0, 1.0),(1.0,  0.5, 0.5)]}
    my_cmap=colors.LinearSegmentedColormap('my_colormap',cdict,256)
    #heatmap = ax.pcolor(data, cmap=plt.cm.Blues)
    heatmap = ax.pcolor(data, cmap=my_cmap, vmin=0, vmax=1)
    cbar = plt.colorbar(heatmap)
    plt.title(name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot()
    #save plot
    f_format = name.split('.')[-1]
    name = name.split('.')[0]
    plt.savefig(name+'.'+f_format, format=f_format, bbox_inches='tight')
    #plt.show()

def plot_lineas_especies(name, x, t_total, labels=['A','B','C','D']):
    #graficar sum xy para todas las especies
    t = np.linspace(0,t_total+1,t_total+1)
    plot_xy(name, x, t, 'especies_vs_tiempo', labels)
    #diagramas de fase
    for i,j in combinations ([n for n in range(len(x[0]))],2):
        plot_xy(name.split('.')[0] + '_'+labels[i]+labels[j]+'.' + name.split('.')[-1], x[:,i], x[:,j], 'fase_'+labels[i]+labels[j])
        
        
    
def heatmaps_especie_en_tiempo(name,poblacion, n=[], label=['A', 'B', 'C', 'D']):
    ter = '.'+name.split('.')[-1]
    name = name.split('.')[0]
    #graficar heatmap especie en un tiempo dado
    #n debe ser un array
    if type(n) == int: n = [n]
    for i in range(n_especies):
        plotHeatmap(name+label[i]+'_t0'+ter, poblacion[0,:,:,i]) #estado inicial
        plotHeatmap(name+label[i]+'_tf'+ter, poblacion[-1,:,:,i]) #estado final
        for j in n: #grafica vector de tiempos
            plotHeatmap(name+label+'_t'+str(j)+ter, poblacion[j,:,:,i])

def heatmap_tipo(name,tipo,m_milpa=.5,m_intensivo=1):
    for i in range(len(tipo)):
        for j in range(len(tipo[0])):
            if tipo[i][j] == 'b': tipo[i][j] = 1.
            if tipo[i][j] == 'm': tipo[i][j] = 1-m_milpa
            if tipo[i][j] == 'i': tipo[i][j] = 1-m_intensivo
    tipo = np.array(tipo)
    plotHeatmap(name, tipo) #estado inicial    
    
    

"""
MEDIDAS
"""
def alpha_shannon(p):
    #recibe poblacion[x][y][sp]
    #regresa float
    #H' = -\sum(p_i ln(p_i))
    #p_i = sp_i / \sum(sp)
    p = np.sum(np.sum(p, axis=0),axis=0) #sumar valores en xy
    t = np.sum(p) #total sp
    #estrictamente hay que dividir entre len(x)*len(y), pero como vamos a dividir entre el total eso puede ser ignorado
    #print p/t
    return -1 * sum(p + np.log(p))

def alpha_pielou(p):
    #recibe poblacion[x][y][sp]
    #regresa float
    #J'=H'/S;   S=num sp
    p = np.sum(np.sum(p, axis=0),axis=0) #sumar valores en xy
    t = np.sum(p) #total sp
    s = len(p) #num sp
    #estrictamente hay que dividir entre len(x)*len(y), pero como vamos a dividir entre el total eso puede ser ignorado
    #print p/t, s
    return (-1 * sum(p + np.log(p)))/s

def alpha_simpson(p):
    #recibe poblacion[x][y][sp]
    #regresa float
    #\gamma = \sum(p_i^2)
    #p_i = sp_i / \sum(sp)
    p = np.sum(np.sum(p, axis=0),axis=0) #sumar valores en xy
    t = np.sum(p) #total sp#estrictamente hay que dividir entre len(x)*len(y), pero como vamos a dividir entre el total eso puede ser ignorado
    #print (p/t)**2
    return np.sum((p/t)**2)

def beta_jacard(a,b,u=0.001):
    #recibe poblacion[x][y][sp]
    #I_AB = interseccion/union
    #     = c / (a+b-c)
    #   a: sp en A
    #   b: sp en B
    #   c: sp en AyB
    #poner en terminos de presencia y ausencia
    a = np.where(a>u,1.,0.)
    b = np.where(b>u,1.,0.)
    #print "interseccion: ", a*b
    #print "union: ", a+b-a*b
    #sum es para sumar las sp
    return np.sum(a*b) / (np.sum(a+b-a*b))








"""
M   M     A  III  N   N
MM MM    AA   I   NN  N
M M M   A A   I   N N N
M   M  AAAA   I   N  NN
M   M A   A  III  N   N
"""


"""
PARAMETROS
"""
n_especies = 4 # numero especies
x_celdas = 5 # numero celdas en x
    #para 1D x=1
y_celdas = 5 # numero celdas en y
              #        si 1 modelo en 1D

r = np.array([ 1.  ,  0.72,  1.53,  1.27]) #taza de crecimiento de las poblaciones
a = np.array([[ 1.  ,  1.09,  1.52,  0.  ], #matriz de competencia
              [ 0.  ,  1.  ,  0.44,  1.36],
              [ 2.33,  0.  ,  1.  ,  0.47],
              [ 1.21,  0.51,  0.35,  1.  ]])

D = 0.8 #coeficiente de difusión/migracion total cte o np.array

m_milpa = 0.3 #taza muerte negra cte o np.array
m_intensivo = 0.6 #taza muerte blanca cte o np.array

h = 0.001 #diferencial de cambio en t (euler y graficas)
t_total = 100 #tiempo total de simulacion


## EJEMPLO DE UNA CORRIDA

##Distribucion de matriz agroecologica
#tipo = genera_tipo_matriz_agroecologica(x_celdas, y_celdas, n_bosque=2, posicion_bosque="extremos", n_milpa=2, posicion_milpa=[(1,2),(3,2)])
#print tipo

##Poblacion inicial
#poblacion_0 = genera_poblacion_inicial(tipo, n_especies, p0_bosque=0.7, p0_milpa=0, p0_intensivo=0)
###print poblacion_0

#poblacion = correr_2D(poblacion_0, tipo, t_total, n_especies, d_competencia, r, a, m_milpa, m_intensivo, D, 'vecinos8')
##print poblacion[-1]

##Graficas varias

#heatmap_tipo('tipo.png',tipo)
#heatmaps_especie_en_tiempo('especie.png',poblacion)

#sum_xy = np.sum(np.sum(poblacion, axis=1),axis=1) #calcula sum en xy para cada especie
#plot_lineas_especies('sum_xy.png', sum_xy, t_total, ['A','B','C','D'])

##Medidas
#print "a-shannon: ", alpha_shannon(poblacion[-1])
#print "a-pielou:  ", alpha_pielou(poblacion[-1])
#print "a-simpson: ", alpha_simpson(poblacion[-1])
#print "b-jacard:  ", beta_jacard(poblacion[-1][0][0],poblacion[-1][-1][-1])#bosques extremos

##CORRIDAS VARIANDO DOS PARAMETROS
#l = 5 #numero total de puntos por variable
#res = [[0 for i in range(l+1)] for j in range(l+1)] #arreglo de resultados

##Distribucion de matriz agroecologica
#tipo = genera_tipo_matriz_agroecologica(x_celdas, y_celdas, n_bosque=2, posicion_bosque="extremos", n_milpa=2, posicion_milpa="stepping")
##for t in tipo: print t
##Poblacion inicial
#poblacion_0 = genera_poblacion_inicial(tipo, n_especies, p0_bosque=0.5, p0_milpa=0, p0_intensivo=0)
##print poblacion_0[:,:,0] #imprime la especieA


#for i in range(l+1): #i es m_milpa
    #for j in range(l+1): #j es m_intensivo
        ##Defines tipo poblacional_0
        ##Correr y variar parametros
        #poblacion = correr_2D(poblacion_0, tipo, t_total, n_especies, d_competencia, r, a, i/float(l), j/float(l), D, 'vecinos8')
        ##Guardar resultados
        #res[i][j] = poblacion[-1][2][2] 
#res = np.array(res)
##print res
##print np.sum(res,axis=2) #la suma de todas las poblaciones para cada par de casos

#heatmap_tipo('tipo.png',tipo)
#plotHeatmap('proporcion_muerte.png', np.sum(res,axis=2) , 'm_intensivo', 'm_milpa', [i/float(l) for i in range(l+1)], [i/float(l) for i in range(l+1)])


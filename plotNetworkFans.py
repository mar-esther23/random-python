import logging as log
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from math import cos, sin, tan, degrees, radians
from colorsys import rgb_to_hsv, hsv_to_rgb
from random import random



def plotNetworkFans(G, root_size=20, fan_radius=200, spread=60, coloring='layer', color_gradient=0.05, color_random=0 , cmap='hsv', plot=False):
    """
    Plot a state transition graph (directed tree) with fans. 

    The graph should be a TREE except for the root, the nodes have only one succesor and many predescesors. Also, the graph should have only one ROOT, the root can be a single node or cycle.

    In particular it was designed for the state transitions graphs that are the result of synchronous directed boolean networks. This representation was inspired by Maximino Aldana's graphs.

    If the color and size of the root nodes are previously defined as attributes it will respect them and use them as a basis for coloring.

    Parameters
    ----------
    G:      nx Digraph to plot
    root_size:  size of root nodes
    fan_radius: radius of fan
    spread:     spread of fan

    coloring:   algorithm for coloring
        'layer':  the layer has the similar color
                  parent color + color_gradient + color_random,
        'random': each fan has a random color
    color_gradient: change between fans color
    color_random:   random change between fans color

    cmap:       
    plot:       if true plots graphs, 
                if string saves in file plot


    Returns
    -------
    G:      nx DiGraph with x, y, size and color attributes.
    """

    # Determine root of tree
    root = getRoot(G)
    log.info("root={}".format(root))
    G = setRoot(G, root, root_size, fan_radius)

    # To avoid problems remove temporally edges between root cycle
    if len(root) > 1: 
        log.info("Remove cycle edges")
        cycle_edges = G.subgraph(root).edges()
        G.remove_edges_from(cycle_edges)

    # Traverse the graph by depth plotting fans
    for n in root:
        log.info("n={}".format(n))
        log.info("predecessors={}".format(  G.predecessors(n) ))
        if G.predecessors(n) > 0: # Determine predecessors
            #This method is recursive depth first!
            createFan(G, n, spread, fan_radius, 
                coloring, color_gradient, color_random) 

    # Return deleted edges
    if len(root) > 1:
        #TODO check if black is a valid argument
        log.info("Return cycle edges")
        G.add_edges_from(cycle_edges, color=0) # also color

    if plot==True or type(plot)==str:
        plotFanNetworkFromAttributes(G, plot, cmap)

    return G



def createFan(G, n, spread=60, r=200, 
    coloring='layer',color_gradient=0.05, color_random=0 ):
    """
    Returns a fan given a base. The fan has possitions, color and size depending in parent. This method is recursive depth first!

    Parameters
    ----------
    G:      nx Digraph to plot
    n:   base node of fan
    spread: total angle of fan
    r:      radius of fan

    coloring:    algorithm for coloring
        'layer': the layer has the parent color + color_gradient + color_random,
        'random': each fan has a random color
    color_gradient: change between fans color
    color_random:   random change between fans color

    Returns
    -------
    Modifies the graph in place.
    """

    #Determine predeccesors
    pre = G.predecessors(n)
    log.info("node={}".format( n ))
    log.info("precesors={}".format( pre ))

    # Generate position of whole fan of predecesors
    pos = createFanPoints(len(pre), G.node[n]['x'], G.node[n]['y'], G.node[n]['angle'], spread, fan_radius)
    for s in zip(pre, pos):
        # Set position for each node
        log.info("node, (x, y, angle)={}".format(s))
        G.node[s[0]]['x'] = s[1][0]
        G.node[s[0]]['y'] = s[1][1]
        G.node[s[0]]['angle'] = s[1][2] 

        # Set color and size
        log.info("color={}".format(G.node[n]['color']))
        if coloring == 'layer': #take parent color and modify slightly
            color = G.node[n]['color'] + color_gradient + (random()-.5)*color_random
        if coloring == 'random':
            color = random()
        G.node[s[0]]['color'] = color
        log.info("color={}".format( G.node[s[0]]['color'] ))

        G.node[s[0]]['size'] = 2 #small is pretty
        log.info("size={}".format( G.node[s[0]]['size'] ))

    # Set edges to father's color
    for e in G.in_edges(n):
        G[e[0]][e[1]]['color'] = G.node[n]['color']

    # This method is recursive depth first!
    for s in pre:
        log.info("n_s={}".format(s))
        log.info("predecessors={}".format(  G.predecessors(s) ))
        if G.predecessors(s) > 0:
            createFan(G, s, spread, fan_radius, 
                coloring, color_gradient, color_random) 


def createFanPoints(n=1, x_0=0, y_0=0, a_0=0, spread=60, r=200):
    """
    Returns the positions and angle (x,y,a) of a fan of nodes with a given base.

    Parameters
    ----------
    n:      (int) number of nodes
    x_0:    position of base of fan
    y_0:    position of base of fan
    a_0:    angle of base of fan
    spread: total angle of fan
    r:      radius of fan

    Returns
    -------
    List of (x,y,a) points
    """
    # log.basicConfig(level=log.DEBUG)
    pos = []
    log.info( "n={0}, spread={1}".format(n,spread))
    for i in range(n):
        if n==1: new_angle = a_0
        elif spread == 360: new_angle = a_0 + spread/float(n) * i - spread/2.0
        else: new_angle = a_0 + spread/float(n-1) * i - spread/2.0
        log.info( "angle={}".format(new_angle) )
        x = r * cos(radians(new_angle)) + x_0
        y = r * sin(radians(new_angle)) + y_0
        log.info( '(x,y)=({0}, {1})'.format(x,y) )
        pos.append((x,y,new_angle))
    return pos


def getRoot(G):
    """
    Determine the root of the tree, return a list with the ordered node(s). The root can be a node or cycle. Remember that nodes have only one succesor.

    """
    # log.basicConfig(level=log.DEBUG)
    # search for steady state attractor
    root = []
    for n in G:
        if G.out_degree(n) == 0: root.append(n)
    log.info( "root_steady={}".format(root) ) 
    if len(root) == 1: return root
    if len(root) > 1: 
        raise IndexError("The graph has more than one root!")
        return None
    # search for cyclic attractor
    else:
        cycle = nx.simple_cycles(G)
        root = cycle.next()
        log.info( "root_cycle={}".format(root) ) 
        try: cycle.next() #verify only one cycle
        except  StopIteration: return root
        raise IndexError("The graph has more than one  cycle!")
        return None


def setRoot(G, root, size=50, radius=200):
    """
    This method assigns x,y, size and color to the root.
    If a node has a previously defined attributed it is respected.
    """
    if len(root) == 1: #if steady state set as root in (0,0)
        # Set position of root
        G.node[root[0]]['x'] = 0
        G.node[root[0]]['y'] = 0
        G.node[root[0]]['angle'] = 0 
        # Set color and size
        if 'color' not in G.node[root[0]]: #if no defined color assign a random one
            G.node[root[0]]['color'] = random()
        if 'size' not in G.node[root[0]]: #if no defined size assign a default
            G.node[root[0]]['size'] = size

    else: # if cycle create fan
        # Determine position of root
        pos = createFanPoints(len(root), 0, 0, 0, 360, radius)
        for n in zip(root, pos):
            # Set position of root
            log.info("node, x, y, angle={}".format(n))
            G.node[n[0]]['x'] = n[1][0]
            G.node[n[0]]['y'] = n[1][1]
            G.node[n[0]]['angle'] = n[1][2] 
            # Set color and size
            if 'color' not in G.node[n[0]]: #if no defined color assign a random one
                G.node[n[0]]['color'] = random()
            if 'size' not in G.node[n[0]]: #if no defined size assign a default
                G.node[n[0]]['size'] = size

    return G


def plotFanNetworkFromAttributes(G, plot=True, cmap='hsv'):
    """
    Take a DiGraph, where position, size and color are node attributes and plot it with draw_networkx.

    This code supposes no labels or attributes.
    This code suposses the attributes are called: x, y, color, size.
    """

    # Convert node attributes to valid format
    log.info("Create node attributes for plotting")
    position = {}
    color, size = [], []
    for n in G:
        position[n] = (G.node[n]['x'], G.node[n]['y'])
        color.append(G.node[n]['color'])
        size.append(G.node[n]['size'])
    #convert color from float to rgba
    color = cm.ScalarMappable(cmap=cmap).to_rgba(color)
    log.info("Plot nodes.")
    nx.draw_networkx_nodes(G, with_labels=False, linewidths=0.10,
         pos=position, node_size=size, node_color=color, 
         )

    #convert edge attributes to valid format
    log.info("Create edge attributes for plotting")
    edge_color = []
    for e in G.edges():
        edge_color.append(G[e[0]][e[1]]['color'])
    #convert color from float to rgba
    edge_color = cm.ScalarMappable(cmap=cmap).to_rgba(edge_color)
    log.info("Plot edges")
    nx.draw_networkx_edges(G, arrows=False, alpha=0.5,
         pos=position, edge_color=edge_color,
         )


    #remove axis
    plt.tick_params(
        axis='both', which='both',
        bottom='off', top='off', left='off', right='off',
        labelbottom='off', labelleft='off'
        )

    if type(plot)==str: plt.savefig(plot)
    else: plt.show()


"""
MAIN
Esto sera quitado algun dia.

Por lo pronto existe para importar grafica CC.
"""
# log.basicConfig(level=log.DEBUG)

fan_radius = 750
root_size = 150
spread = 60

color_gradient = 0.05
color_random = 0.1
coloring = 'layer'
cmap = 'Accent'


# Open network
G = nx.DiGraph()
f = open('StrikingAtCC.csv')
for line in f:
    line = line.strip().split(',')
    G.add_edge(int(line[0]), int(line[1]))
f.close()
print len(G)

# set default colors
G.node[2346 ]['color'] = 0.33 #G1 g
G.node[10508]['color'] = 0.33 #G1 g
G.node[8193 ]['color'] = 0.00 #S  r
G.node[9345 ]['color'] = 0.00 #S  r
G.node[10113]['color'] = 0.00 #S  r
G.node[14273]['color'] = 0.16 #G2 y
G.node[5887 ]['color'] = 0.16 #G2 y
G.node[7165 ]['color'] = 0.16 #G2 y 
G.node[6589 ]['color'] = 0.66 #M  b
G.node[6461 ]['color'] = 0.66 #M  b
G.node[6462 ]['color'] = 0.66 #M  b

# plotNetworkFans(G, root_size, fan_radius, spread, 
#     coloring, color_gradient, color_random , cmap, plot=True)

plotNetworkFans(G, plot=True)


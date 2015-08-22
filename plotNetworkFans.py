import logging as log
import networkx as nx
import matplotlib.pyplot as plt
from math import cos, sin, tan, degrees, radians
from colorsys import rgb_to_hsv, hsv_to_rgb
from random import random


"""
This code was designed to plot directed tree graphs as fans. 

The graph should be a TREE except for the root, the nodes have only one succesor and many predescesors. Also, the graph should have only one ROOT, the root can be a single node or cycle.

In particular it was designed for the state transitions graphs that are the result of synchronous directed boolean networks. This representation was inspired by Maximino Aldana's graphs.
"""




def createFan(G, n, spread=120, r=50, dhsv=0.05):
    """
    Returns a fan given a base. The fan has possitions, color ans size depending in parent. This method is recursive depth first!

    Parameters
    ----------
    G:      nx Digraph to plot
    n:   base node of fan
    spread: total angle of fan
    r:      radius of fan
    dhsv:   hsv gradient

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
        color = rgb_to_hsv(G.node[n]['color'][0], G.node[n]['color'][1], G.node[n]['color'][2])[0] + dhsv
        #hes just like his father (kinda)!
        G.node[s[0]]['color'] = hsv_to_rgb(color,1,1)
        log.info("color={}".format( G.node[s[0]]['color'] ))
        G.node[s[0]]['size'] = 1 #small is pretty, we'll color edges later
        log.info("size={}".format( G.node[s[0]]['size'] ))

    # Set edges to father's color
    for e in G.in_edges(n):
        G[e[0]][e[1]]['color'] = G.node[n]['color']

    # This method is recursive depth first!
    for s in pre:
        log.info("n_s={}".format(s))
        log.info("predecessors={}".format(  G.predecessors(s) ))
        if G.predecessors(s) > 0:
            createFan(G, s, spread=spread, r=r, dhsv=dhsv)


def createFanPoints(n=1, x_0=0, y_0=0, a_0=0, spread=120, r=50):
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
        if spread == 360 or n==1: new_angle = a_0 + spread/float(n) * i - spread/2.0
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


def setRoot(G, root, size=50, radius=250):
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


def plotFanNetworkFromAttributes(G, filename=False):
    """
    Take a DiGraph, where position, size and color are node attributes and plot it with draw_networkx.

    This code supposes no labels or attributes.
    This code suposses the attributes are called: x, y, color, size.
    """
    # Convert node attributes to valid format
    position = {}
    color, size = [], []
    for n in G:
        position[n] = (G.node[n]['x'], G.node[n]['y'])
        color.append(G.node[n]['color'])
        size.append(G.node[n]['size'])
    nx.draw_networkx_nodes(G, with_labels=False,
         pos=position, node_size=size, node_color=color
         )

    #convert edge attributes to valid format
    e_color = []
    for e in G.edges():
        e_color.append(G[e[0]][e[1]]['color'])
    nx.draw_networkx_edges(G, arrows=False,
         pos=position, edge_color=e_color
         )
    if filename: plt.savefig(filename)
    else: plt.show()


"""
MAIN
Esto sera quitado algun dia.
"""
# log.basicConfig(level=log.DEBUG)

fan_radius = 750
hue_gradient = 0.05
root_size = 150
spread = 60


# Open network
G = nx.DiGraph()
f = open('StrikingAtCC.csv')
for line in f:
    line = line.strip().split(',')
    G.add_edge(int(line[0]), int(line[1]))
f.close()
print len(G)

# Determine root of tree
root = getRoot(G)
print root
log.info("root={}".format(root))
G = setRoot(G, root, root_size, fan_radius)

# To avoid problems remove temporally edges between root cycle
if len(root) > 1: 
    cycle_edges = G.subgraph(root).edges()
    G.remove_edges_from(cycle_edges)

# Traverse the graph by depth plotting fans
for n in root:
    log.info("n={}".format(n))
    log.info("predecessors={}".format(  G.predecessors(n) ))
    if G.predecessors(n) > 0: # Determine predecessors
        #This method is recursive depth first!
        createFan(G, n, spread, fan_radius, dhsv=hue_gradient) 

# Return deleted edges
if len(root) > 1:
    G.add_edges_from(cycle_edges, color=(0,0,0)) #colorear de paso

plotFanNetworkFromAttributes(G)

    



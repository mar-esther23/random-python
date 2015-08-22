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
        color = rgb_to_hsv(G.node[n]['color'][0], G.node[n]['color'][1], G.node[n]['color'][2])[0] + dhsv #hes just like his father (kinda)!
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
log.info("root={}".format(root))

# Determine position of root
if len(root) == 1: #if steady state set as root in (0,0)
    G.node[root[0]]['x'] = 0
    G.node[root[0]]['y'] = 0
    G.node[root[0]]['angle'] = 0 
else: # if cycle create fan
    pos = createFanPoints(len(root), 0, 0, 0, 360, fan_radius)
    for n in zip(root, pos):
        log.info("node, x, y, angle={}".format(n))
        G.node[n[0]]['x'] = n[1][0]
        G.node[n[0]]['y'] = n[1][1]
        G.node[n[0]]['angle'] = n[1][2] 
    # To avoid problems remove temporally edges between cycle
    cycle_edges = G.subgraph(root).edges()
    G.remove_edges_from(cycle_edges)


# Determine color and size of root
for n in root:
    try: G.node[n]['color']
    except KeyError: #if no defined color assign a random one
        G.node[n]['color'] = hsv_to_rgb(random(),1,1)
    log.info("color={}".format( G.node[n]['color'] ))
    try: G.node[n]['size']
    except KeyError:
        G.node[n]['size'] = root_size
    log.info("size={}".format( G.node[n]['size'] ))




# Traverse the graph by depth plotting fans
for n in root:
    log.info("n={}".format(n))
    log.info("predecessors={}".format(  G.predecessors(n) ))
    if G.predecessors(n) > 0: # Determine predecessors
        #This method is recursive depth first!
        createFan(G, n, spread, fan_radius, dhsv=hue_gradient) 

# Return deleted edges
if root > 1:
    G.add_edges_from(cycle_edges)
    for e in cycle_edges:
        G[e[0]][e[1]]['color'] = hsv_to_rgb(0,0,0)

        
# Plot
position = {}
color, size = [], []
for n in G:
    position[n] = (G.node[n]['x'], G.node[n]['y'])
    color.append(G.node[n]['color'])
    size.append(G.node[n]['size'])

e_color = []
for e in G.edges():
    e_color.append(G[e[0]][e[1]]['color'])


nx.draw_networkx_nodes(G, with_labels=False,
     pos=position, node_size=size, node_color=color
     )
nx.draw_networkx_edges(G, arrows=False,
     pos=position, edge_color=e_color
     )
plt.show()





# # set positions
# G.position={}
# G.position[0]=(1,1) #trampaaaa
# # fancreateFanPoints(n=1, x_0=0, y_0=0, angle_0=0, spread=120, r=50):
# pos = createFanPoints(len(attr), 1, 1, 0, 120, 100)

# for n in zip(attr, pos):
#     G.position[n[0]] = n[1]

# H = G.subgraph( list(G.predecessors(0)) + [0] )
# nx.draw_networkx(H, with_labels=False, pos=G.position, node_size=20)
# plt.show()


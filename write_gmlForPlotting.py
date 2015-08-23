import networkx as nx
from matplotlib.colors import colorConverter


def write_gmlForPlotting(G, path,
    x='x', y='y', w='w', h='h', typ='type', fill='fill',
    e_fill='fill', e_width='width'
    ):
    """
    Exports G into a plotable gml file.

    Take the graphics arguments defined as attributes of the node/edge and ordered them so that write_gml() can export them correctly as graphics arguments.
    Attributes x, y, w, h, width should be integers
    Fill uses colorConverter.to_rgb(), see the documentation for examples of valid colors.
    Proved with yEd.


    Parameters
    ----------
    G:          networkx graph
    path:       name of file to export

    Node attributes
        x:      attribute that determines position in x
        y:      attribute that determines position in y
        w:      attribute that determines w
        h:      attribute that determines h
        typ:    attribute that determines type
        fill:   attribute that determines fill

    Edge attributes
        e_fill:  attribute that determines e_fill
        e_width: attribute that determines e_width

    Returns
    -------
    path gml archive
    """

    # save node attributes in graphics attribute
    for n in G.nodes():
        if 'graphics' not in G.node[n]: G.node[n]['graphics'] = {}
        if x in G.node[n]: G.node[n]['graphics']['x'] = G.node[n][x]
        if y in G.node[n]: G.node[n]['graphics']['y'] = G.node[n][y]
        if w in G.node[n]: G.node[n]['graphics']['w'] = G.node[n][w]
        if h in G.node[n]: G.node[n]['graphics']['h'] = G.node[n][h]
        if typ in G.node[n]: G.node[n]['graphics']['type'] = G.node[n][typ]
        if fill in G.node[n]: 
            #convert fill to valid hex
            color=colorConverter.to_rgb(G.node[n][fill])
            color='#%02x%02x%02x'%(color[0]*255,color[1]*255,color[2]*255)
            G.node[n]['graphics']['fill'] = color
        # Delete original attributes only after all have been updated
        if x in G.node[n]: del G.node[n][x]
        if y in G.node[n]: del G.node[n][y]
        if w in G.node[n]: del G.node[n][w]
        if h in G.node[n]: del G.node[n][h]
        if typ in G.node[n]:  del G.node[n][typ]
        if fill in G.node[n]: del G.node[n][fill]
        if len(G.node[n]['graphics']) == 0: del G.node[n]['graphics']


    # save edge attributes in graphics attribute
    for s,t in G.edges():
        if 'graphics' not in G[s][t]: G.node[s][t]['graphics'] = {}
        if e_width in G[s][t]: G[s][t]['graphics']['width'] = G[s][t][e_width]
        if e_fill in G[s][t]: 
            #convert fill to valid hex
            color=colorConverter.to_rgb(G.node[s][t][fill])
            color='#%02x%02x%02x'%(color[0]*255,color[1]*255,color[2]*255)
            G[s][t]['graphics']['fill'] = color
        # Delete original attributes only after all have been updated
        if e_width in G[s][t]: del G[s][t][e_width]
        if e_fill in G[s][t]: del G[s][t][e_fill]
        if len(G[s][t]['graphics']) == 0: del G[s][t]['graphics']


    nx.write_gml(G, path)

# # create graph to export

# G = nx.DiGraph()
# G.add_node(1, x=075, y=000, size=60, type="ellipse", color='red')
# G.add_node(2, x=000, y=100, size=60, type="ellipse", color=(0,0,1))
# G.add_node(3, x=150, y=100, size=60, type="ellipse", color="#00FF00")
# G.add_edge(2,1)
# G.add_edge(1,3, width=2)
# G.add_edge(2,3)#, color='black')

# gmlForPlotting(G, 'example.gml',
#     h='size', w='size', fill='color', e_fill='color'
#     )
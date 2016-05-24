def boolnet2networkx(filename, header=True, comment='#'):
    '''
    Takes rules in boolnet format and returns a networkx DiGraph.
    Ignores boolean rules, only returns edges
    '''
    import networkx as nx
    import re

    with open(filename) as f: data = f.read()
    data = data.strip().split('\n')
    if header: data.pop(0) #remove header
    data = [ d for d in data if d[0]!=comment ] #remove commented lines
    rgx = re.compile('[%s]' % '()!|&')
    data = [rgx.sub('', d) for d in data ] #remove boolean operations
    #generate dic of target: [source(s)]
    data = { d.split(',')[0]:d.split(',')[1].split() for d in data }
    G = nx.DiGraph(data) #create DiGraph from dictionary
    G = G.reverse() #correct direction
    return G
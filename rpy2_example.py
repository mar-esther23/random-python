import numpy as np

import rpy2
from rpy2.robjects.packages import importr
boolnet = importr("BoolNet")


nameIn = 'grn.dat'
net = boolnet.loadNetwork(nameIn)
# print net

attr = boolnet.getAttractors(net)
# print attr

# print attr.rx2('attractors')[0].rx2('involvedStates')
names = attr.names
for n in names: 
    for i in range(len(attr.rx2(n))):
        print '\n', n, i
        # print type (attr.rx2(n)[i])
        print list(attr.rx2(n)[i].names)
        for m in attr.rx2(n)[i].names:
            # print type(attr.rx2(n)[i].rx2(m))
            if isinstance(attr.rx2(n)[i].rx2(m), rpy2.rinterface.RNULLType):
                print None
            else: print list(attr.rx2(n)[i].rx2(m))

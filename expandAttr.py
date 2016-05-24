def expandAttr(attr, wildcard='*'):
    '''
    attr is a list of 0, 1, *
    converts * to 0 and 1 and creates new attractors
    '''

    n = attr.count(wildcard)
    expanded = [attr]

    while n > 0:
        add = []
        for e in expanded:
            index = e.index(wildcard)
            e0 = e[:]
            e0[index] = '0'
            e1 = e[:]
            e1[index] = '1'
            add.append(e0)
            add.append(e1)
        expanded += add
        n-=1

    expanded = [e for e in expanded if e.count(wildcard)==0 ]
    return expanded

# with open('minThInsulin-attr-short.csv') as f: data=f.read()
# data = data.strip().split('\n')
# data = [d.strip().split(',') for d in data]


# f = open('minThInsulin-attr.csv','w')
# for d in data:
#     # print d
#     attr = expandAttr(d)
#     for a in attr: f.write(','.join(a) +'\n')
# f.close()
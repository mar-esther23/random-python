#convert to format# id, author, title, journal, year, volume, pages
f = open('ZhuRef.dat', 'r')
ref = []
for line in f:
    data = line.split('.')[:-1]
    data[-1] = data[-1].replace('\xe2\x80\x93','-')
    data = data[0:3] + [''.join(data[3:-1])] + data[-1].replace(';',':').split(':')
    ref.append(data)
f.close()


##export to bibtex
#for r in ref:
    #if r[2].strip().split()[0] != 'A' and r[2].strip().split()[0] != 'The':
        #idref = r[1].split()[0] + r[4] + r[2].strip().split()[0]
        #idref = idref.replace(' ','')
    #else: 
        #idref = r[1].split()[0] + r[4] + r[2].strip().split()[1]
        #idref = idref.replace(' ','')
    #print '\n@article{', idref , ','
    #print 'title={', r[2], '},'
    #print 'author={', r[1], '},'
    #print 'journal={', r[3], '},'
    #print 'year={', r[4], '},'
    #print 'volume={', r[5], '},'
    #print 'pages={', r[6], '},\n}'

    
#crossref
#generate dictionary of article number : bibtex id
crossref = {}
for r in ref:
    if r[2].strip().split()[0] != 'A' and r[2].strip().split()[0] != 'The':
        idref = r[1].split()[0] + r[4] + r[2].strip().split()[0]
        idref = idref.replace(' ','')
    else: 
        idref = r[1].split()[0] + r[4] + r[2].strip().split()[1]
        idref = idref.replace(' ','')
    crossref[r[0]]=idref

    

f = open('crossref.dat', 'r')
for line in f:
    data = line.split()
    for d in data:
        if d == '-': print 'Zhu2010Differentiation',
        else: print crossref[d],
    print
"""
Takes a series of OTUs detemined with diferent identities percentges and compares the counts and taxa determination.
OTUs most be in biom format 
Check the path to OTUs
"""



import biom
import re
import matplotlib.pyplot as plt
from operator import itemgetter



identity = [79, 82, 85, 88, 91]
tables = {}

for iden in identity: 
    with open(  'otus_%d/Data%d/otu_table_L6.biom' % (iden,iden)  ) as f: t = biom.parse_table(f)
    t = [ t.ids(axis='observation'), list(t.sum(axis='observation'))  ]
    t = list(map(list, zip(*t)))
    t = sorted(t, key=itemgetter(1), reverse=True)
    t = list(map(list, zip(*t)))
    tables[iden] = t


#OTU count plots
for iden, tab in tables.items():
    plt.loglog(range(len(tab[1])), tab[1], 'o-', label=str(iden)+'%' )
    plt.legend()
plt.title('Counts per sample')
plt.xlabel('Sample')
plt.ylabel('Count')
plt.show()
plt.close()

for iden, tab in tables.items():
    # print iden, max(tab[1])
    bins =[10**i for i in range(  len(str(int(max(tab[1]))))+1  )]
    plt.hist(tab[1], bins=bins, histtype='step', label=str(iden)+'%')
    plt.legend()
plt.xscale('log', nonposy='clip')
# plt.yscale('log', nonposy='clip')
plt.title('Counts histogram')
plt.xlabel('Samples')
plt.ylabel('Count')
plt.show()
plt.close()


#Taxa determination
for iden, tab in tables.items():
    taxa = []
    for t in tab[0]:
        t = re.sub('[^A-Za-z0-9;_]+', '', t)
        t = t.replace('k__','').replace('p__','').replace('c__','').replace('o__','').replace('f__','').replace('g__','')
        t = t.replace('Other','')
        t = t.split(';')
        t = filter((u'').__ne__, t)
        taxa.append(t)
    l = [len(i) for i in taxa]
    l = sorted(l, reverse=True)
    print iden , l
    plt.hist(l, bins=6, histtype='step', label=str(iden)+'%')
    plt.legend()
plt.title('Taxa histogram')
r = range(1,7)# [.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
l = ['k','p','c','o','f','g']
plt.xticks(r, l, rotation='vertical')
plt.xlabel('Taxa')
plt.ylabel('Samples')
plt.show()
plt.close()

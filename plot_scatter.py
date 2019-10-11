
#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

def plot_xy (x, y, color='k', label='', show=True):
	"""
	Plots pairs of x y arguments as points. Returns plot.
	
	x 		array of x coordinate of points
	y		array of y coordinate of points
	color 	color of points
	label 	label of points
	show	show plot if True
	

	"""
	plt.scatter(x, y, c=color, label=label)
	plt.legend()
	plt.axis('equal') #keeps proportion between axis
	if show: plt.show()
	return plt

def plot_multi_xy (data, colors='k', labels='', show=True):
	"""
	Plots sets of pairs of x y arguments as points. Each set has its own color and label. Returns plot.
	data    array of arrays, contains sets of pairs of x y arrays. 
			[  [[x0],[y0]],  [[x1],[y1]], ...  ]
	color 	array of color of each set of points, default color black
	label 	array of label of each set of points, default label ''
	show	show plot if True
	"""
	colors += ['k'] * (len(data) - len(colors)) #padding colors with default
	labels += [''] * (len(data) - len(labels)) #padding labels with default
	for d in zip(data, colors, labels):
		plt.scatter(d[0][0], d[0][1], c=d[1], label=d[2])
		plt.legend()
		plt.axis('equal') #keeps proportion between axis
	if show: plt.show()
	return plt


N=500
x = np.random.randn(N)
y = np.random.randn(N)
plot_xy(x, y, 'g', 'green')



N=500
data = [ 
	[np.random.randn(N), np.random.randn(N)], 
	[np.random.randn(N), np.random.randn(N)], 
	[np.random.randn(N), np.random.randn(N)], 
	[np.random.randn(N), np.random.randn(N)], 
	[np.random.randn(N), np.random.randn(N)]   ]
c=['g','b', 'r']
l=['a','b','c','d','e','f']

plot_multi_xy (data, c, l)

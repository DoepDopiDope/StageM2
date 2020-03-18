

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from simcado.source import *
from simcado.utils import *
from stagem2.distrib import *


data = np.loadtxt("D:/Users Documents/Documents/Cours/M2 - Astro/Stage/data/kroupa_fig2_2000.txt", delimiter = ";")

kroupx = [data[i][0] for i in range(len(data))]
kroupy = [10**data[i][1] for i in range(len(data))]

plt.scatter(kroupx,kroupy, marker  = "+", color = 'red', label = "Kroupa2000, Fig2, N = 10^6")


##########
number = 1000000
bins = 100  
##########

mass_ranges = [0.01, 0.08, 0.50, 1.0, 1000]
powers = [0.3, 1.3, 2.3, 2.3]


kroup = seg_powerlaw(mass_ranges, powers)

masses = kroup.random(number = number)

logmasses = np.log10(masses)



hist= np.histogram(logmasses, bins = bins)
xvals = [(hist[1][i+1] + hist[1][i])/2 for i in range(len(hist[1])-1)]
yvals = [hist[0][i] for i in range(len(xvals))]
kroupbias = [xvals,yvals]
plt.plot(xvals,np.dot(yvals,1/1000), label = "Kroupa, not corrected for bias")


###############
mass_ranges = [0.01, 0.08, 0.50, 1.0, 1000]
powers = [0.3, 1.8, 2.7, 2.3]


kroup = seg_powerlaw(mass_ranges, powers)

masses = kroup.random(number = number)

logmasses = np.log10(masses)



hist= np.histogram(logmasses, bins = bins)
xvals = [(hist[1][i+1] + hist[1][i])/2 for i in range(len(hist[1])-1)]
yvals = [hist[0][i] for i in range(len(xvals))]
kroupnobias = [xvals,yvals]
plt.plot(xvals,np.dot(yvals,1/1000), label = "Kroupa, corrected for bias")


plt.yscale('log')

plt.ylabel('N(log(M)) * 1/1000')
plt.xlabel('log(Mass)')

plt.title('Comparison of Kroupa2000 and my coded IMF, N = {}'.format(number))
plt.legend()
plt.show()

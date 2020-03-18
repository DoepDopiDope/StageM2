

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from simcado.source import *
from simcado.utils import *
from stagem2.distrib import *



fname = find_file('IMF_1E4.dat')
imf = np.loadtxt(fname)
print(np.min(imf))
print(np.max(imf))

bins = 100
# N = np.histogram(np.log10(imf), bins = bins)

# plt.hist(np.log10(imf), bins = 1000)
# #plt.xscale('log')
# plt.yscale('log')
# plt.show()
number = len(imf)

logmasses = np.log10(imf)
xmin = np.min(logmasses)
xmax = np.max(logmasses)
bins = bins

hist= np.histogram(logmasses, bins = bins)
xvals = [(hist[1][i+1] + hist[1][i])/2 for i in range(len(hist[1])-1)]
yvals = [hist[0][i] for i in range(len(xvals))]
ymin = np.min(yvals)
ymax = np.max(yvals)*5

plt.plot(xvals,yvals, label = 'simcado data')
plt.yscale('log')
plt.xlim((xmin,xmax))
plt.ylim((ymin,ymax))
# plt.hist(logmasses, bins=bins)

# plt.ylabel('log(N/log(M))')
# plt.xlabel('log(Mass)')
# plt.title('simcado_kroupa : Histogram, N = {}'.format(number))
# plt.show()


##########
number = int(1e6)
number = len(imf)
##########

mass_ranges = [0.01, 0.08, 0.50, 1.0, 1000]
powers = [0.3, 1.3, 2.3, 2.3]


kroup = seg_powerlaw(mass_ranges, powers)

masses = kroup.random(number = number)

logmasses = np.log10(masses)

bins = bins

hist= np.histogram(logmasses, bins = bins)
xvals = [(hist[1][i+1] + hist[1][i])/2 for i in range(len(hist[1])-1)]
yvals = [hist[0][i] for i in range(len(xvals))]
plt.plot(xvals,yvals, label = "Kroupa, not corrected for bias")
# plt.hist(logmasses, bins=bins)
plt.yscale('log')
plt.xlim((xmin,xmax))
plt.ylim((ymin,ymax))
plt.ylabel('log(N/log(M))')
plt.xlabel('log(Mass)')
# plt.title('kroup_1 : Randomly generated stars, N = {}'.format(number))
# plt.show()

###############
mass_ranges = [0.01, 0.08, 0.50, 1.0, 1000]
powers = [0.3, 1.8, 2.7, 2.3]


kroup = seg_powerlaw(mass_ranges, powers)

masses = kroup.random(number = number)

logmasses = np.log10(masses)

bins = bins

hist= np.histogram(logmasses, bins = bins)
xvals = [(hist[1][i+1] + hist[1][i])/2 for i in range(len(hist[1])-1)]
yvals = [hist[0][i] for i in range(len(xvals))]
plt.plot(xvals,yvals, label = "Kroupa, corrected for bias")
# plt.hist(logmasses, bins=bins)
plt.yscale('log')
plt.xlim((xmin,xmax))
plt.ylim((ymin,ymax))
plt.ylabel('log(N/log(M))')
plt.xlabel('log(Mass)')
plt.title('Comparison of Simcado and Kroupa IMFs, N = {}'.format(number))
plt.legend()
plt.show()

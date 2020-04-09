

import numpy as np
import scipy.ndimage as spi
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


import simcado as sim

gal = spi.imread("a1.jpg")[:,:,0]

lam, spec = sim.source.SED("M0V", "Ks", 17)
lam, spec = sim.source.scale_spectrum_sb(lam, spec, mag_per_arcsec=17, filter_name="Ks")

src = sim.source.source_from_image(gal, lam, spec, plate_scale=0.004)
im = sim.run(src)
plt.imshow(im[0].data, cmap = "hot")
plt.colorbar()
plt.show()

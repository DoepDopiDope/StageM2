
import simcado as sim
from stagem2.utils import *

filename = "R136.txt"
src = make_simcado_source(filename = filename, distance = 50000)
im = sim.run(src,SCOPE_PSF_FILE="PSF_MCAO.fits",OBS_DIT=300, detector_layout="full")

plot_full_array(im, vmin = 5000, vmax = 50000, savename = filename + ".png", )
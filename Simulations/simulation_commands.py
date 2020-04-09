# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 21:23:51 2020

@author: dddoe
# """


import simcado as sim
from stagem2.utils import *

# filename = "R136.txt"
# src = make_simcado_source(filename = filename, distance = 50000)
# im = sim.run(src,SCOPE_PSF_FILE="PSF_MCAO.fits",OBS_DIT=300, detector_layout="full")

# plot_full_array(im, vmin = 5000, vmax = 50000, savename = filename + ".png", )


# cmd = sim.UserCommands()

# cmd_filename = "D:/Users Documents/Documents/Cours/M2 - Astro/Stage/Simulations/usercomands.txt"

# cmd.writo(cmd_filename)
# cmd = sim.UserCommands("my_cmds.txt")

cmd["OBS_DIT"] = 60  #temps d'observation
cmd["OBS_NDIT"] = 1 #nombre de poses
cmd["OBS_OUTPUT_DIR"] = "./output.fits"  #out filename, or im.writeto("full_detector.fits")
cmd["SCOPE_PSF_FILE"] = "PSF_MCAO.fits"
# cmd["INST_FILTER_TC"]
cmd["SCOPE_PSF_FILE"] = "PSF_MCAO.fits"
cmd["FPA_CHIP_LAYOUT"] = "full"

# simcado.optics.get_filter_set()

# im = sim.run(src, cmds = cmd)
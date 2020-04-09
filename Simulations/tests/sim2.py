

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import simcado
from matplotlib.colors import LogNorm
from simcado import UserCommands
import os
import poppy

#cmds = UserCommands("PSF_SCAO.fits")
#cmds["SCOPE_PSF_FILE"] = filename



#strehl = 0.5
#simcado.psf.poppy_ao_psf(strehl=strehl, mode='wide',  
                                      #pix_res=0.004, size=1024, wavelength=2.2, filename="PSF_strehl.fits")
#fwhm = 0.4 # arcsec

## Creating the PSF and save it to the disk
#simcado.psf.seeing_psf(fwhm=fwhm, psf_type="moffat", size=1024, 
                                    #pix_res=0.004, filename="PSF_seeing.fits")

## Adding keyword WAVELENG to the headers. IMPORTANT for the simulations
#simcado.utils.add_keyword("PSF_seeing.fits", "WAVELENG", 2.2, comment='', ext=0)


src = simcado.source.cluster(mass=10000.0,distance=500000,half_light_radius = 5)
sim = simcado.run(src, OBS_DIT=300)
plt.imshow(sim[0].data, cmap = "gray_r",  norm=LogNorm(vmin=15000, vmax=30000))
plt.show()

exit()

hdu_scao = simcado.run(src, SCOPE_PSF_FILE="PSF_SCAO.fits", INST_FILTER_TC="TC_filter_Ks.dat",
                       OBS_DIT=300)
hdu_strehl = simcado.run(src, SCOPE_PSF_FILE="PSF_strehl.fits", INST_FILTER_TC="TC_filter_Ks.dat",
                         OBS_DIT=300)
hdu_seeing = simcado.run(src, SCOPE_PSF_FILE="PSF_seeing.fits", INST_FILTER_TC="TC_filter_Ks.dat",
                         OBS_DIT=300)
hdu_mcao = simcado.run(src,SCOPE_PSF_FILE="PSF_MCAO.fits", INST_FILTER_TC="TC_filter_Ks.dat",
                         OBS_DIT=300)




fig=plt.figure(figsize=(16,16))
print(np.median(hdu_mcao[0].data), np.max(hdu_mcao[0].data))
#plt.imshow(hdu_mcao[0].data[400:600,400:600], cmap="gray_r", vmin=15000, vmax=116000)

ax1=plt.subplot(221)
ax1.imshow(hdu_scao[0].data, cmap="gray_r", norm=LogNorm(vmin=15000, vmax=30000))
ax1.set_title("with SCAO")

ax2=plt.subplot(222)
ax2.imshow(hdu_strehl[0].data, cmap="gray_r", norm=LogNorm(vmin=15000, vmax=30000))
ax2.set_title("analytical PSF with a Strehl 0.5")


ax3=plt.subplot(223)
ax3.imshow(hdu_seeing[0].data, cmap="gray_r", norm=LogNorm(vmin=15000, vmax=30000))
ax3.set_title("with seeing 0.4 arcsec")

ax4=plt.subplot(224)
ax4.imshow(hdu_mcao[0].data, cmap="gray_r", norm=LogNorm(vmin=15000, vmax=30000))
ax4.set_title("with MCAO")

plt.show()

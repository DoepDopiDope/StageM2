

import numpy as np
import os
import warnings
import astropy.units as u
import astropy.constants as c





def gaussian(number, distance, half_light_radius):
    """
    Returns the star position in the sky as a series of x and y postions (in arcsec)
    This is a copy of the method used in simacdo.source.cluster() function.
    ========
    Input
    - number : number of stars
    - distance : distance to the cluster (in parsec)
    - half_light_radius : hwhm of the cluster (in parsec)
    ========
    """
    distance *= u.pc
    half_light_radius *= u.pc
    hwhm = (half_light_radius/distance*u.rad).to(u.arcsec).value
    sig = hwhm / np.sqrt(2 * np.log(2))
    
    x = np.random.normal(0, sig, number)
    y = np.random.normal(0, sig, number)
    
    return [x,y]

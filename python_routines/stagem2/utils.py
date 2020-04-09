
from astropy.table import Table
import astropy.units as u
import numpy as np
from simcado.source import *
from simcado import utils
from simcado.source import _scale_pickles_to_photons
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import os
import sys
import inspect

__pkg_dir__ = os.path.dirname(inspect.getfile(inspect.currentframe()))



# filename = "eff_25000msol_kroup.txt"
# src = make_simcado_source(filename = filename, distance = 10000)
# im = sim.run(src,SCOPE_PSF_FILE="PSF_MCAO.fits",OBS_DIT=300, detector_layout="full")


def get_table(filename):
    table_data = Table.read(filename, format='ascii')
    return table_data

def get_pos(table):
    x = table["x_[pc]"]
    y = table["y_[pc]"]
    z = table["z_[pc]"]
    return [x,y,z]

def get_vel(table):
    vx = table["vx_[km/s]"]
    vy = table["vy_[km/s]"]
    vz = table["vz_[km(/s]"]
    return [vx,vy,vz]

def get_masses(table):
    masses = table["Mass_[Msun]"]
    return masses


def z_project(pos, distance):
    """
    Projects a population distribution in the (x,y) angle plane, given a distance.
    """
    x = pos[0]
    y = pos[1]
    z = pos[2]
    xproj = (np.arctan(x/(distance + z)) * u.rad).to(u.arcsec).value
    yproj = (np.arctan(y/(distance + z)) * u.rad).to(u.arcsec).value
    
    return [xproj,yproj]

def get_coords(astropy_image):
    """
    Returns the coordinates of a composite SimCADO image
    """
    im = astropy_image
    #List of coordinates of all subimages
    coords = []
    for i in range(len(im)):
        hdr = im[i].header
        # Checking that this contains an image
        if "NAXIS" not in hdr:
            coords.append(None)
            continue
        elif hdr["NAXIS"] == 0:
            coords.append(None)
            continue
        # Coordinates of current image
        coords_list= []
        for j in range(1,hdr["NAXIS"]+1):
            crval = hdr['CRVAL' + str(j)]
            crpix = hdr['CRPIX' + str(j)]
            cdelt = hdr['CDELT' + str(j)]
            #Coords of current image along current axis
            axcoord = [crval + (k-(crpix-1))*cdelt for k in range(hdr["NAXIS"+str(j)])]
            coords_list.append(axcoord)
        coords.append(coords_list)
    
    return coords

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

# crpix = 6304.5
# crval = 90
# cdelt = 1.1111111111111e-06

def plot_full_array(astropy_image, vmin = None, vmax = None, savename = "test.png"):
    """
    Plots a SimCADO composite image consisting of all CCD arrays
    """
    im = astropy_image
    #Removing None coords
    temp_coords = get_coords(im)
    coords = []
    data = []
    for i in range(len(temp_coords)):
        cur = temp_coords[i]
        if cur is not None:
            coords.append(cur)
            data.append(im[i].data)
    
    plt.figure(figsize=(15,15))
    if vmin is None and vmax is None:
      if np.max(data)/np.min(data) > 10000:
        vmax = np.max(data)
        vmin = vmax/10000
    for i in range(len(data)):
        print("Plotting chip {}".format(i))
        x = coords[i][0]
        y = coords[i][1]
        plt.pcolormesh(x,y, data[i], vmin = vmin, vmax = vmax, norm = colors.LogNorm(), cmap = 'inferno')
    plt.colorbar()
    print("Saving figure")
    plt.savefig(savename)


def check_coords(astropy_image):
    """
    Plots the corners of all the CCD arrays of a SimCADO image
    """
    im = astropy_image
    #Removing None coords
    temp_coords = get_coords(im)
    coords = []
    data = []
    for i in range(len(temp_coords)):
        cur = temp_coords[i]
        if cur is not None:
            coords.append(cur)
            data.append(im[i].data)
    xs = []
    ys = []
    test = plt.figure()
    for i in range(len(data)):
        print("Plotting chip {}".format(i))
        x = coords[i][0]
        y = coords[i][1]
        xmin = np.min(x)
        xmax = np.max(x)
        ymin = np.min(y)
        ymax = np.max(y)
        xs.append(xmin)
        xs.append(xmin)
        xs.append(xmax)
        xs.append(xmax)
        ys.append(ymin)
        ys.append(ymax)
        ys.append(ymin)
        ys.append(ymax)
    plt.scatter(xs,ys, marker = "+", color = "red")
    plt.show()

def plot_imf(masses, bins = 100):
    """
    Plots the IMF for a given list of masses
    """
    logmasses = np.log10(masses)
    hist= np.histogram(logmasses, bins = bins)
    xvals = [10**((hist[1][i+1] + hist[1][i])/2) for i in range(len(hist[1])-1)]
    yvals = [hist[0][i] for i in range(len(xvals))]
    plt.plot(xvals,yvals)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('N(log(M))')
    plt.xlabel('Mass')
    plt.show()


def plot_density_profile(table, rmin = 0, rmax = 100, bins = 100, show = False):
    """
    Plots the density profile as a function of radius
    Superposes EFF analytical values. This analytical function is not yet scaled to match rho_0, so heights might differ
    """
    masses = get_masses(table)
    pos = get_pos(table)
    x = pos[0]
    y = pos[1]
    z = pos[2]
    r = [np.sqrt(x[i]**2 + y[i]**2 + z[i]) for i in range(len(x))]
    bins = np.linspace(rmin,rmax, bins+1)
    mean_mass = []
    for i in range(len(bins)-1):
        vol = 4/3 * np.pi*(bins[i+1]**3 - bins[i]**3)
        mass_bin = []
        for j in range(len(masses)):
            if r[j]>= bins[i] and r[j] < bins[i+1]:
                mass_bin.append(masses[j])
        mean_mass.append(np.mean(mass_bin)/vol)
    mid_bin = [(bins[i+1]+bins[i])/2 for i in range(len(bins)-1)]
    plt.plot(mid_bin,mean_mass, label = "MCluster simulated data")
    plt.scatter(mid_bin, EFF_test(mid_bin), marker = "+", color = "red", label = "EFF mass density")
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Radius distance (pc)")
    plt.ylabel("Mean density (Msol/pc^3)")
    plt.legend()
    if show == True:
        plt.show()

def EFF_test(rlist, a = 1, gamma = 2):
    """
    Returns the values of rho density corresponding to the given rlist.
    EFF : rho = rho_0 * (1+ (rlist[i]/a)**2)**(-gamma/2)
    TODO : rho_0 is not yet calculated.
    """
    rho_list = []
    for i in range(len(rlist)):
        rho_list.append((1+ (rlist[i]/a)**2)**(-gamma/2))
    return rho_list
    
def make_simcado_source(filename = None, pos= None, star_masses=None, distance=None):
    """
    Makes a SimCADO source object from a list of stars with position (pos) and star masses (star_masses), and distance.
    If filename is given, will read the parameters in it instead (typically a MCluster output file). Still requires "distance" parameter.
    """
    if filename is not None:
        print("Extracting table from file")
        table = get_table(filename)
        print("Done")
        pos = get_pos(table)
        star_masses = get_masses(table)
    
    xproj,yproj = z_project(pos, distance)
    distances = np.add(pos[2], distance)
    # Assign stellar types to the masses in imf using list of average
    # main-sequence star masses:
    stel_type = [i + str(j) + "V" for i in "OBAFGKM" for j in range(10)]
    masses = _get_stellar_mass(stel_type)
    ref = utils.nearest(masses, star_masses)
    thestars = [stel_type[i] for i in ref] # was stars, redefined function name

    # assign absolute magnitudes to stellar types in cluster
    unique_ref = np.unique(ref)
    unique_type = [stel_type[i] for i in unique_ref]
    unique_Mv = _get_stellar_Mv(unique_type)

    # Mv_dict = {i : float(str(j)[:6]) for i, j in zip(unique_type, unique_Mv)}
    ref_dict = {i : j for i, j in zip(unique_type, np.arange(len(unique_type)))}

    # find spectra for the stellar types in cluster
    lam, spectra = _scale_pickles_to_photons(unique_type)

    # this one connects the stars to one of the unique spectra
    stars_spec_ref = [ref_dict[i] for i in thestars]

    # absolute mag + distance modulus
    m = np.array([unique_Mv[i] for i in stars_spec_ref])
    m += 5 * np.log10(distance) - 5

    # set the weighting
    weight = 10**(-0.4*m)
    
    src = Source(lam=lam, spectra=spectra, x=xproj, y=yproj, ref=stars_spec_ref,
                 weight=weight, units="ph/s/m2")

    src.info["object"] = "cluster"
    src.info["total_mass"] = np.sum(star_masses)
    src.info["masses"] = star_masses
    src.info["half_light_radius"] = None
    src.info["hwhm"] = None
    src.info["distance"] = distance*u.pc
    src.info["stel_type"] = stel_type
    
    return src



def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration >= total: 
        print()

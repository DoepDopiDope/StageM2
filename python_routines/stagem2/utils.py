
from astropy.table import Table
import astropy.units as u
from astropy import constants as const
from astropy.io import ascii
import numpy as np
from simcado.source import *
from simcado import utils
from simcado.source import _scale_pickles_to_photons
import simcado as sim
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import os
import sys
import inspect
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

__pkg_dir__ = os.path.dirname(inspect.getfile(inspect.currentframe()))
stdir = os.path.join(__pkg_dir__, "data")

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
    vz = table["vz_[km/s]"]
    return [vx,vy,vz]

def get_masses(table = None, filename = None):
    if table is None and filename is None:
        raise ValueError("Please give either table or filename as an argument")
    if filename is not None:
        table = fits.open(filename)[1].data
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

def get_coords(astropy_image, relative = False):
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
            if relative == True:
                xmax, xmin= np.max(axcoord[int(len(axcoord)//2)]), np.min(axcoord[int(len(axcoord)//2)])
                xmid = ((xmax + xmin)/2)
                axcoord -= xmid
            coords_list.append(axcoord)
        coords.append(coords_list)
    
    return coords

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def find_nearest_higher(inputData, searchVal):
    diff = inputData - searchVal
    diff[diff<0] = np.inf
    idx = diff.argmin()
    return idx, inputData[idx]

def find_nearest_lower(inputData, searchVal):
    diff = inputData - searchVal
    diff[diff>0] = -np.inf
    idx = diff.argmax()
    return idx, inputData[idx]

# crpix = 6304.5
# crval = 90
# cdelt = 1.1111111111111e-06

def plot_full_array(astropy_image = None,
                    vmin = None,
                    vmax = None,
                    scale = 10000,
                    offset = None,
                    savename = "test.png",
                    filename = None,
                    convert_deg_to_arcsec = True,
                    colormap = 'jet'):
    """
    Plots a SimCADO composite image consisting of all CCD arrays
    """
    if filename is not None:
        im = fits.open(filename)
    else:
        im = astropy_image
    #Removing None coords
    temp_coords = get_coords(im)
    i=0
    data = []
    while i<len(temp_coords):
        if temp_coords[i] is None:
            temp_coords = np.delete(temp_coords,i)
            im = np.delete(im,i)
        else:
            data.append(im[i].data)
            i+=1
    
    #Searching max coords
    x = []
    y = []
    for i in range(len(temp_coords)):
        x.append(temp_coords[i][0])
        y.append(temp_coords[i][1])
    xmax, xmin, ymax, ymin = np.max(x[int(len(x)//2)]), np.min(x[int(len(x)//2)]), np.max(y[int(len(x)//2)]), np.min(y[int(len(x)//2)])
    xmid,ymid = ((xmax + xmin)/2), ((ymax + ymin)/2)

    if convert_deg_to_arcsec == True:
        xmid = (xmid*u.deg).to(u.arcsec).value
        ymid = (ymid*u.deg).to(u.arcsec).value
        # xmax = (xmax*u.deg).to(u.arcsec).value
        # ymax = (ymax*u.deg).to(u.arcsec).value
    # Getting coordinates of each point in each CCD panel
    coords = []
    for i in range(len(temp_coords)):
        cur = temp_coords[i]
        if convert_deg_to_arcsec == True:
            cur = (cur*u.deg).to(u.arcsec).value
        cur[0] -= xmid
        cur[1] -= ymid
        coords.append(cur)
    del temp_coords
    coordmax = np.max([abs(np.max(coords)),abs(np.min(coords))])
        
    
    # figsize = np.dot(100,(6.4,4.8))  # Default is (6.4,4.8)
    # fontsize = np.dot(50,10)        # Default is 10
    # plt.rcParams.update({'font.size': fontsize})
    plt.figure(figsize=(16,12))
    # plt.figure()
    if vmin is None and vmax is None:
        vmax = np.max(data)
        vmin = vmax/scale
    elif vmin is None:
        vmin = vmax/scale
    
    if offset is not None:
        vmax /= offset
        vmin /= offset
    
    for i in range(len(data)):
        print("Plotting chip {}".format(i))
        x = coords[i][0]
        y = coords[i][1]
        plt.pcolormesh(x,y, data[i], vmin = vmin, vmax = vmax, norm = colors.LogNorm(), cmap = colormap)
    
    plt.xlim((-coordmax, coordmax))
    plt.ylim((-coordmax, coordmax))
    plt.xlabel("ΔRA (arcsec)")
    plt.ylabel("ΔDEC (arcsec)")
    plt.colorbar()
    print("Saving figure")
    plt.savefig(savename, dpi = 300)


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


def make_simcado_source(filename = None,
                        pos= None,
                        star_masses = None,
                        distance=8000,
                        mass_lim = 21.695,
                        teff_lim = 35083.26,
                        fileend = "V_scaled.fits",
                        get_spectra = False,
                        get_raw = False,
                        bin_size = None,
                        use_extinction = None,
                        map_extinction_filter = "Ks",
                        target_extinction_filter = "Ks",
                        get_the_position = False,
                        get_the_masses = False):
    """
    Makes a SimCADO source object from a list of stars with position (pos) and star masses (star_masses), and distance.
    If filename is given, will read the parameters in it instead (typically a MCluster output file). Still requires "distance" parameter.
    """
    if filename is not None:
        print("Extracting table from file")
        table = fits.open(filename)[1].data
        pos = get_pos(table)
        star_masses = get_masses(table)
    
    
    # print(np.max(xproj), np.min(xproj))
    # print(np.max(yproj), np.min(yproj))
    sim_masses = []
    sim_pos = [[],[],[]]
    bt_masses = []
    bt_pos = [[],[],[]]
    if pos is None:
        x = [0 for i in range(len(star_masses))]
        y = [0 for i in range(len(star_masses))]
        z = [0 for i in range(len(star_masses))]
        pos = [x,y,z]
    for i in range(len(star_masses)):
        mass = star_masses[i]
        if mass > mass_lim:
            sim_masses.append(mass)
            x = pos[0][i]
            y = pos[1][i]
            z = pos[2][i]
            sim_pos[0].append(x)
            sim_pos[1].append(y)
            sim_pos[2].append(z)
        else:
            bt_masses.append(mass)
            x = pos[0][i]
            y = pos[1][i]
            z = pos[2][i]
            bt_pos[0].append(x)
            bt_pos[1].append(y)
            bt_pos[2].append(z)
    #
    # Building spectra in SimCADO way
    #
    stel_type = [i + str(j) + "V" for i in "OBAFGKM" for j in range(10)]
    masses = _get_stellar_mass(stel_type)
    ref = utils.nearest(masses, sim_masses)
    thestars = [stel_type[i] for i in ref] # was stars, redefined function name

    # assign absolute magnitudes to stellar types in cluster
    unique_ref = np.unique(ref)
    unique_type = [stel_type[i] for i in unique_ref]
    unique_Mv = _get_stellar_Mv(unique_type)

    # Mv_dict = {i : float(str(j)[:6]) for i, j in zip(unique_type, unique_Mv)}
    ref_dict = {i : j for i, j in zip(unique_type, np.arange(len(unique_type)))}

    # find spectra for the stellar types in cluster
    lam, spectra = _scale_pickles_to_photons(unique_type)
    lensimspectra = len(spectra)
    # this one connects the stars to one of the unique spectra
    stars_spec_ref = [ref_dict[i] for i in thestars]
    
    
    #
    # Building BTSettl spectra
    #
    # Reading spectra from data
    print("Reading BT-Settl spectra")
    spectra_path = stdir + '/spectra/bt-settl'
    spectra_bt, spectra_teff = [], []
    iteration = 0
    for file in os.listdir(spectra_path):
        printProgressBar(iteration, len(os.listdir(spectra_path))-1)
        iteration+=1
        if file.endswith(fileend):
            # print("--------------------")
            # print("Reading spectra '{}'".format(file))
            teff = int(file[3:6]) *100
            if teff > teff_lim:
                continue
            else:
                spectra_teff.append(teff)
            
            spec = fits.open(spectra_path + '/' +file)[1].data
            
            # Getting lambda and flux, also removing lambda = 0
            this_lam = np.delete(spec["lambda (um)"], 0)
            this_spectra = np.delete(spec["Flux (erg/cm2/s/um)"],0)
            if get_raw == True:
                # Rebining spectra to the same resolution as simcado ones
                spec_fun = interp1d(this_lam, this_spectra)
                spec_val = spec_fun(lam)
                
                spectra_bt.append(spec_val)
            else:
                # Rebining spectra to the same resolution as simcado ones
                spec_fun = interp1d(this_lam, this_spectra)
                this_spectra = spec_fun(lam)
                this_lam = lam
                
                
                # Converting /cm2 to /m2
                this_spectra = np.dot(this_spectra, 1e4)
                
                
                # Converting erg to Joules
                this_spectra = (this_spectra * u.erg).to(u.joule).value
                
                
                # E = h*nu, energy per photon, so dividing each value per E returns ph/s/m2/um
                E = const.h.value * const.c.value / ((this_lam*u.um).to(u.m).value)
                
                
                this_spectra = np.multiply(1/E, this_spectra)
                # Spectra is in ph/s/m2/um
                
                
                # Multiplying by wavelength return ph/s/m2
                # In case of multiplying by size of the bin instead of the wavelength
                if bin_size is None:
                    dlam = (this_lam[1:] - this_lam[:-1])
                    dlam = np.append(dlam,dlam[-1])
                    # print("Resolution = {}".format(dlam[10]))
                    
                    for i in range(1,len(dlam)):
                        if dlam[i] == 0:
                            dlam[i] = dlam[i-1]
                            
                    this_spectra = np.multiply(this_spectra, dlam)
    
                else:
                    this_spectra = np.dot(this_spectra, bin_size)
                
                # Adding processed spectra to the list
                spectra_bt.append(this_spectra)
            
    
    # Associating each mass to an effective temperature
    mass, T_eff, V_list = get_param_lists_separation_btsettl_parsec()
    mag_fun = interp1d(T_eff, V_list)
    mass_to_teff = interp1d(mass,T_eff)
    associated_teff = mass_to_teff(bt_masses)
    # final_teff = utils.nearest(spectra_teff, associated_teff)
    
    
    # Making the ref list
    ref = [find_nearest(spectra_teff, i)[0]+len(spectra) for i in associated_teff]
    print("Adding BTSettl spectra to the list")
    if len(spectra)!=0:
        spectra = np.append(spectra,spectra_bt, axis = 0)
        stars_spec_ref = np.append(stars_spec_ref, ref)
    else:
        spectra = spectra_bt
        stars_spec_ref = ref
    
    # for i in range(len(spectra)):
    #     plt.plot(lam,spectra[i])
    # plt.yscale("log")
    # plt.show()
    
    
    # Positions
    if sim_pos is None:
        pos = bt_pos
    elif bt_pos is None:
        pos = sim_pos
    else:
        pos = np.append(sim_pos, bt_pos, axis = 1)
    xproj,yproj = z_project(pos, distance)
    distances = np.add(pos[2], distance)
    
    if use_extinction is not None:
        # Loading extinction map
        print("Computing extinction for each star")
        ext = fits.open(use_extinction)
        ext_val = ext[1].data
        ext_coords = ext[2].data
        x = [ext_coords[0][i][0] for i in range(len(ext_coords[0]))]
        y = [ext_coords[i][0][1] for i in range(len(ext_coords))]
        
        
        # Loading Rieke law
        rieke_law = ascii.read(stdir + "/" + "rieke_extinction_law.dat")
        fr = rieke_law["Filter_name"]
        ar = rieke_law["A_lambda/Av"]
        rieke = {fr[i]:ar[i] for i in range(len(fr))}
        
        
        # Computing factor by which to rescale the extinction
        # The Arches map extinction map is in units of A_Ks
        # Thus we multiply the whole map by (A_filter/A_V) * (A_V/A_Ks), ie : rieke["filter"] / rieke["Ks"]
        ext_factor = rieke[target_extinction_filter] / rieke[map_extinction_filter]
        print("Extinction factor, {} to {} : {}".format(map_extinction_filter, target_extinction_filter, ext_factor))
        ext_val = np.dot(ext_val, ext_factor)
        
        
        # Building the list of extinction for each star, depending on its position
        ext_fun = interp2d(x,y,ext_val)
        extinctions = []
        xmin = np.min(xproj)
        xmax = np.max(xproj)
        ymin = np.min(yproj)
        ymax = np.max(yproj)
        for i in range(len(xproj)):
            printProgressBar(i, len(xproj)-1)
            x = xproj[i]
            y = yproj[i]
            if x < xmin or x > xmax or y < ymin or y > ymax:
                extinctions.append(0)
            else:
                extinctions.append(ext_fun(x,y)[0])
        
        
        
        
    # set the weighting
    
    
    print("Computing apparent magnitudes")
    dm = 5 * np.log10(distances) - 5
    # Adding extinction if asked
    if use_extinction is not None:
        print("Adding extinction to apparent magnitude")
        dm += extinctions
    weight = 10**(-0.4*dm)
    
    
    if get_spectra == True:
        print("There are {} SimCADO Spectra. Rest is BTSettl".format(lensimspectra))
        for mass, fref in zip(star_masses, stars_spec_ref):
            print("Mass = {} Msol, spectra Teff = {} K".format(mass,spectra_teff[fref]))
        spectra = [spectra[stars_spec_ref[i]] for i in range(len(stars_spec_ref))]
        return lam, spectra
    
    if get_the_position == True:
        return xproj,yproj
    if get_the_masses == True:
        return np.append(sim_masses,bt_masses)
    lam = np.asarray(lam)
    spectra = np.asarray(spectra)
    src = Source(lam=lam, spectra=spectra, x=xproj, y=yproj, ref=stars_spec_ref,
                 weight=weight, units="ph/s/m2")


    src.info["object"] = "cluster"
    src.info["total_mass"] = np.sum(star_masses)
    src.info["masses"] = star_masses
    src.info["half_light_radius"] = None
    src.info["hwhm"] = None
    src.info["distance"] = distance*u.pc
    # src.info["stel_type"] = stel_type
    
    return src

 
def make_simcado_source_old(filename = None, pos= None, star_masses=None, distance=None, get_spectra = False):
    """
    Makes a SimCADO source object from a list of stars with position (pos) and star masses (star_masses), and distance.
    If filename is given, will read the parameters in it instead (typically a MCluster output file). Still requires "distance" parameter.
    """
    if filename is not None:
        print("Extracting table from file")
        table = fits.open(filename)[1].data
        pos = get_pos(table)
        star_masses = get_masses(table)
    
    if pos is not None:
        xproj,yproj = z_project(pos, distance)
        distances = np.add(pos[2], distance)
    # print(np.max(xproj), np.min(xproj))
    # print(np.max(yproj), np.min(yproj))
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
    if get_spectra == True:
        spectra = [spectra[stars_spec_ref[i]] for i in range(len(stars_spec_ref))]
        print(ref_dict)
        print(stars_spec_ref)
        print("Magnitude : ", unique_Mv)
        return lam, spectra
    # return lam, spectra, stars_spec_ref, weight, masses, m
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



def get_btsettl_V_mag(filename = "BTSettl_VEGA_t0.003.txt"):
    isodir =stdir + "/isochrones"
    btsettlfilename = os.path.join(isodir ,filename)
    table_data = Table.read(btsettlfilename, format='ascii.basic')
    m = table_data["M/Ms"]
    teff= table_data["Teff(K)"]
    v = table_data["V"]
    return m, teff, v

def get_parsec_V_mag(filename = "parsec.txt"):
    isodir =stdir + "/isochrones"
    btsettlfilename = os.path.join(isodir ,filename)
    table_data = Table.read(btsettlfilename, format='ascii.basic')
    m = table_data["Mass"]
    teff= 10**(table_data["logTe"])
    v = table_data["Vmag"]
    return m, teff, v

def get_btsettl_mag(filename = "BTSettl_VEGA_t0.003.txt", filter = "V"):
    isodir =stdir + "/isochrones"
    btsettlfilename = os.path.join(isodir ,filename)
    table_data = Table.read(btsettlfilename, format='ascii.basic')
    m = table_data["M/Ms"]
    teff= table_data["Teff(K)"]
    v = table_data[filter]
    return m, teff, v

def get_parsec_mag(filename = "parsec.txt", filter = "V"):
    isodir =stdir + "/isochrones"
    btsettlfilename = os.path.join(isodir ,filename)
    table_data = Table.read(btsettlfilename, format='ascii.basic')
    m = table_data["Mass"]
    teff= 10**(table_data["logTe"])
    v = table_data["{}mag".format(filter)]
    return m, teff, v


def get_param_lists_separation_btsettl_parsec(mass_lim = 0.8, filter = "V"):
    if filter == "Ks":
        filter = "K"
    # Get list of magnitude, depending on mass / effective temperature
    m_bt, T_effbt, V_bt = get_btsettl_mag(filter = filter)
    m_pars, T_effpars, V_pars = get_parsec_mag(filter = filter)
    
    # We only use parsec isochrone up to Mass=21.695, after which Teff starts going down
    # This corresponds to idx = 176, Teff = 35083.26, Mass = 21.695
    # After this, we will use SimCADO's way of building spectra
    idx = 176
    m_pars  = m_pars[:idx+1]
    T_effpars = T_effpars[:idx+1]
    V_pars = V_pars[:idx+1]
    
    # We need to define a limit mass or effective temperature from which we use Parsec instead of BTSettl isochrones
    # We define it as when Mv(Parsec) > Mv(BTSettl)
    # We use the transition mass : m = 0.8 Msol
    # We thus buid the lists mass, T_eff, V_list
    mass = []
    T_eff = []
    V_list = []
    for i in range(len(m_bt)):
        if m_bt[i] > mass_lim:
            break
        else:
            mass.append(m_bt[i])
            T_eff.append(T_effbt[i])
            V_list.append(V_bt[i])
    for i in range(len(m_pars)):
        if m_pars[i]<mass_lim:
            continue
        else:
            mass.append(m_pars[i])
            T_eff.append(T_effpars[i])
            V_list.append(V_pars[i])
    
    return mass, T_eff, V_list

def get_full_isochrone(mass_lim = 0.8, filter = "V"):
    if filter == "Ks":
        filter = "K"
    # Get list of magnitude, depending on mass / effective temperature
    m_bt, T_effbt, V_bt = get_btsettl_mag(filter = filter)
    m_pars, T_effpars, V_pars = get_parsec_mag(filter = filter)
    
    # We only use parsec isochrone up to Mass=21.695, after which Teff starts going down
    # This corresponds to idx = 176, Teff = 35083.26, Mass = 21.695
    # After this, we will use SimCADO's way of building spectra
    idx = 176
    m_pars  = m_pars[:idx+1]
    T_effpars = T_effpars[:idx+1]
    V_pars = V_pars[:idx+1]
    
    # We need to define a limit mass or effective temperature from which we use Parsec instead of BTSettl isochrones
    # We define it as when Mv(Parsec) > Mv(BTSettl)
    # We use the transition mass : m = 0.8 Msol
    # We thus buid the lists mass, T_eff, V_list
    mass = []
    T_eff = []
    V_list = []
    for i in range(len(m_bt)):
        if m_bt[i] > mass_lim:
            break
        else:
            mass.append(m_bt[i])
            T_eff.append(T_effbt[i])
            V_list.append(V_bt[i])
    for i in range(len(m_pars)):
        if m_pars[i]<mass_lim:
            continue
        else:
            mass.append(m_pars[i])
            T_eff.append(T_effpars[i])
            V_list.append(V_pars[i])
    
    # Adding  SimCADO masses to the list
    
    return mass, T_eff, V_list



def draw_simcado_btsettl_spectra(list_masses = [0.01, 0.1, 1, 10],
                                 get_spectra = True,
                                 fileend = "scaled.fits",
                                 bin_size = None):
    lamsim, specsim = make_simcado_source_old(star_masses = list_masses, distance = 8000, get_spectra = get_spectra)
    lambt, specbt = make_simcado_source(star_masses = list_masses, distance = 8000, get_spectra = get_spectra, bin_size = bin_size, fileend = fileend)
    plt.figure(figsize = (16,12))
    for i in range(len(specsim)):
        plt.plot(lamsim, specsim[i], label = "Simcado, mass = {}".format(list_masses[i]))
        plt.plot(lambt,specbt[i], label = "BTSettl, mass = {}".format(list_masses[i]))
    plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel("lambda (um)")
    plt.ylabel("Flux (ph/s/m2)")
    plt.legend()
    plt.show()





def compare_mag_simcado_btsettl_parsec(apparent = False, distance = None):
    # Simcado
    themasses = np.linspace(0.01,200,4000)
    stel_type = [i + str(j) + "V" for i in "OBAFGKM" for j in range(10)]
    masses = _get_stellar_mass(stel_type)
    ref = utils.nearest(masses, themasses)
    thestars = [stel_type[i] for i in ref] # was stars, redefined function name
    # assign absolute magnitudes to stellar types in cluster
    unique_ref = np.unique(ref)
    unique_type = [stel_type[i] for i in unique_ref]
    unique_Mv = _get_stellar_Mv(unique_type)
    # Mv_dict = {i : float(str(j)[:6]) for i, j in zip(unique_type, unique_Mv)}
    ref_dict = {i : j for i, j in zip(unique_type, np.arange(len(unique_type)))}
    # this one connects the stars to one of the unique spectra
    stars_spec_ref = [ref_dict[i] for i in thestars]
    # absolute mag + distance modulus
    Mv = np.array([unique_Mv[i] for i in stars_spec_ref])
    
    #BTSettl
    mbt, teff, Mvbt = get_btsettl_V_mag()
    
    mpars, teffpars, Mvpars = get_parsec_V_mag()
    
    
    plt.plot(themasses, Mv, label = "Simcado")
    plt.plot(mbt, Mvbt, label = "BTSettl")
    plt.plot(mpars, Mvpars, label = "Parsec")
    plt.xscale("log")
    plt.xlabel("Mass")
    plt.ylabel("Mv")
    plt.legend()
    plt.show()




def compute_magnitude(lam, spectra, filter_name = "V"):
    
    zeropoints = {"V": 3.619e-9} #erg/s/cm^2/angstrom
    
    # Getting filter data
    filter_data = sim.optics.get_filter_curve(filter_name)
    filter_lam = filter_data.lam
    filter_val = filter_data.val
    
    lam_min = np.min(filter_lam) # wavelength in um
    lam_max = np.max(filter_lam)
    
    
    # Raising error if spectra is out of filter known limits
    if lam_min> np.max(lam) or lam_max < np.min(lam):
        raise ValueError("spectra wavelength range is out of filter range")
    
    
    # Filter Function
    filter_fun = interp1d(filter_lam, filter_val)
    
    
    # Zero-point flux
    zeropoint = zeropoints[filter_name]
    zeropoint = zeropoint / ((u.angstrom).to(u.um)) #erg/s/cm^2/um
    zeropoint_flux = np.dot(filter_val, zeropoint) # Convolving zeropoint with the filter
    zeropoint_flux = np.multiply(zeropoint_flux, filter_lam) # Multiplying by lambda
    flux_ref = np.trapz(zeropoint_flux, filter_lam) # Integrating over wavelength
    
    
    # Limits
    spec_min = np.min(lam)
    spec_max = np.max(lam)
    if spec_min > lam_min:
        idxmin = 0
    else:
        idxmin = find_nearest_higher(lam,lam_min)[0]
    if spec_max < lam_max:
        idxmax = len(spectra) -1
    else:
        idxmax = find_nearest_lower(lam,lam_max)[0]
    

    spectra = spectra[idxmin:idxmax +1]
    lam = lam[idxmin:idxmax+1]

    
    # Spectra flux
    # Applying filter
    filter_values_spectra = filter_fun(lam)
    spectra = np.multiply(spectra,filter_values_spectra)
    
    # Multiplying by lambda
    spectra = np.multiply(spectra,lam)
    
    # Integrating
    flux_spectra = np.trapz(spectra,lam)
    
    
    # Computing magnitude
    mag = -2.5 * np.log10(flux_spectra / flux_ref)
    return mag
    
    
    
    

def compute_limiting_mags(filter_names, exptimes = None, cmds = None, limiting_sigma= 5):
    if exptimes is None:
        exptimes = np.linspace(1800,36000,51)
    vals = sim.simulation.limiting_mags(exptimes = exptimes, filter_names = filter_names, cmds = cmds, limiting_sigma = limiting_sigma)
    return exptimes, vals



def plot_limiting_mags(filter_names, exptimes, vals, limiting_sigma = 5, pixscale = 4.0):
    plt.figure(figsize=(9.6,7.2))
    for i in range(len(filter_names)):
        name = filter_names[i]
        mags = vals[i]
        plt.plot(exptimes, mags, label = "Filter {}".format(name))
        plt.xlabel("Exptimes (s)")
        plt.ylabel("Limiting magnitude")
    plt.title("Limiting magnitude at {} sigmas for various filters, {}mas per pixel".format(limiting_sigma, pixscale))
    plt.legend()
    plt.show()
    plt.close()
    
    
    
    

    
def plot_filter_saturation(filters = ["H"], exptimes = [1, 10]):
    limcount = 65000        # Saturation limit
    distance = 8000         # Distance to source
    maxpsf = 0.032          # Maximum value on the PSF
    DM = 5*np.log10(8000)-5
    for i in range(len(filters)):
        fname = filters[i]
        if fname == "K":
            fname = "Ks"
        fname_simcado = "TC_filter_{}.dat".format(fname)
        print("")
        print("")
        zp = sim.simulation.zeropoint(filter_name = fname_simcado)
        if fname == "Ks":
            fname = "K"
        mbt, t, mag_bt = get_btsettl_mag(filter = fname)
        mpars, t, mag_pars = get_parsec_mag(filter = fname)
        
        maglim10s = -2.5*np.log10((limcount/maxpsf)/10) + zp
        maglim1s= -2.5*np.log10((limcount/maxpsf)/1) + zp
        
        mag_bt_app = mag_bt + DM
        mag_pars_app = mag_pars + DM
        
        # Finding corresponding mass into Parsec isochrone
        fun = interp1d(mag_pars_app, mpars)
        mlim10s = fun(maglim10s)
        mlim1s = fun(maglim1s)
        
        plt.figure(figsize=(12,9))
        
        # Plotting absolute mags
        plt.plot(mpars, mag_pars, label="parsec, absolute magnitude")
        plt.plot(mbt ,mag_bt, label="btsettl, absolute magnitude")
        
        # Plotting apparent mags
        plt.plot(mpars, mag_pars_app, label="parsec, apparent magnitude")
        plt.plot(mbt, mag_bt_app, label="btsettl, apparent, magnitude")
        
        # Plotting saturation mags
        plt.hlines(maglim10s, 0, np.max(mpars), label = "exp = 10s, mag = {:.2f}, Mass = {:.2f}".format(maglim10s, mlim10s), color = "blue")
        plt.hlines(maglim1s, 0, np.max(mpars), label = "exp = 1s, mag = {:.2f}, Mass = {:.2f}".format(maglim1s, mlim1s), color = "red")
        
        # Graph layout
        plt.legend()
        plt.xlabel("Mass")
        plt.xscale("log")
        plt.ylabel("{} Magnitude".format(fname))
        plt.title("Saturation limit in filter : {}".format(fname))

 
    

def my_hist(data, bins, bins_scale = "linear", prod_offset = None, label = None, ax = None, div_binsize = False, **kwargs):
    if ax is None:
        ax = plt.gca()
    
    if bins_scale == "log":
        da = np.log10(data)
        if type(bins) is not int:
            bins = np.log10(bins)
        hist, thebins = np.histogram(da, bins = bins)
        thebins = 10**thebins
    else:
        hist, thebins = np.histogram(data, bins = bins)
    
    
    if prod_offset is not None:
        if prod_offset == "sum":
            # hist = np.dot(hist,1/np.sum(np.dot(hist, [np.log10(thebins[i+1]) - np.log10(thebins[i]) for i in range(len(thebins)-1)])))
            hist = np.dot(hist, 1/np.sum(hist))
        else:
            hist = np.dot(hist, prod_offset)
    
    # print(np.sum(hist))
    if div_binsize == True:
        if bins_scale == "log":
            hist = [(hist[i]/((np.log10(thebins[i+1]) - np.log10(thebins[i])))) for i in range(len(hist))]
        else:
            hist = [(hist[i]/((thebins[i+1] - thebins[i]))) for i in range(len(hist))]
    
    for i in range(len(hist)-1):
        xy = (thebins[i], 0)
        width = thebins[i+1] - thebins[i]
        height = hist[i]
        rect = Rectangle(xy, width, height, **kwargs)
        ax.add_patch(rect)
    xy = (thebins[-1], 0)
    width = thebins[-1] - thebins[-2]
    height = hist[-1]
    rect = Rectangle(xy, width, height, label = label, **kwargs)
    ax.add_patch(rect)
    
    return hist, thebins
    
    
    
    
def adu_to_mag(counts,texp, filtername):
    zps = {'H': 29.491, 'J': 29.491, 'I': 29.491, 'Ks': 29.491}
    if filtername not in zps:
        zp = sim.simulation.zeropoint(filter_name='TC_filter_{}.dat'.format(filtername))
    else:
        zp = zps[filtername]
    
    mag = -2.5*np.log10(counts/texp) + zp
    return mag
    
    
def hosek_extinction(xpos,ypos, target_filter, each_coord = False, get_ratio_Ks = False):
    # Loading extinction map
    ext = fits.open(stdir +"/arches_Ks_extinction.fits")
    ext_val = ext[1].data
    ext_coords = ext[2].data
    x = [ext_coords[0][i][0] for i in range(len(ext_coords[0]))]
    y = [ext_coords[i][0][1] for i in range(len(ext_coords))]
    
    # Loading Rieke law
    rieke_law = ascii.read(stdir + "/" + "rieke_extinction_law.dat")
    fr = rieke_law["Filter_name"]
    ar = rieke_law["A_lambda/Av"]
    rieke = {fr[i]:ar[i] for i in range(len(fr))}
    
    if get_ratio_Ks == True:
        print("Ks:", rieke["Ks"] / rieke["Ks"])
        print("H:", rieke["H"] / rieke["Ks"])
        print("J:", rieke["J"] / rieke["Ks"])
        print("I:", rieke["I"] / rieke["Ks"])
        return 0
    # Computing factor by which to rescale the extinction
    # The Arches map extinction map is in units of A_Ks
    # Thus we multiply the whole map by (A_filter/A_V) * (A_V/A_Ks), ie : rieke["filter"] / rieke["Ks"]
    ext_factor = rieke[target_filter] / rieke["Ks"]
    # print("Extinction factor, {} to {} : {}".format("Ks", target_filter, ext_factor))
    ext_val = np.dot(ext_val, ext_factor)
    
    
    # Building the list of extinction for each star, depending on its position
    ext_fun = interp2d(x,y,ext_val, bounds_error = False, fill_value = 0)
    
    # if len(extinctions) == 1:
    #     extinctions = ext_fun(xpos,ypos)
    #     extinctions = extinctions[0]
    #     return extinctions
    
    
    if hasattr(xpos, "__iter__") and hasattr(ypos, "__iter__"):
        if each_coord == True:
            extinctions = []
            for x,y in zip(xpos,ypos):
                ext = ext_fun(x,y)
                extinctions.append(ext[0])
        else:
            extinctions = ext_fun(xpos,ypos)
    else:
        extinctions = ext_fun(xpos,ypos)[0]
    
    return extinctions

def pixels_to_arcsec(x):
    CRPIX = 6144.5
    CRDELT  =  4.1666666666667E-07 * 3600
    CRVAL = 0
    val = np.dot(np.subtract(x, CRPIX), CRDELT)
    return val









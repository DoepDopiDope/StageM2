

import stagem2.distrib as dist
import stagem2.utils as ut
import numpy as np
from astropy.table import Table
from os.path import join
stdatadir = join(ut.__pkg_dir__, "data")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time




    
def cluster(masses = None,
             pos = None,
             pos_labels = None,
             imf_type = None,
             imf_args = {},
             pos_type = "EFF",
             pos_args = {},
             total_mass = None,
             number = None,
             binary = None,
             segregation = None,
             *args, **kwargs):
    
    start = time.time()
    if masses is None:
        if imf_type is None:
            raise ValueError("imf_type should not be None if masses is not provided")
        if total_mass is None and number is None:
            raise ValueError("Please provide one of these arguments : total_mass / number. ")
        print("Generating stars")
        if number is not None:
            masses = eval("dist.{}(**imf_args).rand(number)".format(imf_type))
        elif total_mass is not None:
            masses = eval("dist.{}(**imf_args).mass_population(total_mass)".format(imf_type))
        else:
            raise ValueError("Didn't succeed to make masses list in etiher the 'number' or 'masses' way")
    
    if binary is not None:
        # Pair stars according to Lu2013 Fig 16 with only binary or single systems
        print("___________________________")
        print("Making binary systems")
        masses = np.sort(masses)[::-1]
        systems = make_binary_systems(masses)
    else:
        masses = [[masses[i],None] for i in range(len(masses))]
    
    if pos is None:
        # Generate position according
        print("___________________________")
        print("Computing position")
        masses, pos, multiplicity, linked_to = make_position(systems, pos_args, pos_type, segregation = segregation)
            
    cl = Table([masses, pos[0], pos[1], pos[2], multiplicity, linked_to], names = ("Mass_[Msun]", "x_[pc]", "y_[pc]", "z_[pc]", "Multiplicity", "Linked_to"))
    end = time.time()
    print("Elapsed time : {:.2f} seconds".format(end - start))
    return cl

#Arches
pos_args = {"a":0.13,
            "gamma":2.3,
            "rmax":3}
"mass = 2.44e4 Msol"
"distance = 8000 pc"

#R136
pos_args = {"a":0.025,
            "gamma":1.85,
            "rmax":10}
"mass = 1.03e5 Msol"
"distance = 50000 pc"


def make_arches():
    pos_args = {"a":0.13,
            "gamma":2.3,
            "rmax":3}
    mass = 2.44e4
    distance = 8000
    cl = cluster(imf_type = "kroupa",
                 pos_type = "EFF",
                 pos_args = pos_args,
                 total_mass = mass,
                 binary = 1,
                 segregation = 0.3)
    return cl, distance

def make_r136():
    pos_args = {"a":0.025,
            "gamma":1.85,
            "rmax":10}
    mass = 1.03e5
    distance = 50000
    cl = cluster(imf_type = "kroupa",
                 pos_type = "EFF",
                 pos_args = pos_args,
                 total_mass = mass,
                 binary = 1,
                 segregation = 0.3)
    return cl, distance

def make_position(systems, pos_args, pos_type = "EFF", segregation = None):
    """
    Compute 3D coordinates of the input systems.
    """
    eff = eval("dist.{}(**pos_args)".format(pos_type))
    print("Computing distance to center of each system")
    radius = []
    for i in range(len(systems)):
        radius.append(eff.random())
    
    #If segregation parameter is not None, mass-segregate the cluster
    if segregation is not None:
        print("Mass segregating the cluster")
        systems, radius = mass_segregation(systems, radius, s = segregation)
        
    print("Computing xyz coordinates of each system")
    pos = dist.make_coordinates(radius)
    print("Computing xyz coordinates of binaries")
    masses, pos, multiplicity, linked_to = dist.make_system_pos(systems, pos)
    return masses, pos, multiplicity, linked_to

def mass_segregation(systems, radius, s):
    """
    Make a mass-segregated list of radius as a function of mass, in a similar way as in MCluster
    """
    
    # Sorting radius
    radius = np.sort(radius)
    # Sorting systems by decreasing total mass
    totmass = []
    for i in range(len(systems)):
        if systems[i][-1] is None:
            totmass.append(systems[i][0])
        else:
            lenmax = len(systems[i])
            totmass.append(sum([systems[i][k] for k in range(lenmax -1)]))
    sorted_systems_idx = np.argsort(totmass)[::-1]
    
    out_radius = []
    out_systems = []
    N = len(sorted_systems_idx)
    for i in range(N):
        X = np.random.uniform()
        idx = int((N - i) * (1 - X**(1-s)))  # Randomizing the radius-index of current mass
        count = 0
        for j in range(len(radius)):    # i-th system takes the place of the idx-th first available radius
            if count == idx:
                out_radius.append(radius[j])
                out_systems.append(systems[sorted_systems_idx[i]])
                radius = np.delete(radius, j)
                ut.printProgressBar(i,N-1)
                break
            else:
                count +=1
    if np.nan in out_radius:
        print("   ")
        print("HAHAHAHAHAHAHAHAHAHA")
    return out_systems, out_radius
    
    
    
    

def make_binary_systems(masses):
    """
    Make paired systems. Computing pairing probability, mass ratio and separation.
    Returns : systems = [mass1, mass2, separation(au)]
    """
    print("Computing mass ratios and separation functions")
    binarfun = mass_ratio_and_separation_cluster()  #Mass ratio and separation function
    
    print("Computing pairing probability, separations, mass ratios, and paired systems")
    final_systems = []
    number = 0
    n=0
    lentot = len(masses)
    while len(masses) >= 2:
        n+=1
        mass = masses[0]
        
        # Pairing probability according to Lu+2013 Mass Fraction
        prob = pairing_probability(mass)
        randomize = np.random.uniform(0.0, 1.0)
        
        # If randomize > prob, no pairing, single-star system
        if randomize > prob:
            system = [mass]
            to_delete = [0]
            separation = None
        # If randomize <= prob, pairing, binary system
        else:
            n+=1
            number +=1
            # Separation method
            mratio,separation = binarfun.random(mass)
            masses = np.delete(masses, 0)
            paired_idx, paired_mass = ut.find_nearest(masses, mass*mratio)
            to_delete = [paired_idx]
        
        #Make system
        system = [mass, paired_mass, separation]
        
        # Add system to list of systems
        final_systems.append(system)
        
        # Remove paired star from list of stars
        masses = np.delete(masses, to_delete)
        
        ut.printProgressBar(n,lentot)
    # If there is a remaining single star, add it to final_systems as a single-star system
    if len(masses) == 1:
        n+=1
        final_systems.append([masses[0], None])
        ut.printProgressBar(n,lentot)
    
    print("Binary systems : {}".format(number))
    return final_systems
        
            
class separation:
    """
    Separation class, Used to compute separation and mass ratio of a binary system, through the 'random' method.
    """
    def __init__(self):
        self.duch = dist.sep_low_duchene()
        self.wadu = dist.ward_duong()
        self.raga = dist.raghavan()
    
    def random(self, mass):
        if mass < 0.1:
            val = self.duch.random()
        elif mass >= 0.1 and mass < 0.7:
            val = self.wadu.random()
        else:
            val = self.raga.random()
        
        # Since the distirbutions return the semi-major axis length, we multiply by 2 to average on a circular orbit
        return val*2

class mass_ratio_and_separation_cluster:
    def __init__(self):
        self.lowduch = dist.mass_duchene(gamma = 4.2)
        self.elbadry = dist.el_badry()
        self.midduch = dist.mass_duchene(gamma = -0.5)
        self.separation = separation()
    
    def random(self, mass, sep_lim = 10000):
        #El Badry
        if mass >= 0.1 and mass < 2.5:
            flag = False
            while flag == False:
                sep = self.separation.random(mass)
                if sep >= self.elbadry.sep_min[0] and sep <= self.elbadry.sep_max[-1] and sep <= sep_lim:
                    flag = True
            mratio = self.elbadry.random(mass,sep)
        else:
            flag = False
            while flag == False:
                sep = self.separation.random(mass)
                if sep <= sep_lim:
                    flag = True
            
            # low masses : low duchene : below 0.1 masses
            if mass < 0.1:
                mratio = self.lowduch.random()
            
            # high masses high duchene : over 2.5 Msol
            else:
                mratio = self.midduch.random()
        
        return mratio,sep
        
        
def make_binary_systems_old(masses):
    """
    Make binary systems according to :
            - Mass fraction : Lu+2013, Fig16
            - Mass ratio : Bate+2012 Fig19 for up to 10Msol. Note that this Figure only represents binary systems with solar primaries. For higher masses, stars are paired with the one that has the nearest mass.
    """
    masses = np.sort(masses)[::-1] # Sorts masses in reverse order
    masses = np.copy(masses)
    funhigh = inverted_interpol_function(stdatadir+"/bate2012_fig19_intnorm_high.txt")
    funmid = inverted_interpol_function(stdatadir+"/bate2012_fig19_intnorm_mid.txt")
    funlow = inverted_interpol_function(stdatadir+"/bate2012_fig19_intnorm_low.txt")
    final_systems = []
    number = 0
    while len(masses) >= 2:
        
        mass = masses[0]
        
        # Pairing probability according to Lu+2013 Mass Fraction
        prob = pairing_probability(mass)
        randomize = np.random.uniform(0.0, 1.0)
        
        # If randomize > prob, no pairing, single-star system
        if randomize > prob:
            system = [mass]
            to_delete = [0]
        # If randomize <= prob, pairing, binary system
        else:
            number +=1
            # Defining pairing method depending on the mass
            if mass >= 10:
                fun = "nearest"
            elif mass >= 0.5 and mass < 10:
                fun = funhigh
            elif mass >= 0.1 and mass < 0.5:
                fun = funmid
            else:
                fun = funlow
            
            # Generating mass ratio and choosing paired mass
            masses = np.delete(masses, 0)
            if fun == "nearest":
                # Pairs the star to the one with the nearest mass
                paired_idx, paired_mass = ut.find_nearest(masses, mass)
            else:
                # Pairs the star with the one of the nearest mass from : mass*q, q = mass ratio
                q = fun(np.random.uniform(0.0, 1.0))
                paired_idx, paired_mass = ut.find_nearest(masses, mass*q)
            
            # Make system
            system = [mass, paired_mass]
            to_delete = [paired_idx]
            
        # Add system to list of systems
        final_systems.append(system)
        
        # Remove paired star from list of stars
        masses = np.delete(masses, to_delete)
    
    # If there is a remaining single star, add it to final_systems as a single-star system
    if len(masses) == 1:
        final_systems.append([masses[0]])
    
    print("number of binary systems = " +str(number) )
    return final_systems
    
def pairing_probability(mass):
    """
    Pairing probability accord to Lu+2013 Fig16
    MF(mass) = A * m**gamma, A=0.44, gamma = 0.51
    """
    A = 0.44
    gamma = 0.51
    prob = A * mass**gamma
    if prob > 1.0:
        prob = 1.0
    return prob




def reproduce_bins(x,y, bin_size = 0.2):
    xout = []
    yout = []
    space_size = 100
    for i in range(len(x)):
        xmin = x[i] - bin_size/2
        xmax = x[i] + bin_size/2
        xout = np.concatenate((xout,np.linspace(xmin,xmax,space_size, endpoint = False)))
        yout = np.concatenate((yout, np.repeat(y[i], space_size)))
    xout= np.concatenate((xout, [1]))
    yout = np.concatenate((yout, [y[-1]]))
    times = int(space_size*np.min(xout)/bin_size)
    if np.min(xout) != 0:
        xout = np.concatenate((np.linspace(0, np.min(xout), times, endpoint = False), xout))
        yout = np.concatenate((np.repeat(0, times), yout))
    # plt.scatter(xout,yout, marker = '+')
    return xout,yout

def make_integral_function(x,y):
    integ = []
    for i in range(len(x)):
        val = np.trapz(y[0:i+1], x[0:i+1])
        integ.append(val)
    return integ

def normalize_integral(y):
    ymax = y[-1]
    newy = np.dot(y, 1/ymax)
    if newy[-1] != 1:
        newy[-1] = 1
    return newy

def make_binary_mass_ratio_files():
    funlow = np.loadtxt(stdatadir+'/bate2012_fig19_low.txt', delimiter = ";")
    funmid = np.loadtxt(stdatadir+'/bate2012_fig19_mid.txt', delimiter = ";")
    funhigh = np.loadtxt(stdatadir+'/bate2012_fig19_high.txt', delimiter = ";")
    names = ["low","mid","high"]
    functions = {"low":funlow,
                 "mid":funmid,
                 "high":funhigh}
    for name in names:
        function = functions[name]
        x = [function[i][0] for i in range(len(function))]
        y = [function[i][1] for i in range(len(function))]
        xbins, ybins = reproduce_bins(x, y)
        yint = make_integral_function(xbins, ybins)
        ynorm = normalize_integral(yint)
        ar = [[xbins[i],ynorm[i]] for i in range(len(xbins))]
        np.savetxt(stdatadir+"/bate2012_fig19_intnorm_"+name+".txt",ar, delimiter = ";")

def inverted_interpol_function(filename):
    func = np.loadtxt(filename, delimiter = ";")
    x = [func[i][0] for i in range(len(func))]
    y = [func[i][1] for i in range(len(func))]
    inv_interpolated = interp1d(y,x)
    return inv_interpolated




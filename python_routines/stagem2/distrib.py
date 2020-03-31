
import numpy as np
import warnings


HDR = {"TYPE": "Not initiated",
       "NAME": "Not initiated",
       "ANALYTICAL_FORM": "Not initiated"}


class distrib:
    
    def __init__(self, hdr=HDR.copy()):
        """
        Make HEADER, containing parameters of dsitribution
        """
        self.hdr = hdr
        self.normalize()
    
    def header(self):
        for name in self.hdr:
            print(name + ' : ', self.hdr[name])
    
    def normalize(self, **kwargs):
        """
        Calls normalization method of current distribution
        """
        self.norm()
    
    def check_normalization(self):
        """
        Checks if current distribution has been normalized. Used in evaluate methods.
        Returns K-normalization value. If it has not been normalized, returns 1 instead
        """
        if "NORMALIZATION_FACTOR" in self.hdr:
            K = self.hdr["NORMALIZATION_FACTOR"]
        else:
            K = None
            warnings.warn("WARNING : Distribution is not yet normalized, check_normalization will return a normalization factor equal to 1. This can lead to negative values of distributions, which do not make any physical sense.")
        return K
    
    def random(self,number):
        return self.rand(number)


class position(distrib):
    """
    Class position for the position of an object
    """
    
    def __init__(self, hdr = HDR.copy()):
        hdr["TYPE"] = "Position"
        super().__init__(hdr.copy())
    
    def z_project(self, distance):
        """
        Projects a population distribution in the (x,y) angle plane, given a distance.
        """
        x = self.pos[0]
        y = self.pos[1]
        z = self.pos[2]
        xproj = np.arctan(x/(distance + z))
        yproj = np.arctan(y/(distance + z))
        
        return [xproj,yproj]
    
    def rand(self, number):
        """
        Generate random number between 0 and 1 and return as many masses from curent imf distribution
        """
        if self.hdr["NAME"] == "Gaussian":
            return self.inv(number)
        else:
            c = np.random.uniform(size = number)
            return self.inv(c)
    
class gaussian(position):
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

    def __init__(self, hwhm, hdr = HDR.copy()):
        hdr["NAME"] = "Gaussian"
        hdr["HWHM"] = hwhm
        hdr["ANALYTICAL_FORM"] = "Regular gaussian with sig = hwhm/sqrt(2*log(2))"
        super().__init__(hdr.copy())
    def norm(self, **kwargs):
        """
        Normalizes the distribution to 1 between M_min and M_max
        Also updates M_min and M_max values if given
        """
        self.hdr["NORMALIZATION_FACTOR"] = 1
    
    def inv(self, number):
        """
        Returns random position values following a gaussian distribution
        """
        hwhm = self.hdr["HWHM"]
        sig = hwhm / np.sqrt(2 * np.log(2))
        x = np.random.normal(0, sig, number)
        y = np.random.normal(0, sig, number)
        z = np.random.normal(0, sig, number)
        self.pos =  [x,y,z]


class EFF(position):
    """
    Position distribution along Elson, Fall & Freeman (1987) distribution
    rho(R) = rho_0 * (1 + (R/a)^2)^(-(gamma-1)/2)
    """
    def __init__(self, a, gamma, hdr = HDR.copy()):
        hdr["NAME"] = "Elson, Fall & Freeman"
        hdr["ANALYTICAL_FORM"] = "rho(R) = rho_0 * (1 + (R/a)^2)^(-(gamma-1)/2)"
        hdr["a"] = a
        hdr["GAMMA"] = gamma
        super().__init__(hdr.copy())
    
    def norm(self):
        #TODO
        return 0
    
    def inv(self):
        #TODO
        return 0

class imf(distrib):
    """
    IMF class
    __________
    Each IMF subclass must contain the methods :
        - norm : normalizes the IMF distribution 
        - evaluate : returns the value of the IMF distribution for te specified mass
        - inv_imf : generate a list of random masses according to the considered IMF
    Contains subclasses for different types of IMF :
        - Salpeter
        - Log-Normal (TODO)
        - Segmented power law (TODO)
    """
    def __init__(self, hdr=HDR.copy()):
        """
        Make HEADER, containing parameters of IMF
        """
        hdr["TYPE"] = "IMF -- Initial Mass Function"
        super().__init__(hdr.copy())    
    
    def rand(self,number):
        """
        Generate random number between 0 and 1 and return as many masses from curent imf distribution
        """
        if number == 1:
            number = None
        c = np.random.uniform(size = number)
        masses = self.inv(c)
        return masses
    
    def getminmass(self):
        if "M_min" in self.hdr:
            return self.hdr["M_min"]
        elif "M0" in self.hdr:
            return self.hdr["M0"]
    
    def getmaxmass(self):
        if "M_max" in self.hdr:
            return self.hdr["M_max"]
        elif "M"+str(self.nseg+1) in self.hdr:
            return self.hdr["M"+str(self.nseg+1)]
    
    def mass_population(self, total_mass, M_min = None, M_max = None, *args, **kwargs):
        """
        Generates a randomized mass population, given an IMF distribution object and the total mass of the population (total_mass).
        """
        if M_min is None:
            #M_min = self.hdr["M_min"]
            M_min = self.getminmass()
        if M_max is None:
            #M_max = self.hdr["M_max"]
            M_max = self.getmaxmass()
        M_min
        check = self.check_normalization()
        if check is None:
            self.norm()
        masses = []
        sum = 0
        print(sum)
        while sum < total_mass:
            val = self.random(number = None)
            sum += val
            masses.append(val)
            print(sum)
        
        print("Asked mass = {} M_sol".format(total_mass))
        print("Final mass = {} M_sol".format(np.sum(masses)))
        print("Generated a total of {} stars".format(len(masses)))
        return masses


class seg_powerlaw(imf):
    """
    #TODO Segmented power law, Kroupa 2001 2002
    
    """
    def __init__(self, mass_ranges, power_ranges,  hdr = HDR.copy(), name = "Segmented Power Law"):
        hdr["NAME"] = name
        hdr["ANALYTICAL_FORM"] = "dN/dM = K_i * M^-alpha_i on each segment [m_i : m_i+1]"
        
        if not np.all(np.diff(mass_ranges) >0):
            raise ValueError("masse_ranges should be ordered by increasing mass")
        if len(mass_ranges) -1 != len(power_ranges):
            raise ValueError("mass_ranges should be of length 1+len(power_ranges)")
        
        for i in range(len(power_ranges)):
            hdr["M" + str(i)] = mass_ranges[i]
            hdr["alpha" + str(i)] = power_ranges[i]
            hdr["M" + str(i+1)] = mass_ranges[i+1]
        self.nseg = len(power_ranges)
        super().__init__(hdr.copy())
        
    def norm(self, **kwargs):
        masses, powers = self.get_param()
        nseg = self.nseg
        sum = 0
        for i in range(nseg):
            prod = 1
            for k in range(1,i+1):
                prod *= masses[k]**(powers[k]-powers[k-1])
            powe = 1-powers[i]
            sum += prod * (masses[i+1]**(powe) - masses[i]**(powe))/(powe)
        self.hdr["NORMALIZATION_FACTOR"] = sum
        
        # Integral values for successive segments
        integ = []
        tot = 0
        for i in range(nseg):
            val = 1
            for k in range(1,i+1):
                val *= masses[k]**(powers[k] - powers[k-1])
            val /= self.hdr["NORMALIZATION_FACTOR"]
            val *= (masses[i+1]**(1-powers[i]) - masses[i]**(1-powers[i]))/(1-powers[i])
            integ.append(val + tot)
            tot = integ[-1]
        self.integvals = integ
    
    def get_param(self):
        masses = [self.hdr["M" + str(i)] for i in range(self.nseg+1)]
        powers = [self.hdr["alpha"+str(i)] for i in range(self.nseg)]
        return masses,powers
    
    def evaluate(self, mass):
        """
        Returns the value of the distribution for a given mass
        """
        masses,powers = self.get_param()
        index = 0
        while True:
            if masses[index] <= mass and masses[index+1]>= mass:
                break
            index += 1
        
        prod = 1
        for k in range(1,index+1):
            prod *= masses[k]**(powers[k] - powers[k-1])
        prefact = prod/self.hdr["NORMALIZATION_FACTOR"]
        out = prefact * mass**(-powers[index])
        return out
    
    def evaluate_integ(self, mass):
        """
        Evaluates the integral value in a certain mass
        """
        masses,powers = self.get_param()
        index = 0
        while True:
            if masses[index] <= mass and masses[index+1]>= mass:
                break
            index += 1
        if index == 0:
                base = 0
        else:
            base = self.integvals[index -1]
        prod = 1
        for k in range(1,index+1):
            prod *= masses[k]**(powers[k] - powers[k-1])
        prefact = prod/self.hdr["NORMALIZATION_FACTOR"]
        out = base + prefact/(1-powers[index]) * (mass**(1-powers[index])-masses[index]**(1-powers[index]))
        return out
    
    def inv(self, clist):
        """
        Check in which segment to search for the mass
        """
        
        masses, powers = self.get_param()
        # Check wether it was given a list or a unique number of c's
        try:
            iter(clist)
            testiter = True
            list_length = len(clist)
        except:
            testiter = False
        if testiter == False:
            clist = [clist]
        vallist = []
        
        # Iterates on the list of c's
        count = 0
        for c in clist:
            if testiter == True:
                print("Star : " + str(count+1) + "/" + str(list_length))
            index = 0
            while self.integvals[index] < c:
                index += 1
            if index == 0:
                base = 0
            else:
                base = self.integvals[index -1]
            val = 1
            for k in range(1,index+1):
                val *= masses[k]**(powers[k] - powers[k-1])
            val = ((c-base)*(1-powers[index]) * (self.hdr["NORMALIZATION_FACTOR"] / val) + masses[index]**(1-powers[index]))**(1/(1-powers[index]))
            vallist.append(val)
            count +=1
        
        # Formats the output to the input format : number or list
        if len(vallist) == 1:
            val = vallist[0]
        else:
            val = vallist
        return val

class kroupa(seg_powerlaw):
    """
    Kroupa IMF, as given in equation (2) in Kroupa.2000: On the variation of the initial mass function.
    """
    def __init__(self):
        mass_ranges = [0.01, 0.08, 0.50, 1.0, 1000]
        powers = [0.3, 1.3, 2.3, 2.3]
        seg_powerlaw.__init__(self,mass_ranges= mass_ranges, power_ranges = powers, name = "Kroupa IMF -- Segmented Power Law")

class lognormal(imf):
    """
    #TODO Log Normal, Chabrier 2003 2005
    
    """
    def __init__(self, M_min, M_max, hdr = HDR.copy()):
        hdr["NAME"] = "Log Normal"
        #hdr["ANALYTICAL_FORM"] = "dN/dM = M^-alpha"
        #hdr["alpha"] = 2.35
        hdr["M_min"] = M_min
        hdr["M_max"] = M_max
        super().__init__(hdr.copy())
        
    def norm(self, **kwargs):
        #TODO
        return 0
    
    def evaluate(self, mass):
        #TODO
        return 1
    
    def inv(self, c):
        #TODO
        return 1


class powerlaw(imf):
    """
    Power law IMF object
    ---------
    Follow a distribution : dN/dM = K * M^-alpha
    """
    
    def __init__(self, M_min, M_max, alpha = 2.35, hdr=HDR.copy(), name = "Power Law"):
        """
        Add specific "salpeter" header keys and values
        """
        hdr["NAME"] = name
        hdr["ANALYTICAL_FORM"] = "dN/dM = M^-alpha"
        hdr["alpha"] = 2.35
        hdr["M_min"] = M_min
        hdr["M_max"] = M_max
        super().__init__(hdr.copy())
    
    
    def norm(self, **kwargs):
        """
        Normalizes the distribution to 1 between M_min and M_max
        Also updates M_min and M_max values if given
        """
        keys = {"alpha":None,
                "M_min":None,
                "M_max":None}
        for key in keys:
            if key in kwargs:
                keys[key] = kwargs[key]
            elif key in self.hdr:
                keys[key] = self.hdr[key]
            else:
                raise ValueError("Missing keyword '{}' in **kwargs or hdr. Either give it or initiate the imf object with a predefined one".format(key))
        alpha = keys["alpha"]
        M_min = keys["M_min"]
        M_max = keys["M_max"]
        
        K = (1-alpha)/(M_max**(1-alpha) - M_min**(1-alpha))
        self.hdr["NORMALIZATION_FACTOR"] = K
    
    def evaluate(self, mass):
        """
        Returns the value of the distribution for a given mass.
        Since the ditribution is normalized to 1, it is required to multiply the output by the total mass of the studied star population.
        NOTE : If the output is negative, it is likely that the distribution has not yet been normalized.
        """
        K = self.check_normalization()
        if K is None:
            K = 1
        alpha = self.hdr["alpha"]
        number = K * mass**(-alpha)
        
        return number
    
    def inv(self, c):
        """
        Returns the value of 'mass' for which the integral from M_min to 'mass' is equal to the random value c given in input.
        -> Basically returns a random mass within the current imf probability distribution.
        """
        
        M_min = self.hdr["M_min"]
        M_max = self.hdr["M_max"]
        alpha = self.hdr["alpha"]
        
        power = 1-alpha
        mass = (c*(M_max**power - M_min**power) + M_min**power)**(1/power)
        
        return mass

class salpeter(powerlaw):
    def __init__(self, M_min, M_max):
        powerlaw.__init__(self, M_min = M_min, M_max = M_max, alpha = 2.35, name = "Salpeter")

#def mass_population(distrib, total_mass, M_min = None, M_max = None, *args, **kwargs):
    #"""
    #Generates a randomized mass population, given an IMF distribution object (distrib) and the total mass of the population (total_mass).
    #"""
    #if M_min is None:
        #M_min = distrib.hdr["M_min"]
    #if M_max is None:
        #M_max = distrib.hdr["M_max"]
    #check = distrib.check_normalization()
    #if check is None:
        #distrib.norm()
    #masses = []
    #while np.sum(masses) < total_mass:
        #masses.append(distrib.random(number = None))
    
    #print("Asked mass = {} M_sol".format(total_mass))
    #print("Final mass = {} M_sol".format(np.sum(masses)))
    #print("Generated a total of {} stars".format(len(masses)))
    #return masses











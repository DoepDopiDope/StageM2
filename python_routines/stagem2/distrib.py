
import numpy as np
import warnings
from scipy.interpolate import interp1d
from os.path import join
import stagem2.utils as ut
stdatadir = join(ut.__pkg_dir__, "data")
import matplotlib.pyplot as plt
import astropy .units as u


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

class log_normal(distrib):
    def __init__(self, log_mean, sigma):
        self.mean= log_mean
        self.sigma = sigma
        self.make_distrib()
    
    def evaluate(self, x):
        val =  (1/(self.sigma * np.sqrt(2*np.pi))) * np.exp(-(np.log10(x)-self.mean)**2 / (2 * self.sigma**2))
        return val
    
    def make_distrib(self, low = None, up = None, size = 3, length = 2001):
        if low is None:
            low = self.mean - size*self.sigma
        if up is None:
            up = self.mean + size * self.sigma
        
        oldx = np.linspace(low,up, length, endpoint = True)
        x = [10**(oldx[i]) for i in range(len(oldx))]
        y = [self.evaluate(x[i]) for i in range(len(x))]
        integ = [np.trapz(y[0:k+1], oldx[0:k+1]) for k in range(len(y))]
        norm = integ[-1]
        integ = np.dot(integ, 1/norm)
        fun = interp1d(integ, oldx)
        self.function = fun
    
    def random(self):
        c = np.random.uniform()
        return 10**self.function(c)

class raghavan(log_normal):
    def __init__(self):
        self.log_mean_period = np.log10((10**5.03) * 24*3600)
        # self.sigma_period = np.log10((10**2.28) * 24*3600)
        self.sigma_period = 2.28    #sigma log period in days
        au = 1.5e11
        sig_trans =(np.log10(period_to_semiaxis(10**4)/au) - np.log10(period_to_semiaxis(10**2)/au))/(4-2)
        self.log_mean_semiaxis = np.log10(period_to_semiaxis(10**self.log_mean_period)/au)
        # self.sigma_semiaxis = np.log10(period_to_semiaxis(10**self.sigma_period)/au)
        self.sigma_semiaxis = self.sigma_period * sig_trans
        super().__init__(self.log_mean_semiaxis, self.sigma_semiaxis)


class ward_duong(log_normal):
    def __init__(self):
        self.log_mean_semiaxis = 0.77
        self.sigma_semiaxis = 1.34
        super().__init__(self.log_mean_semiaxis, self.sigma_semiaxis)

class sep_low_duchene(log_normal):
    def __init__(self):
        self.log_mean_semiaxis = np.log10(4.5)  # mean value log in semiaxis
        self.sigma_period = 0.5     #sigma log period in days
        
        au = 1.5e11
        sig_trans =(np.log10(period_to_semiaxis(10**4, M =0.1)/au) - np.log10(period_to_semiaxis(10**2, M = 0.1)/au))/(4-2)
        
        # self.sigma_semiaxis = np.log10(period_to_semiaxis(10**self.sigma_period)/au)
        self.sigma_semiaxis = self.sigma_period * sig_trans
        
        super().__init__(self.log_mean_semiaxis, self.sigma_semiaxis)

class mass_ratio(distrib):
    def __init__(self, hdr = HDR.copy()):
        hdr["TYPE"] = "Mass Ratio"
        super().__init__(hdr.copy())

class mass_duchene(mass_ratio):
    def __init__(self, gamma, hdr = HDR.copy()):
        hdr["NAME"] = "Duchene2013"
        self.gamma = gamma
        super().__init__(hdr.copy())
        
    def evaluate(self, q, normalize = False):
        gamma = self.gamma
        if normalize == True:
            norm = 1
        else:
            norm = self.normfact
        val = 1/norm* q**(gamma)
        return val
    
    def random(self, vallow = 0.1):
        if self.gamma <0:
            c= np.random.uniform()
            gamma = self.gamma
            q = (c*(gamma+1)*self.normfact +vallow**(gamma+1))**(1/(gamma+1))
        else:
            c = np.random.uniform()
            gamma = self.gamma
            q = (c*(gamma+1)*self.normfact)**(1/(gamma+1))
        return q
    
    def norm(self, vallow = 0.1):
        if self.gamma <0:
            q = np.linspace(vallow, 1, 1000)
        else:
            q = np.linspace(0,1,1000)
        vals = self.evaluate(q, normalize = True)
        integ = np.trapz(vals,q)
        self.normfact = integ



class el_badry(mass_ratio):
    """
    Mass ratio distribution as in El-Badry2019. This class uses all parameters from their fits given in the article appendix.
    """
    def __init__(self, hdr = HDR.copy()):
        hdr["NAME"] = "El-Badry2019"
        parameters = self.load_elbadry_fits()
        self.M_min = parameters[0]
        self.M_max = parameters[1]
        self.sep_min = [50,350,600,1000,2500,5000,15000]
        self.sep_max = [350,600,1000,2500,5000,15000,50000]
        self.qbreak = [0.5,0.5,0.5,0.5,0.3]
        self.F_twin = parameters[2]
        self.q_twin = parameters[3]
        self.gamma_large = parameters[4]
        self.gamma_small = parameters[5]
        self.gamma_s = parameters[6]
        super().__init__(hdr.copy())
        self.functions = self.make_array()
    
    def norm(self):
        return 0
    
    def load_elbadry_fits(self):
        filename = stdatadir + "/el-badry_mass_ratio_fits.txt"
        vals =np.loadtxt(filename, delimiter = ";")
        M_min = vals[0]
        M_max = vals[1]
        F_twin = vals[2:9]
        q_twin = vals[9:16]
        gamma_large = vals[16:23]
        gamma_small = vals[23:30]
        gamma_s = vals[30:37]
        return M_min, M_max, F_twin, q_twin, gamma_large, gamma_small, gamma_s
    
    def make_array(self):
        x = np.linspace(0,1,2001, endpoint = True)
        functions = []
        for i in range(len(self.sep_min)):
            sep_bin = []
            for j in range(len(self.M_min)):
                fun = sub_badry(self.qbreak[j], self.F_twin[i][j], self.q_twin[i][j], self.gamma_large[i][j], self.gamma_small[i][j])
                y = [fun.evaluate(x[k]) for k in range(len(x))]
                integ = [np.trapz(y[0:k+1], x[0:k+1]) for k in range(len(y))]
                newfun = interp1d(x, integ)
                sep_bin.append(newfun)
                ut.printProgressBar(i*(len(self.M_min))+j+1, len(self.sep_min)*len(self.M_min))
            functions.append(sep_bin)
        return functions
    
    def random(self, mass, sep, q_min = 0.05):
        # Searching for separation bin
        flag = False
        for i in range(len(self.sep_min)):
            if sep >= self.sep_min[i] and sep<= self.sep_max[i]:
                sep_idx = i
                flag = True
                break
        if flag == False:
            raise ValueError("given separation is out of El-Badry range, given separation : {}".format(sep))
            return 1
        
        flag = False
        # Searching for mass bin
        for i in range(len(self.M_min)):
            if mass >= self.M_min[i] and mass<= self.M_max[i]:
                M_idx = i
                flag = True
                break
        if flag == False:
             warnings.warn("given mass is out of El-Badry range, given mass : {}".format(mass))
             return 1
        
        func = self.functions[sep_idx][M_idx]
        
        flag = False
        while flag == False:
            mass_ratio = func(np.random.uniform())
            if mass_ratio >= q_min:
                flag = True
        return mass_ratio
        
class sub_badry(el_badry):
    def __init__(self, qbreak, ftwin, qtwin, gamlarge, gamsmall):
        self.qbreak = qbreak
        self.ftwin = ftwin
        self.qtwin = qtwin
        self.gaml = gamlarge
        self.gams = gamsmall
        self.norm()
    
    def norm(self):
        qbreak = self.qbreak
        ftwin = self.ftwin
        qtwin = self.qtwin
        gaml = self.gaml
        gams = self.gams
        first = 1/(gams+1) * (0.3**(gams+1) - 0.05**(gams+1))
        second = 1/(1-ftwin) * (1/(gams +1) * (qbreak**(gams+1) - 0.3**(gams+1)) + (qbreak**(gams-gaml) /(gaml+1)) * (1 - qbreak**(gaml+1)))
        self.normfact = first + second
        b = ftwin/(1-qtwin) * second
        self.b = b
    
    def evaluate(self, q):
        qbreak = self.qbreak
        gams = self.gams
        gaml = self.gaml
        qtwin = self.qtwin
        b = self.b
        normfact = self.normfact
        if q>= 0 and q < 0.05:
            return 0
        elif q >= 0.05 and q < qbreak:
            val = (1/normfact) * q**gams
            return val
        elif q >= qbreak and q < qtwin:
            val = qbreak**(gams-gaml) / normfact * q**gaml
            return val
        elif q >= qtwin and q <= 1:
            val = qbreak**(gams-gaml) / normfact * q**gaml + b/normfact
            return val
        else:
            raise ValueError("q must be between 0 and 1")



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
    rho(R) = rho_0 * (1 + (R/a)^2)^(-(gamma+1)/2)
    """
    def __init__(self, a, gamma, rmax, rmin = 0, hdr = HDR.copy()):
        hdr["NAME"] = "Elson, Fall & Freeman"
        hdr["ANALYTICAL_FORM"] = "rho(R) = rho_0 * (1 + (R/a)^2)^(-(gamma+1)/2)"
        hdr["a"] = a
        self.a = a
        hdr["GAMMA"] = gamma
        self.gamma= gamma
        self.rmin = rmin
        self.rmax = rmax
        super().__init__(hdr.copy())
        self.make_distrib()
    
    def norm(self):
        return 0
    
    def evaluate(self, R):
        gamma = self.gamma
        a = self.a
        val = (1 + (R/a)**2)**(-(gamma+1)/2)
        return val
        
    def make_distrib(self, length = 2001):
        x = np.linspace(self.rmin,self.rmax, length, endpoint = True)
        y = [self.evaluate(x[i]) for i in range(len(x))]
        integ = [np.trapz(y[0:k+1], x[0:k+1]) for k in range(len(y))]
        norm = integ[-1]
        integ = np.dot(integ, 1/norm)
        fun = interp1d(integ, x)
        self.function = fun
    
    def random(self):
        c = np.random.uniform()
        val = self.function(c)
        return val


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
        while sum < total_mass:
            val = self.random(number = None)
            masses.append(val)
            sum += val
            ut.printProgressBar(sum, total_mass, length = 50)
        
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
        
        
        self.normfactinteg = self.evaluate_dlogm_value(1, normalize = True)
        
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
    
    
    
    def evaluate_dlogm_value(self, mass, normalize = False, avoid_norm = False):
        """
        Returns the value of the distribution for a given mass
        """
        masses,powers = self.get_param()
        
        if normalize == True:
            m = np.logspace(np.log10(masses[0]), np.log10(masses[-1]), 2000)
            m[0] = masses[0]
            m[-1] = masses[-1]
            vals = [self.evaluate_dlogm_value(mass = m[i], normalize = False, avoid_norm = True) for i in range(len(m))]
            integ = np.trapz(vals, np.log10(m))
            return integ
        
        
        index = 0
        while True:
            if masses[index] <= mass and masses[index+1]>= mass:
                break
            index += 1
        
        if avoid_norm is False:
            normfact = self.normfactinteg
        else:
            normfact = 1
        
        prod = 1
        for k in range(1,index+1):
            prod *= masses[k]**(powers[k] - powers[k-1])
        prefact = prod/normfact
        out = prefact * mass**(-powers[index] +1)
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



def period_to_semiaxis(period, M = 1.5, G = 6.67e-11 ):
    M *= 1.98e30
    a = ( G*M * (period/(2*np.pi))**2)**(1/3)
    return a

def make_coordinates(radius):
    if not hasattr(radius,"__iter__"):
        radius = [radius]
    x, y, z = [], [], []
    for r in radius:
        theta = np.random.uniform() * 2*np.pi
        phi = np.random.uniform() * 2*np.pi
        x.append(r * np.sin(theta) * np.cos(phi))
        y.append(r * np.sin(theta) * np.sin(phi))
        z.append(r * np.cos(theta))
    if np.nan in x:
        print("In x")
    if np.nan in y:
        print("In y")
    if np.nan in z:
        print("In z")
    if len(x) == 1:
        return x[0],y[0],z[0]
    else:
        return x,y,z

def make_system_pos(systems, pos):
    """
    Make systems position depending on multiplicity and separation
    """
    masses = []
    oldx, oldy, oldz = pos[0], pos[1], pos[2]
    x,y,z = [],[],[]
    multiplicity = []
    linked_to = []
    for i in range(len(systems)):
        system = systems[i]
        if system[-1] is None:
            masses.append(system[0])
            x.append(oldx[i])
            y.append(oldy[i])
            z.append(oldz[i])
            multiplicity.append(1)
            linked_to.append(len(linked_to))
        else:
            mass_1 = system[0]
            mass_2 = system[1]
            a = ((system[-1]/2 * u.au).to(u.pc)).value
            theta = np.random.uniform() * 2*np.pi
            phi = np.random.uniform() * 2*np.pi
            dx = a * np.sin(theta) * np.cos(phi)
            dy = a * np.sin(theta) * np.sin(phi)
            dz = a * np.cos(theta)
            x.append(oldx[i] + dx)
            x.append(oldx[i] - dx)
            y.append(oldy[i] + dy)
            y.append(oldy[i] - dy)
            z.append(oldz[i] + dz)
            z.append(oldz[i] - dz)
            masses.append(mass_1)
            masses.append(mass_2)
            multiplicity.append(2)
            multiplicity.append(2)
            linked_to.append(len(linked_to)+1)
            linked_to.append(len(linked_to)-1)
    newpos = [x,y,z]
    return masses,newpos, multiplicity, linked_to
    
    
    
    
def fun(r, a = 0.13, gamma = 2.3):
    val = (1+ (r/a)**2)**(-gamma/2)
    return val
    
    
    





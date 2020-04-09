 
def cluster(mass=1E3, distance=50000, half_light_radius=1):
    """
    Generate a source object for a cluster

    The cluster distribution follows a gaussian profile with the
    ``half_light_radius`` corresponding to the HWHM of the distribution. The
    choice of stars follows a Kroupa IMF, with no evolved stars in the mix. Ergo
    this is more suitable for a young cluster than an evolved custer

    Parameters
    ----------
    mass : float
        [Msun] Mass of the cluster (not number of stars). Max = 1E5 Msun
    distance : float
        [pc] distance to the cluster
    half_light_radius : float
        [pc] half light radius of the cluster

    Returns
    -------
    src : simcado.Source

    Examples
    --------

    Create a ``Source`` object for a young open cluster with half light radius
    of around 0.2 pc at the galactic centre and 100 solar masses worth of stars:

        >>> from simcado.source import cluster
        >>> src = cluster(mass=100, distance=8500, half_light_radius=0.2)


    """
    # IMF is a realisation of stellar masses drawn from an initial mass
    # function (TODO: which one?) summing to 1e4 M_sol.
    if mass <= 1E4:
        fname = find_file("IMF_1E4.dat")
        imf = np.loadtxt(fname)
        imf = imf[0:int(mass/1E4 * len(imf))]
    elif mass > 1E4 and mass < 1E5:
        fname = find_file("IMF_1E5.dat")
        imf = np.loadtxt(fname)
        imf = imf[0:int(mass/1E5 * len(imf))]
    else:
        raise ValueError("Mass too high. Must be <10^5 Msun")

    # Assign stellar types to the masses in imf using list of average
    # main-sequence star masses:
    stel_type = [i + str(j) + "V" for i in "OBAFGKM" for j in range(10)]
    masses = _get_stellar_mass(stel_type)
    ref = utils.nearest(masses, imf)
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

    # draw positions of stars: cluster has Gaussian profile
    distance *= u.pc
    half_light_radius *= u.pc
    hwhm = (half_light_radius/distance*u.rad).to(u.arcsec).value
    sig = hwhm / np.sqrt(2 * np.log(2))

    x = np.random.normal(0, sig, len(imf))
    y = np.random.normal(0, sig, len(imf))

    src = Source(lam=lam, spectra=spectra, x=x, y=y, ref=stars_spec_ref,
                 weight=weight, units="ph/s/m2")

    src.info["object"] = "cluster"
    src.info["total_mass"] = mass
    src.info["masses"] = imf
    src.info["half_light_radius"] = half_light_radius
    src.info["hwhm"] = hwhm
    src.info["distance"] = distance
    src.info["stel_type"] = stel_type

    return src

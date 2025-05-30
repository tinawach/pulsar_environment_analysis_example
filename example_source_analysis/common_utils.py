import sys, os
import pandas as pd
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import EllipseSkyRegion
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import chi2, norm


def model_significance(dataset1, dataset2, df):
    """
    Calculate significance of a source model following a Gaussian assumption

    Parameters:
    ----------
    - dataset1: gammapy.datasets.MapDataset
        Dataset without the source model 
    - dataset2: gammapy.datasets.MapDataset
        Dataset with the source model 
    - df: int
        Number of degrees of freedom of the source model
    """
    S_0 = dataset1.stat_sum()
    S_1 = dataset2.stat_sum()

    ts = S_0 - S_1
    
    """Convert delta ts to sigma"""
    p_value = chi2.sf(ts, df=df)
    sigma = norm.isf(0.5 * p_value)

    
    print('TS of the source model in the given range is:', ts)
    print('Significance of this detection:', sigma )


def plot_model(model, geom, color='black'): 
    """
    Plot the position of a Model

    Parameters:
    ----------
    - model: gammapy.Modeling.Models
        Source Model
    - geom: gammapy.maps.Geom
        Geometry of the map in which the model is plotted 

    Retruns:
    -----------
    astropy.regions.PixelRegion
    """
    
    ra = model.parameters['lon_0'].value
    dec = model.parameters['lat_0'].value
    if model.parameters.names[2] == 'r_0':
        r = model.parameters['r_0'].value
    if model.parameters.names[2] == 'sigma':
        r = model.parameters['sigma'].value
    e = model.parameters['e'].value
    phi = model.parameters['phi'].value
    
    width= 2*r*np.sqrt(1-e**2)*u.deg
    height = 2*r*u.deg
    center=SkyCoord(ra * u.deg, dec * u.deg)
    ellipse = EllipseSkyRegion(center, width, height, phi*u.deg)

    pix_ellipse = ellipse.to_pixel(geom.to_image().wcs)

    return pix_ellipse


def plot_spectrum(model, result, color, label=None):
        """
    Plot the spectrum of a Model

    Parameters:
    ----------
    - model: gammapy.Modeling.Models
        Source Model
    - result: minuit.result
        minimizer used to optimise the fit position
    """
    spec = model.spectral_model
    energy_range = [0.3, 100] * u.TeV
    spec.plot(energy_bounds=energy_range, energy_power=2, label=label, color=color)

    spec.plot_error(energy_bounds=energy_range, energy_power=2, color=color)
    
    #return spec

def plot_significance_distribution(significance_map, clr1 = 'darkred', clr2 = 'tab:orange'):
    """
    Plot a histogram from a map

    Parameters:
    ----------
    - significance_map: gammapy.maps.Map
        Map from which to derive the histogram
    - clr1: str
        Color of theGaussiann fit curve
    - clr2: str
        Color of the reference curve
    """
    bins=np.linspace(-6, 8, 131)
    #data with exclusion regions removed:
    significance_map_off = significance_map
    significance_off = significance_map_off.data[np.isfinite(significance_map_off.data)]
    significance_data = significance_off

    #fit data: 
    n, bins, patches = plt.hist(significance_data, bins=bins, histtype='step', 
                                        color='k', lw=2, zorder=1)


    x_vals = 0.5*(bins[:-1]+bins[1:])
    mu, std = norm.fit(significance_data)
    mu_opt, std_opt = 0,1
    p = norm.pdf(x_vals, mu, std)
    p_opt = norm.pdf(x_vals, mu_opt, std_opt)

    #plot:
    y = norm.pdf(x_vals, mu, std) * sum(n * np.diff(bins))
    y2 = norm.pdf(x_vals, mu_opt, std_opt) * sum(n * np.diff(bins))

    plt.plot(x_vals,y,lw=2, color=clr1 ,label=r"$\mu$ = {:.2f}, $\sigma$ = {:.2f}".format(mu, std),)
    plt.plot(x_vals, y2, lw=2, color= clr2 ,
         label=r"$\mu$ = {:.2f}, $\sigma$ = {:.2f}".format(mu_opt, std_opt), zorder=2)

    plt.grid(ls='--')
    plt.legend(loc=1, fancybox=True, shadow=True)
    plt.xlim(-6, 8)
    plt.ylim(10**0, 10**4)
    plt.ylabel('Counts')
    plt.xlabel('Significance')
    plt.yscale('log')



def galactic_to_radec(l, b):
    ''' 
    convert from galactic coordinates to J2000 coordinates. 
    '''
    from astropy.coordinates import ICRS, Galactic
    from astropy.units import deg
    from astropy.coordinates import SkyCoord
    c = SkyCoord(l=l*deg, b=b*deg, frame='galactic').transform_to('icrs')
    return c.ra.value, c.dec.value

def angle_between(phi1, theta1, phi2, theta2, unit='deg'):
    phi1 = np.asarray(phi1, dtype='float').copy()
    phi2 = np.asarray(phi2, dtype='float').copy()
    theta1 = np.asarray(theta1, dtype='float').copy()
    theta2 = np.asarray(theta2, dtype='float').copy()
    if unit == 'deg':
        phi1 *= np.pi / 180
        phi2 *= np.pi / 180
        theta1 *= np.pi / 180
        theta2 *= np.pi / 180
    ax1 = np.cos(phi1)  * np.cos(theta1)
    ay1 = np.sin(-phi1) * np.cos(theta1)
    az1 = np.sin(theta1)
    ax2 = np.cos(phi2)  * np.cos(theta2)
    ay2 = np.sin(-phi2) * np.cos(theta2)
    az2 = np.sin(theta2)
    res = np.arccos(np.clip(ax1*ax2 + ay1*ay2 + az1*az2, -1, 1))
    if unit == 'deg':
        return res * 180 / np.pi
    return res

class ExclusionRegionSet(list):
    """
    Class to query regions containing known gamma-ray sources
    """
    
    def __init__(self, regions=[]):
        for r in regions:
            if not isinstance(r, ExclusionRegion):
                raise TypeError('Can only append objects of type ExclusionRegion to this set!')
        list.__init__(self, regions)

    def append(self, region):
        if not isinstance(region, ExclusionRegion):
            raise TypeError('Can only append objects of type ExclusionRegion to this set!')
        list.append(self, region)

    def extend(self, regions):
        for r in regions:
            if not isinstance(r, ExclusionRegion):
                raise TypeError('Can only append objects of type ExclusionRegion to this set!')
        list.extend(self, regions)

    @classmethod
    def from_file(cls, filename):
        ers = cls()
        ers.read_from_file(filename)
        return ers

    def read_from_file(self, filename):
        dat = pd.read_csv(filename, comment='#',
                                    names=['shape', 'type', 'system', 'lam', 'beta', 'name',
                                           'r1', 'r2', 'phi1', 'phi2', 'note'],
                                    delim_whitespace=True)

        if not (all(dat['shape'] == 'SEGMENT') and all(dat['type'] == 'EX') and all(dat['r1'] == 0)):
            raise NotImplementedError('Only circular exclusion regions are supported up to now!')

        for r in dat.itertuples():
            ra, dec = r.lam, r.beta
            if r.system == 'GAL':
                ra, dec = galactic_to_radec(r.lam, r.beta)
            self.append(ExclusionRegion(r.name, ra, dec, r.r2))

    @property
    def names(self):
        return [r.name for r in self]

    def contains(self, test_ra, test_dec):
        mask = np.zeros_like(test_ra, dtype='bool')
        hit_regions = []
        for r in self:
            inside = r.contains(test_ra, test_dec)
            if inside.any():
                hit_regions.append(r)
            mask |= inside
        return mask, hit_regions

    def overlaps(self, test_ra, test_dec, radius):
        mask = np.zeros_like(test_ra, dtype='bool')
        hit_regions = []
        for r in self:
            overlap = r.overlaps(test_ra, test_dec, radius)
            if overlap.any():
                hit_regions.append(r)
            mask |= overlap
        return mask, hit_regions

    def get_region(self, name):
        for r in self:
            if r.name == name:
                return r
        raise ValueError('No region with name {} found!'.format(name))


class ExclusionRegion(object):
    def __init__(self, name, ra, dec, radius):
        self.name   = name
        self.ra     = ra
        self.dec    = dec
        self.radius = radius

    def contains(self, test_ra, test_dec):
        return angle_between(self.ra, self.dec, test_ra, test_dec) < self.radius

    def overlaps(self, test_ra, test_dec, test_radius):
        return angle_between(self.ra, self.dec, test_ra, test_dec) < self.radius + test_radius


def get_excluded_regions(ra, dec, radius):
    ers = ExclusionRegionSet()
    ers.read_from_file('/path/to/ExcludedRegions.dat')
    ers.read_from_file('/path/to/ExcludedRegions-stars.dat')
    return ers.overlaps(ra, dec, radius)[-1]

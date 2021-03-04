import numpy as np
import os
from time import time
import colossus.cosmology.cosmology as cosmo
from colossus.halo import concentration, mass_so
from colossus.lss.mass_function import massFunction
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import pandas as pd
from functools import lru_cache
from HaloUtils import Volume

cosmol=cosmo.setCosmology('planck15')  
cosmo.setCurrent(cosmol)

       
class SizeModel(Volume):
    
    def __init__(self, rhalo, Vol=None):
        if Vol is not None:
            self.Vol = Vol
        else:
            super().__init__()
        self.rhalo = rhalo   #common to all size models
        
    def _print_error(self):
        print('Could not compute size function')
        return
    
    def get_distrib(self,Re,bins, Type='Cassata', stars = None, Re0=None):   
    
        compactDef = globals()[Type]
        Re = Re-Re0
        #bins = bins-Re0
        cdef = compactDef(bins,Re,logstars = stars)
        H = cdef.hist()
        bwidth = bins[1]-bins[0]
        Bins = bins[1:]-0.5*bwidth
        try:
            return np.array([Bins, H])
        except:
            self._print_error
            
    def get_number_density_and_compacts(self,Re, bins, Type='Cassata', stars = None, Re0 = None): 
        
        if Type=='Cassata':  #ugly, need to find a better solution
            if Re0 is None:
                raise ValueError("Re0 must be provided if Type=='Cassata'")
            else:
                Re = Re - Re0 
                #  bins = bins -Re0
                
        compactDef = globals()[Type]
        cdef = compactDef(bins,Re,logstars = stars)
       
        N = cdef.number_density()
        Ncompact = cdef.compacts()
        try:
            return N, Ncompact
        except:
            self._print_error        
            
        
class K13Model(SizeModel):
    def __init__(self,rhalo, A_K, sigma_K,Vol=None): 
        super().__init__(rhalo=rhalo, Vol=Vol)
        self.logA_K = np.log10(A_K)
        self.sigma_K = sigma_K
        
    def to_galaxy_size(self):
        return np.random.normal(self.logA_K+self.rhalo, self.sigma_K)
        
class Concentrationmodel(SizeModel):
    def __init__(self,rhalo, A_c, sigma_c, gamma, Vol=None):
        super().__init__(rhalo=rhalo, Vol=Vol)
        self.logA_c = np.log10(A_c)
        self.sigma_c = sigma_c
        self.gamma = gamma
        
    def concentrationDuttonMaccio14(x,z):
    
        x = 10**(x+np.log10(0.7))
        b =-0.097 +0.024*z
        a=0.537 + (1.025-0.537)*np.e**(-0.718*z**1.08)
        logc = a+b*(np.log10(x)-12)
        sc = 0.11
        logc  = np.random.normal(logc,scale=sc)
        return logc

    def to_galaxy_size(self): # A_c(c/10)**(-gamma)*R
        x = self.logA_c - self.gamma*(logc-np.log10(10))  #log10(10)=1, but keep it like this for clarity
        return np.random.normal(x, size=len(self.rhalo)) + self.rhalo
        
    
class LambdaModel(SizeModel):
    def __init__(self,rhalo, A_lambda, Vol=None): 
        super().__init__(rhalo=rhalo, Vol=Vol)
        self.logA_lambda = np.log10(A_lambda)

    def to_galaxy_size(self, skewed=False):

        if not skewed:
            ## Bullock+01
            lambda0 = -1.459
            sigma = 0.268
            return np.random.normal(lambda0+self.logA_lambda,sigma,size=len(self.rhalo)) + self.rhalo
        else:
            return self._sizes_from_lambda_skewed  #rodriguez-puebla+16

    def _sizes_from_lambda_skewed(self):

        lambdas = self._extract_lambdas()
        return self.rhalo+np.log10(lambdas)     
    
    def _extract_lambdas(self):

        x=np.linspace(1.e-4,0.5,1000000)
        width =x[1]-x[0]
        cumul = self._Schechter_cumul(x,width)
        ff = interp1d(cumul,x)
        xx = np.random.uniform(0,1,size= len(R))
        lambdas = ff(xx)
        return lambdas
    
    def _Schechter_cumul(self,x,width):

        y = self._Schechter_like(x,width)
        cumul = np.cumsum(y)

        return cumul/np.max(cumul)
    
    def _Schechter_like(self,x,width):

        ###peebles spin
        alpha =-4.126
        beta=0.610
        lambda0 = -2.919

        first = (x/10**lambda0)**(-alpha)
        second = -(x/10**lambda0)**(beta)
        final = first*np.exp(second)

        area = np.sum(final*width)
        return final/area

        
class CompactDefinition:
    def __init__(self):
        super().__init__()
        self.Vol = (700)**3   #Fix
    def hist(self):
        bwidth = self.bins[1]-self.bins[0]
        return np.histogram(self.quantity, bins=self.bins)[0]/self.Vol/bwidth
    
    def number_density(self):
        h = self.hist()
        bwidth = self.bins[1]-self.bins[0]
        return  np.sum(h)*bwidth #cumtrapz(h,self.bins[1:])[-1]
    
    def compacts(self):
        h = self.hist()
        bwidth = self.bins[1]-self.bins[0]
        if self.threshold < 1:  # bit dirty, but encompasses Cassata and VanDerWel
            mask = np.ma.masked_less(self.bins[1:]-bwidth, self.threshold).mask  
        else:
            mask = np.ma.masked_greater(self.bins[1:]-bwidth, self.threshold).mask
        return np.sum(h[mask])*bwidth # cumtrapz(,self.bins[1:][mask])[-1]


class Cassata(CompactDefinition):
    def __init__(self, bins, Re, logstars=None):
        super().__init__()
        self.bins = bins
        self.quantity = Re
        self.threshold = -0.4   #lower than

        
class VanDerWel(CompactDefinition):
    def __init__(self, bins, Re, logstars):
        super().__init__()    
        self.bins = bins
        self.quantity = Re-0.75*(logstars-11)
        self.threshold = np.log10(2.5) #lower than
        
        
class Gargiulo(CompactDefinition):
    def __init__(self, bins, Re, logstars,):
        super().__init__()
        self.bins = bins
        self.quantity = logstars-2*Re-np.log10(2*np.pi) - 6 # -6 to convert from pc^-2 to kpc^-2
        self.threshold = 3.3 #higher than

        
class Barro(CompactDefinition):
    def __init__(self, bins, Re, logstars):
        super().__init__()        
        self.bins = bins
        self.quantity = logstars-1.5*Re
        self.threshold = 10.3
        
 
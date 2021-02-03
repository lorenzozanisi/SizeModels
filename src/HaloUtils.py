import numpy as np
import os
from time import time
import colossus.cosmology.cosmology as cosmo
from colossus.halo import concentration, mass_so
from colossus.lss.mass_function import massFunction
from scipy.interpolate import interp1d
import pandas as pd
from functools import lru_cache



cosmol=cosmo.setCosmology('planck15')  
cosmo.setCurrent(cosmol)
    
    
class Volume:
    def __init__(self):
        self.Vol = self._Vol
        
    @property
    def _Vol(self):
        return (400)**3 #Mpc^3
    
    
class DarkMatter(Volume):
    def __init__(self, z,Vol=None, nbody = False ):
        if Vol is not None:
            self.Vol = Vol
        else:
            super().__init__()
  
        self.mdef = 'vir'
        self.z = z
        if nbody:
            pass # insert path of N body simulation, select only centrals
        else:
            mass = self._getHaloMass()
            radius = self._getHaloRadius(mass)
            self._df = pd.DataFrame({'Mpeak':mass,'Rh':radius})

    @property
    def df(self):
        return self._df
    
    def _extract_catalog(self,N,M):
        f=interp1d(N,M)
        array_cumul=np.arange(min(N),max(N))
        cat=f(array_cumul)
        return cat
    
    def _getHaloMass(self):
        dlog10m=0.005
        hmf_choice='despali16' 
        Mvir=10**np.arange(11.5,16,dlog10m) #Mh/h
        massfunct =  massFunction(x=Mvir, z=self.z, mdef=self.mdef, model='despali16', q_out='dndlnM')*np.log(10)  #dn/dlog10M
        massfunct = massfunct*(cosmol.h)**3 #convert  from massf*h**3
        Mvir=np.log10(Mvir)
        Mvir=Mvir-np.log10(cosmol.h)  #convert from M/h
        Ncum=self.Vol*(np.cumsum((massfunct*dlog10m)[::-1])[::-1])
        halos=self._extract_catalog(Ncum,Mvir)
        return halos
    
    def _getHaloRadius(self,halos):
        halonew = np.array(10**halos)*cosmol.h #converts from Mvir to Mvir/h
        rhalo = mass_so.M_to_R(halonew,self.z,mdef=self.mdef)/cosmol.h  
        return np.log10(rhalo)
    
    
class Galaxies(DarkMatter):
    def __init__(self,z,dict_SMHM,quench_dict=None,nbody=False, Vol=None):
        super().__init__(z,Vol=Vol,nbody=nbody)
        self.SMHM = grylls19(**dict_SMHM)
        stars = pd.DataFrame({'Mstar':self._get_stars()})
        self.catalog = pd.concat([self.df,stars], axis=1)
        if quench_dict is not None:
            bl = self.catalog.apply(lambda x: np.random.uniform(size=len(self.catalog)) >  self._fred(x,**quench_dict)).values.T[0] 
            self.catalog.loc[bl,'TType'] = 'LTGs'
            red = np.logical_not(bl)
            self.catalog.loc[red,'TType'] = 'ETGs'
            
    def _fred(self,x,M0=1.5,mu=2.5):  
        M = M0 + (1+self.z)**mu
        return 1./(1+M*1.e12/10**x)
    
    def _get_stars(self):
        return self.SMHM(self.df['Mpeak'], self.z, scatter=0.15)
    
     


class grylls19:  
    
    def __init__(self, gamma10=None, gamma11= None, beta10=None, beta11=None,\
                M10=None, SHMnorm10=None, M11=None, SHMnorm11=None,scatterevol=False):
        ''' If both gamma10 and gamma11 , are None it return grylls19 Pymorph. Otherwise 
        they are custom.
        '''
        self.scatterevol = scatterevol
        if gamma10 is not None:
            self.gamma10 = gamma10
        else:
            self.gamma10 = 0.53
            
        if gamma11 is not None:
            self.gamma11 = gamma11
        else:
            self.gamma11 = 0.03
            
        if M10 is not None:
            self.M10 = M10
        else:
            self.M10 = 11.92
            
        if SHMnorm10 is not None:
            self.SHMnorm10 = SHMnorm10
        else:
            self.SHMnorm10 = 0.032
            
        if SHMnorm11 is not None:
            self.SHMnorm11=SHMnorm11
        else:
            self.SHMnorm11=-0.014
        if M11 is not None:
            self.M11=M11
        else:
            self.M11 = 0.58
        if beta10 is not None:
            self.beta10 = beta10
        else:
            self.beta10 = 1.64
        if beta11 is not None:
            self.beta11 = beta11
        else:
            self.beta11 = -0.69
            
   
    def make(self, halos,z,scatter, scatterevol=False):
        zparameter = np.divide(z-0.1, z+1)   
        M = self.M10 + self.M11*zparameter
        N = self.SHMnorm10 + self.SHMnorm11*zparameter
        b = self.beta10 + self.beta11*zparameter
        g = self.gamma10 + self.gamma11*zparameter          
        stars =  np.power(10, halos) * (2*N*np.power( (np.power(np.power(10,halos-M), -b) +\
                                                       np.power(np.power(10,halos-M), g)), -1))

        if self.scatterevol:
            scatt = np.sqrt( (0.1*(z-0.1))**2+scatter**2)
            
        else: 
            scatt = scatter
            
        stars = np.random.normal(np.log10(stars),scale=scatt)
            
        return stars
    
    def __call__(self, halos, z,scatter):
        return self.make(halos,z,scatter)
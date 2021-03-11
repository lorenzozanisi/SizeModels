from HaloUtils import DarkMatter,Galaxies
from SizeUtils import K13Model
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy
import astropy.units as u
import matplotlib.pylab as plt
import pandas as pd
import matplotlib as mpl
from matplotlib.collections import LineCollection
import colorcet as cc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import jit

import cycler
color = plt.cm.viridis(np.linspace(0, 1,10))
plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)


plt.rcParams['ytick.minor.visible']=True
plt.rcParams['xtick.minor.visible']=True
plt.rcParams['axes.linewidth']=5
plt.rcParams['xtick.major.size'] =15
plt.rcParams['ytick.major.size'] =15
plt.rcParams['xtick.minor.size'] =10
plt.rcParams['ytick.minor.size'] =10
plt.rcParams['xtick.major.width'] =5
plt.rcParams['ytick.major.width'] =5
plt.rcParams['xtick.minor.width'] =5
plt.rcParams['ytick.minor.width'] =5
plt.rcParams['axes.titlepad'] = 10

plt.rcParams['font.size']=55
plt.rcParams['figure.figsize']=(12,16)



def multiline(xs, ys, c, bubu=None, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = np.array([np.column_stack([x, y]) for x, y in zip(xs, ys)])
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_title(bubu
                )
    return lc




def smooth(x):
    return pd.Series(x).rolling(window=4).mean()


def make_compacts(redshifts,bins,mu, dictionary_models,N):
    
    out = {model_label: {'star forming':[], 'quenched':[]} for model_label in dictionary_models.keys()}

    for model_label, model in dictionary_models.items():
        quench_dict = {'M0':1,'mu':mu}
        Ncompact_SF = []
        Ncompact_Q = []
        for i,z in enumerate(redshifts):
            galaxies = Galaxies(z=z,dict_SMHM=model, quench_dict=quench_dict)
            df = galaxies.catalog
            df = df.query('11.3<Mstar<12.1')

            Q = df.query("TType=='ETGs'")
            SF = df.query("TType=='LTGs'")

            QsizeModel = K13Model(rhalo=Q['Rh'], A_K=0.018, sigma_K=0.1 )
            SFsizeModel = K13Model(rhalo=SF['Rh'], A_K=0.022*1.45, sigma_K=0.1 )
            ReSF = SFsizeModel.to_galaxy_size()
            ReQ = QsizeModel.to_galaxy_size()

            if i==0:
                Re0 = np.median(ReQ)
            _, Ncompact_SF_ = SFsizeModel.get_number_density_and_compacts(ReSF, bins=bins,Type='Cassata', stars=None, Re0=Re0)
            _, Ncompact_Q_ = QsizeModel.get_number_density_and_compacts(ReQ, bins=bins,Type='Cassata', stars=None, Re0=Re0)
            Ncompact_SF.append(Ncompact_SF_)
            Ncompact_Q.append(Ncompact_Q_)
        out[model_label]['star forming'] = Ncompact_SF
        out[model_label]['quenched'] = Ncompact_Q
        
    #continuity


    n_SF_model = {model_lab : np.zeros((N,len(times))) for model_lab in dictionary_models.keys()}
    for model_lab in n_SF_model.keys():
        for i in range(1,N+1):
            app1 = []
            for t in range(len(times)-1,-1,-1):

                nQ_t_dt = out[model_lab]['quenched'][t-i]
                nQ_t = out[model_lab]['quenched'][t]
                app1.append(nQ_t_dt - nQ_t)
            n_SF_model[model_lab][i-1] = np.flip(app1)
            
    return out, n_SF_model




if __name__=='__main__':
    cosmo = FlatLambdaCDM(H0=70,Om0=0.3)
    model_1 = dict(gamma10=0.5, scatterevol=False, gamma11=0,M11=0,SHMnorm11=0) 
    model_2 = dict(gamma10=0.5, scatterevol=True, gamma11=0,M11=0,SHMnorm11=0) 
    model_3 = dict(gamma10=0.65, scatterevol=False, gamma11=0,M11=0,SHMnorm11=0) 
    model_4 = dict(gamma10=0.65, scatterevol=True, gamma11=0,M11=0,SHMnorm11=0) 
    
    models = [model_1,model_2,model_3,model_4]
    

    
    bins=np.arange(-2,2,0.10)
    mus = [2,3]


    dt = 0.1
    times = np.arange(5,12.5,dt)
    to_z = lambda x: astropy.cosmology.z_at_value(cosmo.lookback_time, x*u.Gyr)
    redshifts = np.array(list(map(to_z, times)))

    
    maxDt = 1
    N_deltas = 10 #int(maxDt/dt)
    deltas = np.arange(dt,N_deltas*dt+dt,dt)
    print(deltas)

    output = []
    n_SF_model_output = []
    fig,axes = plt.subplots(2,2, figsize=(32,32), sharey=True)
    xx = np.arange(1000,1000)
    axes[1][1].plot(xx,xx, color='cyan',ls='-', lw=4, label='star forming')
    axes[1][1].plot(xx,xx, color='red',ls='-', lw=4, label='quenched')
    axes[0][0].plot(xx,xx, lw=4, ls=':', color='black', label='$\mu=${}'.format(mus[0]))
    axes[0][0].plot(xx,xx, lw=4, ls='-',  color='black', label='$\mu=${}'.format(mus[1]))
    for mu,ls in zip(mus, [':','-']):
        dictionary_models = {'model 3':model_3, 'model 4':model_4}#,'model 3':model_3,'model 4':model_4}
        out = {model_label: {'star forming':[], 'quenched':[]} for model_label in dictionary_models.keys()}   
     
        out_, n_SF_model_ = make_compacts(redshifts,bins,mu, dictionary_models,N_deltas)
        output.append(out_)
        n_SF_model_output.append(n_SF_model_)
        
        
    for k,ls in zip([0,1], [':','-']):    
        for ax, mod_label in zip(axes.T[k],output[0].keys()):

            Out = output[k]
            n_SF = n_SF_model_output[k]
            ax.plot(redshifts,smooth(Out[mod_label]['star forming']), color='cyan',ls=ls, lw=8)# label='star forming')
            ax.plot(redshifts,smooth(Out[mod_label]['quenched']), color='red',ls=ls, lw=8)#, label='quenched')

            off = []
            T = []
            for i in range(0,N_deltas):
                off = np.append(off,smooth(n_SF[mod_label][i]))
                   # off = offsets[mod_label][i]
                ax.plot(redshifts,smooth(n_SF[mod_label][i]), ls=ls, lw=4)
                T = np.append(T,redshifts)
            lc = multiline(T,np.asarray(off), c=deltas[::-1]*1000,cmap=mpl.cm.viridis, ax=ax, lw=4)
                #fig.colorbar(im,ax=ax)
            #plt.legend(frameon=False)
            ax.set_yscale('log')
            ax.set_xlim(1,4)
                #ax.set_yscale('symlog',linthreshy=1.e-6)

            ax.set_ylim(1.e-7, 6.e-5)
        
        
        
        
    axes[0][0].set_ylabel('$n_{compact} \ [Mpc^{-3}]$')
    axes[1][0].set_ylabel('$n_{compact} \ [Mpc^{-3}]$')
    axes[1][0].set_xlabel('z')    
    axes[1][1].set_xlabel('z')    

    plt.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7], aspect=10)
    cbar = fig.colorbar(lc, cax=cbar_ax, orientation='vertical')
    cbar.set_label('$\Delta T_{quench} \ [Myr]$', fontsize=50)

#fig.suptitle(f'mu={mu}')
    axes[0][0].legend(frameon=False,fontsize=45, loc='upper right')
    axes[1][1].legend(frameon=False,fontsize=45, loc='upper left')
    axes[0][0].text(1.5,3.e-5,"Model 1")
    axes[1][0].text(1.5, 3.e-5,"Model 2")
    x = [1.1,1.5]
    ys = np.linspace(1.5e-5,2.e-5,20) 
    color = plt.cm.viridis(np.linspace(0, 1,20))


    lines = [ [(x[0], y), (x[1], y)] for y in ys]


    lc = LineCollection(lines, colors=color, linewidths=2)
    axes[0][1].add_collection(lc)
    axes[0][1].text(1.6, 1.e-5, 'star forming \n from continuity')

    #for ax, mod_label in zip(axes.ravel(),dictionary_models.keys()):
    #    ax.set_title(mod_label)


    #fig.suptitle('$\mu=${}'.format(mu))
    plt.savefig('continuity_mu_composite_models3_4.pdf', bbox_inched='tight')
    fig.clf()
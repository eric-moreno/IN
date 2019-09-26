import sys
import os
import numpy as np
import pandas as pd
#import torch
#import torch.nn as nn
#from torch.autograd.variable import *
#import torch.optim as optim
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import rc
from sklearn.metrics import roc_curve, auc, accuracy_score
import scipy
import h5py
import argparse
import glob
import matplotlib.lines as mlines
from sklearn.ensemble import GradientBoostingRegressor
import itertools
#from histo_utilities import create_TH1D, make_ratio_plot


rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rcParams['font.size'] = 22
rcParams['text.latex.preamble'] = [
#       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
#       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{helvet}',    # set the normal font here
       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
]  
rc('text', usetex=True)

def make_dirs(dirname):
    import os, errno
    """
    Ensure that a named directory exists; if it does not, attempt to create it.
    """
    try:
    	os.makedirs(dirname)
    except:
        print("pass")
    
def make_plots(outputDir, dataframes, tdf_train, savedirs=["Plots"], taggerNames=["IN"], eraText=r'2016 (13 TeV)'):
    print("Making standard plots")

    def cut(tdf, ptlow=300, pthigh=2000):
        mlow, mhigh = 40, 200
        cdf = tdf[(tdf.fj_pt < pthigh) & (tdf.fj_pt>ptlow) &(tdf.fj_sdmass < mhigh) & (tdf.fj_sdmass>mlow)]
        
        masses = cdf['fj_sdmass'].values
        bins_sdmass = np.digitize(masses, bins = np.linspace(mlow,mhigh,9))-1
        cdf.insert(4, 'bins_sdmass', bins_sdmass, True)
        return cdf
    
    def cutrho(tdf, rholow=-8, rhohigh=-1):
        tdf = tdf[(tdf.fj_pt>0) & (tdf.fj_sdmass>0)]
        masses = tdf['fj_sdmass'].values 
        pTs = tdf['fj_pt'].values 

        rho = np.log(np.divide(np.square(masses), np.square(pTs)))
        tdf.insert(2, 'fj_rho', rho, True)
        bins_rho = np.digitize(rho, bins = np.linspace(rholow,rhohigh,21))-1                          
        tdf.insert(3, 'bins_rho', bins_rho, True)
        cdf = tdf[(tdf.fj_rho>rholow) & (tdf.fj_rho<rhohigh)]         
        return cdf
    
    def cuteta(tdf, etalow=-2.5, etahigh=2.5):
        #mlow, mhigh = 90, 140
        mlow, mhigh = 40, 200
        cdf = tdf[(tdf.fj_eta < etahigh) & (tdf.fj_eta>etalow) &(tdf.fj_sdmass < mhigh) & (tdf.fj_sdmass>mlow)]
        return cdf

    def roc_input(frame, signal=["HCC"], include = ["HCC", "Light", "gBB", "gCC", "HBB"], norm=False):       
        # Bkg def - filter unwanted
        bkg = np.zeros(frame.shape[0])    
        for label in include:
            bkg = np.add(bkg,  frame['truth'+label].values )
        bkg = [bool(x) for x in bkg]
        tdf = frame[bkg] #tdf for temporary df   
        
        # Signal
        truth = np.zeros(tdf.shape[0])
        predict = np.zeros(tdf.shape[0])
        prednorm = np.zeros(tdf.shape[0])
        predict_IN = np.zeros(tdf.shape[0])
        prednorm_IN = np.zeros(tdf.shape[0])
        for label in signal:
            truth   += tdf['truth'+label].values
            predict += tdf['predict'+label].values
            #predict_IN += tdf['predict_IN'+label].values
        for label in include:
            prednorm += tdf['predict'+label].values
            #prednorm_IN += tdf['predict_IN'+label].values
        db = tdf['fj_doubleb'].values
        if norm == False:
            return truth, predict, db
        else:
            return truth, np.divide(predict, prednorm), db
    
    def plot_rocs(dfs=[], savedir="", names=[], sigs=[["Hcc"]], bkgs=[["Hbb"]], norm=False, plotname=""):
        f, ax = plt.subplots(figsize=(10, 10))
        for frame, name, sig, bkg in zip(dfs, names, sigs, bkgs):
            truth, predict, db =  roc_input(frame, signal=sig, include = sig+bkg, norm=norm)
            fpr, tpr, threshold = roc_curve(truth, predict)
            

            ax.plot(tpr, fpr, lw=2.5, label="{}, AUC = {:.1f}\%".format(name,auc(fpr,tpr)*100))
            ROCtext=open(os.path.join(savedir, "ROCComparison_"+"+".join(sig)+"_vs_"+"+".join(bkg)+".txt"),'w')
            for ind in range(len(tpr)):
                            ROCtext.write(str(tpr[ind])+'\t'+str(fpr[ind])+'\n')
            ROCtext.close()
            print("{}, AUC={}%".format(name, auc(fpr,tpr)*100), "Sig:", sig, "Bkg:", bkg)
         
        ax.set_xlim(0,1)
        ax.set_ylim(0.001,1)
        if len(sig) == 1 and len(sig[0]) == 3 and sig[0][0] in ["H", "Z", "g"]:
            xlab = '{} \\rightarrow {}'.format(sig[0][0], sig[0][-2]+'\\bar{'+sig[0][-1]+'}') 
            ax.set_xlabel(r'Tagging efficiency ($\mathrm{}$)'.format('{'+xlab+'}'), ha='right', x=1.0)
        else: 
            xlab = ['{} \\rightarrow {}'.format(l[0], l[-2]+'\\bar{'+l[-1]+'}') if l[0][0] in ["H", "Z", "g"] else l for l in sig ]
            ax.set_xlabel(r'Tagging efficiency ($\mathrm{}$)'.format("{"+", ".join(xlab)+"}"), ha='right', x=1.0)
        if len(bkg) == 1 and len(bkg[0]) == 3 and bkg[0][0] in ["H", "Z", "g"]:
            ylab = '{} \\rightarrow {}'.format(bkg[0][0], bkg[0][-2]+'\\bar{'+bkg[0][-1]+'}') 
            ax.set_ylabel(r'Mistagging rate ($\mathrm{}$)'.format('{'+ylab+'}'), ha='right', y=1.0)
        else:
            ylab = ['{} \\rightarrow {}'.format(l[0], l[-2]+'\\bar{'+l[-1]+'}') if l[0][0] in ["H", "Z", "g"] else l for l in bkg ]
            ax.set_ylabel(r'Mistagging rate ($\mathrm{}$)'.format("{"+", ".join(ylab)+"}"), ha='right', y=1.0)
        import matplotlib.ticker as plticker
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
        ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.02))
        ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12 )
        ax.tick_params(direction='in', axis='both', which='minor' , length=6)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')    
        ax.semilogy()
        ax.grid(which='minor', alpha=0.5, axis='y', linestyle='dotted')
        ax.grid(which='major', alpha=0.9, linestyle='dotted')
        leg = ax.legend(borderpad=1, frameon=False, loc=2, fontsize=16,
            title = ""+str(int(round((min(frame.fj_pt)))))+" $\mathrm{<\ jet\ p_T\ <}$ "+str(int(round((max(frame.fj_pt)))))+" GeV" \
                        + "\n "+str(int(round((min(frame.fj_sdmass)))))+" $\mathrm{<\ jet\ m_{SD}\ <}$ "+str(int(round((max(frame.fj_sdmass)))))+" GeV"
                       )
        leg._legend_box.align = "left"
        ax.annotate(eraText, xy=(0.75, 1.1), fontname='Helvetica', ha='left',
                    bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$\mathbf{CMS}$', xy=(0.01, 1.1), fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
                    bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$Simulation\ Open\ Data$', xy=(0.115, 1.1), fontsize=18, fontstyle='italic', ha='left',
                    annotation_clip=False)
        if norm:
            f.savefig(os.path.join(savedir, "ROCNormComparison_"+"+".join(sig)+"_vs_"+"+".join(bkg)+".pdf"), dpi=400)
            f.savefig(os.path.join(savedir, "ROCNormComparison_"+"+".join(sig)+"_vs_"+"+".join(bkg)+".png"), dpi=400)
        else:
            f.savefig(os.path.join(savedir, "ROCComparison_"+"+".join(sig)+"_vs_"+"+".join(bkg)+".pdf"), dpi=400)
            f.savefig(os.path.join(savedir, "ROCComparison_"+"+".join(sig)+"_vs_"+"+".join(bkg)+".png"), dpi=400)
        plt.close(f)
    
    def plot_rocs_with_DDT(dfs=[], savedir="", names=[], sigs=[["Hcc"]], bkgs=[["Hbb"]], norm=False, plotname="", DDT_results = [[],[]]):
        f, ax = plt.subplots(figsize=(10, 10))
        colors = ['blue', 'navy', 'cornflowerblue', 'orange', 'orangered']
        line_styles = ['-', '--', '-.' , ':', '--']
        line_widths = [2.5, 2.5, 2.5, 3, 2.5]
        print(len(dfs))
        for frame, name, sig, bkg, color, style, width in zip(dfs, names, sigs, bkgs, colors, line_styles, line_widths):
            truth, predict, db =  roc_input(frame, signal=sig, include = sig+bkg, norm=norm)
            fpr, tpr, threshold = roc_curve(truth, predict)
            
            if color not in ['orange', 'orangered']:
                
                ax.plot(tpr, fpr, lw=width, color=color, linestyle = style, label="{}, AUC = {:.1f}\%".format(name,auc(fpr,tpr)*100))
                ROCtext=open(os.path.join(savedir, "ROCComparison_"+"+".join(sig)+"_vs_"+"+".join(bkg)+".txt"),'w')
                for ind in range(len(tpr)):
                                ROCtext.write(str(tpr[ind])+'\t'+str(fpr[ind])+'\n')
                ROCtext.close()

                print("{}, AUC={}%".format(name, auc(fpr,tpr)*100), "Sig:", sig, "Bkg:", bkg)
         
        ax.plot(DDT_results[0], DDT_results[1], lw=2.5, color='deepskyblue', linestyle = '-', label="{}, AUC = {:.1f}\%".format('Interaction network, DDT',auc(DDT_results[1],DDT_results[0])*100))
        
        for frame, name, sig, bkg, color, style, width in zip(dfs, names, sigs, bkgs, colors, line_styles, line_widths):
            truth, predict, db =  roc_input(frame, signal=sig, include = sig+bkg, norm=norm)
            fpr, tpr, threshold = roc_curve(truth, predict)
            
            if color in ['orange', 'orangered']:
                print(color)
                ax.plot(tpr, fpr, lw=width, color=color, linestyle=style, label="{}, AUC = {:.1f}\%".format(name,auc(fpr,tpr)*100))
                ROCtext=open(os.path.join(savedir, "ROCComparison_"+"+".join(sig)+"_vs_"+"+".join(bkg)+".txt"),'w')
                for ind in range(len(tpr)):
                                ROCtext.write(str(tpr[ind])+'\t'+str(fpr[ind])+'\n')
                ROCtext.close()

                print("{}, AUC={}%".format(name, auc(fpr,tpr)*100), "Sig:", sig, "Bkg:", bkg)
        
        ax.set_xlim(0,1)
        ax.set_ylim(0.001,1)
        if len(sig) == 1 and len(sig[0]) == 3 and sig[0][0] in ["H", "Z", "g"]:
            xlab = '{} \\rightarrow {}'.format(sig[0][0], sig[0][-2]+'\\bar{'+sig[0][-1]+'}') 
            ax.set_xlabel(r'Tagging efficiency ($\mathrm{}$)'.format('{'+xlab+'}'), ha='right', x=1.0)
        else: 
            xlab = ['{} \\rightarrow {}'.format(l[0], l[-2]+'\\bar{'+l[-1]+'}') if l[0][0] in ["H", "Z", "g"] else l for l in sig ]
            ax.set_xlabel(r'Tagging efficiency ($\mathrm{}$)'.format("{"+", ".join(xlab)+"}"), ha='right', x=1.0)
        if len(bkg) == 1 and len(bkg[0]) == 3 and bkg[0][0] in ["H", "Z", "g"]:
            ylab = '{} \\rightarrow {}'.format(bkg[0][0], bkg[0][-2]+'\\bar{'+bkg[0][-1]+'}') 
            ax.set_ylabel(r'Mistagging rate ($\mathrm{}$)'.format('{'+ylab+'}'), ha='right', y=1.0)
        else:
            ylab = ['{} \\rightarrow {}'.format(l[0], l[-2]+'\\bar{'+l[-1]+'}') if l[0][0] in ["H", "Z", "g"] else l for l in bkg ]
            ax.set_ylabel(r'Mistagging rate ($\mathrm{}$)'.format("{"+", ".join(ylab)+"}"), ha='right', y=1.0)
        import matplotlib.ticker as plticker
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
        ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.02))
        ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12 )
        ax.tick_params(direction='in', axis='both', which='minor' , length=6)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')    
        ax.semilogy()
        ax.grid(which='minor', alpha=0.5, axis='y', linestyle='dotted')
        ax.grid(which='major', alpha=0.9, linestyle='dotted')
        leg = ax.legend(borderpad=1, frameon=False, loc=2, fontsize=16,
            title = ""+str(int(round((min(frame.fj_pt)))))+" $\mathrm{<\ jet\ p_T\ <}$ "+str(int(round((max(frame.fj_pt)))))+" GeV" \
                        + "\n "+str(int(round((min(frame.fj_sdmass)))))+" $\mathrm{<\ jet\ m_{SD}\ <}$ "+str(int(round((max(frame.fj_sdmass)))))+" GeV"
                       )
        leg._legend_box.align = "left"
        ax.annotate(eraText, xy=(0.75, 1.1), fontname='Helvetica', ha='left',
                    bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$\mathbf{CMS}$', xy=(0.01, 1.1), fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
                    bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$Simulation\ Open\ Data$', xy=(0.115, 1.1), fontsize=18, fontstyle='italic', ha='left',
                    annotation_clip=False)
        if norm:
            f.savefig(os.path.join(savedir, "ROCNormComparison_withDDT_"+"+".join(sig)+"_vs_"+"+".join(bkg)+".pdf"), dpi=400)
            f.savefig(os.path.join(savedir, "ROCNormComparison_withDDT_"+"+".join(sig)+"_vs_"+"+".join(bkg)+".png"), dpi=400)
        else:
            f.savefig(os.path.join(savedir, "ROCComparison_withDDT_"+"+".join(sig)+"_vs_"+"+".join(bkg)+".pdf"), dpi=400)
            f.savefig(os.path.join(savedir, "ROCComparison_withDDT_"+"+".join(sig)+"_vs_"+"+".join(bkg)+".png"), dpi=400)
        plt.close(f)
    
    def quantile_regression_DDT(dfs, dfs_train, FPR_cut): 
        
        print('Fitting Quantile Reg. of FPR = ' + str(FPR_cut))
        data_train = [[mass] for mass, pT in zip(dfs_train.loc[dfs_train['truth'+'QCD'] == 1]['fj_sdmass'].values , dfs_train.loc[dfs_train['truth'+'QCD'] == 1]['fj_pt'].values)]
        data = [[mass] for mass, pT in zip(dfs['fj_sdmass'].values , dfs['fj_pt'].values)]
        
        
        if FPR_cut > 10: 
            aux_scale = 0.1*float(FPR_cut)/3000.
        
        else: 
            aux_scale = 0.1*float(FPR_cut)/100.
            
        model = GradientBoostingRegressor(loss='quantile', alpha= float(FPR_cut)/100, 
                                          n_estimators=250, 
                                          min_samples_leaf=3, min_samples_split=3, 
                                          max_depth=5, 
                                          validation_fraction=0.2, 
                                          n_iter_no_change=5, tol=1e-4, 
                                          verbose=1)
        model.fit(data_train, dfs_train.loc[dfs_train['truth'+'QCD'] == 1]['predict'+'Hbb'].values/aux_scale)
        cuts = aux_scale*model.predict(data)
        return cuts
    
    def fit_quantile_reg(tdf, FPR_cut=[], savename=""):
            
        from sklearn.ensemble import GradientBoostingRegressor

        print('Setting DDT cut to {}%'.format(FPR_cut))
        big_cuts = {}
        for FPR in range(len(FPR_cut)):
            
            cuts = quantile_regression_DDT(tdf, tdf_train, FPR_cut[FPR])
            big_cuts[str(round(FPR_cut[FPR], 5))] = cuts
            

        np.save(savename+'.npy', big_cuts)
    
    
    
    def plot_jsd_limit(frame=[], savedir="", names=[], sigs=[["Hcc"]], bkgs=[["Hbb"]], norm=False, plotname=""):

        from scipy.spatial import distance
        
        ndatapoints = 300
        nBbins = 10
        f, ax = plt.subplots(figsize=(10, 10))
        temp_frame = frame
        
        mmin = -5
        mmax = 5
        nbins = 20
        frame = np.random.normal(size=188924) 
        spec_a = np.digitize(frame, bins=np.linspace(mmin,mmax,nbins+1), right=False)-1
        spec_ohe_a = np.zeros((spec_a.shape[0],nbins))
        spec_ohe_a[np.arange(spec_a.shape[0]),spec_a] = 1
        spec_ohe_a_sum = np.sum(spec_ohe_a,axis=0)/spec_ohe_a.shape[0]
        nA = spec_ohe_a.shape[0]
        #plt.yscale('log')
        #ax.loglog()
        
        for nA, x, color, color_fill in zip(np.logspace(np.log10(len(frame)) - 2, np.log10(len(frame)), 3), np.logspace(0, 2, 3),  ['orange', 'pink', 'purple'], ['bisque', 'lightpink', 'violet']): 
            
            big_kld = []
            spec_a = np.digitize(frame, bins=np.linspace(mmin,mmax,nbins+1), right=False)-1
            spec_ohe_a = np.zeros((spec_a.shape[0],nbins))
            spec_ohe_a[np.arange(spec_a.shape[0]),spec_a] = 1
            spec_ohe_a_sum = np.sum(spec_ohe_a,axis=0)/spec_ohe_a.shape[0]
            
            
            spec_ohe_a = spec_ohe_a[:int(nA)]                         
            spec_ohe_a_sum = np.sum(spec_ohe_a,axis=0)/spec_ohe_a.shape[0]
            nA = spec_ohe_a.shape[0]
      
            for nB in np.logspace(np.log10(x), np.log10(nA), nBbins):    
                kld = []
                for i in range(ndatapoints): 
                    index = np.random.randint(0, int(nA), int(nB))
                    spec_ohe_b = spec_ohe_a[index] 
                    spec_ohe_b_sum = np.sum(spec_ohe_b,axis=0)/spec_ohe_b.shape[0]
                    kld.append(1./scipy.stats.entropy(spec_ohe_a_sum, spec_ohe_b_sum, base=2))      
                big_kld.append(kld)
                print(nB)
            
            sums = np.array([np.mean(i) for i in big_kld])
            std = [np.std(i) for i in big_kld]
            
            from scipy.optimize import curve_fit
            
            def myExpFunc(x, a, b):
                return a * np.power(x, b)
            popt, pcov = curve_fit(myExpFunc, np.logspace(np.log10(x), np.log10(nA), nBbins)/nA, sums[2])
            
            #plt.plot(np.logspace(np.log10(x), np.log10(nA), nBbins)/nA, myExpFunc(np.logspace(np.log10(x), np.log10(nA), nBbins)/nA, *popt))
            
            print(popt)
       
            
            plt.plot(np.logspace(np.log10(x), np.log10(nA), nBbins)/float(nA), sums, color = color, label = "Normal dist., nA = " + str(nA), linestyle='--')
            plt.fill_between(np.logspace(np.log10(x), np.log10(nA), nBbins)/float(nA), sums-std, sums+std, color = color_fill, alpha=0.50)
        
        
        mmin = 40
        mmax = 200
        nbins = 8

        
        big_kld = []
        
        frame = temp_frame
        
        frame = frame.loc[frame['truth'+bkgs[0][0]] == 1]
        nA = len(frame)
        print(nA)
        mass_a = frame.loc[frame['truth'+bkgs[0][0]] == 1]['fj_sdmass'].values
        spec_a = np.digitize(mass_a, bins=np.linspace(mmin,mmax,nbins+1), right=False)-1
        spec_ohe_a = np.zeros((spec_a.shape[0],nbins))
        spec_ohe_a[np.arange(spec_a.shape[0]),spec_a] = 1
        spec_ohe_a_sum = np.sum(spec_ohe_a,axis=0)/spec_ohe_a.shape[0]
        nA = spec_ohe_a.shape[0]
        
        for nA, x,  color, color_fill in zip(np.logspace(np.log10(len(frame)) - 2, np.log10(len(frame)), 3), np.logspace(0, 2, 3), ['red', 'blue', 'green'], ['salmon', 'lightblue', 'lightgreen']): 
            
            big_kld = []
            
            mass_a = frame.loc[frame['truth'+bkgs[0][0]] == 1]['fj_sdmass'].values
            spec_a = np.digitize(mass_a, bins=np.linspace(mmin,mmax,nbins+1), right=False)-1
            spec_ohe_a = np.zeros((spec_a.shape[0],nbins))
            spec_ohe_a[np.arange(spec_a.shape[0]),spec_a] = 1
            spec_ohe_a_sum = np.sum(spec_ohe_a,axis=0)/spec_ohe_a.shape[0]
            
            
            spec_ohe_a = spec_ohe_a[:int(nA)]                         
            spec_ohe_a_sum = np.sum(spec_ohe_a,axis=0)/spec_ohe_a.shape[0]
            nA = spec_ohe_a.shape[0]
      
            for nB in np.logspace(np.log10(x), np.log10(nA), nBbins):    
                kld = []
                for i in range(ndatapoints): 
                    index = np.random.randint(0, int(nA), int(nB))
                    spec_ohe_b = spec_ohe_a[index] 
                    spec_ohe_b_sum = np.sum(spec_ohe_b,axis=0)/spec_ohe_b.shape[0]
                    kld.append(1./scipy.stats.entropy(spec_ohe_b_sum, spec_ohe_a_sum, base=2))      
                big_kld.append(kld)
                print(nB)
                
            sums = np.array([np.mean(i) for i in big_kld])
            std = [np.std(i) for i in big_kld]
           
            plt.plot(np.logspace(np.log10(x), np.log10(nA), nBbins)/nA, sums, color = color, label = "Mass dist., nA = " + str(nA))
            
            from scipy.optimize import curve_fit
            
            def myExpFunc(x, a, b):
                return a * np.power(x, b)
            popt, pcov = curve_fit(myExpFunc, np.logspace(np.log10(x), np.log10(nA), nBbins)/nA, sums)
            print('look here')
            print(popt)
            
            plt.plot(np.logspace(np.log10(x), np.log10(nA), nBbins)/nA, myExpFunc(np.logspace(np.log10(x), np.log10(nA), nBbins)/nA, *popt))
            plt.fill_between(np.logspace(np.log10(x), np.log10(nA), nBbins)/nA, sums-std, sums+std, color = color_fill, alpha=0.50)
       
        
        ax.set_xlim(0.00052932,1)
        ax.set_ylim(1e-2,70000)
        ax.set_xlabel(r'nB/nA',ha='right', x=1.0)
        ax.set_ylabel(r'1 / $D_\mathrm{JS}$',ha='right', y=1.0)
        
        import matplotlib.ticker as plticker
        
        #ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12 )
        #ax.tick_params(direction='in', axis='both', which='minor' , length=6)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')    
        ax.grid(which='minor', alpha=0.5, linestyle='dotted')
        ax.grid(which='major', alpha=0.9, linestyle='dotted')
        
        leg = ax.legend(borderpad=1, frameon=False, loc='upper left', fontsize=16)
       
       
        leg._legend_box.align = "right"
        
        ax.annotate(eraText, xy=(1e3, 1.1e7), fontname='Helvetica', ha='left',
                    bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$\mathbf{CMS}$', xy=(1.1, 1.1e7), fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
                    bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$Simulation\ Open\ Data$', xy=(3, 1.1e7), fontsize=18, fontstyle='italic', ha='left',
                    annotation_clip=False)
        
        f.savefig(os.path.join(savedir, "JSD_limit" + ".pdf"), dpi=400)
        f.savefig(os.path.join(savedir, "JSD_limit" + ".png"), dpi=400)
        plt.close(f)     
           
    def get_jsd_limit(frame=[], nB=0.1, eps_bkg=0, savedir=""):   
        
        ndatapoints = 2
        
        mmin = 40
        mmax = 200
        nbins = 8
        
        frame = frame.loc[frame['truth'+'QCD'] == 1]
        N = len(frame)

        jsds = []
        for i in range(ndatapoints): 
            nB_exp = np.random.poisson(nB)
            
            if nB_exp > nB: nB_exp = nB
                
            index_b = np.random.choice(int(N), int(nB_exp),replace=False)
            index_a = np.delete(range(N), index_b)

            #frame_a = frame.iloc[index_a]
            frame_b = frame.iloc[index_b] 
            frame_a = frame 
            
            mass_a = frame_a['fj_sdmass'].values
            mass_b = frame_b['fj_sdmass'].values

            # digitze into bins
            spec_pass = np.digitize(mass_a, bins=np.linspace(mmin,mmax,nbins+1), right=False)-1
            spec_fail = np.digitize(mass_b, bins=np.linspace(mmin,mmax,nbins+1), right=False)-1
            # one hot encoding
            spec_ohe_pass = np.zeros((spec_pass.shape[0],nbins))
            spec_ohe_pass[np.arange(spec_pass.shape[0]),spec_pass] = 1
            spec_ohe_pass_sum = np.sum(spec_ohe_pass,axis=0)
            #for i in range(len(spec_ohe_pass_sum)): 
            #    spec_ohe_pass_sum[i] = np.random.poisson(spec_ohe_pass_sum[i])
            spec_ohe_pass_sum = spec_ohe_pass_sum/spec_ohe_pass.shape[0]

            spec_ohe_fail_sum = np.array([int(i*nB_exp)/float(nB_exp) for i in spec_ohe_pass_sum])
            
            
            #spec_ohe_fail = np.zeros((spec_fail.shape[0],nbins))
            #spec_ohe_fail[np.arange(spec_fail.shape[0]),spec_fail] = 1
            #spec_ohe_fail_sum = np.sum(spec_ohe_fail,axis=0)
            #for i in range(len(spec_ohe_fail_sum)): 
            #    spec_ohe_fail_sum[i] = np.random.poisson(spec_ohe_fail_sum[i])
            #spec_ohe_fail_sum = spec_ohe_fail_sum/spec_ohe_fail.shape[0]
            
            
            M = 0.5*spec_ohe_pass_sum+0.5*spec_ohe_fail_sum
            kld_pass = scipy.stats.entropy(spec_ohe_pass_sum,M,base=2)
            kld_fail = scipy.stats.entropy(spec_ohe_fail_sum,M,base=2)
            jsd = 0.5*kld_pass+0.5*kld_fail

            jsds.append(1./jsd)      

        '''
        f, ax = plt.subplots(figsize=(10, 10))
        plt.hist(jsds, bins = 10) 

        ax.set_xlabel(r'JSD',ha='right', x=1.0)
        ax.set_ylabel(r'Count (Events)',ha='right', y=1.0)
        plt.axvline(np.mean(jsds), color = 'red')
        plt.title(r'JSD Distribution, nB/nA = {:0.5f}'.format(eps_bkg))
        f.savefig(os.path.join(savedir,'JSD_Distribution_epsbkg{:0.2f}.png'.format(eps_bkg)), dpi=400)
        '''
        print(np.mean(jsds))
        
        return(np.mean(jsds), np.std(jsds))
        
    
    def bestpartition_JSD(frame, eps, max_iter=None):
        mmin = 40
        mmax = 200
        nbins = 8
        
        frame = frame.loc[frame['truth'+'QCD'] == 1]
        masses = frame['fj_sdmass'].values
        spec_pass = np.digitize(masses, bins=np.linspace(mmin,mmax,nbins+1), right=False)-1
        spec_ohe_pass = np.zeros((spec_pass.shape[0],nbins))
        spec_ohe_pass[np.arange(spec_pass.shape[0]),spec_pass] = 1
        spec_ohe_pass_sum = np.sum(spec_ohe_pass,axis=0)
        h = spec_ohe_pass_sum
        print(h)
        
        A_base = np.asarray(h*eps, dtype=int)
        B_base = h - A_base
        
        nBins = len(h)
        ## all ways to go 
        j=[]
        for i,swap in enumerate(itertools.product([0,-1,1], repeat=len(h))):
            A = np.maximum(A_base + swap, np.zeros(nBins))
            B = np.maximum(B_base - swap, np.zeros(nBins))
            if np.sum(A)==0 or np.sum(B)==0: continue
            A = A/np.sum(A)
            B = B/np.sum(B)
            j.append(JSD(A,B))
            #print(swap,j[-1])
            if max_iter and i>max_iter: break

        return j
    
    
    def JSD(A,B):
        S = A/2. + B/2.
        eA = scipy.stats.entropy( A, S, base=2)
        eB = scipy.stats.entropy( B, S, base=2)
        jsd = eA/2. + eB/2.
        return 1./jsd
    
    def plot_jsd(dfs=[], savedir="", names=[], sigs=[["Hcc"]], bkgs=[["Hbb"]], norm=False, plotname=""):

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        
        def find_nearest(array,value):
            idx = (np.abs(array-value)).argmin()
            return idx, array[idx]
        
        rhomin = -8
        rhomax = -1
        bins = 20

        dfs.insert(3, dfs[0])
        temp_names = np.copy(names)
        temp_names = np.insert(temp_names, 3, 'Interaction network, DDT')
        sigs.append('Hbb')
        bkgs.append('QCD')
        min_jsd_eff = 2e-4
        mmin = 40
        mmax = 200
        nbins = 8
        f, ax = plt.subplots(figsize=(10, 10))
        ax.loglog()
        line_styles = [':', ':', '--', '-', '-', '-.']
        
        for frame, name, sig, bkg, color, style in zip(dfs, temp_names, sigs, bkgs, colors, line_styles):
            
            if name not in ['Interaction network, DDT']:
                
                truth, predict, db =  roc_input(frame, signal=sig, include = sig+bkg, norm=norm)
                fpr, tpr, threshold = roc_curve(truth, predict)
                print("{}, AUC={}%".format(name, auc(fpr,tpr)*100), "Sig:", sig, "Bkg:", bkg)
                print("{}, Acc={}%".format(name, accuracy_score(truth,predict>0.5)*100), "Sig:", sig, "Bkg:", bkg)

                cuts = {}
                jsd_plot = []
                eb_plot = []
                print(len(frame))
                for wp,marker in zip([0.5,0.9,0.95],['^','s','o','v']): # % signal eff.
                    idx, val = find_nearest(tpr, wp)
                    cuts[str(wp)] = threshold[idx] # threshold for deep double-b corresponding to ~1% mistag rate

                    mask_pass = (frame['predict'+sig[0]] > cuts[str(wp)]) & frame['truth'+bkg[0]]
                    mask_fail = (frame['predict'+sig[0]] < cuts[str(wp)]) & frame['truth'+bkg[0]]
                    mass = frame['fj_sdmass'].values
                    mass_pass = mass[mask_pass]
                    mass_fail = mass[mask_fail]


                    #mass_pass_temp = mass_pass[np.random.randint(0, len(mass_pass), int(min_jsd_eff*len(mass_pass)) + 1)]
                    #mass_fail_temp = mass_fail[np.random.randint(0, len(mass_fail), int(min_jsd_eff*len(mass_fail)) + 1)]

                    #mass_pass = mass_pass[:int(min_jsd_eff*len(mass_pass)) + 1]
                    #mass_fail = mass_fail[:int(min_jsd_eff*len(mass_fail))]


                    # digitze into bins
                    spec_pass = np.digitize(mass_pass, bins=np.linspace(mmin,mmax,nbins+1), right=False)-1
                    spec_fail = np.digitize(mass_fail, bins=np.linspace(mmin,mmax,nbins+1), right=False)-1
                    # one hot encoding
                   
                    spec_ohe_pass = np.zeros((spec_pass.shape[0],nbins))
                    spec_ohe_pass[np.arange(spec_pass.shape[0]),spec_pass] = 1
                    spec_ohe_pass_sum = np.sum(spec_ohe_pass,axis=0)/spec_ohe_pass.shape[0]

                    spec_ohe_fail = np.zeros((spec_fail.shape[0],nbins))
                    spec_ohe_fail[np.arange(spec_fail.shape[0]),spec_fail] = 1
                    spec_ohe_fail_sum = np.sum(spec_ohe_fail,axis=0)/spec_ohe_fail.shape[0]
                    M = 0.5*spec_ohe_pass_sum+0.5*spec_ohe_fail_sum

                    kld_pass = scipy.stats.entropy(spec_ohe_pass_sum,M,base=2)
                    kld_fail = scipy.stats.entropy(spec_ohe_fail_sum,M,base=2)

                    jsd = 0.5*kld_pass+0.5*kld_fail

                    print('eS = %.2f%%, eB = %.2f%%, 1/eB=%.2f, jsd = %.2f, 1/jsd = %.2f'%(tpr[idx]*100,fpr[idx]*100,1/fpr[idx],jsd,1/jsd))
                    eb_plot.append(1/fpr[idx])
                    jsd_plot.append(1/jsd)
                    ax.plot([1/fpr[idx]],[1/jsd],marker=marker,markersize=12,color=color)
                    
            else:
                
                truth, predict, db =  roc_input(frame, signal=sig, include = sig+bkg, norm=norm)
                fpr, tpr, threshold = roc_curve(truth, predict)
                print("{}, AUC={}%".format(name, auc(fpr,tpr)*100), "Sig:", sig, "Bkg:", bkg)
                print("{}, Acc={}%".format(name, accuracy_score(truth,predict>0.5)*100), "Sig:", sig, "Bkg:", bkg)

                cuts = {}
                jsd_plot = []
                eb_plot = []
          

                aux_scale = 1/100.
               
                model = GradientBoostingRegressor(loss='ls',
                                                  n_estimators=200, 
                                                  min_samples_leaf=5, min_samples_split=5, 
                                                  max_depth=5, 
                                                  verbose=1)
         
                data_train = [[i] for i in TPR_range]
            
                print(data_train)
                print(FPR_range)
                
                model.fit(data_train, FPR_range/aux_scale)
                
                big_cuts = np.load('DDT_TPRcuts_sdmass.npy')
                
                for wp,marker in zip([50.,90.,95.],['^','s','o', 'v']): # % signal eff.
                    idx, val = find_nearest(tpr, wp/100)

                    mask_pass = (frame['predict'+sig[0]] > big_cuts.item().get(str(wp))) & frame['truth'+bkg[0]]
                    mask_fail = (frame['predict'+sig[0]] < big_cuts.item().get(str(wp))) & frame['truth'+bkg[0]]
                    mass = frame['fj_sdmass'].values
                    mass_pass = mass[mask_pass]
                    mass_fail = mass[mask_fail]

                    #mass_pass_temp = mass_pass[np.random.randint(0, len(mass_pass), int(min_jsd_eff*len(mass_pass)) + 1)]
                    #mass_fail_temp = mass_fail[np.random.randint(0, len(mass_fail), int(min_jsd_eff*len(mass_fail)) + 1)]

                    #mass_pass = mass_pass[:int(min_jsd_eff*len(mass_pass))]
                    #mass_fail = mass_fail[:int(min_jsd_eff*len(mass_fail))]

                    # digitze into bins
                    spec_pass = np.digitize(mass_pass, bins=np.linspace(mmin,mmax,nbins+1), right=False)-1
                    spec_fail = np.digitize(mass_fail, bins=np.linspace(mmin,mmax,nbins+1), right=False)-1
                    # one hot encoding
                    
                    spec_ohe_pass = np.zeros((spec_pass.shape[0],nbins))
                    spec_ohe_pass[np.arange(spec_pass.shape[0]),spec_pass] = 1
                    spec_ohe_pass_sum = np.sum(spec_ohe_pass,axis=0)/spec_ohe_pass.shape[0]
                    
                    spec_ohe_fail = np.zeros((spec_fail.shape[0],nbins))
                    spec_ohe_fail[np.arange(spec_fail.shape[0]),spec_fail] = 1
                    spec_ohe_fail_sum = np.sum(spec_ohe_fail,axis=0)/spec_ohe_fail.shape[0]
                    M = 0.5*spec_ohe_pass_sum+0.5*spec_ohe_fail_sum

                    kld_pass = scipy.stats.entropy(spec_ohe_pass_sum,M,base=2)
                    kld_fail = scipy.stats.entropy(spec_ohe_fail_sum,M,base=2)
                    jsd = 0.5*kld_pass+0.5*kld_fail
                    
                    FPR_wp = 1./(model.predict([[wp/100.]])[0]*aux_scale)
                    
                    print('eS = %.2f%%, eB = %.2f%%, 1/eB=%.2f, jsd = %.2f, 1/jsd = %.2f'%(tpr[idx]*100,fpr[idx]*100,1/fpr[idx],jsd,1/jsd))
                    
                    
                    print(FPR_wp)
                    
                    eb_plot.append(FPR_wp)
                    jsd_plot.append(1/jsd)
                    ax.plot([FPR_wp],[1/jsd],marker=marker,markersize=12,color=color)
                    
             
                   
            ax.plot(eb_plot,jsd_plot,linestyle=style,label=name,color=color)
        
        
        
        min_jsd = []
        size = len(dfs[0].loc[dfs[0]['truth'+'QCD'] == 1])
        for eps_bkg in np.logspace(0.1,4, 10): 
            #min_jsd.append(get_jsd_limit(dfs[0], size/eps_bkg, 1/eps_bkg, savedir))
            bestpart = bestpartition_JSD(dfs[0], 1/eps_bkg)
            min_jsd.append([max(bestpart)])
        
        
        ax.plot(np.logspace(0.1, 4, 10), [i[0] for i in min_jsd], lw = 5, linestyle='--', label='Statistical Limit JSD')
        #plt.fill_between(np.logspace(0, 4, 10), [i[0]-i[1] for i in min_jsd],[i[0]+i[1] for i in min_jsd], color = 'lightblue', alpha=0.50)
        
        #size = len(dfs[0].loc[dfs[0]['truth'+'QCD'] == 1])
        #min_jsd = get_jsd_limit(dfs[0], nB=min_jsd_eff*size)
        #print("[5.26173191e+04 1.03417247e+00]101347.09877133417")
        
        '''
        def myExpFunc(x):
            return 101000 * np.power(x, 1.03417247e+00)
        
        min_jsd = []
        size = len(dfs[0].loc[dfs[0]['truth'+'QCD'] == 1])
        for nB in size/np.logspace(np.log10(1), np.log10(1e4), 10): 
            print(nB)
            min_jsd.append(myExpFunc(nB/size))
        
        print(min_jsd)
        
        ax.plot(np.logspace(0, 4, 10), min_jsd, linestyle='--', label='Statistical Limit JSD')
        print(get_jsd_limit(dfs[0], size))

        '''
        #min_jsd = [get_jsd_limit(frame=dfs[0], nB=min_jsd_eff * len(dfs[0].loc[dfs[0]['truth'+'QCD'] == 1]))[0] for i in range(10)]
        
        #ax.plot([1, 1e4], [np.mean(min_jsd), np.mean(min_jsd)], linestyle='--', label='Statistical Limit JSD')
        #plt.fill_between([1, 1e4], [min_jsd[0]-min_jsd[1], min_jsd[0]-min_jsd[1]], [min_jsd[0]+min_jsd[1], min_jsd[0]+min_jsd[1]], color = 'lightblue', alpha=0.50)
        
        ax.set_xlim(1,2e3)
        ax.set_ylim(1,1e9)
        ax.set_xlabel(r'Background rejection (QCD) 1 / $\varepsilon_\mathrm{bkg}$',ha='right', x=1.0)
        ax.set_ylabel(r'Mass decorrelation 1 / $D_\mathrm{JS}$',ha='right', y=1.0)
        
        import matplotlib.ticker as plticker
        ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12 )
        ax.tick_params(direction='in', axis='both', which='minor' , length=6)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')    
        ax.loglog()
        ax.grid(which='minor', alpha=0.5, linestyle='dotted')
        ax.grid(which='major', alpha=0.9, linestyle='dotted')
        leg = ax.legend(borderpad=1, frameon=False, loc='upper left', fontsize=16,
            title = ""+str(int(round((min(frame.fj_pt)))))+" $\mathrm{<\ jet\ p_T\ <}$ "+str(int(round((max(frame.fj_pt)))))+" GeV" \
                        + "\n "+str(int(round((min(frame.fj_sdmass)))))+" $\mathrm{<\ jet\ m_{SD}\ <}$ "+str(int(round((max(frame.fj_sdmass)))))+" GeV"
                       )
        leg._legend_box.align = "right"

        circle = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                                            markersize=12, label=r'$\varepsilon_\mathrm{sig}$ = 95\%%')
        square = mlines.Line2D([], [], color='gray', marker='s', linestyle='None',
                                                             markersize=12, label=r'$\varepsilon_\mathrm{sig}$ = 90\%%')
        utriangle = mlines.Line2D([], [], color='gray', marker='^', linestyle='None',
                                                                  markersize=12, label=r'$\varepsilon_\mathrm{sig}$ = 50\%%')
        #dtriangle = mlines.Line2D([], [], color='gray', marker='v', linestyle='None',
        #                                                          markersize=12, label=r'$\varepsilon_\mathrm{sig}$ = 30\%%')
        plt.gca().add_artist(leg)
        #leg2 = ax.legend(handles=[circle, square, utriangle, dtriangle],fontsize=16,frameon=False,borderpad=1,loc='upper right')
        leg2 = ax.legend(handles=[circle, square, utriangle],fontsize=16,frameon=False,borderpad=1,loc='upper right')
        leg2._legend_box.align = "right"
        plt.gca().add_artist(leg2)
        ax.annotate(eraText, xy=(2.5e2, 1.1e9), fontname='Helvetica', ha='left',
                    bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$\mathbf{CMS}$', xy=(1.1, 1.1e9), fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
                    bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$Simulation\ Open\ Data$', xy=(2.3, 1.1e9), fontsize=18, fontstyle='italic', ha='left',
                    annotation_clip=False)
        
        f.savefig(os.path.join(savedir, "JSD_"+"+".join(sig)+"_vs_"+"+".join(bkg)+".pdf"), dpi=400)
        f.savefig(os.path.join(savedir, "JSD_"+"+".join(sig)+"_vs_"+"+".join(bkg)+".png"), dpi=400)
        plt.close(f)
    
    
    def plot_jsd_sig(dfs=[], savedir="", names=[], sigs=[["Hcc"]], bkgs=[["Hbb"]], norm=False, plotname=""):

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        
        def find_nearest(array,value):
            idx = (np.abs(array-value)).argmin()
            return idx, array[idx]

        rhomin = -8
        rhomax = -1
        bins = 20
             
        dfs.insert(3, dfs[0])
        temp_names = np.copy(names)
        temp_names = np.insert(temp_names, 3, 'Interaction network, DDT')
        sigs.append('Hbb')
        bkgs.append('QCD')
        
        mmin = 40
        mmax = 200
        nbins = 8
        min_jsd_eff = 0.01
        f, ax = plt.subplots(figsize=(10, 10))
        ax.semilogy()
        for frame, name, sig, bkg, color in zip(dfs, temp_names, sigs, bkgs, colors):
            
            if name not in ['Interaction network, DDT']:
                
                #frame = frame[:int(len(frame)*min_jsd_eff)]\
                truth, predict, db =  roc_input(frame, signal=sig, include = sig+bkg, norm=norm)
                fpr, tpr, threshold = roc_curve(truth, predict)
                print("{}, AUC={}%".format(name, auc(fpr,tpr)*100), "Sig:", sig, "Bkg:", bkg)
                print("{}, Acc={}%".format(name, accuracy_score(truth,predict>0.5)*100), "Sig:", sig, "Bkg:", bkg)
                cuts = {}
                jsd_plot = []
                es_plot = []
       
                #for wp,marker in zip([0.005,0.01,0.05,0.1],['v','^','s','o']): # % bkg rej.
                for wp,marker in zip([0.01,0.05,0.1],['^','s','o']): # % bkg rej.    
                    idx, val = find_nearest(fpr, wp)                    
                    cuts[str(wp)] = threshold[idx] # threshold 
                    
                    jsds = []
                    
                    for i in range(100): 
                        index = np.random.randint(0, len(frame), int(min_jsd_eff*len(frame)))
                    
                        mask_pass = (frame.iloc[index]['predict'+sig[0]] > cuts[str(wp)]) & frame.iloc[index]['truth'+bkg[0]]
                        mask_fail = (frame.iloc[index]['predict'+sig[0]] < cuts[str(wp)]) & frame.iloc[index]['truth'+bkg[0]]
                        mass = frame.iloc[index]['fj_sdmass'].values
                        mass_pass = mass[mask_pass]
                        mass_fail = mass[mask_fail]

                        #mass_pass_temp = mass_pass[np.random.randint(0, len(mass_pass), int(min_jsd_eff*len(mass_pass)))]
                        #mass_fail_temp = mass_fail[np.random.randint(0, len(mass_fail), int(min_jsd_eff*len(mass_fail)))]

                        #mass_pass = mass_pass[:int(min_jsd_eff*len(mass_pass)) + 1]
                        #mass_fail = mass_fail[:int(min_jsd_eff*len(mass_fail))]


                        # digitze into bins
                        spec_pass = np.digitize(mass_pass, bins=np.linspace(mmin,mmax,nbins+1), right=False)-1
                        spec_fail = np.digitize(mass_fail, bins=np.linspace(mmin,mmax,nbins+1), right=False)-1
                        # one hot encoding
                        spec_ohe_pass = np.zeros((spec_pass.shape[0],nbins))
                        spec_ohe_pass[np.arange(spec_pass.shape[0]),spec_pass] = 1
                        spec_ohe_pass_sum = np.sum(spec_ohe_pass,axis=0)/spec_ohe_pass.shape[0]
                        
                        spec_ohe_fail = np.zeros((spec_fail.shape[0],nbins))
                        spec_ohe_fail[np.arange(spec_fail.shape[0]),spec_fail] = 1
                        spec_ohe_fail_sum = np.sum(spec_ohe_fail,axis=0)/spec_ohe_fail.shape[0]
                       
                        M = 0.5*spec_ohe_pass_sum+0.5*spec_ohe_fail_sum

                        kld_pass = scipy.stats.entropy(spec_ohe_pass_sum,M,base=2)
                        kld_fail = scipy.stats.entropy(spec_ohe_fail_sum,M,base=2)
                        jsd = 0.5*kld_pass+0.5*kld_fail
                        jsds.append(jsd)
                        
                    jsd = np.mean(jsds)    
                    
                    print('eS = %.2f%%, eB = %.2f%%, 1/eB=%.2f, jsd = %.2f, 1/jsd = %.2f'%(tpr[idx]*100,fpr[idx]*100,1/fpr[idx],jsd,1/jsd))
                    es_plot.append(tpr[idx])
                    jsd_plot.append(1/jsd)
                    ax.plot([tpr[idx]],[1/jsd],marker=marker,markersize=12,color=color)
             
                ax.plot(es_plot,jsd_plot,linestyle='-',label=name,color=color)
            
            else: 
                #frame_size = int(len(frame)*min_jsd_eff)
                #frame = frame[:frame_size]
                truth, predict, db =  roc_input(frame, signal=sig, include = sig+bkg, norm=norm)
                fpr, tpr, threshold = roc_curve(truth, predict)
                
                print("{}, AUC={}%".format(name, auc(fpr,tpr)*100), "Sig:", sig, "Bkg:", bkg)
                print("{}, Acc={}%".format(name, accuracy_score(truth,predict>0.5)*100), "Sig:", sig, "Bkg:", bkg)
                cuts = {}
                jsd_plot = []
                es_plot = []
                big_cuts = np.load('DDT_FPRcuts_sdmass.npy')
                    
                #for wp,marker in zip([0.5,1.,5.,10.],['v','^','s','o']): # % bkg rej.
                for wp,marker in zip([1.,5.,10.],['^','s','o']): # % bkg rej.
                    idx, val = find_nearest(fpr, wp/100)
                    '''
                    jsds = []
                    
                    for i in range(100): 
                        index = np.random.randint(0, len(frame), int(min_jsd_eff*len(frame)))
                        
                        mask_pass = (frame.iloc[index]['predict'+sig[0]] > big_cuts.item().get(str(wp))[index]) & frame.iloc[index]['truth'+bkg[0]]
                        mask_fail = (frame.iloc[index]['predict'+sig[0]] < big_cuts.item().get(str(wp))[index]) & frame.iloc[index]['truth'+bkg[0]]
                        mass = frame.iloc[index]['fj_sdmass'].values
                        mass_pass = mass[mask_pass]
                        mass_fail = mass[mask_fail]

                        #mass_pass_temp = mass_pass[np.random.randint(0, len(mass_pass), int(min_jsd_eff*len(mass_pass)))]
                        #mass_fail_temp = mass_fail[np.random.randint(0, len(mass_fail), int(min_jsd_eff*len(mass_fail)))]

                        #mass_pass = mass_pass[:int(min_jsd_eff*len(mass_pass))]
                        #mass_fail = mass_fail[:int(min_jsd_eff*len(mass_fail))]

                        # digitze into bins
                        spec_pass = np.digitize(mass_pass, bins=np.linspace(mmin,mmax,nbins+1), right=False)-1
                        spec_fail = np.digitize(mass_fail, bins=np.linspace(mmin,mmax,nbins+1), right=False)-1
                        # one hot encoding
                        spec_ohe_pass = np.zeros((spec_pass.shape[0],nbins))
                        spec_ohe_pass[np.arange(spec_pass.shape[0]),spec_pass] = 1
                        spec_ohe_pass_sum = np.sum(spec_ohe_pass,axis=0)/spec_ohe_pass.shape[0]
                        spec_ohe_fail = np.zeros((spec_fail.shape[0],nbins))
                        spec_ohe_fail[np.arange(spec_fail.shape[0]),spec_fail] = 1
                        spec_ohe_fail_sum = np.sum(spec_ohe_fail,axis=0)/spec_ohe_fail.shape[0]
                        M = 0.5*spec_ohe_pass_sum+0.5*spec_ohe_fail_sum

                        kld_pass = scipy.stats.entropy(spec_ohe_pass_sum,M,base=2)
                        kld_fail = scipy.stats.entropy(spec_ohe_fail_sum,M,base=2)
                        jsd = 0.5*kld_pass+0.5*kld_fail
                        jsds.append(jsd)
                    jsd = np.mean(jsds) 
                    '''
                    
                    mask_pass = (frame['predict'+sig[0]] > big_cuts.item().get(str(wp))) & frame['truth'+bkg[0]]
                    mask_fail = (frame['predict'+sig[0]] < big_cuts.item().get(str(wp))) & frame['truth'+bkg[0]]
                    mass = frame['fj_sdmass'].values
                    mass_pass = mass[mask_pass]
                    mass_fail = mass[mask_fail]

                    #mass_pass_temp = mass_pass[np.random.randint(0, len(mass_pass), int(min_jsd_eff*len(mass_pass)))]
                    #mass_fail_temp = mass_fail[np.random.randint(0, len(mass_fail), int(min_jsd_eff*len(mass_fail)))]

                    #mass_pass = mass_pass[:int(min_jsd_eff*len(mass_pass))]
                    #mass_fail = mass_fail[:int(min_jsd_eff*len(mass_fail))]

                    # digitze into bins
                    spec_pass = np.digitize(mass_pass, bins=np.linspace(mmin,mmax,nbins+1), right=False)-1
                    spec_fail = np.digitize(mass_fail, bins=np.linspace(mmin,mmax,nbins+1), right=False)-1
                    # one hot encoding
                    spec_ohe_pass = np.zeros((spec_pass.shape[0],nbins))
                    spec_ohe_pass[np.arange(spec_pass.shape[0]),spec_pass] = 1
                    spec_ohe_pass_sum = np.sum(spec_ohe_pass,axis=0)/spec_ohe_pass.shape[0]
                    spec_ohe_fail = np.zeros((spec_fail.shape[0],nbins))
                    spec_ohe_fail[np.arange(spec_fail.shape[0]),spec_fail] = 1
                    spec_ohe_fail_sum = np.sum(spec_ohe_fail,axis=0)/spec_ohe_fail.shape[0]
                    M = 0.5*spec_ohe_pass_sum+0.5*spec_ohe_fail_sum

                    kld_pass = scipy.stats.entropy(spec_ohe_pass_sum,M,base=2)
                    kld_fail = scipy.stats.entropy(spec_ohe_fail_sum,M,base=2)
                    jsd = 0.5*kld_pass+0.5*kld_fail
                    
                    print(make_TPR_DDT(frame, wp, siglab='QCD', sculp_label='fj_sdmass', taggerName=name))
                    print('eS = %.2f%%, eB = %.2f%%, 1/eB=%.2f, jsd = %.2f, 1/jsd = %.2f'%(tpr[idx]*100,fpr[idx]*100,1/fpr[idx],jsd,1/jsd))
                    es_plot.append(make_TPR_DDT(frame, wp, siglab='QCD', sculp_label='fj_sdmass', taggerName=name))
                    jsd_plot.append(1/jsd)
                    ax.plot([make_TPR_DDT(frame, wp, siglab='QCD', sculp_label='fj_sdmass', taggerName=name)],[1/jsd],marker=marker,markersize=12,color=color)
                    
                    
                ax.plot(es_plot,jsd_plot,linestyle='-',label=name,color=color)
        
        '''
        def find_nearest(array,value):
            idx = (np.abs(array-value)).argmin()
            return idx, array[idx]
        
        def myExpFunc(x):
            return 101000 * np.power(x, 1.03417247e+00)
        
        truth, predict, db =  roc_input(dfs[0], signal=['Hbb'], include = ['Hbb', 'QCD'], norm=norm)
        fpr, tpr, threshold = roc_curve(truth, predict)
        
        min_jsd = []
        size = len(dfs[0].loc[dfs[0]['truth'+'QCD'] == 1])
        for nB in np.linspace(0, 1, 30): 
            idx, arr = find_nearest(tpr, nB)
            
            print(fpr[idx])
            min_jsd.append(myExpFunc(fpr[idx]))
               
        ax.plot(np.linspace(0, 1, 30), min_jsd, linestyle='--', label='Statistical Limit JSD')
        '''
        
        min_jsd = [get_jsd_limit(frame=dfs[0], nB=min_jsd_eff* len(dfs[0].loc[dfs[0]['truth'+'QCD'] == 1]))[0] for i in range(10)]
        
        ax.plot([0, 1], [np.mean(min_jsd), np.mean(min_jsd)], linestyle='--', label='Statistical Limit JSD')
        
        
        '''
        size = len(dfs[0].loc[dfs[0]['truth'+'QCD'] == 1])
        min_jsd = get_jsd_limit(dfs[0], nB=min_jsd_eff*size)
        ax.plot([0, 1], [min_jsd[0], min_jsd[0]], linestyle='--', label='Statistical Limit JSD')
        plt.fill_between([0, 1], [min_jsd[0]-min_jsd[1], min_jsd[0]-min_jsd[1]], [min_jsd[0]+min_jsd[1], min_jsd[0]+min_jsd[1]], color = 'lightblue', alpha=0.50)
        '''
        ax.set_xlim(0,1)
        ax.set_ylim(1,5e4)
        ax.set_xlabel(r'Tagging efficiency ($\mathrm{H \rightarrow b\bar{b}}$) $\varepsilon_\mathrm{sig}$',ha='right', x=1.0)
        ax.set_ylabel(r'Mass decorrelation 1 / $D_\mathrm{JS}$',ha='right', y=1.0)
        
        import matplotlib.ticker as plticker
        ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12 )
        ax.tick_params(direction='in', axis='both', which='minor' , length=6)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')    
        ax.semilogy()
        ax.grid(which='minor', alpha=0.5, linestyle='dotted')
        ax.grid(which='major', alpha=0.9, linestyle='dotted')
        leg = ax.legend(borderpad=1, frameon=False, loc='upper left', fontsize=16,
            title = ""+str(int(round((min(frame.fj_pt)))))+" $\mathrm{<\ jet\ p_T\ <}$ "+str(int(round((max(frame.fj_pt)))))+" GeV" \
                        + "\n "+str(int(round((min(frame.fj_sdmass)))))+" $\mathrm{<\ jet\ m_{SD}\ <}$ "+str(int(round((max(frame.fj_sdmass)))))+" GeV"
                       )
        leg._legend_box.align = "left"

        circle = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                                            markersize=12, label=r'$\varepsilon_\mathrm{bkg}$ = 10\%%')
        square = mlines.Line2D([], [], color='gray', marker='s', linestyle='None',
                                                             markersize=12, label=r'$\varepsilon_\mathrm{bkg}$ = 5\%%')
        utriangle = mlines.Line2D([], [], color='gray', marker='^', linestyle='None',
                                                                  markersize=12, label=r'$\varepsilon_\mathrm{bkg}$ = 1\%%')
        #dtriangle = mlines.Line2D([], [], color='gray', marker='v', linestyle='None',
        #                                                          markersize=12, label=r'$\varepsilon_\mathrm{bkg}$ = 0.5\%%')
        plt.gca().add_artist(leg)
        #leg2 = ax.legend(handles=[circle, square, utriangle, dtriangle],fontsize=16,frameon=False,borderpad=1,loc='center left')
        leg2 = ax.legend(handles=[circle, square, utriangle],fontsize=16,frameon=False,borderpad=1,loc='upper right')
        leg2._legend_box.align = "left"
        plt.gca().add_artist(leg2)
        ax.annotate(eraText, xy=(0.75, 5.1e4),fontname='Helvetica', ha='left',
                    bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$\mathbf{CMS}$', xy=(0, 5.1e4), fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
                    bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$Simulation\ Open\ Data$', xy=(0.105, 5.1e4), fontsize=18, fontstyle='italic', ha='left',
                    annotation_clip=False)
        
        f.savefig(os.path.join(savedir, "JSD_sig_"+"+".join(sig)+"_vs_"+"+".join(bkg)+".pdf"), dpi=400)
        f.savefig(os.path.join(savedir, "JSD_sig_"+"+".join(sig)+"_vs_"+"+".join(bkg)+".png"), dpi=400)
        plt.close(f)
    
    def make_TPR_DDT(tdf, FPR_cut=5, siglab="Hcc", sculp_label='Light', savedir="", taggerName=""): 
        
        
        NBINS= 8 # number of bins for loss function
        MMAX = 200. # max value
        MMIN = 40. # min value


        weight = tdf['truth'+'Hbb'].values
        bins = np.linspace(40,200,NBINS+1)

        correct = sum(weight)
        if siglab == sculp_label: return 
        def find_nearest(array,value):
            idx = (np.abs(array-value)).argmin()
            return idx, array[idx]

        if siglab[0] in ["H", "Z", "g"] and len(siglab) == 3:
            legend_siglab = '{} \\rightarrow {}'.format(siglab[0], siglab[-2]+'\\bar{'+siglab[-1]+'}') 
            legend_siglab = '$\mathrm{}$'.format('{'+legend_siglab+'}')
        else:
            legend_siglab = siglab
        if sculp_label[0] in ["H", "Z", "g"] and len(sculp_label) == 3:
            legend_bkglab = '{} \\rightarrow {}'.format(sculp_label[0], sculp_label[-2]+'\\bar{'+sculp_label[-1]+'}') 
            legend_bkglab = '$\mathrm{}$'.format('{'+legend_bkglab+'}')
        else: legend_bkglab = sculp_label


        # Placing events in bins to detemine threshold for each individual bin

        ctdf = tdf.copy()
        ctdf = ctdf.head(0)

        #ctdf = quantile_regression_DDT(tdf, tdf_train, FPR_cut)
        big_cuts = np.load('DDT_FPRcuts_sdmass.npy')

        cuts = big_cuts.item().get(str(round(FPR_cut, 5)))
        
        ctdf = ctdf.append(tdf.loc[tdf['predict'+'Hbb'].values > cuts[:len(tdf)]])
        
        
        weight = ctdf['truth'+'Hbb'].values
        selected = sum(weight) 

        return(float(selected)/correct)
    
    def make_FPR_DDT(tdf, TPR_cut=5, siglab="Hcc", sculp_label='Light', savedir="", taggerName=""): 
        
        
        NBINS= 8 # number of bins for loss function
        MMAX = 200. # max value
        MMIN = 40. # min value


        weight = tdf['truth'+'QCD'].values
        bins = np.linspace(40,200,NBINS+1)

        correct = sum(weight)
        if siglab == sculp_label: return 
        def find_nearest(array,value):
            idx = (np.abs(array-value)).argmin()
            return idx, array[idx]

        if siglab[0] in ["H", "Z", "g"] and len(siglab) == 3:
            legend_siglab = '{} \\rightarrow {}'.format(siglab[0], siglab[-2]+'\\bar{'+siglab[-1]+'}') 
            legend_siglab = '$\mathrm{}$'.format('{'+legend_siglab+'}')
        else:
            legend_siglab = siglab
        if sculp_label[0] in ["H", "Z", "g"] and len(sculp_label) == 3:
            legend_bkglab = '{} \\rightarrow {}'.format(sculp_label[0], sculp_label[-2]+'\\bar{'+sculp_label[-1]+'}') 
            legend_bkglab = '$\mathrm{}$'.format('{'+legend_bkglab+'}')
        else: legend_bkglab = sculp_label


        # Placing events in bins to detemine threshold for each individual bin

        ctdf = tdf.copy()
        ctdf = ctdf.head(0)

        cuts = quantile_regression_DDT(tdf, tdf_train, TPR_cut)
        #big_cuts = np.load('DDT_TPRcuts_sdmass.npy')
        #print(TPR_cut)
        
        #cuts = big_cuts.item().get(str(round(TPR_cut, 5)))
        print(cuts)
        ctdf = ctdf.append(tdf.loc[tdf['predict'+'Hbb'].values > cuts])
        
   
        weight = ctdf['truth'+'QCD'].values
        selected = sum(weight) 
        
        return(float(selected)/correct)
        
    def make_DDT(tdf, FPR_cut=[], siglab="Hcc", sculp_label='Light', savedir="", taggerName=""): 
        from sklearn.ensemble import GradientBoostingRegressor

        NBINS= 8 # number of bins for loss function
        MMAX = 200. # max value
        MMIN = 40. # min value
        bins = np.linspace(MMIN,MMAX,NBINS+1)
        
        '''
        weight = tdf['truth'+'Hbb'].values
        bins = np.linspace(40,200,NBINS+1)
        values, bins, _ = plt.hist(tdf['fj_sdmass'].values, bins=bins, weights = weight, lw=2, normed=False,
                        histtype='step',label='{}\% FPR Cut QCD'.format(FPR_cut))
        correct = sum(values)
        '''
        
        print('Setting DDT cut to {}%'.format(FPR_cut))
        if siglab == sculp_label: return 
        def find_nearest(array,value):
            idx = (np.abs(array-value)).argmin()
            return idx, array[idx]
        
        if siglab[0] in ["H", "Z", "g"] and len(siglab) == 3:
            legend_siglab = '{} \\rightarrow {}'.format(siglab[0], siglab[-2]+'\\bar{'+siglab[-1]+'}') 
            legend_siglab = '$\mathrm{}$'.format('{'+legend_siglab+'}')
        else:
            legend_siglab = siglab
        if sculp_label[0] in ["H", "Z", "g"] and len(sculp_label) == 3:
            legend_bkglab = '{} \\rightarrow {}'.format(sculp_label[0], sculp_label[-2]+'\\bar{'+sculp_label[-1]+'}') 
            legend_bkglab = '$\mathrm{}$'.format('{'+legend_bkglab+'}')
        else: legend_bkglab = sculp_label
                  
                    
        f, ax = plt.subplots(figsize=(10,10))
        big_cuts = np.load('DDT_FPRcuts_sdmass.npy')
        #big_cuts = {}
        ctdf = tdf.copy()
        ctdf = ctdf.head(0)
        
        dataframes_cut = [ctdf for i in range(len(FPR_cut))]
        for FPR in range(len(FPR_cut)):
            
            cuts = big_cuts.item().get(str(round(FPR_cut[FPR], 5)))
            cuts = quantile_regression_DDT(tdf, tdf_train, FPR_cut[FPR])
            #big_cuts[str(round(FPR_cut[FPR], 5))] = cuts
            dataframes_cut[FPR] = dataframes_cut[FPR].append(tdf.loc[tdf['predict'+'Hbb'].values > cuts])

        #np.save('DDT_TPRcuts_sdmass.npy', big_cuts)
        #sys.exit()

        f, ax = plt.subplots(figsize=(10,10))
        h_list = []
        
        weight_uncut = tdf['truth'+'QCD'].values.astype(float)
        
        
        
        n, binEdges = np.histogram(tdf.loc[tdf['truth'+'QCD'] == 1]['fj_sdmass'].values, bins=bins)
        #h_list.append(create_TH1D(tdf['fj_sdmass'].values, name='No tagging applied', title=None, binning=bins, weights=weight_uncut/np.sum(weight_uncut), h2clone=None, axis_title = ['Soft-Drop Mass', 'Normalized Scale (QCD)'], opt='', color = 1))
        #h_list[-1].SetMarkerStyle(20)
        #h_list[-1].SetMarkerColor(1)
        #h_list[-1].SetStats(0)
        #h_list[-1].SetLineWidth(2)
        
        colorcode = [2, 5, 6, 8, 9]
        for FPR in range(len(FPR_cut)):
            weight = dataframes_cut[FPR]['truth'+'QCD'].values.astype(float)
            ax.hist(dataframes_cut[FPR]['fj_sdmass'].values, bins=bins, 
                    weights=weight/np.sum(weight), 
                    lw=2, 
                    normed=False,
                    histtype='step',
                    label='{}\% mistagging rate'.format(FPR_cut[FPR]),
                    color='C'+str(FPR)
                   )
            n, binEdges = np.histogram(dataframes_cut[FPR].loc[dataframes_cut[FPR]['truth'+'QCD'] == 1]['fj_sdmass'].values, bins=bins)
            #h_list.append(create_TH1D(dataframes_cut[FPR]['fj_sdmass'].values, name='{}% mistagging rate'.format(FPR_cut[FPR]), title=None, binning=bins, weights=weight/np.sum(weight), h2clone=None, axis_title = ['Soft-Drop Mass', 'Normalized Scale (QCD)'], opt='', color = colorcode[FPR]))
            #h_list[-1].SetStats(0)
            #h_list[-1].SetLineWidth(2)
            err = np.sqrt(n)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            #plt.errorbar(bincenters, n/np.sum(n), yerr=err/np.sum(n), fmt='none', ecolor='C'+str(FPR))
  
        #plot = make_ratio_plot(h_list, title = "", label = "", in_tags = None, ratio_bounds = [0.6, 1.4], draw_opt = 'E1')
        
        #plot.SaveAs(os.path.join(savedir,'Ratio_Plot_SDmass_QCD.pdf'))
        
        ax.hist(tdf['fj_sdmass'].values, bins=bins, weights = weight_uncut/np.sum(weight_uncut), lw=2, normed=False,
                        histtype='step',label='No tagging applied')
        
        
        #values, bins, _ = plt.hist(ctdf['fj_sdmass'].values, bins=bins, weights = weight, lw=2, normed=False,
        #                histtype='step',label='{}\% FPR Cut QCD'.format(FPR_cut))
        #selected = sum(values)
        #print('TPR IS: ' + str(selected/correct))                        
                                
        ax.set_xlabel(r'$\mathrm{m_{SD}\ [GeV]}$', ha='right', x=1.0)
        ax.set_ylabel(r'Normalized scale ({})'.format('QCD'), ha='right', y=1.0)
        import matplotlib.ticker as plticker
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=20))
        ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=10))
        ax.yaxis.set_minor_locator(plticker.AutoMinorLocator(5))
        ax.set_xlim(40, 200)
        ax.set_ylim(0, 0.45)
        ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12)#, labelleft=False )

        ax.tick_params(direction='in', axis='both', which='minor' , length=6)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')    
        #ax.grid(which='minor', alpha=0.5, axis='y', linestyle='dotted')
        ax.grid(which='major', alpha=0.9, linestyle='dotted')
        leg = ax.legend(borderpad=1, frameon=False, loc='best', fontsize=16,
            title = ""+str(int(round((min(frame.fj_pt)))))+" $\mathrm{<\ jet\ p_T\ <}$ "+str(int(round((max(frame.fj_pt)))))+" GeV" \
              + "\n "+str(int(round((min(frame.fj_sdmass)))))+" $\mathrm{<\ jet\ m_{SD}\ <}$ "+str(int(round((max(frame.fj_sdmass)))))+" GeV"
                  + "\n {} tagging {}".format(taggerName, legend_siglab)           )
        leg._legend_box.align = "right"
        ax.annotate(eraText, xy=(0.75, 1.015), xycoords='axes fraction', fontname='Helvetica', ha='left',
                        bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$\mathbf{CMS}$', xy=(0, 1.015), xycoords='axes fraction', fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
                        bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$Simulation\ Open\ Data$', xy=(0.105, 1.015), xycoords='axes fraction', fontsize=18, fontstyle='italic', ha='left',

                        annotation_clip=False)
        f.savefig(os.path.join(savedir,'DDT_QCD_tag'+'QCD_cut' + '.png'), dpi=400)
        f.savefig(os.path.join(savedir,'DDT_QCD_tag'+'QCD_cut' + '.pdf'), dpi=400)
        
        # Plot Hbb distribution with the same 5% QCD cut
        f, ax = plt.subplots(figsize=(10,10))
        ctdf = tdf.copy()
        ctdf = ctdf.head(0)
        
        f, ax = plt.subplots(figsize=(10,10))
        weight_uncut = tdf['truth'+'Hbb'].values.astype(float)
        
        '''
        h_list=[]
        n, binEdges = np.histogram(tdf.loc[tdf['truth'+'QCD'] == 1]['fj_sdmass'].values, bins=bins)
        h_list.append(create_TH1D(tdf['fj_sdmass'].values, name='No tagging applied', title=None, binning=bins, weights=weight_uncut/np.sum(weight_uncut), h2clone=None, axis_title = ['Soft-Drop Mass', 'Normalized Scale (Hbb)'], opt='', color = 1))
        h_list[-1].SetMarkerStyle(20)
        h_list[-1].SetMarkerColor(1)
        h_list[-1].SetStats(0)
        '''
        
        for FPR in range(len(FPR_cut)): 
            weight = dataframes_cut[FPR]['truth'+'Hbb'].values.astype(float)
            ax.hist(dataframes_cut[FPR]['fj_sdmass'].values, bins=bins, 
                    weights=weight/np.sum(weight), 
                    lw=2, 
                    normed=False,
                    histtype='step',
                    label='{}\% mistagging rate'.format(FPR_cut[FPR]),
                    color='C'+str(FPR)
                   )

            #n, binEdges = np.histogram(dataframes_cut[FPR].loc[dataframes_cut[FPR]['truth'+'Hbb'] == 1]['fj_sdmass'].values, bins=bins)
            #h_list.append(n)
            #h_list.append(create_TH1D(dataframes_cut[FPR]['fj_sdmass'].values, name='{}% mistagging rate'.format(FPR_cut[FPR]), title=None, binning=bins, weights=weight/np.sum(weight), h2clone=None, axis_title = ['Soft-Drop Mass', 'Normalized Scale (Hbb)'], opt='', color = colorcode[FPR]))
            #h_list[-1].SetStats(0)
            #err = np.sqrt(n)
            #bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            #plt.errorbar(bincenters, n/np.sum(n), yerr=err/np.sum(n), fmt='none', ecolor='C'+str(FPR))
       
        #plot = make_ratio_plot(h_list, title = "", label = "", in_tags = None, ratio_bounds = [0.8, 1.2], draw_opt = 'E1')
        #plot.SaveAs(os.path.join(savedir,'Ratio_Plot_SDmass_Hbb.pdf'))
                  
        ax.hist(tdf['fj_sdmass'].values, bins=bins, weights = weight_uncut/np.sum(weight_uncut), lw=2, normed=False,
                        histtype='step',label='No tagging applied')
                                                    
        ax.set_xlabel(r'$\mathrm{m_{SD}\ [GeV]}$', ha='right', x=1.0)
        ax.set_ylabel(r'Normalized scale ({})'.format('Hbb'), ha='right', y=1.0)
        import matplotlib.ticker as plticker
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=20))
        ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=10))
        ax.yaxis.set_minor_locator(plticker.AutoMinorLocator(5))
        ax.set_xlim(40, 200)
        ax.set_ylim(0, 0.6)
        ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12)#, labelleft=False )

        ax.tick_params(direction='in', axis='both', which='minor' , length=6)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')    
        #ax.grid(which='minor', alpha=0.5, axis='y', linestyle='dotted')
        ax.grid(which='major', alpha=0.9, linestyle='dotted')
        leg = ax.legend(borderpad=1, frameon=False, loc='best', fontsize=16,
            title = ""+str(int(round((min(frame.fj_pt)))))+" $\mathrm{<\ jet\ p_T\ <}$ "+str(int(round((max(frame.fj_pt)))))+" GeV" \
              + "\n "+str(int(round((min(frame.fj_sdmass)))))+" $\mathrm{<\ jet\ m_{SD}\ <}$ "+str(int(round((max(frame.fj_sdmass)))))+" GeV"
                  + "\n {} tagging {}".format(taggerName, legend_siglab)           )
        leg._legend_box.align = "right"
        ax.annotate(eraText, xy=(0.75, 1.015), xycoords='axes fraction', fontname='Helvetica', ha='left',
                        bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$\mathbf{CMS}$', xy=(0, 1.015), xycoords='axes fraction', fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
                        bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$Simulation\ Open\ Data$', xy=(0.105, 1.015), xycoords='axes fraction', fontsize=18, fontstyle='italic', ha='left',

                        annotation_clip=False)
        f.savefig(os.path.join(savedir,'DDT_Hbb_tag'+'QCD_cut' + '.png'), dpi=400)
        f.savefig(os.path.join(savedir,'DDT_Hbb_tag'+'QCD_cut' + '.pdf'), dpi=400)
    
    def sculpting_multiple_taggers(tdf_array, names_array, siglab="Hbb", sculp_label='QCD', savedir="", sculpt_wp = 0.01):
        
        #if siglab == sculp_label: return 
        def find_nearest(array,value):
            idx = (np.abs(array-value)).argmin()
            return idx, array[idx]
        
        line_styles = ['--', '-.', ':', '--', '-.']
        line_widths = [2.5, 2.5, 3, 2.5, 2.5]
        
        if siglab[0] in ["H", "Z", "g"] and len(siglab) == 3:
            legend_siglab = '{} \\rightarrow {}'.format(siglab[0], siglab[-2]+'\\bar{'+siglab[-1]+'}') 
            legend_siglab = '$\mathrm{}$'.format('{'+legend_siglab+'}')
        else:
            legend_siglab = siglab
        if sculp_label[0] in ["H", "Z", "g"] and len(sculp_label) == 3:
            legend_bkglab = '{} \\rightarrow {}'.format(sculp_label[0], sculp_label[-2]+'\\bar{'+sculp_label[-1]+'}') 
            legend_bkglab = '$\mathrm{}$'.format('{'+legend_bkglab+'}')
        else: legend_bkglab = sculp_label
        
        f, ax = plt.subplots(figsize=(10,10))
        for tdf, name, style, width in zip(tdf_array, names_array, line_styles, line_widths):
           
            if name == "Interaction network": 
                
                truth, predict, db = roc_input(tdf, signal=[siglab], include = [siglab, 'QCD'])
                fpr, tpr, threshold = roc_curve(truth, predict)

                cuts = {}
                for wp in [1., sculpt_wp]: # % mistag rate
                    idx, val = find_nearest(fpr, wp)
                    cuts[str(wp)] = threshold[idx] # threshold for deep double-b corresponding to ~1% mistag rate


                bins = np.linspace(40,240,11)
                for wp, cut in reversed(cuts.items()):

                    ctdf = tdf[tdf['predict'+siglab].values > cut]
                    weight = ctdf['truth'+sculp_label].values
                    print(float(len(ctdf))/len(tdf))
                    
                    
                    if str(wp) == '1.0':
                        ax.hist(ctdf['fj_sdmass'].values, bins=bins, weights = weight/float(np.sum(weight)), 
                            linestyle='-', lw=4, normed=False,
                            histtype='stepfilled',label='No tagging applied', alpha=0.3)
                        
                    else:   
                        ax.hist(ctdf['fj_sdmass'].values, bins=bins, weights = weight/float(np.sum(weight)), lw=width, linestyle = style, normed=False,
                            histtype='step',label=' {}'.format(name))
                        
                 
                # DDT
                big_cuts = np.load('DDT_FPRcuts_sdmass.npy')
                ctdf = tdf.copy()
                ctdf = ctdf.head(0)
                FPR_cut = [float(sculpt_wp)]
                bins = np.linspace(40,240,11)
                dataframes_cut = [ctdf for i in range(len(FPR_cut))]
                for FPR in range(len(FPR_cut)):
                    cuts = big_cuts.item().get(str(round(FPR_cut[FPR], 5)))
            
                    dataframes_cut[FPR] = dataframes_cut[FPR].append(tdf.loc[tdf['predict'+siglab].values > cuts])

                h_list = []

                weight_uncut = tdf['truth'+sculp_label].values.astype(float)
                                    
                print(float(len(dataframes_cut[FPR]))/len(tdf))

                for FPR in range(len(FPR_cut)):
                    weight = dataframes_cut[FPR]['truth'+sculp_label].values.astype(float)
                    ax.hist(dataframes_cut[FPR]['fj_sdmass'].values, bins=bins, 
                            weights=weight/np.sum(weight), 
                            lw=3, 
                            normed=False,
                            histtype='step',
                            linestyle = '--', 
                            label='{}'.format('Interaction network, DDT')
                           )
            else: 
                print(name)
                truth, predict, db = roc_input(tdf, signal=[siglab], include = [siglab, 'QCD'])
                fpr, tpr, threshold = roc_curve(truth, predict)

                cuts = {}
                for wp in [sculpt_wp]: # % mistag rate
                    idx, val = find_nearest(fpr, wp)
                    cuts[str(wp)] = threshold[idx] # threshold for deep double-b corresponding to ~1% mistag rate


                bins = np.linspace(40,240,11)
                for wp, cut in reversed(cuts.items()):
                    
                    ctdf = tdf[tdf['predict'+siglab].values > cut]
                    weight = ctdf['truth'+sculp_label].values
                    
                    print(float(len(ctdf))/len(tdf))

                    if str(wp) != '1.0':
                        ax.hist(ctdf['fj_sdmass'].values, bins=bins, weights = weight/float(np.sum(weight)), 
                                lw=width, linestyle=style, normed=False,
                                histtype='step',label=' {}'.format(name))

                    #else: 
                     
                        #if name == "Deep double-b mass decor.":
                            #ax.hist(ctdf['fj_sdmass'].values, bins=bins, weights = weight/float(np.sum(weight)), lw=2, normed=False,
                            #        histtype='step',label='No tagging applied')
            
        

        frame = tdf_array[0]
        ax.set_xlabel(r'$\mathrm{m_{SD}\ [GeV]}$', ha='right', x=1.0)
        ax.set_ylabel(r'Normalized scale ({})'.format(legend_bkglab), ha='right', y=1.0)
        import matplotlib.ticker as plticker
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=20))
        ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=10))
        ax.yaxis.set_minor_locator(plticker.AutoMinorLocator(5))
        ax.set_xlim(40, 240)
        ax.set_ylim(0, 0.6)
        ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12)#, labelleft=False )
        ax.tick_params(direction='in', axis='both', which='minor' , length=6)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')    
        #ax.grid(which='minor', alpha=0.5, axis='y', linestyle='dotted')
        ax.grid(which='major', alpha=0.9, linestyle='dotted')
        leg = ax.legend(borderpad=1, frameon=False, loc='upper right', fontsize=16,
            title = ""+str(int(round((min(frame.fj_pt)))))+" $\mathrm{<\ jet\ p_T\ <}$ "+str(int(round((max(frame.fj_pt)))))+" GeV" \
              + "\n "+str(int(round((min(frame.fj_sdmass)))))+" $\mathrm{<\ jet\ m_{SD}\ <}$ "+str(int(round((max(frame.fj_sdmass)))))+" GeV" + 
                        '\n {}\% mistagging rate'.format(str(100*sculpt_wp))
                           )
        leg._legend_box.align = "right"
        ax.annotate(eraText, xy=(0.75, 1.015), xycoords='axes fraction', fontname='Helvetica', ha='left',
                        bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$\mathbf{CMS}$', xy=(0, 1.015), xycoords='axes fraction', fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
                        bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$Simulation\ Open\ Data$', xy=(0.105, 1.015), xycoords='axes fraction', fontsize=18, fontstyle='italic', ha='left',
                        annotation_clip=False)

        f.savefig(os.path.join(savedir,'M_sculpting_tag'+siglab+"_"+sculp_label+'.png'), dpi=400)
        f.savefig(os.path.join(savedir,'M_sculpting_tag'+siglab+"_"+sculp_label+'.pdf'), dpi=400)
        
        
    def sculpting(tdf, siglab="Hcc", sculp_label='Light', savedir="", taggerName=""):
        if siglab == sculp_label: return 
        def find_nearest(array,value):
            idx = (np.abs(array-value)).argmin()
            return idx, array[idx]
        
        if siglab[0] in ["H", "Z", "g"] and len(siglab) == 3:
            legend_siglab = '{} \\rightarrow {}'.format(siglab[0], siglab[-2]+'\\bar{'+siglab[-1]+'}') 
            legend_siglab = '$\mathrm{}$'.format('{'+legend_siglab+'}')
        else:
            legend_siglab = siglab
        if sculp_label[0] in ["H", "Z", "g"] and len(sculp_label) == 3:
            legend_bkglab = '{} \\rightarrow {}'.format(sculp_label[0], sculp_label[-2]+'\\bar{'+sculp_label[-1]+'}') 
            legend_bkglab = '$\mathrm{}$'.format('{'+legend_bkglab+'}')
        else: legend_bkglab = sculp_label

        truth, predict, db = roc_input(tdf, signal=[siglab], include = [siglab, sculp_label])
        fpr, tpr, threshold = roc_curve(truth, predict)

        cuts = {}
        for wp in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]: # % mistag rate
            idx, val = find_nearest(fpr, wp)
            cuts[str(wp)] = threshold[idx] # threshold for deep double-b corresponding to ~1% mistag rate
        
        f, ax = plt.subplots(figsize=(10,10))
        bins = np.linspace(40,200,9)
        for wp, cut in reversed(sorted(cuts.items())):
            
            
            ctdf = tdf[tdf['predict'+siglab].values > cut]
            weight = ctdf['truth'+sculp_label].values
            
            f, ax = plt.subplots(figsize=(10,10))

            if str(wp)=='1.0':
                ax.hist(ctdf['fj_sdmass'].values, bins=bins, weights = weight/np.sum(weight), lw=2, normed=False,
                        histtype='step',label='No tagging applied')
            else:
                ax.hist(ctdf['fj_sdmass'].values, bins=bins, weights = weight/np.sum(weight), lw=2, normed=False,
                        histtype='step',label='{}\%  mistagging rate'.format(float(wp)*100.))

        ax.set_xlabel(r'$\mathrm{m_{SD}\ [GeV]}$', ha='right', x=1.0)
        ax.set_ylabel(r'Normalized scale ({})'.format(legend_bkglab), ha='right', y=1.0)
        import matplotlib.ticker as plticker
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=20))
        ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=10))
        ax.yaxis.set_minor_locator(plticker.AutoMinorLocator(5))
        ax.set_xlim(40, 200)
        ax.set_ylim(0, 0.45)
        ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12)#, labelleft=False )
        ax.tick_params(direction='in', axis='both', which='minor' , length=6)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')    
        #ax.grid(which='minor', alpha=0.5, axis='y', linestyle='dotted')
        ax.grid(which='major', alpha=0.9, linestyle='dotted')
        leg = ax.legend(borderpad=1, frameon=False, loc='best', fontsize=16,
            title = ""+str(int(round((min(frame.fj_pt)))))+" $\mathrm{<\ jet\ p_T\ <}$ "+str(int(round((max(frame.fj_pt)))))+" GeV" \
              + "\n "+str(int(round((min(frame.fj_sdmass)))))+" $\mathrm{<\ jet\ m_{SD}\ <}$ "+str(int(round((max(frame.fj_sdmass)))))+" GeV"
                  + "\n {} tagging {}".format(taggerName, legend_siglab)           )
        leg._legend_box.align = "right"
        ax.annotate(eraText, xy=(0.75, 1.015), xycoords='axes fraction', fontname='Helvetica', ha='left',
                        bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$\mathbf{CMS}$', xy=(0, 1.015), xycoords='axes fraction', fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
                        bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$Simulation\ Open\ Data$', xy=(0.105, 1.015), xycoords='axes fraction', fontsize=18, fontstyle='italic', ha='left',
                        annotation_clip=False)
        f.savefig(os.path.join(savedir,'M_sculpting_tag'+siglab+"_"+sculp_label+'.png'), dpi=400)
        f.savefig(os.path.join(savedir,'M_sculpting_tag'+siglab+"_"+sculp_label+'.pdf'), dpi=400)
        #return
        f, ax = plt.subplots(figsize=(10,10))
        bins = np.linspace(300,2000,20)
        for wp, cut in reversed(sorted(cuts.items())):
            ctdf = tdf[tdf['predict'+siglab].values > cut]
            weight = ctdf['truth'+sculp_label].values
            ax.hist(ctdf['fj_pt'].values, bins=bins, weights = weight/np.sum(weight), lw=2, normed=False,
                    histtype='step',label='{}\%  mistagging rate'.format(float(wp)*100.))
        
        ax.set_xlabel(r'$\mathrm{p_T\ [GeV]}$', ha='right', x=1.0)
        #ax.set_ylabel(r'Normalized scale ({})'.format(sculp_label.replace("Hcc", r"$\mathrm{H \rightarrow c\bar{c}}$").replace("Hbb", r"$\mathrm{H \rightarrow b\bar{b}}$")), ha='right', y=1.0)
        ax.set_ylabel(r'Normalized scale ({})'.format(legend_bkglab), ha='right', y=1.0)
        import matplotlib.ticker as plticker
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=200))
        ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=50))
        ax.yaxis.set_minor_locator(plticker.AutoMinorLocator(5))
        ax.set_xlim(300, 2000)
        ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12)#, labelleft=False )
        ax.tick_params(direction='in', axis='both', which='minor' , length=6)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')    
        #ax.grid(which='minor', alpha=0.5, axis='y', linestyle='dotted')
        ax.grid(which='major', alpha=0.9, linestyle='dotted')
        leg = ax.legend(borderpad=1, frameon=False, loc='best', fontsize=16,
            title = ""+str(int(round((min(frame.fj_pt)))))+" $\mathrm{<\ jet\ p_T\ <}$ "+str(int(round((max(frame.fj_pt)))))+" GeV" \
                        + "\n "+str(int(round((min(frame.fj_sdmass)))))+" $\mathrm{<\ jet\ m_{SD}\ <}$ "+str(int(round((max(frame.fj_sdmass)))))+" GeV"\
                + "\n Tagging {}".format(legend_siglab)           )
        leg._legend_box.align = "right"
        ax.annotate(eraText, xy=(0.75, 1.015), xycoords='axes fraction', fontname='Helvetica', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$\mathbf{CMS}$', xy=(0, 1.015), xycoords='axes fraction', fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$Simulation\ Open\ Data$', xy=(0.105, 1.015), xycoords='axes fraction', fontsize=18, fontstyle='italic', ha='left',
            annotation_clip=False)

        
        f.savefig(os.path.join(savedir,'Pt_sculpting_tag'+siglab+"_"+sculp_label+'.png'), dpi=400)
        
        f, ax = plt.subplots(figsize=(10,10))
        bins = np.linspace(-8,-1,20)
        for wp, cut in reversed(sorted(cuts.items())):
            ctdf = tdf[tdf['predict'+siglab].values > cut]
            weight = ctdf['truth'+sculp_label].values
            if str(wp)=='1.0':
                ax.hist(ctdf['fj_rho'].values, bins=bins, weights = weight/np.sum(weight), lw=2, normed=False,
                        histtype='step',label='No tagging applied')
            else:
                ax.hist(ctdf['fj_rho'].values, bins=bins, weights = weight/np.sum(weight), lw=2, normed=False,
                        histtype='step',label='{}\%  mistagging rate'.format(float(wp)*100.))
        
        ax.set_xlabel(r'$\mathrm{\rho}$', ha='right', x=1.0)
        ax.set_ylabel(r'Normalized scale ({})'.format('QCD'), ha='right', y=1.0)
        import matplotlib.ticker as plticker
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1))
        ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=10))
        ax.yaxis.set_minor_locator(plticker.AutoMinorLocator(5))
        ax.set_xlim(-8, -1)
        ax.set_ylim(0, 0.2)
        ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12)#, labelleft=False )

        ax.tick_params(direction='in', axis='both', which='minor' , length=6)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')    
        #ax.grid(which='minor', alpha=0.5, axis='y', linestyle='dotted')
        ax.grid(which='major', alpha=0.9, linestyle='dotted')
        leg = ax.legend(borderpad=1, frameon=False, loc='best', fontsize=16,
            title = ""+str(int(round((min(tdf.fj_rho)))))+" $\mathrm{< rho <}$ "+str(int(round((max(tdf.fj_rho)))))+" GeV" \
                  + "\n {} tagging {}".format(taggerName, legend_siglab)           )
        leg._legend_box.align = "right"
        ax.annotate(eraText, xy=(0.75, 1.015), xycoords='axes fraction', fontname='Helvetica', ha='left',
                        bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$\mathbf{CMS}$', xy=(0, 1.015), xycoords='axes fraction', fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
                        bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$Simulation\ Open\ Data$', xy=(0.105, 1.015), xycoords='axes fraction', fontsize=18, fontstyle='italic', ha='left',

                        annotation_clip=False)
        f.savefig(os.path.join(savedir,'DDTrho_QCD_tag'+'QCD_cut' + '.png'), dpi=400)
        f.savefig(os.path.join(savedir,'DDTrho_QCD_tag'+'QCD_cut' + '.pdf'), dpi=400)
        
        
        plt.close(f)
        
    def eta_dep(xdf, sig_label="Hcc", bkg_label="", savedir="", taggerName=""):
        if sig_label == bkg_label: return
            
        if sig_label[0] in ["H", "Z", "g"] and len(sig_label) == 3:
            legend_siglab = '{} \\rightarrow {}'.format(sig_label[0], sig_label[-2]+'\\bar{'+sig_label[-1]+'}') 
            legend_siglab = '$\mathrm{}$'.format('{'+legend_siglab+'}')
        else:
            legend_siglab = sig_label
            
        def roc(xdf, etalow=300, etahigh=2500, verbose=False):
            tdf = xdf[(xdf.fj_eta < etahigh) & (xdf.fj_eta>etalow)]
            truth, predict, db = roc_input(tdf, signal=[sig_label], include = [sig_label, bkg_label])
            fpr, tpr, threshold = roc_curve(truth, predict)
            return fpr, tpr
        
        step = float(5)/10
        pts = np.arange(-2.5,2.5,step)


        efftight, effloose = [], []
        mistight, misloose = [], []
        def find_nearest(array,value):
                idx = (np.abs(array-value)).argmin()
                return idx, array[idx]

        for et in pts:
            fpr, tpr = roc(xdf, etalow=et, etahigh=et+step)
            ix, mistag =  find_nearest(fpr, 0.1)
            effloose.append(tpr[ix])
            ix, mistag =  find_nearest(fpr, 0.01)
            efftight.append(tpr[ix])
            ix, eff =  find_nearest(tpr, 0.9)
            misloose.append(fpr[ix])
            ix, eff =  find_nearest(tpr, 0.5)
            mistight.append(fpr[ix])

        # Pad endpoints
        pts = np.concatenate((np.array([-2.5]), pts))
        pts = np.concatenate((pts, np.array([2.5])))
        effloose = [effloose[0]] + effloose + [effloose[-1]]
        efftight = [efftight[0]] + efftight + [efftight[-1]]
        misloose = [misloose[0]] + misloose + [misloose[-1]]
        mistight = [mistight[0]] + mistight + [mistight[-1]]

        f, ax = plt.subplots(figsize=(10,10))
        ax.step(pts, effloose, lw=2, label='10\% mistagging rate', c='black', where='post')
        ax.step(pts, efftight, lw=2, label='1\% mistagging rate', c='black', linestyle='dashed')
        ax.step([],[], label='90\% efficiency', c='red', where='post')
        ax.step([],[], label='50\% efficiency', c='red', where='post', linestyle='dashed')
        
        ax2 = ax.twinx()
        ax2.step(pts, mistight, lw=2, label='40\% efficiency', c='red', where='post', linestyle='dashed')
        ax2.step(pts, misloose, lw=2, label='80\% efficiency', c='red', where='post')
        #ax2.yaxis.set_major_locator(plticker.MultipleLocator(base=0.05))
        #ax2.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.01))
        ax2.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12)
        ax2.tick_params(direction='in', axis='both', which='minor' , length=6)
        ax2.yaxis.set_ticks_position('right')
        ax2.tick_params('y', which='both', colors='red')
        ax2.semilogy()
        ax2.set_ylim(0.0002, 1)
        
        ax.set_xlabel(r'$\mathrm{\eta}$', ha='right', x=1.0)
        ylab1 = ['{} \\rightarrow {}'.format(l[0], l[-2]+'\\bar{'+l[-1]+'}') if len(l) == 3 and l[0] in ["H", "Z", "g"] else l for l in [sig_label] ][0]
        ax.set_ylabel(r'Tagging efficiency ($\mathrm{}$)'.format("{"+ylab1+"}"), ha='right', y=1.0, color='black')
        ylab2 = ['{} \\rightarrow {}'.format(l[0], l[-2]+'\\bar{'+l[-1]+'}') if len(l) == 3 and l[0] in ["H", "Z", "g"] else l for l in [bkg_label] ][0]
        ax2.set_ylabel(r'Mistagging rate ($\mathrm{}$)'.format("{"+ylab2+"}"), ha='right', y=1.0, color='red')
        ax2.get_yaxis().set_label_coords(1.05,1)
        import matplotlib.ticker as plticker
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
        ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.1))
        ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
        ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.02))
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0, 1.3)
        ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12)
        ax.tick_params(direction='in', axis='both', which='minor' , length=6)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('left')    
        #ax.grid(which='minor', alpha=0.5, axis='y', linestyle='dotted')
        ax.grid(which='major', alpha=0.9, linestyle='dotted')
        leg = ax.legend(borderpad=1, frameon=False, loc=1, fontsize=16 )
        mpt = ""+str(int(round((min(frame.fj_pt)))))+" $\mathrm{<\ jet\ p_T\ <}$ "+str(int(round((max(frame.fj_pt)))))+" GeV"       + "\n "+str(int(round((min(frame.fj_sdmass)))))+" $\mathrm{<\ jet\ m_{SD}\ <}$ "+str(int(round((max(frame.fj_sdmass)))))+" GeV" + "\n {}".format(taggerName)
        ax.annotate( mpt, xy=(0.05, 0.85), xycoords='axes fraction', fontname='Helvetica', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        leg._legend_box.align = "right"
        ax.annotate(eraText, xy=(0.75, 1.015), xycoords='axes fraction', fontname='Helvetica', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$\mathbf{CMS}$', xy=(0, 1.015), xycoords='axes fraction', fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$Simulation\ Open\ Data$', xy=(0.105, 1.015), xycoords='axes fraction', fontsize=18, fontstyle='italic', ha='left',
            annotation_clip=False)

        f.savefig(os.path.join(savedir,'eta_dep_'+sig_label+"_vs_"+bkg_label+'.png'), dpi=400)
        f.savefig(os.path.join(savedir,'eta_dep_'+sig_label+"_vs_"+bkg_label+'.pdf'), dpi=400)
        plt.close(f)
        

    def pt_dep(xdf, sig_label="Hcc", bkg_label="", savedir="", taggerName=""):
        if sig_label == bkg_label: return
            
        if sig_label[0] in ["H", "Z", "g"] and len(sig_label) == 3:
            legend_siglab = '{} \\rightarrow {}'.format(sig_label[0], sig_label[-2]+'\\bar{'+sig_label[-1]+'}') 
            legend_siglab = '$\mathrm{}$'.format('{'+legend_siglab+'}')
        else:
            legend_siglab = sig_label
            
        def roc(xdf, ptlow=300, pthigh=2500, verbose=False):
            tdf = xdf[(xdf.fj_pt < pthigh) & (xdf.fj_pt>ptlow)]
            truth, predict, db = roc_input(tdf, signal=[sig_label], include = [sig_label, bkg_label])
            fpr, tpr, threshold = roc_curve(truth, predict)
            return fpr, tpr
        
        step = float((2000-300))/10
        pts = np.arange(300,2000,step)


        efftight, effloose = [], []
        mistight, misloose = [], []
        def find_nearest(array,value):
                idx = (np.abs(array-value)).argmin()
                return idx, array[idx]

        for pt in pts:
            fpr, tpr = roc(xdf, pt,pt+step)
            ix, mistag =  find_nearest(fpr, 0.1)
            effloose.append(tpr[ix])
            ix, mistag =  find_nearest(fpr, 0.01)
            efftight.append(tpr[ix])
            ix, eff =  find_nearest(tpr, 0.9)
            misloose.append(fpr[ix])
            ix, eff =  find_nearest(tpr, 0.5)
            mistight.append(fpr[ix])

        # Pad endpoints
        pts = np.concatenate((np.array([300]), pts))
        pts = np.concatenate((pts, np.array([2000])))
        effloose = [effloose[0]] + effloose + [effloose[-1]]
        efftight = [efftight[0]] + efftight + [efftight[-1]]
        misloose = [misloose[0]] + misloose + [misloose[-1]]
        mistight = [mistight[0]] + mistight + [mistight[-1]]

        f, ax = plt.subplots(figsize=(10,10))
        
        ax.step(pts, effloose, lw=2, label='10\% mistagging rate', c='black', where='post')
        ax.step(pts, efftight, lw=2, label='1\% mistagging rate', c='black', linestyle='dashed')
        ax.step([],[], label='90\% efficiency', c='red', where='post')
        ax.step([],[], label='50\% efficiency', c='red', where='post', linestyle='dashed')
        
        ax2 = ax.twinx()
        ax2.step(pts, mistight, lw=2, label='50\% efficiency', c='red', where='post', linestyle='dashed')
        ax2.step(pts, misloose, lw=2, label='90\% efficiency', c='red', where='post')
        ax2.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12)
        ax2.tick_params(direction='in', axis='both', which='minor' , length=6)
        ax2.yaxis.set_ticks_position('right')
        ax2.tick_params('y', which='both', colors='red')
        ax2.semilogy()
        ax2.set_ylim(0.0002, 1)
        
        ax.set_xlabel(r'$\mathrm{p_T\ [GeV]}$', ha='right', x=1.0)
        ylab1 = ['{} \\rightarrow {}'.format(l[0], l[-2]+'\\bar{'+l[-1]+'}') if len(l) == 3 and l[0] in ["H", "Z", "g"] else l for l in [sig_label] ][0]
        ax.set_ylabel(r'Tagging efficiency ($\mathrm{}$)'.format("{"+ylab1+"}"), ha='right', y=1.0, color='black')
        #ax.set_ylabel(r'Tagging efficiency ({})'.format(sig_label.replace("Hcc", r"$\mathrm{H \rightarrow c\bar{c}}$").replace("Hbb", r"$\mathrm{H \rightarrow b\bar{b}}$")), ha='right', y=1.0, color='black')
        ylab2 = ['{} \\rightarrow {}'.format(l[0], l[-2]+'\\bar{'+l[-1]+'}') if len(l) == 3 and l[0] in ["H", "Z", "g"] else l for l in [bkg_label] ][0]
        ax2.set_ylabel(r'Mistagging rate ($\mathrm{}$)'.format("{"+ylab2+"}"), ha='right', y=1.0, color='red')
        #ax2.set_ylabel(r'Mistagging rate ({})'.format(bkg_label.replace("Hcc", r"$\mathrm{H \rightarrow c\bar{c}}$").replace("Hbb", r"$\mathrm{H \rightarrow b\bar{b}}$")), ha='right', y=1.0, color='red')
        ax2.get_yaxis().set_label_coords(1.05,1)
        import matplotlib.ticker as plticker
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=200))
        ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=50))
        ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
        ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.02))
        ax.set_xlim(300, 2000)
        ax.set_ylim(0, 1.3)
        ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12)
        ax.tick_params(direction='in', axis='both', which='minor' , length=6)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('left')    
        #ax.grid(which='minor', alpha=0.5, axis='y', linestyle='dotted')
        ax.grid(which='major', alpha=0.9, linestyle='dotted')
        leg = ax.legend(borderpad=1, frameon=False, loc=1, fontsize=16 )
        mpt = ""+str(int(round((min(frame.fj_pt)))))+" $\mathrm{<\ jet\ p_T\ <}$ "+str(int(round((max(frame.fj_pt)))))+" GeV"       + "\n "+str(int(round((min(frame.fj_sdmass)))))+" $\mathrm{<\ jet\ m_{SD}\ <}$ "+str(int(round((max(frame.fj_sdmass)))))+" GeV" + "\n {}".format(taggerName)
        ax.annotate( mpt, xy=(0.05, 0.85), xycoords='axes fraction', fontname='Helvetica', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        leg._legend_box.align = "right"
        ax.annotate(eraText, xy=(0.75, 1.015), xycoords='axes fraction', fontname='Helvetica', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$\mathbf{CMS}$', xy=(0, 1.015), xycoords='axes fraction', fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$Simulation\ Open\ Data$', xy=(0.105, 1.015), xycoords='axes fraction', fontsize=18, fontstyle='italic', ha='left',
            annotation_clip=False)

        
        f.savefig(os.path.join(savedir,'Pt_dep_'+sig_label+"_vs_"+bkg_label+'.png'), dpi=400)
        f.savefig(os.path.join(savedir,'Pt_dep_'+sig_label+"_vs_"+bkg_label+'.pdf'), dpi=400)
        plt.close(f)

    def multiple_pu_dep(tdf_array, names_array, sig_label="Hcc", bkg_label="", savedir="", sculpt_wp = 0.01):
        if sig_label == bkg_label: return
            
        if sig_label[0] in ["H", "Z", "g"] and len(sig_label) == 3:
            legend_siglab = '{} \\rightarrow {}'.format(sig_label[0], sig_label[-2]+'\\bar{'+sig_label[-1]+'}') 
            legend_siglab = '$\mathrm{}$'.format('{'+legend_siglab+'}')
        else:
            legend_siglab = sig_label
         
        def cut_pu(xdf, pulow=0, puhigh=50, verbose=False):
            # npv
            tdf = xdf[(xdf.npv < puhigh) & (xdf.npv >= pulow)]
            return tdf 
        
        def roc(xdf, pulow=0, puhigh=50, verbose=False):
            # npv
            tdf = xdf[(xdf.npv < puhigh) & (xdf.npv >= pulow)]
            # ntrueInt
            #tdf = xdf[(xdf.ntrueInt < puhigh) & (xdf.ntrueInt >= pulow)]            
            truth, predict, db = roc_input(tdf, signal=[sig_label], include = [sig_label, bkg_label])
            fpr, tpr, threshold = roc_curve(truth, predict)
            return fpr, tpr
        
        line_styles = ['--', '-.', ':', '--', '-.']
        line_widths = [2.5, 2.5, 3, 2.5, 2.5]
        
        print(names_array)
        
        f, ax = plt.subplots(figsize=(10,10))
        for xdf, taggerName, style, width in zip(tdf_array, names_array, line_styles, line_widths):
        
         
            step = 5
            pts = np.arange(0,50,step)

        
        
            efftight, effloose = [], []
            mistight, misloose = [], []
            def find_nearest(array,value):
                    idx = (np.abs(array-value)).argmin()
                    return idx, array[idx]

            for pu in pts:
                fpr, tpr = roc(xdf, pu,pu+step)
                ix, mistag =  find_nearest(fpr, 0.01)
                efftight.append(tpr[ix])
            

            # Pad endpoints
            pts = np.concatenate((np.array([0]), pts))
            pts = np.concatenate((pts, np.array([50])))

            efftight = [efftight[0]] + efftight + [efftight[-1]]
            ax.step(pts, efftight, lw=width, label=taggerName, linestyle=style)

            if taggerName == 'Interaction network':
                
                step = 5
                pts = np.arange(0,50,step)



                efftight, effloose = [], []
                mistight, misloose = [], []
                

                for pu in pts:
                    
                    efftight.append(make_TPR_DDT(cut_pu(xdf,pu,pu+step), 1., siglab='QCD', sculp_label='fj_sdmass', savedir=savedir, taggerName=taggerName))

                pts = np.concatenate((np.array([0]), pts))
                pts = np.concatenate((pts, np.array([50])))

                efftight = [efftight[0]] + efftight + [efftight[-1]]
                ax.step(pts, efftight, lw=width, label='Interaction network, DDT', linestyle='-')   
            
        
        ax.set_xlabel(r'Reconstructed primary vertices', ha='right', x=1.0)
       
        ylab1 = ['{} \\rightarrow {}'.format(l[0], l[-2]+'\\bar{'+l[-1]+'}') if len(l) == 3 and l[0] in ["H", "Z", "g"] else l for l in [sig_label] ][0]
        ax.set_ylabel(r'Tagging efficiency ($\mathrm{}$)'.format("{"+ylab1+"}"), ha='right', y=1.0, color='black')
        #ax.set_ylabel(r'Tagging efficiency ({})'.format(sig_label.replace("Hcc", r"$\mathrm{H \rightarrow c\bar{c}}$").replace("Hbb", r"$\mathrm{H \rightarrow b\bar{b}}$")), ha='right', y=1.0, color='black')
    
        import matplotlib.ticker as plticker
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=10))
        ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=2))
        ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
        ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.02))
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 1.3)
        ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12)
        ax.tick_params(direction='in', axis='both', which='minor' , length=6)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('left')    
        #ax.grid(which='minor', alpha=0.5, axis='y', linestyle='dotted')
        ax.grid(which='major', alpha=0.9, linestyle='dotted')
        leg = ax.legend(borderpad=1, frameon=False, loc=1, fontsize=14 )
        mpt = ""+str(int(round((min(xdf.fj_pt)))))+" $\mathrm{<\ jet\ p_T\ <}$ "+str(int(round((max(xdf.fj_pt)))))+" GeV"       + "\n "+str(int(round((min(xdf.fj_sdmass)))))+" $\mathrm{<\ jet\ m_{SD}\ <}$ "+str(int(round((max(xdf.fj_sdmass)))))+" GeV" + "\n {}".format('1\% mistagging rate')
        ax.annotate( mpt, xy=(0.05, 0.85), xycoords='axes fraction', fontname='Helvetica', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        leg._legend_box.align = "right"
        ax.annotate(eraText, xy=(0.75, 1.015), xycoords='axes fraction', fontname='Helvetica', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$\mathbf{CMS}$', xy=(0, 1.015), xycoords='axes fraction', fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$Simulation\ Open\ Data$', xy=(0.105, 1.015), xycoords='axes fraction', fontsize=18, fontstyle='italic', ha='left',
            annotation_clip=False)

        
        f.savefig(os.path.join(savedir,'PU_dep_'+sig_label+"_vs_"+bkg_label+'.png'), dpi=400)
        f.savefig(os.path.join(savedir,'PU_dep_'+sig_label+"_vs_"+bkg_label+'.pdf'), dpi=400)
        plt.close(f)
    
    
    
    def pu_dep(xdf, sig_label="Hcc", bkg_label="", savedir="", taggerName=""):
        if sig_label == bkg_label: return
            
        if sig_label[0] in ["H", "Z", "g"] and len(sig_label) == 3:
            legend_siglab = '{} \\rightarrow {}'.format(sig_label[0], sig_label[-2]+'\\bar{'+sig_label[-1]+'}') 
            legend_siglab = '$\mathrm{}$'.format('{'+legend_siglab+'}')
        else:
            legend_siglab = sig_label
            
        def roc(xdf, pulow=0, puhigh=50, verbose=False):
            # npv
            tdf = xdf[(xdf.npv < puhigh) & (xdf.npv >= pulow)]
            # ntrueInt
            #tdf = xdf[(xdf.ntrueInt < puhigh) & (xdf.ntrueInt >= pulow)]            
            truth, predict, db = roc_input(tdf, signal=[sig_label], include = [sig_label, bkg_label])
            fpr, tpr, threshold = roc_curve(truth, predict)
            return fpr, tpr
        
        step = 5
        pts = np.arange(0,50,step)


        efftight, effloose = [], []
        mistight, misloose = [], []
        def find_nearest(array,value):
                idx = (np.abs(array-value)).argmin()
                return idx, array[idx]

        for pu in pts:
            fpr, tpr = roc(xdf, pu,pu+step)
            ix, mistag =  find_nearest(fpr, 0.1)
            effloose.append(tpr[ix])
            ix, mistag =  find_nearest(fpr, 0.01)
            efftight.append(tpr[ix])
            ix, eff =  find_nearest(tpr, 0.9)
            misloose.append(fpr[ix])
            ix, eff =  find_nearest(tpr, 0.5)
            mistight.append(fpr[ix])

        # Pad endpoints
        pts = np.concatenate((np.array([0]), pts))
        pts = np.concatenate((pts, np.array([50])))
        effloose = [effloose[0]] + effloose + [effloose[-1]]
        efftight = [efftight[0]] + efftight + [efftight[-1]]
        misloose = [misloose[0]] + misloose + [misloose[-1]]
        mistight = [mistight[0]] + mistight + [mistight[-1]]

        f, ax = plt.subplots(figsize=(10,10))
        
        ax.step(pts, effloose, lw=2, label='10\% mistagging rate', c='black', where='post')
        ax.step(pts, efftight, lw=2, label='1\% mistagging rate', c='black', linestyle='dashed')
        ax.step([],[], label='90\% efficiency', c='red', where='post')
        ax.step([],[], label='50\% efficiency', c='red', where='post', linestyle='dashed')
        
        ax2 = ax.twinx()
        ax2.step(pts, mistight, lw=2, label='90\% efficiency', c='red', where='post', linestyle='dashed')
        ax2.step(pts, misloose, lw=2, label='50\% efficiency', c='red', where='post')
        ax2.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12)
        ax2.tick_params(direction='in', axis='both', which='minor' , length=6)
        ax2.yaxis.set_ticks_position('right')
        ax2.tick_params('y', which='both', colors='red')
        ax2.semilogy()
        ax2.set_ylim(0.0002, 1)
        
        ax.set_xlabel(r'Reconstructed primary vertices', ha='right', x=1.0)
        ylab1 = ['{} \\rightarrow {}'.format(l[0], l[-2]+'\\bar{'+l[-1]+'}') if len(l) == 3 and l[0] in ["H", "Z", "g"] else l for l in [sig_label] ][0]
        ax.set_ylabel(r'Tagging efficiency ($\mathrm{}$)'.format("{"+ylab1+"}"), ha='right', y=1.0, color='black')
        #ax.set_ylabel(r'Tagging efficiency ({})'.format(sig_label.replace("Hcc", r"$\mathrm{H \rightarrow c\bar{c}}$").replace("Hbb", r"$\mathrm{H \rightarrow b\bar{b}}$")), ha='right', y=1.0, color='black')
        ylab2 = ['{} \\rightarrow {}'.format(l[0], l[-2]+'\\bar{'+l[-1]+'}') if len(l) == 3 and l[0] in ["H", "Z", "g"] else l for l in [bkg_label] ][0]
        ax2.set_ylabel(r'Mistagging rate ($\mathrm{}$)'.format("{"+ylab2+"}"), ha='right', y=1.0, color='red')
        #ax2.set_ylabel(r'Mistagging rate ({})'.format(bkg_label.replace("Hcc", r"$\mathrm{H \rightarrow c\bar{c}}$").replace("Hbb", r"$\mathrm{H \rightarrow b\bar{b}}$")), ha='right', y=1.0, color='red')
        ax2.get_yaxis().set_label_coords(1.05,1)
        import matplotlib.ticker as plticker
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=10))
        ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=2))
        ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
        ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.02))
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 1.3)
        ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12)
        ax.tick_params(direction='in', axis='both', which='minor' , length=6)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('left')    
        #ax.grid(which='minor', alpha=0.5, axis='y', linestyle='dotted')
        ax.grid(which='major', alpha=0.9, linestyle='dotted')
        leg = ax.legend(borderpad=1, frameon=False, loc=1, fontsize=16 )
        mpt = ""+str(int(round((min(frame.fj_pt)))))+" $\mathrm{<\ jet\ p_T\ <}$ "+str(int(round((max(frame.fj_pt)))))+" GeV"       + "\n "+str(int(round((min(frame.fj_sdmass)))))+" $\mathrm{<\ jet\ m_{SD}\ <}$ "+str(int(round((max(frame.fj_sdmass)))))+" GeV" + "\n {}".format(taggerName)
        ax.annotate( mpt, xy=(0.05, 0.85), xycoords='axes fraction', fontname='Helvetica', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        leg._legend_box.align = "right"
        ax.annotate(eraText, xy=(0.75, 1.015), xycoords='axes fraction', fontname='Helvetica', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$\mathbf{CMS}$', xy=(0, 1.015), xycoords='axes fraction', fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$Simulation\ Open\ Data$', xy=(0.105, 1.015), xycoords='axes fraction', fontsize=18, fontstyle='italic', ha='left',
            annotation_clip=False)

        
        f.savefig(os.path.join(savedir,'PU_dep_'+sig_label+"_vs_"+bkg_label+'.png'), dpi=400)
        f.savefig(os.path.join(savedir,'PU_dep_'+sig_label+"_vs_"+bkg_label+'.pdf'), dpi=400)
        plt.close(f)
        
    def overlay_distribution(xdf, savedir="", feature='pt' ,truths=["Hcc", "QCD"], app_weight=True):  
            all_labels = ["QCD", "Light", "Hbb", "Hcc", "gbb", "gcc", "Zbb", "Zcc"]    
            trus = []
            for tru in truths:
                if tru != "":
                    t = xdf["truth"+tru].values
                else: t = np.zeros(len(xdf))
                trus.append(t)
            
            # Get histo values
            if feature == "pt":
                pred = xdf['fj_pt'].values
            elif feature == "m":
                pred = xdf['fj_sdmass'].values
            elif feature == "eta":
                pred = xdf['fj_eta'].values
            elif feature in all_labels:
                pred = xdf['predict'+feature].values
            else: 
                return

            # Set Alpha        
            if np.sum([ 1 for tru in truths if len(tru) > 1]) > 4: a = 0.5
            elif np.sum([ 1 for tru in truths if len(tru) > 1]) > 2: a = 0.7
            else: a = 0.8

            legend_labels = ['{} \\rightarrow {}'.format(l[0], l[-2]+'\\bar{'+l[-1]+'}') if len(l) == 3 and l[0] in ["H", "Z", "g"] else l for l in all_labels ]

            f, ax = plt.subplots(figsize=(10,10))
            if feature == "m":
                ax.set_xlim(40,200)
                bins = np.linspace(40,200,33)
            elif feature == "eta":
                ax.set_xlim(-3.,3)
                bins = np.linspace(-3,3,33)
            elif feature == "pt":
                ax.set_xlim(300,2000)
                bins = np.linspace(300,2000,33)
            elif feature in all_labels:
                ax.set_xlim(0,1)
                bins = np.linspace(0,1,33)
            else:
                return

            n_max = 0
            for tru, weight in zip(truths, trus):
                if app_weight:
                    if tru == "":
                        ax.hist(pred, bins=bins, weights = weight*xdf.Weight/np.sum(weight*xdf.Weight),
                                alpha=a, normed=False, label="")# Mock plot
                    else:
                        ax.hist(pred, bins=bins, weights = weight*xdf.Weight/np.sum(weight*xdf.Weight),
                                alpha=a, normed=False,
                            label=r'Weighted $\mathrm{}$'.format('{'+legend_labels[all_labels.index(tru)]+'}') )    
                else: 
                    if tru == "":
                        ax.hist(pred, bins=bins, weights = weight/np.sum(weight),
                                alpha=a, normed=False, label="")# Mock plot
                    else:
                        ax.hist(pred, bins=bins, weights = weight/np.sum(weight),
                                alpha=a, normed=False, 
                        label=r'$\mathrm{}$'.format('{'+legend_labels[all_labels.index(tru)]+'}') )    

            if feature == "m":
                ax.set_xlabel(r'$\mathrm{m_{SD}\ [GeV]}$', ha='right', x=1.0)
            elif feature == "eta":
                ax.set_xlabel(r'$\mathrm{\eta}$', ha='right', x=1.0)
            elif feature == "pt":
                ax.set_xlabel(r'$\mathrm{p_T\ [GeV]}$', ha='right', x=1.0)
            elif feature in all_labels:
                try: ax.set_xlabel(r'Dicriminator $\mathrm{}$'.format('{'+legend_labels[all_labels.index(feature)]+'}'), ha='right', x=1.0)
                except: return
            else:
                return

            ax.set_ylabel(r'Normalized scale', ha='right', y=1.0)
            import matplotlib.ticker as plticker
            if feature == "m":
                ax.xaxis.set_major_locator(plticker.MultipleLocator(base=20))
                ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=5))
                ax.yaxis.set_minor_locator(plticker.AutoMinorLocator(5))
            elif feature == "eta":
                ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1))
                ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.2))
                ax.yaxis.set_minor_locator(plticker.AutoMinorLocator(5))
            elif feature == "pt":
                ax.xaxis.set_major_locator(plticker.MultipleLocator(base=200))
                ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=50))
                ax.yaxis.set_minor_locator(plticker.AutoMinorLocator(5))
            elif feature in all_labels:
                ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.2))
                ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.04))
                ax.yaxis.set_minor_locator(plticker.AutoMinorLocator(5))
            else:
                return
            ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12 )
            ax.tick_params(direction='in', axis='both', which='minor' , length=6)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')    
            #ax.semilogy()
            ax.grid(which='minor', alpha=0.5, axis='y', linestyle='dotted')
            ax.grid(which='major', alpha=0.9, linestyle='dotted')
            leg = ax.legend(borderpad=1, frameon=False, loc=1, fontsize=16,
                title = ""+str(int(round((min(frame.fj_pt)))))+" $\mathrm{<\ jet\ p_T\ <}$ "+str(int(round((max(frame.fj_pt)))))+" GeV" \
                  + "\n "+str(int(round((min(frame.fj_sdmass)))))+" $\mathrm{<\ jet\ m_{SD}\ <}$ "+str(int(round((max(frame.fj_sdmass)))))+" GeV"
                            + "\n "+taggerName)
            leg._legend_box.align = "left"
            ax.annotate(eraText, xy=(0.75, 1.015), xycoords='axes fraction', fontname='Helvetica', ha='left',
                bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
            ax.annotate('$\mathbf{CMS}$', xy=(0, 1.015), xycoords='axes fraction', fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
                bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
            ax.annotate('$Simulation\ Open\ Data$', xy=(0.105, 1.015), xycoords='axes fraction', fontsize=18, fontstyle='italic', ha='left',
                annotation_clip=False)

            if app_weight:
                f.savefig(os.path.join(savedir, 'dist_weight_'+feature+'_'+"-".join(t for t in truths if len(t)>1)+'.pdf'), dpi=400)
                f.savefig(os.path.join(savedir, 'dist_weight_'+feature+'_'+"-".join(t for t in truths if len(t)>1)+'.png'), dpi=400)
            else:
                f.savefig(os.path.join(savedir, 'dist_'+feature+'_'+"-".join(t for t in truths if len(t)>1)+'.pdf'), dpi=400)
                f.savefig(os.path.join(savedir, 'dist_'+feature+'_'+"-".join(t for t in truths if len(t)>1)+'.png'), dpi=400)
            plt.close(f)                                
            print()
    
    
    # plot BIG comparison ROC - hardcoded for now
    make_dirs(os.path.join(outputDir,'Plots'))

    plot_rocs(dfs=[cut(frame) for frame in dataframes],
              savedir=os.path.join(outputDir,'Plots'),
              names=taggerNames,
              sigs=[['Hbb'],['Hbb'],['Hbb'],['Hbb']],
              bkgs=[['QCD'],['QCD'],['QCD'],['QCD']])
    
   
    
    tdf_train = cut(cutrho(tdf_train))
    
    FPR_range = np.logspace(-1, 1.9999999999, 20)
    TPR_range = [] 
    for frame,savedir,taggerName in zip(dataframes,savedirs,taggerNames):
        
        labels = [n[len("truth"):] for n in frame.keys() if n.startswith("truth")]
        savedir = os.path.join(outputDir,savedir)
        make_dirs(savedir)
        
        for FPR in range(len(FPR_range)): 
            TPR_range.append(make_TPR_DDT(cut(cutrho(frame)), FPR_range[FPR], siglab='QCD', sculp_label='fj_sdmass', savedir=savedir, taggerName=taggerName))
            FPR_range[FPR] = float(FPR_range[FPR])/100      
        
        break
            
    plot_rocs_with_DDT(dfs=[cut(cutrho(frame)) for frame in dataframes],
              savedir=os.path.join(outputDir,'Plots'),
              names=taggerNames,
             sigs=[['Hbb'],['Hbb'],['Hbb'],['Hbb'],['Hbb']],
              bkgs=[['QCD'],['QCD'],['QCD'],['QCD'],['QCD']], DDT_results = [TPR_range, FPR_range])
    
   
    plot_jsd(dfs=[cut(cutrho(frame)) for frame in dataframes],
             savedir=os.path.join(outputDir,'Plots'),
             names=taggerNames,
             sigs=[['Hbb'],['Hbb'],['Hbb'],['Hbb'],['Hbb'],['Hbb']],
             bkgs=[['QCD'],['QCD'],['QCD'],['QCD'],['QCD'],['QCD']])
 
    plot_jsd_sig(dfs=[cut(cutrho(frame)) for frame in dataframes],
             savedir=os.path.join(outputDir,'Plots'),
             names=taggerNames,
             sigs=[['Hbb'],['Hbb'],['Hbb'],['Hbb'],['Hbb'],['Hbb']],
             bkgs=[['QCD'],['QCD'],['QCD'],['QCD'],['QCD'],['QCD']])
    
    sculpting_multiple_taggers([cut(cutrho(frame)) for frame in dataframes], taggerNames, siglab='Hbb', sculp_label='QCD', savedir=os.path.join(outputDir,'Plots'), sculpt_wp = 0.01)
    
    
    sculpting_multiple_taggers([cut(cutrho(frame)) for frame in dataframes], taggerNames, siglab='Hbb', sculp_label='Hbb', savedir=os.path.join(outputDir,'Plots'), sculpt_wp = 0.01)  
    multiple_pu_dep([cut(cutrho(frame)) for frame in dataframes], taggerNames, sig_label='Hbb', bkg_label='QCD', savedir=os.path.join(outputDir,'Plots'), sculpt_wp = 0.01)

    tdf_train = cut(tdf_train)
    for frame,savedir,taggerName in zip(dataframes,savedirs,taggerNames):
        labels = [n[len("truth"):] for n in frame.keys() if n.startswith("truth")]
        frame = cut(cutrho(frame))
        
        if taggerName == "Interaction network": 
            make_dirs(os.path.join(outputDir,"Plots/IN_DDT"))
            
            array_cuts = [50., 10., 5., 1.]
            
            '''
            # DDT FPR Cuts
            for FPR in np.logspace(-1, 1.9999999999, 20):
                if FPR not in array_cuts: 
                    array_cuts.append(FPR)
            
            for FPR in [0.5, 1.,5.,10.]: 
                if FPR not in array_cuts: 
                    array_cuts.append(FPR)     
            
            # DDT TPR Cuts
            array_cuts = [95, 90, 50, 30]
            
            fit_quantile_reg(frame, 
                             FPR_cut=array_cuts,
                             savename='DDT_TPRcuts_sdmass')            
            '''

            make_DDT(frame, 
                     FPR_cut=array_cuts, 
                     siglab="Hbb", 
                     savedir=os.path.join(outputDir,"Plots/IN_DDT"), 
                     taggerName="Interaction Network, DDT")

            
        savedir = os.path.join(outputDir,savedir)
        make_dirs(savedir)
        
        for label in labels:
            for label2 in labels:
                if label == label2: continue        
                multiple_pu_dep(frame, savedir=savedir, sig_label=label, bkg_label=label2, taggerName=taggerName)
                plot_rocs(dfs=[frame], savedir=savedir, names=[taggerName], 
                          sigs=[[label]], 
                          bkgs=[[label2]])
                sculpting(frame, siglab=label, sculp_label=label2, savedir=savedir, taggerName=taggerName)
                pt_dep(frame, savedir=savedir, sig_label=label, bkg_label=label2, taggerName=taggerName)
                eta_dep(frame, savedir=savedir, sig_label=label, bkg_label=label2, taggerName=taggerName)
                                
            plot_rocs(dfs=[frame], savedir=savedir, names=[taggerName], 
                      sigs=[[label]],
                      bkgs=[[l for l in labels if l != label]])

        for feature in labels+["m", "pt", "eta"]:
            overlay_distribution(frame, savedir=savedir, feature=feature , truths=labels, app_weight=False)    
            for i, lab in enumerate(labels):
                truths = [""]*len(labels)
                truths[i] = lab
                overlay_distribution(frame, savedir=savedir, feature=feature , truths=truths, app_weight=False)
                
    print("Finished Plotting")

def main(args):
    evalDir = args.outdir 

    df = pd.read_pickle('output.pkl')
    df_dec = pd.read_pickle('output_dec.pkl')
    df_in = df.copy(deep=True)
    prediction_in = np.load('%s/prediction_new.npy'%(args.indir))
    df_in['predictHbb'] = prediction_in[:,1]
    df_in['predictQCD'] = prediction_in[:,0]
    df_in_adv = df.copy(deep=True)
    prediction_in_adv = np.load('%s/prediction_new.npy'%(args.inadvdir))
    df_in_adv['predictHbb'] = prediction_in_adv[:,1]
    df_in_adv['predictQCD'] = prediction_in_adv[:,0]
    df_in_rwgt = df.copy(deep=True)
    prediction_in_rwgt = np.load('%s/prediction_new.npy'%(args.inrwgtdir))
    df_in_rwgt['predictHbb'] = prediction_in_rwgt[:,1]
    df_in_rwgt['predictQCD'] = prediction_in_rwgt[:,0]
    
    
    df_train = pd.read_pickle('output_train.pkl')
    df_in_train = df_train.copy(deep=True)
    prediction_in_train = np.load('%s/prediction_train_new.npy'%(args.indir))
    df_in_train['predictHbb'] = prediction_in_train[:,1]
    df_in_train['predictQCD'] = prediction_in_train[:,0]
    
    print(len(df_in))
    print(len(df_in_train))
    
    # to add ntrueInt
    #save_path = '/bigdata/shared/BumbleB/convert_20181121_ak8_80x_deepDoubleB_db_pf_cpf_sv_dl4jets_test_moreinfo/'
    #test_spec_arrays = []
    #for test_file in sorted(glob.glob(save_path + 'test_*_spectators_0.npy')):
    #    test_spec_arrays.append(np.load(test_file))
    #    test_spec = np.concatenate(test_spec_arrays)
    #test_spec = np.swapaxes(test_spec, 1, 2)
    #ntrueInt = test_spec[:,-1,0]
    #df['ntrueInt'] = ntrueInt
    #df_dec['ntrueInt'] = ntrueInt
    #df_in['ntrueInt'] = ntrueInt
    #df_in_dec['ntrueInt'] = ntrueInt
    #print(df[['npv','ntrueInt']])
          
    # Generate Loss Plot
    def plot_loss(indir,outdir, taggerName="", eraText=""):
        loss_vals_training = np.load('%s/loss_vals_training_new.npy'%indir)
        loss_vals_validation = np.load('%s/loss_vals_validation_new.npy'%indir)
        loss_std_training = np.load('%s/loss_std_training_new.npy'%indir)
        loss_std_validation = np.load('%s/loss_std_validation_new.npy'%indir)
        epochs = np.array(range(len(loss_vals_training)))
        f, ax = plt.subplots(figsize=(10, 10))
        ax.plot(epochs, loss_vals_training, label='Training')
        ax.plot(epochs, loss_vals_validation, label='Validation', color = 'green')
        #ax.fill_between(epochs, loss_vals_validation - loss_std_validation,
        #                 loss_vals_validation + loss_std_validation, color = 'lightgreen',
        #                 label = r'Validation $\pm$ 1 std. dev.')
        #ax.fill_between(epochs, loss_vals_training - loss_std_training,
        #                 loss_vals_training + loss_std_training, color = 'lightblue',
        #                 label = 'Training $\pm$ 1 std. dev.')
        leg = ax.legend(loc='upper right', title=taggerName, borderpad=1, frameon=False, fontsize=16)
        leg._legend_box.align = "right"
        
        
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.set_xlim(0,np.max(epochs))
        #ymin = min(np.min(loss_vals_training),np.min(loss_vals_validation))
        #ymax = max(np.min(loss_vals_training),np.max(loss_vals_validation))
        #ax.set_ylim(ymin, ymax)
        
        ax.annotate(eraText, xy=(0.75, 1.015), xycoords='axes fraction', fontname='Helvetica', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$\mathbf{CMS}$', xy=(0, 1.015), xycoords='axes fraction', fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$Simulation\ Open\ Data$', xy=(0.105, 1.015), xycoords='axes fraction', fontsize=18, fontstyle='italic', ha='left',
            annotation_clip=False)
        
        f.savefig('%s/Loss_%s.png'%(outdir,indir.replace('/','')))
        f.savefig('%s/Loss_%s.pdf'%(outdir,indir.replace('/','')))
        plt.close(f)

    #plot_loss(args.indir,args.outdir, taggerName="Interaction network", eraText=r'2016 (13 TeV)')W
    #plot_loss(args.indecdir,args.outdir, taggerName="Interaction network mass decor.", eraText=r'2016 (13 TeV)')
    
    make_plots(evalDir,
               [df_in, df_in_adv, df_in_rwgt, df,df_dec], df_in_train,
               savedirs=["Plots/IN", "Plots/IN_adversarial", "Plots/IN_rwgt", "Plots/DDB","Plots/DDB_dec"],
               taggerNames=["Interaction network", "Interaction network, adversarial", "Interaction network, QCD reweight", "Deep double-b", "Deep double-b, mass decor."],
               eraText=r'2016 (13 TeV)')

    #make_plots(evalDir,
    #           [df_in,df,df_dec],
    #           savedirs=["Plots/IN","Plots/DDB","Plots/DDB_dec"],
    #           taggerNames=["Interaction network", "Deep double-b", "Deep double-b mass decor."],
    #           eraText=r'2016 (13 TeV)')

    
    print('made plots?')

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()
    
    # Required positional arguments
    parser.add_argument("indir", help="IN results dir")
    parser.add_argument("inadvdir", help="IN adversarial results dir")
    parser.add_argument("inrwgtdir", help="IN QCD reweight results dir")
    parser.add_argument("-o", "--outdir", action='store', dest='outdir', default = 'IN_Run2', help="outdir")

    args = parser.parse_args()
    main(args)

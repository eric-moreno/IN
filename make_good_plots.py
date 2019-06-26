import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd.variable import *
import torch.optim as optim
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import rc
import setGPU
from sklearn.metrics import roc_curve, auc, accuracy_score
import scipy
import h5py
import argparse
import glob
import matplotlib.lines as mlines

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
    
def make_plots(outputDir, dataframes, savedirs=["Plots"], taggerNames=["IN"], eraText=r'2016 (13 TeV)'):
    print("Making standard plots")

    def cut(tdf, ptlow=300, pthigh=2000):
        mlow, mhigh = 40, 200
        cdf = tdf[(tdf.fj_pt < pthigh) & (tdf.fj_pt>ptlow) &(tdf.fj_sdmass < mhigh) & (tdf.fj_sdmass>mlow)]
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

        
    def plot_jsd(dfs=[], savedir="", names=[], sigs=[["Hcc"]], bkgs=[["Hbb"]], norm=False, plotname=""):

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        
        def find_nearest(array,value):
            idx = (np.abs(array-value)).argmin()
            return idx, array[idx]

        mmin = 40
        mmax = 200
        nbins = 8
        f, ax = plt.subplots(figsize=(10, 10))
        ax.loglog()
        for frame, name, sig, bkg, color in zip(dfs, names, sigs, bkgs, colors):
            truth, predict, db =  roc_input(frame, signal=sig, include = sig+bkg, norm=norm)
            fpr, tpr, threshold = roc_curve(truth, predict)
            print("{}, AUC={}%".format(name, auc(fpr,tpr)*100), "Sig:", sig, "Bkg:", bkg)
            print("{}, Acc={}%".format(name, accuracy_score(truth,predict>0.5)*100), "Sig:", sig, "Bkg:", bkg)

            cuts = {}
            jsd_plot = []
            eb_plot = []
            for wp,marker in zip([0.3,0.5,0.9,0.95],['v','^','s','o']): # % signal eff.
                idx, val = find_nearest(tpr, wp)
                cuts[str(wp)] = threshold[idx] # threshold for deep double-b corresponding to ~1% mistag rate
                mask_pass = (frame['predict'+sig[0]] > cuts[str(wp)]) & frame['truth'+bkg[0]]
                mask_fail = (frame['predict'+sig[0]] < cuts[str(wp)]) & frame['truth'+bkg[0]]
                mass = frame['fj_sdmass'].values
                mass_pass = mass[mask_pass]
                mass_fail = mass[mask_fail]
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
            ax.plot(eb_plot,jsd_plot,linestyle='-',label=name,color=color)

        ax.set_xlim(1,1e4)
        ax.set_ylim(1,1e5)
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
        leg = ax.legend(borderpad=1, frameon=False, loc='upper right', fontsize=16,
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
        dtriangle = mlines.Line2D([], [], color='gray', marker='v', linestyle='None',
                                                                  markersize=12, label=r'$\varepsilon_\mathrm{sig}$ = 30\%%')
        plt.gca().add_artist(leg)
        leg2 = ax.legend(handles=[circle, square, utriangle, dtriangle],fontsize=16,frameon=False,borderpad=1,loc='center right')
        leg2._legend_box.align = "right"
        plt.gca().add_artist(leg2)
        ax.annotate(eraText, xy=(1e3, 1.1e5), fontname='Helvetica', ha='left',
                    bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$\mathbf{CMS}$', xy=(1.1, 1.1e5), fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
                    bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$Simulation\ Open\ Data$', xy=(3, 1.1e5), fontsize=18, fontstyle='italic', ha='left',
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

        mmin = 40
        mmax = 200
        nbins = 8
        f, ax = plt.subplots(figsize=(10, 10))
        ax.semilogy()
        for frame, name, sig, bkg, color in zip(dfs, names, sigs, bkgs, colors):
            truth, predict, db =  roc_input(frame, signal=sig, include = sig+bkg, norm=norm)
            fpr, tpr, threshold = roc_curve(truth, predict)
            print("{}, AUC={}%".format(name, auc(fpr,tpr)*100), "Sig:", sig, "Bkg:", bkg)
            print("{}, Acc={}%".format(name, accuracy_score(truth,predict>0.5)*100), "Sig:", sig, "Bkg:", bkg)

            cuts = {}
            jsd_plot = []
            es_plot = []
            for wp,marker in zip([0.005,0.01,0.05,0.1],['v','^','s','o']): # % bkg rej.
                idx, val = find_nearest(fpr, wp)
                cuts[str(wp)] = threshold[idx] # threshold 
                mask_pass = (frame['predict'+sig[0]] > cuts[str(wp)]) & frame['truth'+bkg[0]]
                mask_fail = (frame['predict'+sig[0]] < cuts[str(wp)]) & frame['truth'+bkg[0]]
                mass = frame['fj_sdmass'].values
                mass_pass = mass[mask_pass]
                mass_fail = mass[mask_fail]
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
                es_plot.append(tpr[idx])
                jsd_plot.append(1/jsd)
                ax.plot([tpr[idx]],[1/jsd],marker=marker,markersize=12,color=color)
            ax.plot(es_plot,jsd_plot,linestyle='-',label=name,color=color)

        ax.set_xlim(0,1)
        ax.set_ylim(1,1e5)
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
        dtriangle = mlines.Line2D([], [], color='gray', marker='v', linestyle='None',
                                                                  markersize=12, label=r'$\varepsilon_\mathrm{bkg}$ = 0.5\%%')
        plt.gca().add_artist(leg)
        leg2 = ax.legend(handles=[circle, square, utriangle, dtriangle],fontsize=16,frameon=False,borderpad=1,loc='center left')
        leg2._legend_box.align = "left"
        plt.gca().add_artist(leg2)
        ax.annotate(eraText, xy=(0.75, 1.1e5), fontname='Helvetica', ha='left',
                    bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$\mathbf{CMS}$', xy=(0, 1.1e5), fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
                    bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$Simulation\ Open\ Data$', xy=(0.105, 1.1e5), fontsize=18, fontstyle='italic', ha='left',
                    annotation_clip=False)
        
        f.savefig(os.path.join(savedir, "JSD_sig_"+"+".join(sig)+"_vs_"+"+".join(bkg)+".pdf"), dpi=400)
        f.savefig(os.path.join(savedir, "JSD_sig_"+"+".join(sig)+"_vs_"+"+".join(bkg)+".png"), dpi=400)
        plt.close(f)
        #sys.exit()

    def make_DDT(tdf, FPR_cut, siglab="Hcc", sculp_label='Light', savedir="", taggerName=""): 
        
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
            
        # Placing events in bins to detemine threshold for each individual bin
        binned_events = []
        
        NBINS= 8 # number of bins for loss function
        MMAX = 200. # max value
        MMIN = 40. # min value
        
        for nbin in range(NBINS): 
            binned_events.append([]) 
           
        binWidth = (MMAX - MMIN) / NBINS
        masses = tdf['fj_sdmass'].values
        for event in range(len(tdf)):
            if masses[event] < MMIN: 
                binned_events[0].append(event)
            elif masses[event] > MMAX: 
                binned_events[-1].append(event)
            else:
                binned_events[int((masses[event]-MMIN)/binWidth)].append(tdf.index[event])
        
        fprs = []
        tprs = []
        thresholds = []
  
        for i in range(NBINS):
            truth, predict, db = roc_input(tdf.reindex(binned_events[i]), signal=['QCD'], include = ['QCD', 'Hbb'])
            fpr, tpr, threshold = roc_curve(truth, predict)
            fprs.append(fpr)
            tprs.append(tpr)
            thresholds.append(threshold)
       
        # Determining cuts for each bin
        big_cuts = []
        bins = np.linspace(40,200,9)
        for i in range(NBINS): 
            cuts = {}
            for wp in [FPR_cut/100]: # % mistag rate
                idx, val = find_nearest(fprs[i], wp)
                cuts[str(wp)] = thresholds[i][idx] # threshold for tagger corresponding to ~0.05% mistag rate
            big_cuts.append(cuts)
   
        f, ax = plt.subplots(figsize=(10,10))
        
        
        # Plot QCD distribution at 5% cut
        ctdf = tdf.copy()
        ctdf = ctdf.head(0)
        for i in range(NBINS): 
            for wp, cut in reversed(sorted(big_cuts[i].items())):
                ctdf = ctdf.append(tdf.loc[binned_events[i]][tdf.loc[binned_events[i]]['predict'+'QCD'].values > cut])
        
        weight = ctdf['truth'+'QCD'].values
        ax.hist(ctdf['fj_sdmass'].values, bins=bins, weights = weight/np.sum(weight), lw=2, normed=False,
                        histtype='step',label='{}\% FPR Cut QCD'.format(FPR_cut))
       
        weight_uncut = tdf['truth'+'QCD'].values
        ax.hist(tdf['fj_sdmass'].values, bins=bins, weights = weight_uncut/np.sum(weight_uncut), lw=2, normed=False,
                        histtype='step',label='No FPR Cut QCD')
                                
                                
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
        for i in range(NBINS): 
            for wp, cut in reversed(sorted(big_cuts[i].items())):
                ctdf = ctdf.append(tdf.loc[binned_events[i]][tdf.loc[binned_events[i]]['predict'+'Hbb'].values > cut])
        
        weight = ctdf['truth'+'Hbb'].values
        ax.hist(ctdf['fj_sdmass'].values, bins=bins, weights = weight/np.sum(weight), lw=2, normed=False,
                        histtype='step',label='{}\% FPR Cut QCD'.format(FPR_cut))
       
        weight_uncut = tdf['truth'+'Hbb'].values
        ax.hist(tdf['fj_sdmass'].values, bins=bins, weights = weight_uncut/np.sum(weight_uncut), lw=2, normed=False,
                        histtype='step',label='No FPR Cut QCD')
                                
                                
        ax.set_xlabel(r'$\mathrm{m_{SD}\ [GeV]}$', ha='right', x=1.0)
        ax.set_ylabel(r'Normalized scale ({})'.format('Hbb'), ha='right', y=1.0)
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
        f.savefig(os.path.join(savedir,'DDT_Hbb_tag'+'QCD_cut' + '.png'), dpi=400)
        f.savefig(os.path.join(savedir,'DDT_Hbb_tag'+'QCD_cut' + '.pdf'), dpi=400)
        
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
        for wp in [0.01, 0.05, 0.1, 0.5, 1.0]: # % mistag rate
            idx, val = find_nearest(fpr, wp)
            cuts[str(wp)] = threshold[idx] # threshold for deep double-b corresponding to ~1% mistag rate
        
        f, ax = plt.subplots(figsize=(10,10))
        bins = np.linspace(40,200,9)
        for wp, cut in reversed(sorted(cuts.items())):
            ctdf = tdf[tdf['predict'+siglab].values > cut]
            weight = ctdf['truth'+sculp_label].values
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
    plot_jsd(dfs=[cut(frame) for frame in dataframes],
             savedir=os.path.join(outputDir,'Plots'),
             names=taggerNames,
             sigs=[['Hbb'],['Hbb'],['Hbb'],['Hbb']],
             bkgs=[['QCD'],['QCD'],['QCD'],['QCD']])

    plot_jsd_sig(dfs=[cut(frame) for frame in dataframes],
             savedir=os.path.join(outputDir,'Plots'),
             names=taggerNames,
             sigs=[['Hbb'],['Hbb'],['Hbb'],['Hbb']],
             bkgs=[['QCD'],['QCD'],['QCD'],['QCD']])

    plot_rocs(dfs=[cut(frame) for frame in dataframes],
              savedir=os.path.join(outputDir,'Plots'),
              names=taggerNames,
              sigs=[['Hbb'],['Hbb'],['Hbb'],['Hbb']],
              bkgs=[['QCD'],['QCD'],['QCD'],['QCD']])

        
    for frame,savedir,taggerName in zip(dataframes,savedirs,taggerNames):
        labels = [n[len("truth"):] for n in frame.keys() if n.startswith("truth")]
        savedir = os.path.join(outputDir,savedir)
        make_dirs(savedir)
        frame = cut(frame)
 
        for label in labels:
            for label2 in labels:
                if label == label2: continue
                    
                pu_dep(frame, savedir=savedir, sig_label=label, bkg_label=label2, taggerName=taggerName)
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
    
        make_DDT(frame, 5, siglab='QCD', sculp_label='Hbb', savedir=savedir, taggerName=taggerName)
    
    print("Finished Plotting")

def main(args):
    evalDir = args.outdir 

    df = pd.read_pickle('output.pkl')
    df_dec = pd.read_pickle('output_dec.pkl')
    df_in = df.copy(deep=True)
    prediction_in = np.load('%s/prediction_new.npy'%(args.indir))
    df_in['predictHbb'] = prediction_in[:,1]
    df_in['predictQCD'] = prediction_in[:,0]
    df_in_dec = df.copy(deep=True)
    prediction_in_dec = np.load('%s/prediction_new.npy'%(args.indecdir))
    df_in_dec['predictHbb'] = prediction_in_dec[:,1]
    df_in_dec['predictQCD'] = prediction_in_dec[:,0]

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

    #plot_loss(args.indir,args.outdir, taggerName="Interaction network", eraText=r'2016 (13 TeV)')
    #plot_loss(args.indecdir,args.outdir, taggerName="Interaction network mass decor.", eraText=r'2016 (13 TeV)')
    
    make_plots(evalDir,
               [df_in,df_in_dec,df,df_dec],
               savedirs=["Plots/IN", "Plots/IN_dec","Plots/DDB","Plots/DDB_dec"],
               taggerNames=["Interaction network", "Interaction network mass decor.", "Deep double-b", "Deep double-b mass decor."],
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
    parser.add_argument("indecdir", help="IN decor results dir")
    
    parser.add_argument("-o", "--outdir", action='store', dest='outdir', default = 'IN_Run2', help="outdir")

    args = parser.parse_args()
    main(args)

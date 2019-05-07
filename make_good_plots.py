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
from sklearn.metrics import roc_curve, auc
import h5py
import argparse

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
          + "\n "+str(int(round((min(frame.fj_sdmass)))))+" $\mathrm{<\ jet\ m_{sd}\ <}$ "+str(int(round((max(frame.fj_sdmass)))))+" GeV"
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
              + "\n "+str(int(round((min(frame.fj_sdmass)))))+" $\mathrm{<\ jet\ m_{sd}\ <}$ "+str(int(round((max(frame.fj_sdmass)))))+" GeV"\
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
            ix, eff =  find_nearest(tpr, 0.8)
            misloose.append(fpr[ix])
            ix, eff =  find_nearest(tpr, 0.4)
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
        ax.step([],[], label='80\% efficiency', c='red', where='post')
        ax.step([],[], label='40\% efficiency', c='red', where='post', linestyle='dashed')
        
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
        ax2.set_ylim(0.001, 2)
        
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
        ax.set_ylim(0, 1.2)
        ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12)
        ax.tick_params(direction='in', axis='both', which='minor' , length=6)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('left')    
        #ax.grid(which='minor', alpha=0.5, axis='y', linestyle='dotted')
        ax.grid(which='major', alpha=0.9, linestyle='dotted')
        leg = ax.legend(borderpad=1, frameon=False, loc=1, fontsize=16 )
        mpt = ""+str(int(round((min(frame.fj_pt)))))+" $\mathrm{<\ jet\ p_T\ <}$ "+str(int(round((max(frame.fj_pt)))))+" GeV"       + "\n "+str(int(round((min(frame.fj_sdmass)))))+" $\mathrm{<\ jet\ m_{SD}\ <}$ "+str(int(round((max(frame.fj_sdmass)))))+" GeV"
        ax.annotate( mpt, xy=(0.05, 0.88), xycoords='axes fraction', fontname='Helvetica', ha='left',
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
            ix, eff =  find_nearest(tpr, 0.8)
            misloose.append(fpr[ix])
            ix, eff =  find_nearest(tpr, 0.4)
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
        ax.step([],[], label='80\% efficiency', c='red', where='post')
        ax.step([],[], label='40\% efficiency', c='red', where='post', linestyle='dashed')
        
        ax2 = ax.twinx()
        ax2.step(pts, mistight, lw=2, label='40\% efficiency', c='red', where='post', linestyle='dashed')
        ax2.step(pts, misloose, lw=2, label='80\% efficiency', c='red', where='post')
        ax2.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12)
        ax2.tick_params(direction='in', axis='both', which='minor' , length=6)
        ax2.yaxis.set_ticks_position('right')
        ax2.tick_params('y', which='both', colors='red')
        ax2.semilogy()
        ax2.set_ylim(0.001, 2)
        
        ax.set_xlabel(r'$\mathrm{p_T\ [GeV]}$', ha='right', x=1.0)
        ylab1 = ['{} \\rightarrow {}'.format(l[0], l[-2]+'\\bar{'+l[-1]+'}') if len(l) == 3 and l[0] in ["H", "Z", "g"] else l for l in [sig_label] ][0]
        ax.set_ylabel(r'{} tagging efficiency ($\mathrm{}$)'.format(taggerName,"{"+ylab1+"}"), ha='right', y=1.0, color='black')
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
        ax.set_ylim(0, 1.2)
        ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12)
        ax.tick_params(direction='in', axis='both', which='minor' , length=6)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('left')    
        #ax.grid(which='minor', alpha=0.5, axis='y', linestyle='dotted')
        ax.grid(which='major', alpha=0.9, linestyle='dotted')
        leg = ax.legend(borderpad=1, frameon=False, loc=1, fontsize=16 )
        mpt = ""+str(int(round((min(frame.fj_pt)))))+" $\mathrm{<\ jet\ p_T\ <}$ "+str(int(round((max(frame.fj_pt)))))+" GeV"       + "\n "+str(int(round((min(frame.fj_sdmass)))))+" $\mathrm{<\ jet\ m_{SD}\ <}$ "+str(int(round((max(frame.fj_sdmass)))))+" GeV"
        ax.annotate( mpt, xy=(0.05, 0.88), xycoords='axes fraction', fontname='Helvetica', ha='left',
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
                               )
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
              bkgs=[['QCD'],['QCD'],['QCD'],['Hbb']])
    
    for frame,savedir,taggerName in zip(dataframes,savedirs,taggerNames):
        labels = [n[len("truth"):] for n in frame.keys() if n.startswith("truth")]
        savedir = os.path.join(outputDir,savedir)
        make_dirs(savedir)
        frame = cut(frame)
 
        for label in labels:
            for label2 in labels:
                if label == label2: continue
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
    df_in_dec = df.copy(deep=True)
    prediction_in_dec = np.load('%s/prediction_new.npy'%(args.indecdir))
    df_in_dec['predictHbb'] = prediction_in_dec[:,1]
    df_in_dec['predictQCD'] = prediction_in_dec[:,0]
    make_plots(evalDir,
               [df_in,df_in_dec,df,df_dec],
               savedirs=["Plots/IN", "Plots/IN_dec","Plots/DDB","Plots/DDB_dec"],
               taggerNames=["Interaction network", "Interaction network decor.", "Deep double-b", "Deep double-b mass decor."],
               eraText=r'2016 (13 TeV)')
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

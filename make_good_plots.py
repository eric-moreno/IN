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
from argparse import ArgumentParser

parser = ArgumentParser(description ='Script to evaluate training')
parser.add_argument("-i", help="Predictions for test dataset (numpy array)", default=None, metavar="FILE")
parser.add_argument("-o",  help="Eval output directory", default=None, metavar="PATH")
opts=parser.parse_args()

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

def evaluate(testd, trainData, model_output, outputDir, storeInputs=False, adv=False):
    	NENT = 1  # Can skip some events
    	filelist=[]
        i=0
        x = 0
        for s in testd.samples:
        #for s in testd.samples[0:1]:
            spath = testd.getSamplePath(s)
            filelist.append(spath)
            h5File = h5py.File(spath)
            f = h5File
            #features_val = [h5File['x%i_shape'%j][()] for j in range(0, h5File['x_listlength'][()][0])]
            features_val = [h5File['x%i'%j][()] for j in range(0, h5File['x_listlength'][()][0])]
            #features_val=testd.getAllFeatures()
            predict_test_i = np.load(model_output)
            predict_test_i = predict_test_i[x:200000 + x]
            labels_val_i = h5File['y0'][()][::NENT,:]
            spectators_val_i = h5File['z0'][()][::NENT,0,:]
            if storeInputs: raw_features_val_i = h5File['z1'][()][::NENT,0,:]
            if i==0:
                predict_test = predict_test_i
                labels_val = labels_val_i
                spectators_val = spectators_val_i
                if storeInputs: raw_features_val = raw_features_val_i                                                    
            else:
                predict_test = np.concatenate((predict_test,predict_test_i))
                labels_val = np.concatenate((labels_val, labels_val_i))
                spectators_val = np.concatenate((spectators_val, spectators_val_i))
                if storeInputs: raw_features_val = np.concatenate((raw_features_val, raw_features_val_i))
            i+=1
            x += 200000
        
	# Labels
	print testd.dataclass.branches
	feature_names = testd.dataclass.branches[1]
	spectator_names = testd.dataclass.branches[0]
        #truthnames = testd.getUsedTruth()
        
	from DeepJetCore.DataCollection import DataCollection                 
    	traind=DataCollection()
	traind.readFromFile(trainData)
	truthnames = traind.getUsedTruth()
	# Store features                                            
	print "Coulmns", spectator_names                   
        df = pd.DataFrame(spectators_val, columns = spectator_names)

	if storeInputs: 
		for i, tname in enumerate(feature_names):
			df[tname] = raw_features_val[:,i]

	# Add predictions
	print truthnames
	print predict_test.shape
	for i, tname in enumerate(truthnames):
		df['truth'+tname] = labels_val[:,i]
		#print "Mean 0th label predict predict of ", tname, np.mean(predict_test[:,0]), ", Stats:", np.sum(labels_val[:,i]), "/", len(labels_val[:,i])
                if adv:
		    df['predict'+tname] = predict_test[:,NBINS+i]
                    #df['predict_IN'+tname] = predict_test_IN[:,NBINS+i]
                    for j in range(NBINS):
                        df['predict_massbin%i'%j] = predict_test[:,j+i]
                        #df['predict_IN_massbin%i'%j] = predict_test_IN[:,j+i]
                else:
                    df['predict'+tname] = predict_test[:,i]
                    #df['predict_IN'+tname] = predict_test_IN[:,i]
                    
	print "Testing prediction:"
	print "Total: ", len(predict_test[:,0])
	for lab in truthnames:
		print lab, ":", sum(df['truth'+lab].values)

	#df.to_pickle(outputDir+'/output.pkl')    #to save the dataframe, df to 123.pkl
	return df
	print "Finished storing dataframe"	

def make_dirs(dirname):
    import os, errno
    """
    Ensure that a named directory exists; if it does not, attempt to create it.
    """
    try:
    	os.makedirs(dirname)
    except OSError, e:
        if e.errno != errno.EEXIST:
        	raise
    
def make_plots(outputDir, dataframe, savedir="Plots", taggerName="IN", eraText=r'2016 (13 TeV)'):
    print "Making standard plots"	
    frame = dataframe
    labels = [n[len("truth"):] for n in frame.keys() if n.startswith("truth")]
    savedir = os.path.join(outputDir,savedir)
    make_dirs(savedir)

    def cut(tdf, ptlow=300, pthigh=2000):
        mlow, mhigh = 40, 200
        cdf = tdf[(tdf.fj_pt < pthigh) & (tdf.fj_pt>ptlow) &(tdf.fj_sdmass < mhigh) & (tdf.fj_sdmass>mlow)]
        return cdf

    def cuteta(tdf, etalow=-2.5, etahigh=2.5):
        #mlow, mhigh = 90, 140
        mlow, mhigh = 40, 200
        cdf = tdf[(tdf.fj_eta < etahigh) & (tdf.fj_eta>etalow) &(tdf.fj_sdmass < mhigh) & (tdf.fj_sdmass>mlow)]
        return cdf

    frame = cut(frame)
 
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
            #np.save('fpr2017simulation_adv', fpr)
            #np.save('tpr2017simulation_adv', tpr)
            #np.save('threshold2017simulation_adv', threshold)
            #fpr_IN_noSV, tpr_IN_noSV, threshold_IN = roc_curve(truth, predict_IN)
            #fpr_IN = np.load('fpr_IN.npy')
            #tpr_IN = np.load('tpr_IN.npy')
            #fpr = np.load('fpr_DDB_opendata_withoutAdversarial.npy')
            #tpr = np.load('tpr_DDB_opendata_withoutAdversarial.npy')
            #tpr_DDB_Adversarial = np.load('tpr_DDB_opendata.npy')
            #fpr_DDB_Adversarial = np.load('fpr_DDB_opendata.npy')
            #tpr_IN_Adversarial = np.load('tpr_IN_adversarial.npy')
            #fpr_IN_Adversarial = np.load('fpr_IN_adversarial.npy')
            #tpr2017simulation = np.load('tpr2017simulation.npy')
            #fpr2017simulation = np.load('fpr2017simulation.npy')
            
            ax.plot(tpr, fpr, lw=2.5, label="Interaction Network, AUC = {:.1f}\%".format(auc(fpr,tpr)*100))
            #ax.plot(tpr_DDB_Adversarial, tpr_DDB_Adversarial, lw=2.5, label="DeepDouble{} w/ Adv.".format(name))
            #ax.plot(tpr, fpr, lw=2.5, label="DeepDoubleB Opendata, AUC = {:.1f}\%".format(auc(fpr,tpr)*100))
            #ax.plot(tpr_IN_noSV, fpr_IN_noSV, lw=2.5, label="Interaction Network w/o Vertices, AUC = {:.1f}\%".format(auc(fpr_IN_noSV,tpr_IN_noSV)*100))
            #ax.plot(tpr2017simulation, fpr2017simulation, lw=2.5, label="DeepDoubleB 2017 Simulation, AUC = {:.1f}\%".format(auc(fpr2017simulation, tpr2017simulation)*100))
            #ax.plot(tpr_IN_Adversarial, fpr_IN_Adversarial, lw=2.5, label="Interaction Network w/ Adv., AUC = {:.1f}\%".format(auc(fpr_IN_Adversarial,tpr_IN_Adversarial)*100))
            ROCtext=open(os.path.join(savedir, "ROCComparison_"+"+".join(sig)+"_vs_"+"+".join(bkg)+".txt"),'w')
            for ind in range(len(tpr)):
                            ROCtext.write(str(tpr[ind])+'\t'+str(fpr[ind])+'\n')
            ROCtext.close()
            print "{}, AUC={}%".format(name, auc(fpr,tpr)*100), "Sig:", sig, "Bkg:", bkg
            
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
        ax.annotate(eraText, xy=(0.80, 1.1), fontname='Helvetica', ha='left',
                    bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$\mathbf{CMS}$', xy=(0.01, 1.1), fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
                    bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$Simulation\ Opendata$', xy=(0.115, 1.1), fontsize=18, fontstyle='italic', ha='left',
                    annotation_clip=False)
        if norm: f.savefig(os.path.join(savedir, "ROCNormComparison_"+"+".join(sig)+"_vs_"+"+".join(bkg)+".pdf"), dpi=400)
        else: f.savefig(os.path.join(savedir, "ROCComparison_"+"+".join(sig)+"_vs_"+"+".join(bkg)+".pdf"), dpi=400)
        if norm: f.savefig(os.path.join(savedir, "ROCNormComparison_"+"+".join(sig)+"_vs_"+"+".join(bkg)+".png"), dpi=400)
        else: f.savefig(os.path.join(savedir, "ROCComparison_"+"+".join(sig)+"_vs_"+"+".join(bkg)+".png"), dpi=400)

    for label in labels:
        for label2 in labels:
            if label == label2: continue
            plot_rocs(dfs=[frame], savedir=savedir, names=[taggerName], 
                 sigs=[[label]], 
                 bkgs=[[label2]])

        plot_rocs(dfs=[frame], savedir=savedir, names=[taggerName], 
                 sigs=[[label]], 
                 bkgs=[[l for l in labels if l != label]])
    	
    def sculpting(tdf, siglab="Hcc", sculp_label='Light', savedir=""):
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
        for wp in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]: # % mistag rate
            idx, val = find_nearest(fpr, wp)
            cuts[str(wp)] = threshold[idx] # threshold for deep double-b corresponding to ~1% mistag rate
        
        f, ax = plt.subplots(figsize=(10,10))
        bins = np.linspace(40,200,17)
        for wp, cut in reversed(sorted(cuts.iteritems())):
            ctdf = tdf[tdf['predict'+siglab].values > cut]
            weight = ctdf['truth'+sculp_label].values
            ax.hist(ctdf['fj_sdmass'].values, bins=bins, weights = weight, lw=2, normed=True,histtype='step',label='{}\%  mistagging rate'.format(float(wp)*100.))
        
        ax.set_xlabel(r'$\mathrm{m_{SD}\ [GeV]}$', ha='right', x=1.0)
        ax.set_ylabel(r'Normalized scale ({})'.format(legend_bkglab), ha='right', y=1.0)
        import matplotlib.ticker as plticker
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=20))
        ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=10))
        ax.yaxis.set_minor_locator(plticker.AutoMinorLocator(5))
        ax.set_xlim(40, 200)
        ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12, labelleft=False )
        ax.tick_params(direction='in', axis='both', which='minor' , length=6)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')    
        #ax.grid(which='minor', alpha=0.5, axis='y', linestyle='dotted')
        ax.grid(which='major', alpha=0.9, linestyle='dotted')
        leg = ax.legend(borderpad=1, frameon=False, loc='best', fontsize=16,
            title = ""+str(int(round((min(frame.fj_pt)))))+" $\mathrm{<\ jet\ p_T\ <}$ "+str(int(round((max(frame.fj_pt)))))+" GeV" \
              + "\n "+str(int(round((min(frame.fj_sdmass)))))+" $\mathrm{<\ jet\ m_{SD}\ <}$ "+str(int(round((max(frame.fj_sdmass)))))+" GeV"
                  + "\n Tagging {}".format(legend_siglab)           )
        leg._legend_box.align = "right"
        ax.annotate(eraText, xy=(0.8, 1.015), xycoords='axes fraction', fontname='Helvetica', ha='left',
                        bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$\mathbf{CMS}$', xy=(0, 1.015), xycoords='axes fraction', fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
                        bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$Simulation\ Opendata$', xy=(0.105, 1.015), xycoords='axes fraction', fontsize=18, fontstyle='italic', ha='left',
                        annotation_clip=False)
        f.savefig(os.path.join(savedir,'M_sculpting_tag'+siglab+"_"+sculp_label+'.png'), dpi=400)
        f.savefig(os.path.join(savedir,'M_sculpting_tag'+siglab+"_"+sculp_label+'.pdf'), dpi=400)
        #return
        f, ax = plt.subplots(figsize=(10,10))
        bins = np.linspace(300,2000,35)
        for wp, cut in reversed(sorted(cuts.iteritems())):
            ctdf = tdf[tdf['predict'+siglab].values > cut]
            weight = ctdf['truth'+sculp_label].values
            ax.hist(ctdf['fj_pt'].values, bins=bins, weights = weight, lw=2, normed=True,histtype='step',label='{}\%  mistagging rate'.format(float(wp)*100.))
        
        ax.set_xlabel(r'$\mathrm{p_T\ [GeV]}$', ha='right', x=1.0)
        #ax.set_ylabel(r'Normalized scale ({})'.format(sculp_label.replace("Hcc", r"$\mathrm{H \rightarrow c\bar{c}}$").replace("Hbb", r"$\mathrm{H \rightarrow b\bar{b}}$")), ha='right', y=1.0)
        ax.set_ylabel(r'Normalized scale ({})'.format(legend_bkglab), ha='right', y=1.0)
        import matplotlib.ticker as plticker
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=200))
        ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=50))
        ax.yaxis.set_minor_locator(plticker.AutoMinorLocator(5))
        ax.set_xlim(300, 2000)
        ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12, labelleft=False )
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
        ax.annotate(eraText, xy=(0.8, 1.015), xycoords='axes fraction', fontname='Helvetica', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$\mathbf{CMS}$', xy=(0, 1.015), xycoords='axes fraction', fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$Simulation\ Opendata$', xy=(0.105, 1.015), xycoords='axes fraction', fontsize=18, fontstyle='italic', ha='left',
            annotation_clip=False)

        
        f.savefig(os.path.join(savedir,'Pt_sculpting_tag'+siglab+"_"+sculp_label+'.png'), dpi=400)

    for label in labels:  
        for label2 in labels:
            if label == label2: continue
            sculpting(frame, siglab=label, sculp_label=label2, savedir=savedir) 


    def eta_dep(xdf, sig_label="Hcc", bkg_label="", savedir=""):
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
        ax.annotate(eraText, xy=(0.8, 1.015), xycoords='axes fraction', fontname='Helvetica', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$\mathbf{CMS}$', xy=(0, 1.015), xycoords='axes fraction', fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$Simulation\ Opendata$', xy=(0.105, 1.015), xycoords='axes fraction', fontsize=18, fontstyle='italic', ha='left',
            annotation_clip=False)

        f.savefig(os.path.join(savedir,'eta_dep_'+sig_label+"_vs_"+bkg_label+'.png'), dpi=400)
        f.savefig(os.path.join(savedir,'eta_dep_'+sig_label+"_vs_"+bkg_label+'.pdf'), dpi=400)
        

    def pt_dep(xdf, sig_label="Hcc", bkg_label="", savedir=""):
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
        ax.annotate(eraText, xy=(0.8, 1.015), xycoords='axes fraction', fontname='Helvetica', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$\mathbf{CMS}$', xy=(0, 1.015), xycoords='axes fraction', fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
        ax.annotate('$Simulation\ Opendata$', xy=(0.105, 1.015), xycoords='axes fraction', fontsize=18, fontstyle='italic', ha='left',
            annotation_clip=False)

        
        f.savefig(os.path.join(savedir,'Pt_dep_'+sig_label+"_vs_"+bkg_label+'.png'), dpi=400)
        f.savefig(os.path.join(savedir,'Pt_dep_'+sig_label+"_vs_"+bkg_label+'.pdf'), dpi=400)

    for label in labels:  
        for label2 in labels:
            if label == label2: continue
            pt_dep(frame, savedir=savedir, sig_label=label, bkg_label=label2)
            eta_dep(frame, savedir=savedir, sig_label=label, bkg_label=label2)


    for label in labels:  
        for label2 in labels:
            if label == label2: continue
            sculpting(frame, siglab=label, sculp_label=label2, savedir=savedir) 

   
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
                        ax.hist(pred, bins=bins, weights = weight*xdf.Weight, alpha=a, normed=True, label="")# Mock plot
                    else:
                        ax.hist(pred, bins=bins, weights = weight*xdf.Weight, alpha=a, normed=True, 
                            label=r'Weighted $\mathrm{}$'.format('{'+legend_labels[all_labels.index(tru)]+'}') )    
                else: 
                    if tru == "":
                        ax.hist(pred, bins=bins, weights = weight, alpha=a, normed=True, label="")# Mock plot
                    else:
                        ax.hist(pred, bins=bins, weights = weight, alpha=a, normed=True, 
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
            ax.annotate(eraText, xy=(0.8, 1.015), xycoords='axes fraction', fontname='Helvetica', ha='left',
                bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
            ax.annotate('$\mathbf{CMS}$', xy=(0, 1.015), xycoords='axes fraction', fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
                bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
            ax.annotate('$Simulation\ Opendata$', xy=(0.105, 1.015), xycoords='axes fraction', fontsize=18, fontstyle='italic', ha='left',
                annotation_clip=False)

            if app_weight:
                f.savefig(os.path.join(savedir, 'dist_weight_'+feature+'_'+"-".join(t for t in truths if len(t)>1)+'.pdf'), dpi=400)
                f.savefig(os.path.join(savedir, 'dist_weight_'+feature+'_'+"-".join(t for t in truths if len(t)>1)+'.png'), dpi=400)
            else:
                f.savefig(os.path.join(savedir, 'dist_'+feature+'_'+"-".join(t for t in truths if len(t)>1)+'.pdf'), dpi=400)
                f.savefig(os.path.join(savedir, 'dist_'+feature+'_'+"-".join(t for t in truths if len(t)>1)+'.png'), dpi=400)
            print 

    for feature in labels+["m", "pt", "eta"]:
        overlay_distribution(frame, savedir=savedir, feature=feature , truths=labels, app_weight=False)    
        for i, lab in enumerate(labels):
            truths = [""]*len(labels)
            truths[i] = lab
            overlay_distribution(frame, savedir=savedir, feature=feature , truths=truths, app_weight=False)



    print "Finished Plotting"	    
    
from DeepJetCore.DataCollection import DataCollection
inputTestDataCollection = '/bigdata/shared/BumbleB/convert_20181121_ak8_80x_deepDoubleB_db_cpf_sv_reduced_dl4jets_test/dataCollection.dc'
inputTrainDataCollection = '/bigdata/shared/BumbleB/convert_20181121_ak8_80x_deepDoubleB_db_cpf_sv_reduced_dl4jets_train_val/dataCollection.dc'
prediction = opts.i
evalDir = opts.o

testd=DataCollection()
testd.readFromFile(inputTestDataCollection)
df = evaluate(testd, inputTrainDataCollection, prediction, evalDir, storeInputs=False, adv=False)
make_plots(evalDir, df, savedir="Plots", taggerName="IN", eraText=r'2016 (13 TeV)')

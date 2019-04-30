import matplotlib as mpl
mpl.use('agg')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import torch
softmax = torch.nn.Softmax(dim=1)
import glob
import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import rc
import pandas as pd

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rcParams['font.size'] = 22
rcParams['text.latex.preamble'] = [
    #       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
    #       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
           r'\usepackage{helvet}',    # set the normal font here
           r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
           r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
    ]
#rcParams["figure.facecolor"] = "white"
#rcParams["savefig.facecolor"] = "white"
rc('text', usetex=True)

save_path = '/bigdata/shared/BumbleB/convert_20181121_ak8_80x_deepDoubleB_db_pf_cpf_sv_dl4jets_test/'
test_0_arrays = []
test_2_arrays = []
test_3_arrays = []
test_spec_arrays = []
target_test_arrays = []

for test_file in sorted(glob.glob(save_path + 'test_*_features_0.npy')):
    test_0_arrays.append(np.load(test_file))
test_0 = np.concatenate(test_0_arrays)

for test_file in sorted(glob.glob(save_path + 'test_*_features_2.npy')):
    test_2_arrays.append(np.load(test_file))
test_2 = np.concatenate(test_2_arrays)

for test_file in sorted(glob.glob(save_path + 'test_*_features_3.npy')):
    test_3_arrays.append(np.load(test_file))
test_3 = np.concatenate(test_3_arrays)

for test_file in sorted(glob.glob(save_path + 'test_*_spectators_0.npy')):
    test_spec_arrays.append(np.load(test_file))
test_spec = np.concatenate(test_spec_arrays)

for test_file in sorted(glob.glob(save_path + 'test_*_truth_0.npy')):
    target_test_arrays.append(np.load(test_file))
target_test = np.concatenate(target_test_arrays)

del test_0_arrays
del test_2_arrays
del test_3_arrays
del test_spec_arrays
del target_test_arrays
test_0 = np.swapaxes(test_0, 1, 2)
test_2 = np.swapaxes(test_2, 1, 2)
test_3 = np.swapaxes(test_3, 1, 2)
test_spec = np.swapaxes(test_spec, 1, 2)
print(test_0.shape)
print(test_2.shape)
print(test_3.shape)
print(target_test.shape)
print(test_spec.shape)
print(target_test.shape)
fj_pt = test_spec[:,0,0]
fj_eta = test_spec[:,1,0]
fj_sdmass = test_spec[:,2,0]

min_pt = 300
max_pt = 2000
min_eta = -999 # no cut
max_eta = 999 # no cut
min_msd = 40
max_msd = 200

test_0 = test_0 [ (fj_sdmass > min_msd) & (fj_sdmass < max_msd) & (fj_eta > min_eta) & (fj_eta < max_eta) & (fj_pt > min_pt) & (fj_pt < max_pt) ]
test_2 = test_2 [ (fj_sdmass > min_msd) & (fj_sdmass < max_msd) & (fj_eta > min_eta) & (fj_eta < max_eta) & (fj_pt > min_pt) & (fj_pt < max_pt) ]
test_3 = test_3 [ (fj_sdmass > min_msd) & (fj_sdmass < max_msd) & (fj_eta > min_eta) & (fj_eta < max_eta) & (fj_pt > min_pt) & (fj_pt < max_pt) ]
test_spec = test_spec [ (fj_sdmass > min_msd) & (fj_sdmass < max_msd) & (fj_eta > min_eta) & (fj_eta < max_eta) & (fj_pt > min_pt) & (fj_pt < max_pt) ]
target_test = target_test [ (fj_sdmass > min_msd) & (fj_sdmass < max_msd) & (fj_eta > min_eta) & (fj_eta < max_eta) & (fj_pt > min_pt) & (fj_pt < max_pt) ]
print(test_0.shape)
print(test_2.shape)
print(test_3.shape)
print(target_test.shape)
print(test_spec.shape)
print(target_test.shape)

spectators = ['fj_pt',
              'fj_eta',
              'fj_sdmass',
              'fj_n_sdsubjets',
              'fj_doubleb',
              'fj_tau21',
              'fj_tau32',
              'npv',
              'npfcands',
              'ntracks',
              'nsv']

params_0 = ['fj_jetNTracks',
          'fj_nSV',
          'fj_tau0_trackEtaRel_0',
          'fj_tau0_trackEtaRel_1',
          'fj_tau0_trackEtaRel_2',
          'fj_tau1_trackEtaRel_0',
          'fj_tau1_trackEtaRel_1',
          'fj_tau1_trackEtaRel_2',
          'fj_tau_flightDistance2dSig_0',
          'fj_tau_flightDistance2dSig_1',
          'fj_tau_vertexDeltaR_0',
          'fj_tau_vertexEnergyRatio_0',
          'fj_tau_vertexEnergyRatio_1',
          'fj_tau_vertexMass_0',
          'fj_tau_vertexMass_1',
          'fj_trackSip2dSigAboveBottom_0',
          'fj_trackSip2dSigAboveBottom_1',
          'fj_trackSip2dSigAboveCharm_0',
          'fj_trackSipdSig_0',
          'fj_trackSipdSig_0_0',
          'fj_trackSipdSig_0_1',
          'fj_trackSipdSig_1',
          'fj_trackSipdSig_1_0',
          'fj_trackSipdSig_1_1',
          'fj_trackSipdSig_2',
          'fj_trackSipdSig_3',
          'fj_z_ratio'
          ]

params_1 = ['pfcand_ptrel',
          'pfcand_erel',
          'pfcand_phirel',
          'pfcand_etarel',
          'pfcand_deltaR',
          'pfcand_puppiw',
          'pfcand_drminsv',
          'pfcand_drsubjet1',
          'pfcand_drsubjet2',
          'pfcand_hcalFrac'
         ]

params_2 = ['track_ptrel',     
          'track_erel',     
          'track_phirel',     
          'track_etarel',     
          'track_deltaR',
          'track_drminsv',     
          'track_drsubjet1',     
          'track_drsubjet2',
          'track_dz',     
          'track_dzsig',     
          'track_dxy',     
          'track_dxysig',     
          'track_normchi2',     
          'track_quality',     
          'track_dptdpt',     
          'track_detadeta',     
          'track_dphidphi',     
          'track_dxydxy',     
          'track_dzdz',     
          'track_dxydz',     
          'track_dphidxy',     
          'track_dlambdadz',     
          'trackBTag_EtaRel',     
          'trackBTag_PtRatio',     
          'trackBTag_PParRatio',     
          'trackBTag_Sip2dVal',     
          'trackBTag_Sip2dSig',     
          'trackBTag_Sip3dVal',     
          'trackBTag_Sip3dSig',     
          'trackBTag_JetDistVal'
         ]

params_3 = ['sv_ptrel',
          'sv_erel',
          'sv_phirel',
          'sv_etarel',
          'sv_deltaR',
          'sv_pt',
          'sv_mass',
          'sv_ntracks',
          'sv_normchi2',
          'sv_dxy',
          'sv_dxysig',
          'sv_d3d',
          'sv_d3dsig',
          'sv_costhetasvpv'
         ]


#Convert two sets into two branch with one set in both and one set in only one (Use for this file)
#training = training_2
test = test_2
params = params_2
#training_sv = training_3
test_sv = test_3
params_sv = params_3
N = test.shape[2]
N_sv = test_sv.shape[2]

outdir = 'out'
label = 'new'

# Generate Loss Plot
loss_vals_training = np.load('%s/loss_vals_training_%s.npy'%(outdir,label))
loss_vals_validation = np.load('%s/loss_vals_validation_%s.npy'%(outdir,label))
loss_std_validation = np.load('%s/loss_std_validation_%s.npy'%(outdir,label))
loss_std_training = np.load('%s/loss_std_training_%s.npy'%(outdir,label))
acc_vals_validation = np.load('%s/acc_vals_validation_%s.npy'%(outdir,label))

epochs = np.array(range(len(loss_vals_training)))
fig = plt.figure(figsize = (12,10))
ax1 = fig.add_subplot(111)
ax1.plot(epochs, loss_vals_training, label='training')
ax1.plot(epochs, loss_vals_validation, label='validation', color = 'green')
ax1.fill_between(epochs, loss_vals_validation - loss_std_validation/2, loss_vals_validation + loss_std_validation/2, color = 'lightgreen', label = 'Validation +/- 0.5 Std')
ax1.fill_between(epochs, loss_vals_training - loss_std_training/2, loss_vals_training + loss_std_training/2, color = 'lightblue', label = 'Training +/- 0.5 Std')
plt.legend(loc='upper right')
plt.title('Loss Plot Plain IN (Data Generator)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('%s/Loss_SV_tracks_data_generator_%s.png'%(outdir,label))
plt.savefig('%s/Loss_SV_tracks_data_generator_%s.pdf'%(outdir,label))

plt.figure(figsize=(12, 10), dpi = 200)
plt.plot(acc_vals_validation)
sns.set()
plt.title('Accuracy Plain IN (Data Generator)')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.savefig("%s/Accuracy_SV_tracks_dataGenerator_%s.png"%(outdir,label))
plt.savefig("%s/Accuracy_SV_tracks_dataGenerator_%s.pdf"%(outdir,label))

# Generate ROC Plot

prediction = np.array([])
IN_out = np.array([])
batch_size = 128
torch.cuda.empty_cache()
test = test_2
params = params_2
#training_sv = training_3
test_sv = test_3
params_sv = params_3
N = test.shape[2]
N_sv = test_sv.shape[2]
n_targets = target_test.shape[1]
from gnn import GraphNet
gnn = GraphNet(N, n_targets, len(params), 15, N_sv, len(params_sv), vv_branch=False)
gnn.load_state_dict(torch.load('%s/gnn_%s_best.pth'%(outdir,label)))

for j in tqdm.tqdm(range(0, target_test.shape[0], batch_size)):
    out_test = softmax(gnn(torch.from_numpy(test[j:j + batch_size]).cuda(), torch.from_numpy(test_sv[j:j + batch_size]).cuda()))
    out_test = out_test.cpu().data.numpy()
    if j==0:
        prediction = out_test
    else:
        prediction = np.concatenate((prediction, out_test),axis=0)        
    del out_test

np.save('%s/truth_%s.npy'%(outdir,label),prediction)
np.save('%s/prediction_%s.npy'%(outdir,label),prediction)
print(prediction)
print(target_test)

fpr, tpr, thresholds = roc_curve(target_test[:,1], prediction[:,1])
auc = roc_auc_score(target_test[:,1], prediction[:,1])

#fpr_DeepDoubleB = np.load('fpr_DeepDoubleB.npy')
#tpr_DeepDoubleB = np.load('tpr_DeepDoubleB.npy')
#dfpr_BDT = np.load('dfpr_BDT.npy')
#dtpr_BDT = np.load('dtpr_BDT.npy')

frame = pd.read_pickle('output.pkl')

#print(frame)

sig=["Hbb"]
bkg=["QCD"]
truth_ddb = frame['truth%s'%sig[0]].values
predict_ddb = frame['predict%s'%sig[0]].values

fpr_ddb, tpr_ddb, thresholds_ddb = roc_curve(truth_ddb, predict_ddb)
auc_ddb = roc_auc_score(truth_ddb, predict_ddb)

f, ax = plt.subplots(figsize=(10, 10))
ax.set_frame_on(True)
#fpr_DeepDoubleB = np.load('fpr_DDB_opendata.npy')
#tpr_DeepDoubleB = np.load('tpr_DDB_opendata.npy')
#plt.figure(figsize=(12,10), dpi = 200)
lw = 2
ax.plot(tpr, fpr, color='darkorange',
                 lw=lw, label='Interaction network, AUC = %.1f'%(auc*100.))
ax.plot(tpr_ddb, fpr_ddb, color='navy', lw=lw, label='Deep double-b, AUC = %.1f'%(auc_ddb*100.))
#ax.set_facecolor('white')
ax.set_xlim(0,1)
ax.set_ylim(0.001,1)

eraText=r'2016 (13 TeV)'
xlab = '{} \\rightarrow {}'.format(sig[0][0], sig[0][-2]+'\\bar{'+sig[0][-1]+'}')
ax.set_xlabel(r'Tagging efficiency ($\mathrm{}$)'.format('{'+xlab+'}'), ha='right', x=1.0)
ylab = ['{} \\rightarrow {}'.format(l[0], l[-2]+'\\bar{'+l[-1]+'}') if l[0][0] in ["H", "Z", "g"] else l for l in bkg ]
ax.set_ylabel(r'Mistagging rate ($\mathrm{}$)'.format("{"+", ".join(ylab)+"}"), ha='right', y=1.0)
leg = ax.legend(facecolor='white',borderpad=1, frameon=False, loc=2, fontsize=16,
                            title = ""+str(min_pt)+" $\mathrm{<\ jet\ p_T\ <}$ "+str(max_pt)+" GeV" \
                          + "\n "+str(min_msd)+" $\mathrm{<\ jet\ m_{SD}\ <}$ "+str(max_msd)+" GeV"
                                       )
import matplotlib.ticker as plticker
ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.02))
ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12 )
ax.tick_params(direction='in', axis='both', which='minor' , length=6)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.semilogy()
ax.grid(which='minor', alpha=0.5, axis='y', linestyle='dotted',color='black')
ax.grid(which='major', alpha=0.9, linestyle='dotted',color='black')
ax.annotate(eraText, xy=(0.80, 1.1), fontname='Helvetica', ha='left',
                        bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
ax.annotate('$\mathbf{CMS}$', xy=(0.01, 1.1), fontname='Helvetica', fontsize=24, fontweight='bold', ha='left',
                        bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0, 'pad':13}, annotation_clip=False)
ax.annotate('$Simulation\ Open\ Data$', xy=(0.115, 1.1), fontsize=18, fontstyle='italic', ha='left',
                        annotation_clip=False)
f.savefig('%s/ROC_curve_data_generator_%s.png'%(outdir,label),dpi=400)
f.savefig('%s/ROC_curve_data_generator_%s.pdf'%(outdir,label),dpi=400)

np.save('%s/tpr_%s.npy'%(outdir,label), tpr)
np.save('%s/fpr_%s.npy'%(outdir,label), fpr)
np.save('%s/thresholds_%s.npy'%(outdir,label), thresholds)

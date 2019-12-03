Interaction Network: a graph-based neural network for particle physics applications
======================================================================================

The data used in this package was produced from [CMS open simulated data](http://doi.org/10.7483/OPENDATA.CMS.JGJX.MS7Q) using [HiggsToBBNtupleProducerTool](https://github.com/cms-opendata-analyses/HiggsToBBNtupleProducerTool). The interaction network natively runs in PyTorch, but can be exported to other formats, as explained below.

E. Moreno et al., Interaction networks for the identification of boosted Higgs to bb decays, [arXiv:1909.12285](https://arxiv.org/abs/1909.12285) [hep-ex]

Setup
======================================================================================
Clone the repository and setup the libraries. Convert the data to h5 files which can be loaded through the H5Data class in [data.py](data.py) (TODO).

```
git clone https://github.com/eric-moreno/IN.git
cd IN
source install_miniconda.sh
source install.sh
source setup.sh # every time
```

Training
======================================================================================

Change the `test_path` and `train_path` in [IN_dataGenerator.py](IN_dataGenerator.py) to reflect the directories of the test and training datasets (in converted h5 format). Change the file loading to reflect the naming scheme of your h5 files in [IN_dataGenerator.py](IN_dataGenerator.py).

Determine the parameters needed for the IN. For example: 

  - Output directory = IN_training
  - Vertex-particle branch = 1 (turned on)
  - Vertex-vertex branch = 0 (turned off)
  - De = 20 
  - Do = 24
  - Hidden = 60

Would be executed by running:

```bash
python IN_dataGenerator.py IN_training 1 0 --De 20 --Do 24 --hidden 60 
```

For adversarial training to decorrelate from soft-drop jet mass, you must have a full pretrained IN to preload with the same parameters
as the IN you are trying to adversarially train. For adversarial training you must specify the previous parameters as well as some 
additional parameters (preloaded IN directory, lambda weight, and mass bins): 

```bash
python IN_dataGenerator_adv.py IN_training_adv 0 --De 20 --Do 24 --hidden 60 --preload IN_training --lambda 10 --nbins 40  
```

Alternatively, there is also an option to decorrelate using the DDT-technique. This is performed after training with a normal IN in the [make_good_plots.py](make_good_plots.py) script. 

Evaluation 
=====================================================================================

Change the save path for the test dataset under `save_path` in IN_eval.py. Next call IN_eval.py given the network parameters and save 
location: 

```bash
python IN_eval.py IN_training 1 0 --De 20 --Do 24 --hidden 60 
```

To make various plots (ROC, pT dep, PU dep, sculpting, distributions, etc.), run [make_good_plots.py](make_good_plots.py) giving both the regular IN and mass decorrelated IN and an output directory: 

```bash
xrdcp root://eosuser.cern.ch//eos/user/w/woodson/IN/output.pkl .
xrdcp root://eosuser.cern.ch//eos/user/w/woodson/IN/output_dec.pkl .
xrdcp root://eosuser.cern.ch//eos/user/w/woodson/IN/IN_training.tar.gz .
xrdcp root://eosuser.cern.ch//eos/user/w/woodson/IN/IN_training_adv.tar.gz .
tar xvzf IN_training_adv.tar.gz 
tar xvzf IN_training.tar.gz 
rm IN_training_adv.tar.gz IN_training.tar.gz
python make_good_plots.py IN_training IN_training_adv --outdir eval_IN_training 
```
This script ([make_good_plots.py](make_good_plots.py)) will also create a DDT-version of the IN that decorrelates based on a mass-dependent threshold cut. It is *usually* better decorrelated than an adversarial training. 

Exporting to TensorFlow, MXNet, etc.
====================================================================================
This export uses [ONNX](https://github.com/onnx/onnx), an open ecosystem for interchangable ML models. It works with both mass-sculpting and mass-decorrelated models. Change the save path for the test dataset under `save_path` in [IN_onnx.py](IN_onnx.py).

To use ONNX, you must have an already trained model and must provide [IN_onnx.py](IN_onnx.py) with the parameters of this trained model: 

```bash
python IN_onnx.py IN_training 1 0 --De 20 --Do 24 --hidden 60 
```

This will save the ONNX model in the directory where the trained IN is located, which can then be easily converted into most ML frameworks. 

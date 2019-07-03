Interaction Network - A graph-based neural network for particle physics applications
======================================================================================

This data format used in this package was produced from CMS Opendata (http://opendata-dev.web.cern.ch/record/12102) and converted 
Ntuples using https://github.com/cms-opendata-analyses/HiggsToBBNtupleProducerTool. The interaction
network natively runs using pytorch, but can be exported to other formats, as explained below.

Setup
======================================================================================
Clone all the files in the repository into a directory. Save data as a number of hdf5 files which can be loaded through the H5Data 
class in data.py.

Training
======================================================================================

Change the test_path and train_path inside IN_dataGenerator.py to reflect the directories of the test and training datasets 
(in hdf5 format). Change the file loading to reflect the naming scheme of your hdf5 files inside IN_dataGenerator.py

Determine the parameters needed for the IN. For example: 

  - Output Directory = IN_training

  - Vertex-Vertex branch = 0 (turned off)

  - De = 20 

  - Do = 24

  - Hidden = 60


Would be run using :

```
python IN_dataGenerator.py IN_training 0 --De 20 --Do 24 --hidden 60 
```


For adversarial training to decorrelate from soft-drop jet mass, you must have a full pretrained IN to preload with the same parameters
as the IN you are trying to adversarially train. For adversarial training you must specify the previous parameters as well as some 
additional parameters (preloaded IN directory, lambda weight, and mass bins): 

```
python IN_dataGenerator_adv.py IN_training 0 --De 20 --Do 24 --hidden 60 --preload IN_training --lambda 10, --nbins 40  
```

Alternatively, there is also an option to decorrelate using the DDT-technique. This is performed after training with a normal IN in the make_good_plots.py script. 

Evaluation 
=====================================================================================

Change the save path for the test dataset under save_path in IN_eval.py. Next call IN_eval.py given the network parameters and save 
location: 

```
python IN_eval.py IN_training 0 --De 20 --Do 24 --hidden 60 
```

To make various plots (ROC, pT dep, PU dep, sculpting, distributions, etc.) run make_good_plots.py giving both the regular IN and 
mass decorrelated IN and an output directory: 

```
python make_good_plots.py IN_training IN_training_adv --outdir eval_IN_training 
```
This script (make_good_plots.py) will also create a DDT-version of the IN that decorrelates based on a mass-dependent threshold cut. It is *usually* better decorrelated than an adversarial training. 

Exporting to TensorFlow, MXNet, etc.
====================================================================================
This export uses ONNX (https://github.com/onnx/onnx), an open ecosystem for interchangable ML models. 
It works with both mass-sculpting and mass-decorrelated models. Change the save path for the test dataset under save_path in IN_onnx.py.

To use ONNX, you must have an already trained model and must provide IN_onnx.py with the parameters of this trained model: 

```
python IN_onnx.py IN_training 0 --De 20 --Do 24 --hidden 60 
```

This will save the ONNX model in the directory where the trained IN is located, which can then be easily converted into most ML frameworks. 

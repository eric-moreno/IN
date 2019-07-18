conda create -n pytorch-training python=3.6 matplotlib=2.0.0 pandas numpy scipy tqdm scikit-learn h5py
source activate pytorch-training
# if you have access to GPU (e.g. caltech)
#conda install pytorch torchvision cudatoolkit=7.5 -c pytorch
#pip install setGPU
# if not:
conda install pytorch-cpu torchvision-cpu -c pytorch

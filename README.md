# MoSeVa: model selection with spectral variability based on manifold learning.

This is the repo for MoSeVa: an automatic identification and quantification algorithm which considers spectral variability for gamma-ray spectrometry.
![ ](illustrations/example_moseva.PNG)

The code is organized as follows:
-  The Code folder contains the source code for the IAE and the MoSeVa,P-OMP algorithm
-  The Data folder contains the dataset of 96 spectral signatures of 12 radionuclides as a function of steel thickness.
-  The Notebooks folder contains two jupyter notebook files for training an IAE model and using MoSeVa to identify and quantify the radionuclides
      - The Models folder contains the pre-trained IAE model.
 ## Package requirements
MoSeVa was coded using Pytorch. To use MoSeVa, you will need the packages listed in environment.yml. To create and activate a conda environment with all the imports needed, do:
-  conda env create -f environment.yml
-  conda activate pytorch
Another way is to use the requirements.txt file:
-  conda install --yes --file requirements.txt
##  Test MoSeVa code
-  Run IAE_CNN_joint_gamma_spectrometry.ipynb if you want to see how IAE works and train a new IAE model with your new data.
-  Run Identification_quantification_variability.ipynb 




   

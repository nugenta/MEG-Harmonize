# MEG-Harmonize
Code from the paper "Multi-site Harmonization for Magnetoenceophalography Spectral Power Data"

The script is meant for the comparison of multiple harmonization methods. If your interest is solely in 
harmonizing MEG data created using the ENIGMA pipeline (or similarly formatted MEG data), we reccommend that 
you use the harmonization script available in the ENIGMA pipeline. That script, located in the enigmeg/group 
folder of the repository, is called enigma_harmonize.py and performs GAM-ComBat using the NeuroHarmonize 
package, which is our recommended harmonization method. 

The ENIGMA pipeline is available here: https://github.com/nih-megcore/enigma_MEG/

The script here, enigma_compare_harmonize.py, requires installation of neuroCovHarmonize, a fork of 
neuroHarmonize that also incorporates code for CovBat-Harmonization. 

neuroCovHarmonize is available here:

https://github.com/nugenta/neuroCovHarmonize

Installation of neuroHarmonize, neuroCombat, and rpy2 are also required.

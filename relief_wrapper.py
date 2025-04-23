#!/usr/bin/env python

## top ############################
## 06/2024 Justin Rajendra
## run the RELIEF function from the neuroCombat_Rpackage 
## https://github.com/Jfortin1/neuroCombat_Rpackage
## uses rpy2 to run the RELIEF function in an embedded R instance
## original RELIEF.R code is here:
## https://github.com/junjypark/RELIEF/blob/master/R/RELIEF.R
## this was tested with the RELIEF.R from ~6/2024
## there are lots of commented out print functions to test as you go
## this uses some faked data as shown on the neuroCombat_Rpackage help page
## need to bring in real data from python
## just need scanner ids and a matrix of the data 
## need 1 scanner id matching each column in the matrix

## you will need to change the path to the local RELIEF.R
reliefSource = 'RELIEF.R'

## libraries (not really needed)
# import sys, os, glob, subprocess, csv, re, shutil, random, math
# import numpy as np

## make sure you have rpy2 installed (I had to do it from source)
## https://rpy2.github.io/doc/v3.5.x/html/overview.html#installation
## this tested with rpy2 version 3.5
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
import rpy2.robjects.packages as rpackages

## import R packages
base = importr('base')
utils = importr('utils')

## packages needed for RELIEF
packnames = ('denoiseR','MASS','Matrix')
utils.chooseCRANmirror(ind=1)
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

## if the above doesn't work, try this:
# utils.install_packages('denoiseR')
# utils.install_packages('MASS')
# utils.install_packages('Matrix')

## packages for relief
denoiseR = importr('denoiseR')
MASS = importr('MASS')
Matrix = importr('Matrix')

## source the RELIEF functions from local .R file
r = robjects.r
r.source(reliefSource)   ## path to the local file

## bring the function into python
relief = robjects.globalenv['relief']
# print(relief.r_repr())

## fake data ####################
p = 10000   ## data. number of rows
n = 10      ## number of sites = number of columns
pn = p*n    ## duh
ran_num = robjects.r['runif'](pn)
dat = robjects.r['matrix'](ran_num,p,n)
# print(dat.r_repr())

## scanner id. length should equal the number of columns (n)
## as integers. these will be converted to a categorical variable
batch = robjects.IntVector([1,1,1,1,1,2,2,2,2,2])

## or you can do strings
# batch = robjects.StrVector(['a','a','a','a','a','b','b','b','b','b'])
# print(batch.r_repr())

## run the function and save to python variable #######################
## arguments to the relief function are:
## dat, batch=NULL, mod=NULL, scale.features=T, eps=1e-3, max.iter=1000, verbose=T
output = relief(dat,batch)

######################################################
## the data returned is a list of matrices, lists, and original data
## below is the output from the relief function separated out if you need them

## "dat.relief" is the output data as a matrix
dat_relief = output.rx2('dat.relief')
# print(dat_relief)

## "estimates" is a list. Below are elements of the list ############
estimates = output.rx2('estimates')
# print(estimates)

## "Xbeta" is a matrix 
Xbeta = output.rx2('estimates').rx2('Xbeta')
# print(Xbeta)

## "gamma" is a matrix 
gamma = output.rx2('estimates').rx2('gamma')
# print(gamma)

## "sigma.mat" is a matrix 
sigma_mat = output.rx2('estimates').rx2('sigma.mat')
# print(sigma_mat)

## "sigma.mat.batch" is a matrix 
sigma_mat_batch = output.rx2('estimates').rx2('sigma.mat.batch')
# print(sigma_mat_batch)

## "sigma.harnomized" is one number
sigma_harnomized = output.rx2('estimates').rx2('sigma.harnomized')
# print(sigma_harnomized)

## "R" is a matrix 
R = output.rx2('estimates').rx2('R')
# print(R)

## "I" is a matrix
I = output.rx2('estimates').rx2('I')
# print(I)

##  "E.scaled" is a matrix
E_scaled = output.rx2('estimates').rx2('E.scaled')
# print(type(E_scaled))

## "E.original" is a matrix
E_original = output.rx2('estimates').rx2('E.original')
# print(E_original)

#### end estimates list #########

## "dat.original" is the input data matrix
dat_original = output.rx2('dat.original')
# print(dat_original)

## "batch" is the input batch as a factor variable
batch_output = output.rx2('batch')
# print(batch_output)
# A density transformer is required to bridge the WC model to the radiative transfer model
# The reason for this is that the WC model returns depth-resolved densities over a fixed total depth
# whereas the RTM expects a single layer with variable depth. This script translates the former
# into the latter so that the output of the WC model can feed into the parameterised RTM.


import numpy as np
import collections
from SNICAR_feeder import snicar_feeder
import matplotlib.pyplot as plt


def density_transformer(thicknesses, densities):

    # trim densities > 915 (these are all replaced by underlying ice in snicar)
    # then trim matching elements in thickness array
    dens = densities[densities<915]
    thick = thicknesses[0:len(dens)]
    
    # calculate total thickness of porous layers ("WC")
    tot_thick = sum(thick)

    # transformations so distances below surface can act as weightings
    distances = thick * 100
    distances = distances.astype(int)

    # transform density data into N evenly spaced 1cm layers
    dens_evenspaced = np.repeat(dens,distances)
    weights= [] # empty list to append to

    # iterate through evenly spaced density list
    # each element's weight is 1-distance below surface in cm
    for i in range(len(dens_evenspaced)):
        weights.append(int((1-(i/len(dens_evenspaced)))*100))

    # average density is the mean of the evenly spaced densities
    # multiplied by their weights (i.e. for each element if the weight 
    # is e.g. 20 create 20 instances of that element in the 
    # array before averaging)
    density_av = np.mean(np.repeat(dens_evenspaced,weights)).astype(int)

    # return the parameters required to feed the snicar 
    # parameterisation: mean density and WC thickness
    
    return density_av, tot_thick




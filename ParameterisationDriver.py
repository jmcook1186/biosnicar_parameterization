"""
#####################################################################
################# BioSNICAR PARAMETERISATION ########################
##########################  DRIVER  #################################

add brief explanation

NOTE: To resolve dependencies:
runs in BioSNICAR_py Python 3.6 environment
runs inside top level directory in BioSNICAR_GO_PY respository

######################################################################
######################################################################

"""

from ParameterisationFuncs import generate_snicar_dataset_exponential,\
    call_snicar, generate_snicar_params_exponential,\
    generate_snicar_dataset_single_layer, save_model, regression_exponential,\
    regression_single_layer, evaluate_model_performance, abs_model_fit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

###########################
# SET USER DEFINE VARIABLES
###########################


single_layer_model = False
exponential_model = True


if exponential_model:
    dz = [0.001, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.1, 0.15, 0.2]
    algs_surface = [0, 1000, 5000, 10000, 15000]
    densities_surface = [300, 400, 500, 600, 700]
    zeniths = [30, 40, 45, 50, 60]#, 50, 55, 60]#, 60, 70, 80]
    savepath = '/home/joe/Code/BioSNICAR_GO_PY/'
    path_to_data = str(savepath+'snicar_data.csv')
    poly_deg = 2

    # generate dataset
    # generate_snicar_dataset_exponential(dz, algs_surface,\
    #      densities_surface, zeniths, savepath)

    # generate regression model
    #var = variable to regress against ('BBA', 'lyr1_abs', 'lyr2_abs.... lyr11_abs')
    var = 'lyr1_abs'
    model = regression_exponential(var, dz, algs_surface, densities_surface,\
        zeniths,savepath, path_to_data)

    print(model.summary())

    # save model to pickle
    save_model(model,str(savepath+'parameterisation_model_{}.pkl'.format(var)))


    ## test model
    # test_algs = [0, 1000, 6000, 8000, 14000]
    # test_dens = [350, 400, 500, 550, 650]
    # test_zeniths = [35, 40, 50, 55, 60]

    # test_results = test_model_exponential(dz, test_algs, test_dens, test_zeniths, savepath)

    # plt.scatter(test_results.snicarBBA, df.param_BBA,marker='x',color='k'),
    # plt.ylim(0.4,0.8),plt.xlim(0.4,0.8),plt.ylabel('parameterisation BBA'),plt.xlabel('snicar BBA')
    # plt.text(0.05,0.8,"SZA = 40 degrees")

    # determine subsurface exponential model for absorbed flux
    abs_model_fit(dz, path_to_data)

if single_layer_model:

    dzs = [0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]
    algs = [0, 1000, 5000, 10000, 15000, 20000]
    densities = [400, 500, 600, 700, 800, 850]
    zeniths = [40, 50, 60]
    savepath = '/home/joe/Code/BioSNICAR_GO_PY/'
    path_to_data = str(savepath+'snicar_data_single_layer.csv')
    
    # generate dataset
    generate_snicar_dataset_single_layer(densities, dzs, algs, zeniths,savepath)

    # generate regression model
    model = regression_single_layer(path_to_data)

    # save model to pickle
    save_model(model,str(savepath+'parameterisation_model_single_layer.pkl'),savepath)

    ## test model
    test_dzs = [0.02, 0.03, 0.05, 0.07, 0.1 , 0.15, 0.2]
    test_algs = [0, 1000, 6000, 8000, 14000, 18000]
    test_dens = [400, 500, 600, 700, 800]
    test_zeniths = [40, 45, 50, 60]

    test_results = test_model_single_layer(test_dens, test_dzs, test_algs,\
        test_zeniths, model, savepath)

    plt.scatter(test_results.snicar_BBA, test_results.param_BBA,marker='x',color='k'),
    plt.ylim(0,1),plt.xlim(0,1),plt.ylabel('parameterisation BBA'),plt.xlabel('snicar BBA')


    test_results = test_results[abs(test_results.snicar_BBA-test_results.param_BBA)<0.05]
    X = test_results.snicar_BBA
    y = test_results.param_BBA

    model = sm.OLS(y,X).fit()
    print(model.rsquared)


evaluate_model_performance('/home/joe/Code/BioSNICAR_GO_PY/parameterisation_tests_exponential.csv')

from ParameterisationFuncs import generate_snicar_dataset_single_layer, save_model,\
    regression_single_layer, test_model_single_layer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

###########################
# SET USER DEFINE VARIABLES
###########################

dzs = [0.1, 0.15, 0.2, 0.3, 0.5, 0.6, 0.7]
algs = [0, 3000, 5000, 7000, 9000, 11000, 13000, 15000, 20000]
densities = [400, 500, 600, 700, 800, 850, 900]
zeniths = [30, 40, 50, 60, 70, 80]
savepath = '/home/joe/Code/BioSNICARParameterization/'
path_to_data = str(savepath+'snicar_data_single_layer.csv')

# generate dataset
generate_snicar_dataset_single_layer(densities, dzs, algs, zeniths, savepath)

# generate regression model
modelBBA = regression_single_layer(path_to_data, 'BBA')
modelABS = regression_single_layer(path_to_data, 'abs')

# save model to pickle
save_model(modelBBA,str(savepath+'parameterisation_model_BBA_single_layer.pkl'),"BBA",savepath)
save_model(modelABS,str(savepath+'parameterisation_model_ABS_single_layer.pkl'),"ABS",savepath)

## test model
test_dzs = [0.18, 0.25, 0.35, 0.45, 0.55, 0.65]
test_algs = [0, 2000, 4000, 8000, 10000, 12000, 14000, 16000, 20000]
test_dens = [425, 450, 550, 650, 750, 775, 825]
test_zeniths = [35, 45, 55, 65, 75]

test_results = test_model_single_layer(test_dens, test_dzs, test_algs,\
    test_zeniths, modelBBA, modelABS, savepath)

test_results.param_BBA = test_results.param_BBA.astype(float)
test_results.param_ABS = test_results.param_ABS.astype(float)

test_results.to_csv('param_test_results.csv')

plt.figure()
plt.scatter(test_results.snicar_BBA, test_results.param_BBA, marker='^',facecolor='None',color='g')
plt.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1), color='k')
plt.ylabel('parameterisation BBA'),plt.xlabel('snicar BBA')
plt.savefig(str(savepath+"BBA_test_results.png"))

plt.figure()
plt.scatter(test_results.snicar_ABS, test_results.param_ABS,marker='o',facecolor='None',color='b')
plt.plot(np.arange(0,260,1),np.arange(0,260,1), color='k')
plt.ylabel('parameterisation abs (W/m2)'),plt.xlabel('snicar abs (W/m2)')
plt.savefig(str(savepath+"ABS_test_results.png"))


BBA_abs_error = np.mean(abs(test_results.snicar_BBA - test_results.param_BBA))
BBA_abs_error_std = np.std(abs(test_results.snicar_BBA - test_results.param_BBA))
ABS_abs_error = np.mean(abs(test_results.snicar_ABS - test_results.param_ABS))
ABS_abs_error_std = np.std(abs(test_results.snicar_ABS - test_results.param_ABS))

print(f"absolute error BBA = {BBA_abs_error} +/- {BBA_abs_error_std}")
print(f"absolute error ABS = {ABS_abs_error} +/- {ABS_abs_error_std}")

XBBA = test_results.snicar_BBA
yBBA = test_results.param_BBA

modelBBA = sm.OLS(yBBA,XBBA).fit()
print("BBA model r2 = ", modelBBA.rsquared)

XABS = test_results.snicar_ABS
yABS = test_results.param_ABS

modelABS = sm.OLS(yABS,XABS).fit()
print("Abs model r2 = ", modelABS.rsquared)



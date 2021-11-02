import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

"""

Script calculates the spectral extinction coefficient for ice with varying 
algal concentrations.

"""

dir_base = '/home/joe/Code/BioSNICAR_GO_PY/'
refidx_file = xr.open_dataset(dir_base+'Data/rfidx_ice.nc')
alg_file = xr.open_dataset(dir_base\
    +'Data/Mie_files/480band/lap/Cook2020_glacier_algae_4_40.nc')
irr_file = xr.open_dataset(dir_base\
    +'/Data/Mie_files/480band/fsds/swnb_480bnd_smm_clr_SZA45.nc')
wvl = np.arange(0.2, 4.999, 0.01)
density_WC=350 #variable
density_algae=650 #fixed
alg_list = [0, 1000e-9, 2000e-9, 5000e-9, 7500e-9, 10000e-9,\
    12500e-9, 15000e-9, 17500e-9, 20000e-9, 22500e-9, 25000e-9]


def calculate_ext_coeff(alg_list, refidx_file, alg_file,\
    irr_file, wvl, density_WC, density_algae):
    BBcoeff = []

    for alg in alg_list:
        mass_conc_algae=alg #kg alg/kg ice, variable
        abs_ice=4*np.pi*refidx_file['im_Pic16'].values/(wvl*1e-6)
        abs_algae=(alg_file['ext_cff_mss'].values*650)#m2/kg *kg /m3 = m-1
        vol_algae=mass_conc_algae*density_WC/density_algae #(m3 alg/m3 ice)
        vol_ice=1-mass_conc_algae*density_algae/density_WC
        vol_tot=vol_algae+vol_ice
        abs_coeff=abs_ice*vol_ice/vol_tot+abs_algae*vol_algae/vol_tot

        # weighted av
        abs_coeff_weighted =\
            np.sum(abs_coeff[15:100]*irr_file['flx_frc_sfc'].values[15:100])\
                /np.sum(irr_file['flx_frc_sfc'].values[15:100])
        
        # plot spectral coeff inside loop
        plt.figure(1)
        plt.plot(wvl[15:100]*1000, abs_coeff[15:100])
        
        # append BB value to list
        BBcoeff.append(abs_coeff_weighted)

    return abs_coeff, BBcoeff



def regression(alg_list, BBcoeff):

    import statsmodels.api as sm  
    X = np.array(alg_list)
    X = sm.add_constant(X)

    y = BBcoeff

    model = sm.OLS(y,X).fit()

    return model



# call func
abs_coeff, BBcoeff = calculate_ext_coeff(alg_list, refidx_file, alg_file,\
    irr_file, wvl, density_WC, density_algae)

model = regression(alg_list, BBcoeff)



plt.xlabel("Wavelength (nm)"), plt.ylabel("ext_cff (m-1)")
plt.savefig("WVL_SPECTRALEXT.jpg")

plt.figure(2)
alg_list_labels = [i*1e9 for i in alg_list]
plt.plot(alg_list_labels,BBcoeff),plt.xticks(rotation=45)
plt.ylabel("ext_cff (m-1)"), plt.xlabel("Algal concentration ppb")
plt.savefig("ALG_BBEXT.jpg")
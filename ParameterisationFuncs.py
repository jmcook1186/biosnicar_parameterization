from SNICAR_feeder import snicar_feeder
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm  
import numpy as np
import pandas as pd
import collections
import dask


def generate_snicar_params_single_layer(density, dz, alg, solzen):
    
    rho_layers = [density,density]
    grain_rds = [density,density]
    layer_type = [1,1]
    mss_cnc_glacier_algae = [alg,0]
    dz = [0.001, dz]

    params = collections.namedtuple("params","rho_layers, grain_rds, layer_type, dz, mss_cnc_glacier_algae, solzen")
    params.grain_rds = grain_rds
    params.rho_layers = rho_layers
    params.layer_type = layer_type
    params.dz = dz
    params.mss_cnc_glacier_algae = mss_cnc_glacier_algae
    params.solzen = solzen

    return params


def generate_snicar_params_exponential(dz, alg_surface, density_surface, zenith):

    cum_dz = np.cumsum(dz)
    rho_layers = np.zeros(len(dz))
    mss_cnc_glacier_algae = np.zeros(len(dz))
    layer_type = list(np.ones(len(dz)))

    mss_cnc_glacier_algae[0] = alg_surface

    for layer in np.arange(0,len(dz),1):
        rho_layers[layer] = density_surface*np.exp(cum_dz[layer])
    
    rho_layers = (np.round(rho_layers,0)).astype(int)
    rho_layers[rho_layers>900]=900
    grain_rds = rho_layers

    params = collections.namedtuple("params","rho_layers, grain_rds, layer_type, dz, mss_cnc_glacier_algae, solzen")
    params.grain_rds = grain_rds
    params.rho_layers = rho_layers
    params.layer_type = layer_type
    params.dz = dz
    params.mss_cnc_glacier_algae = mss_cnc_glacier_algae
    params.solzen = zenith

    return params


def generate_snicar_dataset_exponential(dz, algs_surface, densities_surface, zeniths, savepath):

    data = []
    BBAlist = []
    Klist = []
    lyr1_abs = []
    lyr2_abs = []
    lyr3_abs = []
    lyr4_abs = []
    lyr5_abs = []
    lyr6_abs = []
    lyr7_abs = []
    lyr8_abs = []
    lyr9_abs = []
    lyr10_abs = []
    lyr11_abs = []
    # change due to density, zenith, algae & thickness

    # @dask.delayed
    # def run_snicar(dz,density_surface,zenith,alg_surface):

    #     params = generate_snicar_params_exponential(dz, alg_surface, density_surface, zenith)             
    #     albedo, BBA = call_snicar(params)

    #     return BBA


    for i in np.arange(0,len(densities_surface),1):
        for j in np.arange(0,len(zeniths),1):
            for k in np.arange(0,len(algs_surface),1):

                dz = dz
                density_surface = densities_surface[i]
                zenith = zeniths[j]
                alg_surface = algs_surface[k]

                params = generate_snicar_params_exponential(dz, alg_surface, density_surface, zenith)             
                albedo, BBA, F_abs = call_snicar(params)
                
                #BBA = run_snicar(dz,density_surface,zenith,alg_surface)
                data.append((density_surface,zenith,alg_surface))
                BBAlist.append(BBA)

                lyr1_abs.append(F_abs[0])
                lyr2_abs.append(F_abs[1])
                lyr3_abs.append(F_abs[2])
                lyr4_abs.append(F_abs[3])
                lyr5_abs.append(F_abs[4])
                lyr6_abs.append(F_abs[5])
                lyr7_abs.append(F_abs[6])
                lyr8_abs.append(F_abs[7])
                lyr9_abs.append(F_abs[8])
                lyr10_abs.append(F_abs[9])
                lyr11_abs.append(F_abs[10])

    # result = dask.compute(*BBAlist, scheduler='processes')
    print(data)
    out = pd.DataFrame()
    density, zen, alg = zip(*data)

    out['density (kg m^-3)'] = density
    out['zenith (deg)'] = zen
    out['algae (ppb)'] = alg
    out['BBA'] = BBAlist
    out['lyr1_abs'] = lyr1_abs
    out['lyr2_abs'] = lyr2_abs
    out['lyr3_abs'] = lyr3_abs
    out['lyr4_abs'] = lyr4_abs
    out['lyr5_abs'] = lyr5_abs
    out['lyr6_abs'] = lyr6_abs
    out['lyr7_abs'] = lyr7_abs
    out['lyr8_abs'] = lyr8_abs
    out['lyr9_abs'] = lyr9_abs
    out['lyr10_abs'] = lyr10_abs
    out['lyr11_abs'] = lyr11_abs
    out.to_csv(str(savepath+'snicar_data.csv'))

    return




def generate_snicar_dataset_single_layer(densities, dzs, algs, solzens, savepath):
    
    data = []
    BBAlist = []
    # change due to density, zenith, algae & thickness

    # @dask.delayed
    def run_snicar(dz,density,zen,alg):

        params = generate_snicar_params_single_layer(density, dz, alg, zen)             
        albedo, BBA = call_snicar(params)

        return BBA


    for i in np.arange(0,len(dzs),1):
        for j in np.arange(0,len(densities),1):
            for k in np.arange(0,len(solzens),1):
                for p in np.arange(0,len(algs),1):

                    dz = dzs[i]
                    density = densities[j]
                    zen = solzens[k]
                    alg = algs[p]
                    
                    BBA = run_snicar(dz,density,zen,alg)
                    data.append((dz, density, zen, alg))
                    BBAlist.append(BBA)
    
    #result = dask.compute(*BBAlist, scheduler='processes')

    out = pd.DataFrame()
    dz, density, zen, alg = zip(*data)
    out['dz'] = dz
    out['density'] = density
    out['zenith'] = zen
    out['algae'] = alg
    out['BBA'] = BBAlist
    out.to_csv(str(savepath+'snicar_data_single_layer.csv'))

    return


def regression_exponential(var, dz, algs_surface, densities_surface,zeniths,savepath, path_to_data):

    data = pd.read_csv(path_to_data)

    X = data[['density (kg m^-3)','algae (ppb)','zenith (deg)']]
    X = sm.add_constant(X)
    y = data[var]

    model = sm.OLS(y,X).fit()

    return model


def abs_model_fit(dz, path_to_data):

    data = pd.read_csv(path_to_data)
    dzc = np.cumsum(dz)

    y = data.iloc[0].to_numpy()
    y = y[5:]
    y[0] = 10 # set to constant 10 to avoid step in abs

    # layer thicknesses in mm
    corr = [1, 20, 20, 20, 20, 20, 50, 50, 100, 150, 200]

    # normalise for variable layer thickness
    y = y/corr

    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    popt, pcov = sci.optimize.curve_fit(func, dzc, y)

    print("Exponential decay function from surface to lower boundary:")
    print("y represents absorption in Wm/2")
    print(f"\ny = {popt[0]} * np.exp(-{popt[1]} * dz) + {popt[2]}\n")


    return



def regression_single_layer(path_to_data):
    
    """
    prints model equation to console and returns the model object
    so that other functions can make predictions using model.predict()

    to see coefficients examine model.summary() - coefficients are labelled x1 - xn
    to see order of parameters examine polynomial_features.get_features()
    Then create a linear combination of coef(n) * param(n) to give final model
    
    """

    df = pd.read_csv(path_to_data)
    df = df[df.zenith>35]
    # df['density_scaled'] = (df.density-df.density.min()) / (df.density.max()-df.density.min())
    # df['zenith_scaled'] = (df.zenith-df.zenith.min()) / (df.zenith.max()-df.zenith.min())
    # df['algae_scaled'] = (df.algae-df.algae.min()) / (df.algae.max()-df.algae.min())
    # df['dz_scaled'] = (df.dz-df.dz.min()) / (df.dz.max()-df.dz.min())

    X = df[['density','dz','zenith','algae']]
    X = sm.add_constant(X)
    y = df['BBA']

    model = sm.OLS(y,X).fit()

    param1 = X.columns[1]
    param2 = X.columns[2]
    param3 = X.columns[3]
    param4 = X.columns[4]

    coef1 = np.round(model.params[1],8)
    coef2 = np.round(model.params[2],8)
    coef3 = np.round(model.params[3],8)
    coef4 = np.round(model.params[4],8)
    const = np.round(model.params[0],8)
    
    r2 = np.round(model.rsquared,4)

    print(f"r^2 value = {r2}")

    print("\nMODEL EQUATION:\n")


    model_equation = f"({coef1} * {param1})+ \
    ({coef2} * {param2}) + \
    ({coef3} * {param3}) + \
    ({coef4} * {param4}) + \
    {const}"

    print(model_equation)

    return model


def test_model_exponential(dz, algs_surface, densities_surface, zeniths, model, savepath):
    
    BBAlist =[]
    model_BBA = []
    zenlist = []
    alglist = []
    denslist =[]
    lyr1_abs = []
    lyr2_abs = []
    lyr3_abs = []
    lyr4_abs = []
    lyr5_abs = []
    lyr6_abs = []
    lyr7_abs = []
    lyr8_abs = []
    lyr9_abs = []
    lyr10_abs = []
    lyr11_abs = []

    for i in np.arange(0,len(densities_surface),1):
        for j in np.arange(0,len(zeniths),1):
            for k in np.arange(0,len(algs_surface),1):

                dz = dz
                density_surface = densities_surface[i]
                zenith = zeniths[j]
                alg_surface = algs_surface[k]

                params = generate_snicar_params_exponential(dz, alg_surface, density_surface, zenith)             
                albedo, BBA, F_abs = call_snicar(params)
                SNICARBBA.append(BBA)
                model_BBA.append(float(model.predict([1,density_surface,alg_surface,zenith])))
                zenlist.append(zenith)
                denslist.append(density_surface)
                alglist.append(alg_surface)
                lyr1_abs.append(F_abs[0])
                lyr2_abs.append(F_abs[1])
                lyr3_abs.append(F_abs[2])
                lyr4_abs.append(F_abs[3])
                lyr5_abs.append(F_abs[4])
                lyr6_abs.append(F_abs[5])
                lyr7_abs.append(F_abs[6])
                lyr8_abs.append(F_abs[7])
                lyr9_abs.append(F_abs[8])
                lyr10_abs.append(F_abs[9])
                lyr11_abs.append(F_abs[10])

                
                # trigonometry to calculate path length based on total ice thickness and SZA
                # cumdz = np.cumsum(dz)
                # depth = cumdz[-1]
                # PL = depth/np.sin(np.radians(zenith))
                # Klist.append((-1/F_dwn[0]) * ((F_dwn[0] - F_dwn[-1])/PL))

    df = pd.DataFrame()
    df['density (kg m-3)'] = denslist
    df['zenith (deg)'] = zenlist
    df['algae (ppb)'] = alglist
#    df['K (1/m)'] = K
    df['snicar_BBA'] = SNICARBBA
    df['param_BBA'] = model_BBA
    df['lyr1_abs'] = lyr0_abs
    df['lyr2_abs'] = lyr1_abs
    df['lyr3_abs'] = lyr2_abs
    df['lyr4_abs'] = lyr3_abs
    df['lyr5_abs'] = lyr4_abs
    df['lyr6_abs'] = lyr5_abs
    df['lyr7_abs'] = lyr6_abs
    df['lyr8_abs'] = lyr7_abs
    df['lyr9_abs'] = lyr8_abs
    df['lyr10_abs'] = lyr9_abs
    df['lyr11_abs'] = lyr10_abs
    
    df = df[df.snicar<0.8]
    df.to_csv(str(savepath+'parameterisation_tests.csv'))

    return df



def test_model_single_layer(test_densities, test_dzs, test_algs, test_zeniths, model, savepath):


    BBAlist =[]
    model_BBA = []
    zenlist = []
    alglist = []
    denslist =[]
    dzlist =[]

    # change due to density, zenith, algae & thickness

    # @dask.delayed
    def run_snicar(dz,density,zen,alg):

        params = generate_snicar_params_single_layer(density, dz, alg, zen)             
        albedo, BBA = call_snicar(params)

        return BBA


    for i in np.arange(0,len(test_dzs),1):
        for j in np.arange(0,len(test_densities),1):
            for k in np.arange(0,len(test_zeniths),1):
                for p in np.arange(0,len(test_algs),1):

                    dz = test_dzs[i]
                    density = test_densities[j]
                    zen = test_zeniths[k]
                    alg = test_algs[p]
                    
                    BBA = run_snicar(dz,density,zen,alg)
                    model_BBA.append(model.predict([1, density, dz, zen, alg]))
                    denslist.append(density)
                    dzlist.append(dz)
                    zenlist.append(zen)
                    alglist.append(alg)
                    BBAlist.append(BBA)
    
    
    df = pd.DataFrame()
    df['density (kg m-3)'] = denslist
    df['dz (m)'] = dzlist
    df['zenith (deg)'] = zenlist
    df['algae (ppb)'] = alglist
    df['snicar_BBA'] = BBAlist
    df['param_BBA'] = model_BBA
    
    df = df[df.snicar_BBA<0.8]
    df.to_csv(str(savepath+'parameterisation_tests_single_layer.csv'))

    return df


def save_model(model, filename, savepath):

    # save model summary to text file
    with open(filename, 'w') as fh:
        fh.write(model.summary().as_text())
    
    # save model instance to pickle
    model.save(str(savepath)+"parameterisation_model.pkl")

    return 



def evaluate_model_performance(path_to_data):

    data = pd.read_csv(path_to_data)
    data = data[data.abs_error<0.1] #remove invalid snicar runs
    mean_err = np.round(data.abs_error.mean(),4)
    std_err = np.round(data.abs_error.std(),4)

    print(f"absolute error = {mean_err} +/- {std_err}")

    print("Linear regression btwn SNICAR and param predictions:\n")
    x = data.SNICARBBA
    y = data.paramBBA

    model = sm.OLS(y,x).fit()
    print("MODEL PARAMS = \n",model.params)
    print(f"MODEL R2 = {model.rsquared}")

    return


def call_snicar(params):

    inputs = collections.namedtuple('inputs',['dir_base',\
        'rf_ice', 'incoming_i', 'DIRECT', 'layer_type',\
        'APRX_TYP', 'DELTA', 'solzen', 'TOON', 'ADD_DOUBLE', 'R_sfc', 'dz', 'rho_layers', 'grain_rds',\
        'side_length', 'depth', 'rwater', 'nbr_lyr', 'nbr_aer', 'grain_shp', 'shp_fctr', 'grain_ar',\
        'mss_cnc_soot1', 'mss_cnc_soot2', 'mss_cnc_brwnC1', 'mss_cnc_brwnC2', 'mss_cnc_dust1',\
        'mss_cnc_dust2', 'mss_cnc_dust3', 'mss_cnc_dust4', 'mss_cnc_dust5', 'mss_cnc_ash1', 'mss_cnc_ash2',\
        'mss_cnc_ash3', 'mss_cnc_ash4', 'mss_cnc_ash5', 'mss_cnc_ash_st_helens', 'mss_cnc_Skiles_dust1', 'mss_cnc_Skiles_dust2',\
        'mss_cnc_Skiles_dust3', 'mss_cnc_Skiles_dust4', 'mss_cnc_Skiles_dust5', 'mss_cnc_GreenlandCentral1',\
        'mss_cnc_GreenlandCentral2', 'mss_cnc_GreenlandCentral3', 'mss_cnc_GreenlandCentral4',\
        'mss_cnc_GreenlandCentral5', 'mss_cnc_Cook_Greenland_dust_L', 'mss_cnc_Cook_Greenland_dust_C',\
        'mss_cnc_Cook_Greenland_dust_H', 'mss_cnc_snw_alg', 'mss_cnc_glacier_algae', 'FILE_soot1',\
        'FILE_soot2', 'FILE_brwnC1', 'FILE_brwnC2', 'FILE_dust1', 'FILE_dust2', 'FILE_dust3', 'FILE_dust4', 'FILE_dust5',\
        'FILE_ash1', 'FILE_ash2', 'FILE_ash3', 'FILE_ash4', 'FILE_ash5', 'FILE_ash_st_helens', 'FILE_Skiles_dust1', 'FILE_Skiles_dust2',\
        'FILE_Skiles_dust3', 'FILE_Skiles_dust4', 'FILE_Skiles_dust5', 'FILE_GreenlandCentral1',\
        'FILE_GreenlandCentral2', 'FILE_GreenlandCentral3', 'FILE_GreenlandCentral4', 'FILE_GreenlandCentral5',\
        'FILE_Cook_Greenland_dust_L', 'FILE_Cook_Greenland_dust_C', 'FILE_Cook_Greenland_dust_H', 'FILE_snw_alg', 'FILE_glacier_algae',\
        'tau', 'g', 'SSA', 'mu_not', 'nbr_wvl', 'wvl', 'Fs', 'Fd', 'L_snw', 'flx_slr'])


    ##############################
    ## 2) Set working directory 
    ##############################

    # set dir_base to the location of the BioSNICAR_GO_PY folder
    inputs.dir_base = '/home/joe/Code/BioSNICAR_GO_PY/'
    savepath = inputs.dir_base # base path for saving figures

    ################################
    ## 3) Choose plot/print options
    ################################

    show_figs = True # toggle to display spectral albedo figure
    save_figs = False # toggle to save spectral albedo figure to file
    print_BBA = True # toggle to print broadband albedo to terminal
    print_band_ratios = False # toggle to print various band ratios to terminal
    smooth = True # apply optional smoothing function (Savitzky-Golay filter)
    window_size = 9 # if applying smoothing filter, define window size
    poly_order = 3 # if applying smoothing filter, define order of polynomial

    #######################################
    ## 4) RADIATIVE TRANSFER CONFIGURATION
    #######################################

    inputs.DIRECT   = 1       # 1= Direct-beam incident flux, 0= Diffuse incident flux
    inputs.APRX_TYP = 1        # 1= Eddington, 2= Quadrature, 3= Hemispheric Mean
    inputs.DELTA    = 1        # 1= Apply Delta approximation, 0= No delta
    inputs.solzen   = params.solzen      # if DIRECT give solar zenith angle between 0 and 89 degrees (from 0 = nadir, 90 = horizon)

    # CHOOSE ATMOSPHERIC PROFILE for surface-incident flux:
    #    0 = mid-latitude winter
    #    1 = mid-latitude summer
    #    2 = sub-Arctic winter
    #    3 = sub-Arctic summer
    #    4 = Summit,Greenland (sub-Arctic summer, surface pressure of 796hPa)
    #    5 = High Mountain (summer, surface pressure of 556 hPa)
    #    6 = Top-of-atmosphere
    # NOTE that clear-sky spectral fluxes are loaded when direct_beam=1,
    # and cloudy-sky spectral fluxes are loaded when direct_beam=0
    inputs.incoming_i = 4

    ###############################################################
    ## 4) SET UP ICE/SNOW LAYERS
    # For granular layers only, choose TOON
    # For granular layers + Fresnel layers below, choose ADD_DOUBLE
    ###############################################################

    inputs.TOON = False # toggle Toon et al tridiagonal matrix solver
    inputs.ADD_DOUBLE = True # toggle adding-doubling solver

    inputs.dz = params.dz # thickness of each vertical layer (unit = m)
    inputs.nbr_lyr = len(params.dz)  # number of snow layers
    inputs.layer_type = params.layer_type # Fresnel layers for the ADD_DOUBLE option, set all to 0 for the TOON option
    inputs.rho_layers = params.rho_layers # density of each layer (unit = kg m-3) 
    inputs.nbr_wvl=480 
    #inputs.R_sfc = np.array([0.1 for i in range(inputs.nbr_wvl)]) # reflectance of undrlying surface - set across all wavelengths
    inputs.R_sfc = np.genfromtxt('/home/joe/Code/BioSNICAR_GO_PY/Data/rain_polished_ice_spectrum.csv', delimiter = 'csv')

    ###############################################################################
    ## 5) SET UP OPTICAL & PHYSICAL PROPERTIES OF SNOW/ICE GRAINS
    # For hexagonal plates or columns of any size choose GeometricOptics
    # For sphere, spheroids, koch snowflake with optional water coating choose Mie
    ###############################################################################

    inputs.rf_ice = 2 # define source of ice refractive index data. 0 = Warren 1984, 1 = Warren 2008, 2 = Picard 2016

    # Ice grain shape can be 0 = sphere, 1 = spheroid, 2 = hexagonal plate, 3 = koch snowflake, 4 = hexagonal prisms
    # For 0,1,2,3:
    inputs.grain_shp =[0]*len(params.dz) # grain shape(He et al. 2016, 2017)
    inputs.grain_rds = params.grain_rds # effective grain radius of snow/bubbly ice
    inputs.rwater = [0]*len(params.dz) # radius of optional liquid water coating
    
    # For 4:
    inputs.side_length = 0 
    inputs.depth = 0

    # Shape factor = ratio of nonspherical grain effective radii to that of equal-volume sphere
    ### only activated when sno_shp > 1 (i.e. nonspherical)
    ### 0=use recommended default value (He et al. 2017)
    ### use user-specified value (between 0 and 1)
    inputs.shp_fctr = [0]*len(params.dz) 

    # Aspect ratio (ratio of width to length)
    inputs.grain_ar = [0]*len(params.dz) 

    #######################################
    ## 5) SET LAP CHARACTERISTICS
    #######################################

    # Define total number of different LAPs/aerosols in model
    inputs.nbr_aer = 30

    # Set names of files containing the optical properties of these LAPs:
    inputs.FILE_soot1  = 'mie_sot_ChC90_dns_1317.nc'
    inputs.FILE_soot2  = 'miecot_slfsot_ChC90_dns_1317.nc'
    inputs.FILE_brwnC1 = 'brC_Kirch_BCsd.nc'
    inputs.FILE_brwnC2 = 'brC_Kirch_BCsd_slfcot.nc'
    inputs.FILE_dust1  = 'dust_balkanski_central_size1.nc'
    inputs.FILE_dust2  = 'dust_balkanski_central_size2.nc'
    inputs.FILE_dust3  = 'dust_balkanski_central_size3.nc'
    inputs.FILE_dust4  = 'dust_balkanski_central_size4.nc'
    inputs.FILE_dust5 = 'dust_balkanski_central_size5.nc'
    inputs.FILE_ash1  = 'volc_ash_eyja_central_size1.nc'
    inputs.FILE_ash2 = 'volc_ash_eyja_central_size2.nc'
    inputs.FILE_ash3 = 'volc_ash_eyja_central_size3.nc'
    inputs.FILE_ash4 = 'volc_ash_eyja_central_size4.nc'
    inputs.FILE_ash5 = 'volc_ash_eyja_central_size5.nc'
    inputs.FILE_ash_st_helens = 'volc_ash_mtsthelens_20081011.nc'
    inputs.FILE_Skiles_dust1 = 'dust_skiles_size1.nc'
    inputs.FILE_Skiles_dust2 = 'dust_skiles_size2.nc'
    inputs.FILE_Skiles_dust3 = 'dust_skiles_size3.nc'
    inputs.FILE_Skiles_dust4 = 'dust_skiles_size4.nc'
    inputs.FILE_Skiles_dust5 = 'dust_skiles_size5.nc'
    inputs.FILE_GreenlandCentral1 = 'dust_greenland_central_size1.nc'
    inputs.FILE_GreenlandCentral2 = 'dust_greenland_central_size2.nc'
    inputs.FILE_GreenlandCentral3 = 'dust_greenland_central_size3.nc'
    inputs.FILE_GreenlandCentral4 = 'dust_greenland_central_size4.nc'
    inputs.FILE_GreenlandCentral5  = 'dust_greenland_central_size5.nc'
    inputs.FILE_Cook_Greenland_dust_L = 'dust_greenland_Cook_LOW_20190911.nc'
    inputs.FILE_Cook_Greenland_dust_C = 'dust_greenland_Cook_CENTRAL_20190911.nc'
    inputs.FILE_Cook_Greenland_dust_H = 'dust_greenland_Cook_HIGH_20190911.nc'
    inputs.FILE_snw_alg  = 'snw_alg_r025um_chla020_chlb025_cara150_carb140.nc'
    inputs.FILE_glacier_algae = 'Cook2020_glacier_algae_4_40.nc'

    # Add more glacier algae (not functional in current code)
    # (optical properties generated with GO), not included in the current model
    # algae1_r = 6 # algae radius
    # algae1_l = 60 # algae length
    # FILE_glacier_algae1 = str(dir_go_lap_files + 'RealPhenol_algae_geom_{}_{}.nc'.format(algae1_r,algae1_l))
    # algae2_r = 2 # algae radius
    # algae2_l = 10 # algae length
    # FILE_glacier_algae2 = str(dir_go_lap_files + 'RealPhenol_algae_geom_{}_{}.nc'.format(algae2_r,algae2_l))

        
    inputs.mss_cnc_soot1 = [0]*len(params.dz)    # uncoated black carbon (Bohren and Huffman, 1983)
    inputs.mss_cnc_soot2 = [0]*len(params.dz)    # coated black carbon (Bohren and Huffman, 1983)
    inputs.mss_cnc_brwnC1 = [0]*len(params.dz)   # uncoated brown carbon (Kirchstetter et al. (2004).)
    inputs.mss_cnc_brwnC2 = [0]*len(params.dz)   # sulfate-coated brown carbon (Kirchstetter et al. (2004).)
    inputs.mss_cnc_dust1 = [0]*len(params.dz)    # dust size 1 (r=0.05-0.5um) (Balkanski et al 2007)
    inputs.mss_cnc_dust2 = [0]*len(params.dz)    # dust size 2 (r=0.5-1.25um) (Balkanski et al 2007)
    inputs.mss_cnc_dust3 = [0]*len(params.dz)    # dust size 3 (r=1.25-2.5um) (Balkanski et al 2007)
    inputs.mss_cnc_dust4 = [0]*len(params.dz)    # dust size 4 (r=2.5-5.0um)  (Balkanski et al 2007)
    inputs.mss_cnc_dust5 = [0]*len(params.dz)    # dust size 5 (r=5.0-50um)  (Balkanski et al 2007)
    inputs.mss_cnc_ash1 = [0]*len(params.dz)    # volcanic ash size 1 (r=0.05-0.5um) (Flanner et al 2014)
    inputs.mss_cnc_ash2 = [0]*len(params.dz)    # volcanic ash size 2 (r=0.5-1.25um) (Flanner et al 2014)
    inputs.mss_cnc_ash3 = [0]*len(params.dz)    # volcanic ash size 3 (r=1.25-2.5um) (Flanner et al 2014)
    inputs.mss_cnc_ash4 = [0]*len(params.dz)    # volcanic ash size 4 (r=2.5-5.0um) (Flanner et al 2014)
    inputs.mss_cnc_ash5 = [0]*len(params.dz)    # volcanic ash size 5 (r=5.0-50um) (Flanner et al 2014)
    inputs.mss_cnc_ash_st_helens = [0]*len(params.dz)   # ash from Mount Saint Helen's
    inputs.mss_cnc_Skiles_dust1 = [0]*len(params.dz)    # Colorado dust size 1 (Skiles et al 2017)
    inputs.mss_cnc_Skiles_dust2 = [0]*len(params.dz)    # Colorado dust size 2 (Skiles et al 2017)
    inputs.mss_cnc_Skiles_dust3 = [0]*len(params.dz)    # Colorado dust size 3 (Skiles et al 2017)
    inputs.mss_cnc_Skiles_dust4 = [0]*len(params.dz)  # Colorado dust size 4 (Skiles et al 2017)
    inputs.mss_cnc_Skiles_dust5 = [0]*len(params.dz)  # Colorado dust size 5 (Skiles et al 2017)
    inputs.mss_cnc_GreenlandCentral1 = [0]*len(params.dz) # Greenland Central dust size 1 (Polashenski et al 2015)
    inputs.mss_cnc_GreenlandCentral2 = [0]*len(params.dz) # Greenland Central dust size 2 (Polashenski et al 2015)
    inputs.mss_cnc_GreenlandCentral3 = [0]*len(params.dz) # Greenland Central dust size 3 (Polashenski et al 2015)
    inputs.mss_cnc_GreenlandCentral4 = [0]*len(params.dz) # Greenland Central dust size 4 (Polashenski et al 2015)
    inputs.mss_cnc_GreenlandCentral5 = [0]*len(params.dz) # Greenland Central dust size 5 (Polashenski et al 2015)
    inputs.mss_cnc_Cook_Greenland_dust_L = [0]*len(params.dz)
    inputs.mss_cnc_Cook_Greenland_dust_C = [0]*len(params.dz)
    inputs.mss_cnc_Cook_Greenland_dust_H = [0]*len(params.dz)
    inputs.mss_cnc_snw_alg = [0]*len(params.dz)    # Snow Algae (spherical, C nivalis) (Cook et al. 2017)
    inputs.mss_cnc_glacier_algae = params.mss_cnc_glacier_algae   # glacier algae type1 (Cook et al. 2020)

    nbr_aer = 30
    
    outputs = snicar_feeder(inputs)
    

    return outputs.albedo, outputs.BBA, outputs.abs_slr

# -*- coding: utf-8 -*-
"""
Created on 29th of November 2023
Author: Ebbe Kyhl Gøtske

This script contains functions to parameterize the curtailment of wind energy and solar PV resources 
based on outputs from PyPSA-Eur at forced renewable penetration levels.

It contains the two key functions:

    1) base_curtailment: calculates the base curtailment of wind and solar, i.e., the curtailment
                         at which no renewable integration support measures are taken (e.g., storage 
                         deployment, transmission expansion, flexible demand, etc.). This representation
                         includes the dependence of wind curtailment on solar penetration and vice versa.

    2) technology_term: calculates the impact on the curtailment level by installing a technology 
                        (e.g., storage, transmission, heat pump, etc.) relative to the activity of 
                        the technology. For storage, the activity is the dispatch of the storage.

"""
import pandas as pd
import numpy as np

def base_curtailment(df,var,data_points,demand,x_name,base_name):
    # ---------------------------------------------------
    # This function calculates the parameters used to represent the 
    # base curtailment of wind energy and solar PV resources.
    # As the curtailment of wind energy resources not only depend 
    # on the wind penetration level but also on the solar penetration, 
    # this function calculates the curtailment parameters that account
    # for this 2D dependence.
    #
    # Inputs:
    #     df = dataframe containing the aggregated outputs of PyPSA-Eur
    #     var = name of the variable, e.g., "solar absolute curt"
    #     data_points = list containing the penetration levels used in PyPSA, 
    #                   e.g., [0.1,0.3,0.4,0.5,0.6,0.7,0.9]
    #     demand = electricity demand (including both exogenous and endogenous)
    #     x_name = name of the renewbale energy source considered (either "wind" or "solar")
    #     base_name = name of the base scenario 
    #
    # Outputs:
    #     gamma_ij_series = marginal curtailment rates
    #     x_share_df_series = the penetration level as percentage of electricity demand 
    # ---------------------------------------------------

    # configure naming convention of indexes. Here, index "i" refers to the primary variable and "j" the secondary. 
    # I.e., if we are looking at wind curtailment, index "i" refers to the wind share and "j" to the solar share.
    ij_names = {"wind": {"i_name":"w",
                         "j_name":"s"},
                "solar": {"i_name":"s",
                          "j_name":"w"}}
    globals().update(ij_names[x_name])

    # Organize data by penetration levels
    lvls = list(df[base_name][var].index.unique(level=0)) # bins used in PyPSA-Eur

    data_dic = {}
    if x_name == "solar":
        df_lvls = df[base_name][var].reorder_levels(['wind','solar']).sort_index()
    else:
        df_lvls =  df[base_name][var]
        
    # insert zeros at x_i = 0 and x_j = 0 
    df_copy = df_lvls.copy()
    for i in data_points:
        df_copy.loc[i,0] = 0
    for j in data_points:
        df_copy.loc[0,j] = df_copy.loc[0.1,j]
    df_copy = df_copy.sort_index()
    df_lvls = df_copy.copy()

    lvl_count = 0
    for lvl in data_points:
        data_dic[lvl] = df_lvls.loc[lvl]
        lvl_count += 1
    
    # create dataframe with outputs from PyPSA-Eur
    f = pd.DataFrame.from_dict(data_dic) # curtailment f(x_i,x_j)

    # Initialize dictionaries
    x_share = {} # the penetration level of x_i
    curtailment_rate_i = {} # curtailment rate in the x_i-direction
    curtailment_rate_j = {} # curtailment rate in the x_j-direction
    marg_curtailment_rate_i = {} # marginal curtailment rate in the x_i-direction
    marg_curtailment_rate_j = {} # marginal curtailment rate in the x_j-direction
    gamma_ij = {} # resulting marginal curtailment parameters 

    for j in range(len(data_points)): # loop over the data points in the j-direction 
        x_j = f.columns[j] # x_j coordinate
        for i in range(len(data_points)): # loop over the data points in the i-direction
            x_i = data_points[i] # x_i coordinate
            
            # 1) Calculate the curtailment rates
            
            # 1.1) Calculate the curtailment rate in the x_i-direction 
            if i > 0: 
                x_im1 = f.index[i-1] # previous x_i coordinate
                delta_f_i = f.loc[x_i,x_j] - f.loc[x_im1,x_j] # change in f(x_i)
                delta_x_i = (x_i - x_im1)*demand # step size in wind direction
                
                curtailment_rate_i[x_j,x_i] = delta_f_i/delta_x_i

            # 1.2) Calculate the curtailment rate in the x_j-direction
            if j > 0:
                x_jm1 = f.columns[j-1]
                delta_f_j = f.loc[x_i,x_j] - f.loc[x_i,x_jm1] # additional curtailment from increased solar share
                delta_x_j = (x_j- x_jm1)*demand 

                curtailment_rate_j[x_j,x_i] = delta_f_j/delta_x_j

            # 2) Calculate the marginal curtailment rates

            # 2.1) Calculate the marginal curtailment rate in the x_i-direction
            if i == 1:
                marg_curtailment_rate_i[x_j,x_i] = curtailment_rate_i[x_j,x_i] # in the first step, it is the same as the curtailment rate
            elif i > 1: # in the subsequent steps, we calculate the difference between the curtailment rate at x_i and x_im1
                x_im1 = f.index[i-1] # previous x_i-coordinate
                marg_curtailment_rate_i[x_j,x_i] = curtailment_rate_i[x_j,x_i] - curtailment_rate_i[x_j,x_im1]
            else:
                marg_curtailment_rate_i[x_j, x_i] = np.nan

            # 2.2) calculate the marginal curtailment rate in the x_j-direction
            if j == 1:
                marg_curtailment_rate_j[x_j,x_i] = curtailment_rate_j[x_j,x_i].copy() # in the first step, it is the same as the curtailment rate
            elif j > 1:
                x_jm1 = f.columns[j-1] # previous x_j-coordinate
                marg_curtailment_rate_j[x_j,x_i] = curtailment_rate_j[x_j,x_i] - curtailment_rate_j[x_jm1,x_i] # marginal curtailment rate in the x_j direction
            else:
                marg_curtailment_rate_j[x_j, x_i] = np.nan

    # calculate a marginal of the marginal_curtailment_rate_j in the x_i direction 
    ##### Explanation: We now have a representation of how much the wind curtailment increases
    ##### proportional to the solar penetration. To make this representation compatible with
    ##### the desired format, we need to calculate how much this proportion changes when 
    ##### moving in the wind-direction. 
    marg_curtailment_rate_j_copy = marg_curtailment_rate_j.copy()
    for j in range(len(lvls)):
        x_j = f.columns[j] 
        for i in range(len(lvls)): 
            x_i = lvls[i]
            if i > 0 and j > 0:
                x_im1 = lvls[i-1]
                marg_curtailment_rate_j_copy[x_j,x_i] = marg_curtailment_rate_j[x_j,x_i] - marg_curtailment_rate_j[x_j,x_im1] 

    # 3) Allocate the calculated marginal curtailment rates
    for j in range(len(lvls)):
        x_j = f.columns[j] 
        for i in range(len(lvls)): 
            x_i = lvls[i]
            
            if np.isnan(marg_curtailment_rate_i[(x_j, x_i)]): # if the marginal curtailment rate is nan, we do not need to allocate it.
                continue
            
            elif j == 0: # base wind curtailment disregarding solar penetration
                gamma_ij[x_name + "_curtailment_" + i_name +  str(i) + j_name + str(j)] = marg_curtailment_rate_i[(x_j, x_i)]
                x_share[x_name + "_curtailment_" + i_name + str(i) + j_name + str(j)] = x_i 
           
            elif j > 0: # additional curtailment accounting for solar penetration 
                x_jp1 = f.columns[j+1]
                gamma_ij[x_name + "_curtailment_" + i_name +  str(i) + j_name + str(j)] = marg_curtailment_rate_j_copy[(x_jp1, x_i)]
                x_share[x_name + "_curtailment_" + i_name + str(i) + j_name + str(j)] = x_i 
            
    # convert dictionaries to pandas series
    gamma_ij_series = pd.Series(gamma_ij)
    x_share_df_series = pd.Series(x_share)
    
    return gamma_ij_series, x_share_df_series

def technology_term(df, base_0, base_scenario, tech_scenario, tech_name, tech_label, tech_efficiency, renewable="wind"):
    
    # Scenario names for the different PyPSA-Eur scenarios, here for the case of LDES:
    # ---> base_0 = "new_base_co2_lim" # this is the scenario for which base curtailment is calculated
    # ---> base_scenario = "new_SDES_co2_lim" # this is the scenario that represents curtailment w/o LDES
    # ---> tech_scenario = "new_SDES_LDES_co2_lim" # this is the scenario that represents curtailment w/ LDES

    # 1. Calculate curtailment for the different scenarios:
    curt_base_0 = df[base_0][renewable + " absolute curt"] # curtailment in the base scenario
    curt_A = df[base_scenario][renewable + " absolute curt"] # curtailment in the scenario w/o the technology
    curt_B = df[tech_scenario][renewable + " absolute curt"] # curtailment in the scenario w/ the technology
    
    # 2. calculate the activity of the technology (including energy losses):
    act = df[tech_scenario][tech_name + " " + tech_label]/tech_efficiency # in case of energy storage, tech_efficiency is the round-trip efficiency

    # If we are looking at energy curtailment from solar PV, we need to reorder the indexes:
    if renewable == "solar":
        curt_base_0 = curt_base_0.reorder_levels([1,0]).sort_index()
        curt_A = curt_A.reorder_levels([1,0]).sort_index()
        curt_B = curt_B.reorder_levels([1,0]).sort_index()
        act = act.reorder_levels([1,0]).sort_index()

    # 3. calculate the curtailment reduction/increase per activity of the considered technology
    beta = ((curt_B - curt_A)/act) 

    # 4. scale the curtailment reduction to the curtailment of the base scenario
    # Explanation: The base curtailment is derived based on the "new_base_co2_lim" PyPSA scenario
    # We want to isolate the impact of installing the technology, which we can do by comparing the 
    # "new_SDES_co2_lim" and "new_SDES_LDES_co2_lim" scenarios if we are considering the impact of LDES. 
    # However, the curtailment in the "new_SDES_co2_lim" scenario is lower than in the "new_base_co2_lim" scenario. 
    # Thus, we need to scale the curtailment reduction such that we can represent it as a counteracting term to 
    # the base curtailment. Note that base_0 and base_scenario are the same for SDES. 
    beta = beta/curt_A*curt_base_0  
    beta[0.1,0.1] = 0 # curtailment is 0, so the impact is also 0

    # 5. calculate marginal curtailment (when increasing renewable penetration) per activity of the technology
    # NB! We cannot make use of the similar marginal approach as before since this would lead to nonlinearity 
    # in the curtailment reduction term (i.e., it would be a product of wind resources (variable of the optimization) 
    # and the activity of the technology (which is also a variable of the optimization). The commented lines below 
    # show the approach that would be used if we were to calculate the marginal curtailment per activity of the 
    # technology - ignoring that the constraints would be nonlinear. 

    x_js = beta.index.get_level_values(0).unique()
    x_is = beta.index.get_level_values(1).unique()

    curtailment_reduction = pd.DataFrame(index=x_js,columns=x_is) # curtailment reduction per activity of the considered technology
    curtailment_reduction.loc[:,:] = np.nan
    
    for j in range(len(x_js)): # solar index
        x_j = x_js[j]
        for i in range(len(x_is)): # wind index
            x_i = x_is[i]
            if (x_j,x_i) in beta.index:
                curtailment_reduction.loc[x_j,x_i] = beta.loc[x_j,x_i]

    return curtailment_reduction #, act # beta_mrg, beta_format

def convert_series_into_2D_matrix(series,lvls,x_i_str,x_j_str):
    curtailment_df = pd.DataFrame(index=lvls)
    for ind in curtailment_df.index:
        df_curt = series.loc[series.index.str.contains(x_i_str[0] + str(ind))]
        length = len(df_curt)
        length_dif = length - len(curtailment_df.index)
        if length_dif < 0:
            for i in range(abs(length_dif)):
                bin_add = x_i_str + str(ind) + x_j_str[0] + str(length + i) 
                df_curt_T = df_curt.T
                df_curt_T[bin_add] = np.nan
        curtailment_df[ind] = df_curt.values
    return curtailment_df
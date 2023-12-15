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

def base_curtailment(df,var,data_points,demand,x_name,base_name, continuous_axis="secondary"):
    """
    This function calculates the parameters used to represent the 
    base curtailment of wind energy and solar PV resources.
    As the curtailment of wind energy resources not only depend 
    on the wind penetration level but also on the solar penetration, 
    this function calculates the curtailment parameters that account
    for this 2D dependence.
    
    Inputs:
        df = dataframe containing the aggregated outputs of PyPSA-Eur
        var = name of the variable, e.g., "solar absolute curt"
        data_points = list containing the penetration levels used in PyPSA, 
                      e.g., [0.1,0.3,0.4,0.5,0.6,0.7,0.9]
        demand = electricity demand (including both exogenous and endogenous)
        x_name = name of the renewbale energy source considered (either "wind" or "solar")
        base_name = name of the base scenario 
        continuous_axis = which axis is continuous (either "primary" or "secondary") of the secondary term
    
    Outputs:
        gamma_ij_series = marginal curtailment rates
        x_share_df_series = the penetration level as percentage of electricity demand 
    """

    # configure naming convention of indexes. Here, index "i" refers to the primary variable and "j" the secondary. 
    # I.e., if we are looking at wind curtailment, index "i" refers to the wind share and "j" to the solar share.
    ij_names = {"wind": {"i_name":"w",
                         "j_name":"s"},
                "solar": {"i_name":"s",
                          "j_name":"w"}}
    globals().update(ij_names[x_name])

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

    # calculate marginal curtailment rate in the second direction
    if continuous_axis == "secondary":
        marg_ij = marg_curtailment_rate_j.copy()
        marg_j_copy = marg_ij.copy()

    elif continuous_axis == "primary":
        marg_ji = marg_curtailment_rate_i.copy()
        marg_j_copy = marg_ji.copy()

    # bins used in PyPSA-Eur
    lvls = list(df[base_name][var].index.unique(level=0)) 

    for j in range(len(lvls)):
        x_j = f.columns[j] if continuous_axis == "secondary" else lvls[j]
        for i in range(len(lvls)): 
            x_i = lvls[i]

            condition_i = True # if continuous_axis == "secondary" else True
            condition_j = j > 0

            if continuous_axis == "secondary" and condition_i and condition_j:
                x_im1 = lvls[i-1] if i > 0 else 0
                marg_j_copy[x_j,x_i] = marg_ij[x_j,x_i] - marg_ij[x_j,x_im1] 

            elif continuous_axis == "primary" and condition_i and condition_j:
                x_jm1 = lvls[j-1]
                marg_j_copy[x_j,x_i] = marg_ji[x_j,x_i] - marg_ji[x_jm1,x_i]
            else: 
                marg_j_copy[x_j,x_i] = np.nan

    if continuous_axis == "secondary":
        for j in range(len(lvls)):
            x_j = f.columns[j] 
            for i in range(len(lvls)):
                x_i = lvls[i]

                # moving a share of the solar proportionalities from the second to the first row
                share = 0 # 0.5 # this is to allocate a fraction of the increase in the first wind-row to the increase in solar (from 0 to 10%)
                if j == 1:
                    x_jp1 = f.columns[j+1]
                    marg_j_copy[x_j,x_i] = share*marg_j_copy[x_jp1,x_i]

                if j == 2:
                    marg_j_copy[x_j,x_i] = (1-share)*marg_j_copy[x_j,x_i]

    # 3) Allocate the calculated marginal curtailment rates
    bins = [0,0.1,0.3,0.4,0.5,0.6,0.7,0.9]
    for j in range(len(lvls)):
        x_j = f.columns[j] 
        for i in range(len(lvls)): 
            x_i = lvls[i]
            
            if np.isnan(marg_curtailment_rate_i[(x_j, x_i)]): # if the marginal curtailment rate is nan, we do not need to allocate it.
                continue
            
            x_jp1 = f.columns[j+1]
            x_j_coord = x_j if continuous_axis == "primary" and j == 0 else x_jp1
            x_im1 = bins[i]
            x_share[x_name + "_curtailment_" + i_name + str(i) + j_name + str(j)] = x_im1 
            gamma_ij[x_name + "_curtailment_" + i_name +  str(i) + j_name + str(j)] = marg_j_copy[(x_j_coord, x_i)]

    # convert dictionaries to pandas series
    gamma_ij_series = pd.Series(gamma_ij)
    x_share_df_series = pd.Series(x_share)
    
    return gamma_ij_series, x_share_df_series

def technology_term(df, base, ref_scenario, tech_scenario, tech_name, tech_label, tech_efficiency, demand, renewable="wind"):
    """ Calculation of the technology term in the curtailment parameterization.
        Inputs:
        - df = dataframe containing the aggregated outputs of PyPSA-Eur
        - base = name of the scenario for which base curtailment is calculated
        - ref_scenario = name of the scenario w/o the technology
        - tech_scenario = name of the scenario w/ the technology
        - tech_name = name of the technology
        - tech_label = label of the technology corresponding to the variable name in calculated metrics from PyPSA-Eur
        - tech_efficiency = energy efficiency of the technology (e.g., round-trip efficiency of storage)
        - renewable = name of the renewable energy source considered (either "wind" or "solar")

        Outputs:
        - beta = curtailment reduction per activity of the considered technology

    """

    # configure naming convention of indexes. Here, index "i" refers to the primary variable and "j" the secondary. 
    # I.e., if we are looking at wind curtailment, index "i" refers to the wind share and "j" to the solar share.
    ij_names = {"wind": {"i_name":"w",
                         "j_name":"s"},
                "solar": {"i_name":"s",
                          "j_name":"w"}}
    globals().update(ij_names[renewable])

    # 1. Calculate curtailment for the different scenarios:
    curt_base = df[base][renewable + " absolute curt"] # curtailment in the base scenario
    curt_A = df[ref_scenario][renewable + " absolute curt"] # curtailment in the scenario w/o the technology
    curt_B = df[tech_scenario][renewable + " absolute curt"] # curtailment in the scenario w/ the technology
    
    # 2. calculate the activity of the technology (including energy losses):
    # the units of storage dispath are MWh of electricity (in the discharging stage, accounting for energy conversion losses), 
    # so we need to rewind to the charging stage by dividing by the round-trip efficiency
    act = df[tech_scenario][tech_name + " " + tech_label]/tech_efficiency # in case of energy storage, tech_efficiency is the round-trip efficiency

    if tech_name == "transmission": # transmission expansion is calculated relative to the base transmission volume
        base_transmission = df[base][tech_name + " " + tech_label]/tech_efficiency # base transmission volume
        transmission_expansion = act - base_transmission # subtracting the base transmission volume such that the "act" variable corresponds to the transmission expansion
        normalized_transmission_expansion = transmission_expansion/base_transmission*100 # normalized transmission expansion in percentage of base transmission
        act = normalized_transmission_expansion 

    # If we are looking at energy curtailment from solar PV, we need to reorder the indexes:
    if renewable == "solar":
        curt_base = curt_base.reorder_levels([1,0]).sort_index()
        curt_A = curt_A.reorder_levels([1,0]).sort_index()
        curt_B = curt_B.reorder_levels([1,0]).sort_index()
        act = act.reorder_levels([1,0]).sort_index()

    # 3. calculate the curtailment reduction/increase per activity of the considered technology
    delta_curt_AB = curt_B - curt_A # difference in curtailment between the scenario w/o and w/ the technology 
    if tech_name == "transmission":
        delta_curt_AB = delta_curt_AB/demand*100 # we normalize the curtailment reduction by the demand

    activity_threshold = 0.01 # we only consider the impact of the technology if the activity is above this threshold
    act_norm = act/demand*100 # normalized activity in percentage of demand
    act = act.where(act_norm >= activity_threshold)

    relative_curtailment_reduction = delta_curt_AB/act
    # for transmission, we calculate the curtailment reduction [% of demand] per transmission expansion [% of base transmission]
    # for every other case, we calculate the curtailment reduction [MWh] per activity of the technology [MWh]

    # 4. scale the curtailment reduction to the curtailment of the base scenario
    # Explanation: The base curtailment is derived based on the "new_base_co2_lim" PyPSA scenario
    # We want to isolate the impact of installing the technology, which we can do by comparing the 
    # "new_SDES_co2_lim" and "new_SDES_LDES_co2_lim" scenarios if we are considering the impact of LDES. 
    # However, the curtailment in the "new_SDES_co2_lim" scenario is lower than in the "new_base_co2_lim" scenario. 
    # Thus, we need to scale the curtailment reduction such that we can represent it as a counteracting term to 
    # the base curtailment. Note that base and ref_scenario are the same for SDES. 
    relative_curtailment_reduction = relative_curtailment_reduction/curt_A*curt_base
    relative_curtailment_reduction[0.1,0.1] = 0

    x_js = relative_curtailment_reduction.index.get_level_values(0).unique()
    x_is = relative_curtailment_reduction.index.get_level_values(1).unique()

    beta_marginal = pd.DataFrame(index=x_js,columns=x_is) # curtailment reduction per activity of the considered technology
    beta_marginal.loc[:,:] = np.nan
    for j in range(len(x_js)): # secondary axis
        x_j = x_js[j]
        for i in range(len(x_is)): # primary axis
            x_i = x_is[i]
            if (x_j,x_i) in relative_curtailment_reduction.index:
                
                if j == 0:
                    if i == 0:
                        beta_marginal.loc[x_j,x_i] = relative_curtailment_reduction.loc[x_j,x_i]
                    elif i > 0:
                        beta_marginal.loc[x_j,x_i] = relative_curtailment_reduction.loc[x_j,x_i] - relative_curtailment_reduction.loc[x_j,x_is[i-1]]
            
                if j > 0:
                    beta_marginal.loc[x_j,x_i] = relative_curtailment_reduction.loc[x_j,x_i] - relative_curtailment_reduction.loc[x_js[j-1],x_i]
                
    beta_marginal_ji = beta_marginal.loc[x_js[1]:,:].diff(axis=1)
    beta_marginal_ji.loc[:,x_is[0]] = beta_marginal.loc[:,x_is[0]]
    beta_marginal.loc[x_js[1]:,:] = beta_marginal_ji

    beta_to_message = {}
    beta_for_comparison = {}
    for j in range(len(x_js)):
        x_j = x_js[j]
        for i in range(len(x_is)):
            x_i = x_is[i]
            if (x_j,x_i) in relative_curtailment_reduction.index:
                beta_to_message[renewable + "_curtailment_" + i_name +  str(i) + j_name + str(j)] = beta_marginal.loc[x_j,x_i]
                beta_for_comparison[renewable + "_curtailment_" + i_name +  str(i) + j_name + str(j)] = relative_curtailment_reduction.loc[x_j,x_i]

    # convert dictionaries to pandas series
    beta_series_1 = pd.Series(beta_to_message)
    beta_series_2 = pd.Series(beta_for_comparison)
    
    return beta_series_1, beta_series_2

def convert_series_into_2D_matrix(series,lvls,x_i_str,x_j_str):
    curtailment_df = pd.DataFrame(index=lvls)
    for ind in lvls:
        df_curt = series.loc[series.index.str.contains(x_i_str[0] + str(ind))]
        length = len(df_curt)
        length_dif = length - len(lvls)
        if length_dif < 0:
            for i in range(abs(length_dif)):
                bin_add = x_i_str + str(ind) + x_j_str[0] + str(length + i) 
                df_curt_T = df_curt.T
                df_curt_T[bin_add] = np.nan
                df_curt = df_curt_T.T
        
        curtailment_df[ind] = df_curt.values
    return curtailment_df

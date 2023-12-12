# -*- coding: utf-8 -*-
"""
Created on 4th of December 2023
Author: Ebbe Kyhl Gøtske

This script contains two key functions to calculate the curtailment based on a set of parameters
obtained from PyPSA-Eur scenarios. The first function, include_base_curtailment, calculates the
curtailment based on two terms: 
    1) the curtailment as function of the primary resource (e.g., wind), 
    2) the curtailment as function of the secondary resource (e.g., solar). 
    
The second function, include_tech_term, calculates the curtailment reduction based on the activity 
of a given technology.
"""

import pandas as pd
import numpy as np

def include_base_curtailment(renewable,primary_resource,secondary_resource,demand,continuous_axis="secondary"):
    filenames = {"wind":"gamma_ij_wind",
                 "solar":"gamma_ij_solar"}
    gamma_ij = pd.read_csv("MESSAGEix_GLOBIOM/" + filenames[renewable] + ".csv",index_col=0)
    gamma_ij = convert_index(gamma_ij,renewable)

    phi_i = [0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
    phi_j = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7]

    # term representing curtailment of the primary resource (disregarding the impact of the secondary resource)
    curtailment_primary = {}
    for i in range(len(phi_i)-1):
        curtailment_primary[0,i] = gamma_ij.loc[0,i]*(primary_resource - phi_i[i]*demand) if primary_resource > phi_i[i]*demand else 0
    curtailment_primary_df = pd.DataFrame.from_dict(curtailment_primary).T.sort_index()
    
    primary_term_sum = curtailment_primary_df.sum() 

    # term representing the impact of the secondary resource
    curtailment_ij = {}
    for i in range(len(phi_i)-1):
        for j in range(len(phi_j)):

            if (j+1,i) in gamma_ij.index:
                phi_i_limit = phi_i[i]

                if continuous_axis == "secondary":
                    curtailment = gamma_ij.loc[j+1,i]*(secondary_resource - phi_j[j]*demand) if secondary_resource > phi_j[j]*demand else 0
                elif continuous_axis == "primary":
                    curtailment = gamma_ij.loc[j+1,i]*(primary_resource - phi_i[i]*demand) if secondary_resource > phi_j[j]*demand else 0

                curtailment_ij[j,i] = curtailment if primary_resource > phi_i_limit*demand else 0
    
    try:
        curtailment_ij_df = pd.DataFrame.from_dict(curtailment_ij).T.sort_index()
        secondary_term_sum = curtailment_ij_df.sum()
        # print(curtailment_ij)
    except:
      secondary_term_sum = 0

    return primary_term_sum, secondary_term_sum

def include_tech_term(techs,act_techs,renewable,primary_resource,secondary_resource,demand):
    
    tech_term_sum = {}
    for tech in techs:
        act_tech = act_techs[tech]

        if act_tech == 0:
            tech_term_sum[tech] = 0
            continue

        df = pd.read_csv("MESSAGEix_GLOBIOM/beta_" + tech + "_" + renewable + ".csv",index_col=0)
        df = convert_index(df,renewable)
        # convert multiindex dataframe to 2D array
        df = df.unstack()
        df.index = [0.1,0.3,0.4,0.5,0.6,0.7,0.9]
        df.columns = [0.1,0.3,0.4,0.5,0.6,0.7,0.9]

        # tech term 
        curtailment_tech_ij = {}
        bins = [0.,0.1,0.3,0.4,0.5,0.6,0.7,0.9,2] 

        C = primary_resource
        D = secondary_resource

        N = len(df.columns)
        M = len(df.index)
        for i in range(1,N+1):
            for j in range(1,M+1):
                x_i_lower = bins[i-1] 
                x_j_lower = bins[j-1] 
                
                # Two conditions must be True for the beta coefficient to be applied
                
                # 1. primary resource lower bound
                condition_i_lower = C >= x_i_lower*demand # primary resource should be greater than or equal to the i-coordinate of the considered bin
                
                # 2. secondary resource lower bound
                condition_j_lower = D > x_j_lower*demand # secondary resource should be greater than or equal to the j-coordinate of the considered bin

                conditions = condition_i_lower and condition_j_lower

                if conditions:
                    beta_ij = df.loc[bins[j],bins[i]]
                    curtailment_tech_ij[j,i] = beta_ij*act_tech

        tech_term_series = pd.Series(curtailment_tech_ij).fillna(0)
        tech_term_sum[tech] = tech_term_series.sum()

    return tech_term_sum

def convert_index(df,renewable):
    df["ind"] = df.index
    df["ws_ind"] = df.ind.str.split("_",expand=True)[2]
    if renewable == "wind":
        df["solar"] = df["ws_ind"].str.split("s",expand=True)[1].astype(int)
        df["w_ind"] = df["ws_ind"].str.split("s",expand=True)[0]
        df["wind"] = df["w_ind"].str.split("w",expand=True)[1].astype(int)
        df.drop(columns=["ind","ws_ind","w_ind"],inplace=True)
        df.set_index(["solar","wind"],inplace=True)
    elif renewable == "solar":
        df["wind"] = df["ws_ind"].str.split("w",expand=True)[1].astype(int)
        df["s_ind"] = df["ws_ind"].str.split("w",expand=True)[0]
        df["solar"] = df["s_ind"].str.split("s",expand=True)[1].astype(int)
        df.drop(columns=["ind","ws_ind","s_ind"],inplace=True)
        df.set_index(["wind","solar"],inplace=True)

    return df


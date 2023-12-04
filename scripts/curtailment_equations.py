import pandas as pd
import numpy as np

def include_base_curtailment(renewable,primary_resource,secondary_resource,demand):
    filenames = {"wind":"gamma_ij_wind",
                 "solar":"gamma_ij_solar"}
    gamma_ij = pd.read_csv("MESSAGEix_GLOBIOM/" + filenames[renewable] + ".csv",index_col=0)
    gamma_ij["ind"] = gamma_ij.index
    gamma_ij["ws_ind"] = gamma_ij.ind.str.split("_",expand=True)[2]
    if renewable == "wind":
        gamma_ij["solar"] = gamma_ij["ws_ind"].str.split("s",expand=True)[1].astype(int)
        gamma_ij["w_ind"] = gamma_ij["ws_ind"].str.split("s",expand=True)[0]
        gamma_ij["wind"] = gamma_ij["w_ind"].str.split("w",expand=True)[1].astype(int)
        gamma_ij.drop(columns=["ind","ws_ind","w_ind"],inplace=True)
        gamma_ij.set_index(["solar","wind"],inplace=True)
    elif renewable == "solar":
        gamma_ij["wind"] = gamma_ij["ws_ind"].str.split("w",expand=True)[1].astype(int)
        gamma_ij["s_ind"] = gamma_ij["ws_ind"].str.split("w",expand=True)[0]
        gamma_ij["solar"] = gamma_ij["s_ind"].str.split("s",expand=True)[1].astype(int)
        gamma_ij.drop(columns=["ind","ws_ind","s_ind"],inplace=True)
        gamma_ij.set_index(["wind","solar"],inplace=True)

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
                curtailment = gamma_ij.loc[j+1,i]*(secondary_resource - phi_j[j]*demand) if secondary_resource > phi_j[j]*demand else 0
                curtailment_ij[j,i] = curtailment if primary_resource >= phi_i[i+1]*demand else 0
    try:
        curtailment_ij_df = pd.DataFrame.from_dict(curtailment_ij).T.sort_index()
        secondary_term_sum = curtailment_ij_df.sum()
    except:
        secondary_term_sum = 0

    return primary_term_sum, secondary_term_sum

def include_tech_term(techs,act_techs,renewable,primary_resource,secondary_resource,demand):
    
    tech_term_sum = {}
    for tech in techs:
        act_tech = act_techs[tech]

        # Read beta coefficients
        beta_tech_renewable = pd.read_csv("MESSAGEix_GLOBIOM/beta_" + tech + "_" + renewable + ".csv",index_col=0)
        beta_tech_renewable.columns = [0,0.1,0.3,0.4,0.5,0.6,0.7]
        beta_tech_renewable.index = [0,0.1,0.3,0.4,0.5,0.6,0.7]

        # tech term 
        curtailment_tech_ij = {}
        bins_i = [0.,0.1,0.3,0.4,0.5,0.6,0.7,0.9] #beta_tech_renewable.columns
        bins_j = [0.,0.1,0.3,0.4,0.5,0.6,0.7,0.9] #beta_tech_renewable.index

        # if renewable == "wind" and tech == "ldes":
        #     print(primary_resource,secondary_resource)

        N = len(beta_tech_renewable.columns)
        M = len(beta_tech_renewable.index)
        for i in range(N):
            for j in range(M):
                
                lt_renewable_cutoff = round(bins_i[i+1] + bins_j[j+1],2) < 1.3

                condition_i_lower = primary_resource >= bins_i[i]*demand 
                condition_j_lower = secondary_resource > bins_j[j]*demand 

                condition_i_upper = primary_resource < bins_i[i+1]*demand if i+1 < N and lt_renewable_cutoff else True 
                condition_j_upper = secondary_resource <= bins_j[j+1]*demand if j+1 < M and lt_renewable_cutoff else True 

                conditions = condition_i_lower and condition_j_lower and condition_i_upper and condition_j_upper

                # if renewable == "wind" and tech == "ldes" and bins_i[i+1] == 0.4 and bins_j[j+1] == 0.9:
                #     print(condition_i_lower,condition_j_lower,condition_i_upper,condition_j_upper)

                if conditions:
                    beta_ij = beta_tech_renewable.loc[bins_j[j],bins_i[i]]

                    curtailment_tech_ij[j,i] = beta_ij*act_tech

                    # if renewable == "wind" and tech == "ldes":
                    #     print(bins_j[j+1],bins_i[i+1],beta_ij)

        tech_term_series = pd.Series(curtailment_tech_ij).fillna(0)
        
        # count number of non-zero values
        tech_terms = tech_term_series[tech_term_series != 0]

        if len(tech_terms) > 0:
            tech_term_sum[tech] = tech_terms.iloc[0]
        else:
            tech_term_sum[tech] = tech_term_series.sum()

    return tech_term_sum
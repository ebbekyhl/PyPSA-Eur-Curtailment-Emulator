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

        if act_tech == 0:
            tech_term_sum[tech] = 0
            continue

        # Read beta coefficients
        beta_tech_renewable = pd.read_csv("MESSAGEix_GLOBIOM/beta_" + tech + "_" + renewable + ".csv",index_col=0)
        beta_tech_renewable.columns = beta_tech_renewable.columns.astype(float)
        beta_tech_renewable.index = beta_tech_renewable.index.astype(float)

        # tech term 
        curtailment_tech_ij = {}
        bins = [0.,0.1,0.3,0.4,0.5,0.6,0.7,0.9,2] 

        C = primary_resource
        D = secondary_resource

        N = len(beta_tech_renewable.columns)
        M = len(beta_tech_renewable.index)
        for i in range(1,N+1):
            for j in range(1,M+1):

                x_i_lower = bins[i-1] 
                x_i_upper = bins[i]

                x_j_lower = bins[j-1] #if (C >= 10 and D < 30) or (C >= 10 and D < 30) else beta_tech_renewable.index[j]
                x_j_upper = bins[j]

                bounds = {1:7,2:7,3:7,4:6,5:6,6:5,7:3,}
                bound_i = i == bounds[j] # upper bound of the i-coordinate of the considered bin
                bound_j = j == bounds[i] # upper bound of the j-coordinate of the considered bin

                bound_diag = round(bins[i] + bins[j],2) >= 1.3 # diagonal boundary of scenarios corresponding to the 130% VRE share cutouff

                # Four conditions must be True for the beta coefficient to be applied
                
                # 1. primary resource lower bound
                condition_i_lower = C >= x_i_lower*demand # primary resource should be equal to or greater than the i-coordinate of the considered bin
                
                # 2. primary resource upper bound
                condition_i_upper = C < x_i_upper*demand if not bound_i and not bound_diag else True 
                
                # 3. + 4. secondary resource lower and upper bounds
                if j == 1:
                    condition_j_lower = True 
                    condition_j_upper = D <= x_j_upper*demand if not bound_j and not bound_diag else True 
                elif j == 2:
                    condition_j_lower = D > x_j_lower*demand # secondary resource should be equal to or greater than the j-coordinate of the considered bin
                    condition_j_upper = D < x_j_upper*demand if not bound_j and not bound_diag else True 
                else:
                    condition_j_lower = D >= x_j_lower*demand # secondary resource should be equal to or greater than the j-coordinate of the considered bin
                    condition_j_upper = D < x_j_upper*demand if not bound_j and not bound_diag else True 

                conditions = condition_i_lower and condition_j_lower and condition_i_upper and condition_j_upper

                if conditions:
                    beta_ij = beta_tech_renewable.loc[bins[j],bins[i]]
                    curtailment_tech_ij[j,i] = beta_ij*act_tech

        if len(curtailment_tech_ij) > 0:
            tech_term_series = pd.Series(curtailment_tech_ij).fillna(0)
        
            # count number of non-zero values
            tech_terms = tech_term_series[tech_term_series != 0]
            if len(tech_terms) > 1:
                # if there are multiple values, take the value at the highest primary resource level           
                # first, reorder the index to have the primary resource level as the first level
                tech_terms = tech_terms.swaplevel(0,1)

                # then take the value at the highest primary resource level
                tech_terms_max_coord = max(tech_terms.index.get_level_values(0))
                
                tech_term_sum[tech] = tech_terms.loc[tech_terms_max_coord,:].sum()

            else:
                tech_term_sum[tech] = tech_term_series.sum()

        else:
            print("Issues with " + tech + " ," + renewable + " (" + str(D) + "," + str(C) + ") !")
            tech_term_sum[tech] = 0

    return tech_term_sum
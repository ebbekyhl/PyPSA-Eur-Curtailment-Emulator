import pandas as pd
from itertools import product
from message_ix.utils import make_df

def keep_existing_tech_contributions(sc_ref, message_techs_in_curtailment_rels, renewable,beta_tech_renewable, shift_beta=False):
    # interregional elcetricity flow (e.g., from Europe to North Africa) is something that we cannot 
    # represent in this softlinkage. For this reason, we want to keep the coefficients from the original
    # representation. We do this by first initializing the 2D array corresponding to the wind and solar 
    # bins used in PyPSA-Eur and then we copy the coefficients from the original representation to the
    # 2D array. This is only done for the first row in the array, corresponding to wind penetration without 
    # concurrent solar penetration (i.e., solar penetration = 0), to comply with the original representation.
    penetration_lvl_pypsa = {0:0,
                            1:10,
                            2:30,
                            3:40,
                            4:50,
                            5:60,
                            6:70}

    # Acquire parameters for the technology not included (yet) in the softlinkage
    tech_par = sc_ref.par("relation_activity",
                        filters={"relation": [renewable + "_curtailment_1",
                                                renewable + "_curtailment_2",
                                                renewable + "_curtailment_3"],
                                    "node_loc": ["R11_WEU"],
                                    "year_rel": [2050],
                                    "technology": message_techs_in_curtailment_rels,
                                    })

    # Parameters for the electricity demand defining the renewable penetration bins 
    elec_par = sc_ref.par("relation_activity",
                        filters={"relation": [renewable + "_curtailment_1",
                                                renewable + "_curtailment_2",
                                                renewable + "_curtailment_3"],
                                    "node_loc": ["R11_WEU"],
                                    "year_rel": [2050],
                                    "technology": ["elec_t_d"],
                                    })

    beta_tech = beta_tech_renewable["LDES"].copy()
    beta_tech.columns = ["beta"]
    beta_tech.loc[:,:] = 0
    solar_index = pd.Index([int(x.split("_")[2].split("s")[1].split("w")[0]) for x in beta_tech.index])
    wind_index = pd.Index([int(x.split("_")[2].split("w")[1].split("s")[0]) for x in beta_tech.index])
    beta_tech["wind_share"] = [penetration_lvl_pypsa[x] for x in wind_index]
    beta_tech["solar_share"] = [penetration_lvl_pypsa[x] for x in solar_index]

    beta_techs = {}
    for t in message_techs_in_curtailment_rels:
        tech_par_t = tech_par.query("technology == @t")
        beta_tech_insert = beta_tech.copy()

        penetration_lvl_message = (elec_par.value.abs()*100).to_list()
        beta_message = tech_par_t.value.to_list()

        secondary_index = solar_index if renewable == "wind" else wind_index

        for lvl in range(len(beta_message)):
            penetration_lvl_message_lvl = penetration_lvl_message[lvl]

            primary_col = "wind_share" if renewable == "wind" else "solar_share"
            secondary_col = "solar_share" if renewable == "wind" else "wind_share"
            beta_tech_insert_1d = beta_tech_insert.loc[beta_tech_insert[secondary_col] == 0]
            index_lvl = beta_tech_insert_1d.loc[beta_tech_insert_1d[primary_col] <= penetration_lvl_message_lvl].index[-1]
            
            beta_tech_insert.loc[index_lvl,"beta"] = beta_message[lvl]

        beta_tech_insert = beta_tech_insert.loc[beta_tech_insert.beta.drop_duplicates().index]
        beta_tech.loc[beta_tech_insert.index,"beta"] = beta_tech_insert

        beta_tech_t_df = pd.DataFrame(beta_tech.beta)
        beta_tech_t_df.columns = [0]

        if shift_beta:
            # here, we try to shift the coefficients of the first bin in the original representation
            # this is because, the 0 coefficient in the first bin leads to very low VRE penetration
            if renewable == "wind":
                beta_tech_t_df.loc["wind_curtailment_w0s0"] = beta_tech_t_df.loc["wind_curtailment_w2s0"]
                beta_tech_t_df.loc["wind_curtailment_w2s0"] = 0
            elif renewable == "solar":
                beta_tech_t_df.loc["solar_curtailment_s0w0"] = beta_tech_t_df.loc["solar_curtailment_s2w0"]
                beta_tech_t_df.loc["solar_curtailment_s2w0"] = 0

        beta_techs[t] = beta_tech_t_df

        beta_tech_renewable_extended = beta_tech_renewable.copy()
        beta_tech_renewable_extended.update(beta_techs)

    return beta_tech_renewable_extended

def split_wind(df,cname,str1=""):
    df["solar"] = df["index"].str.split("s",expand=True)[1].astype(int) + 1
    df["wind"] = df["index"].str.split("s",expand=True)[0].str.split("w",expand=True)[2].astype(int) + 1
    df_wind_only = df.query("solar == 1")

    df_wind_only["prefix"] = "wind_curtailment"
    df_wind_only["name"] = df_wind_only["prefix"] + str1 + df_wind_only["wind"].astype(str)
    df_wind_only.set_index("name",inplace=True)

    wind_only_dict = df_wind_only[cname].to_dict()

    return wind_only_dict

def split_solar(df,cname,str1=""):
    df["wind"] = df["index"].str.split("w",expand=True)[1].astype(int) + 1
    df["solar"] = df["index"].str.split("w",expand=True)[0].str.split("s",expand=True)[2].astype(int) + 1

    df_solar_only = df.query("wind == 1")
    df_solar_only["prefix"] = "solar_curtailment"
    df_solar_only["name"] = df_solar_only["prefix"] + str1 + df_solar_only["solar"].astype(str)
    df_solar_only.set_index("name", inplace=True)

    solar_only_dict = df_solar_only[cname].to_dict()

    return solar_only_dict

def add_storage_tech(sc, sc_ref, tech, years, capacity_factor, inv_cost, lifetime, efficiency, region, df_beta_solar_SDES, df_beta_solar_LDES, df_beta_wind_LDES, df_beta_wind_SDES, change_structure=False):

    df_input = make_df('input',
                       node_loc=region,
                       technology= tech,
                       year_vtg=years,
                       year_act=years,
                       mode="M1",
                       node_origin=region,
                       commodity="electr",
                       level="secondary",
                       time="year",
                       time_origin="year",
                       value = 1 - efficiency,
                       unit = "GWa")

    df_output = make_df('output',
                            node_loc = region,
                            technology = tech,
                            year_vtg = years,
                            year_act = years,
                            mode = "M1",
                            node_dest = region,
                            commodity = 'exports',
                            level = "secondary",
                            time = "year",
                            time_dest = "year",
                            value = 1,
                            unit = "GWa",
                            )

    df_CF = make_df('capacity_factor',
                            node_loc = region,
                            technology = tech,
                            year_vtg = years,
                            year_act = years,
                            time = "year",
                            value = capacity_factor/100,
                            unit = "%",
                            )

    df_inv_cost = make_df('inv_cost',
                            node_loc=region,
                            technology=tech,
                            year_vtg=years,
                            value=inv_cost,
                            unit="USD/GWa",
                            )

    df_lifetime = make_df('technical_lifetime',
                            node_loc=region,
                            technology=tech,
                            year_vtg=years,
                            value=lifetime,
                            unit="y")
    

    # Now that we have added the LDES and SDES as technologies in the scenario, 
    # we also need to define their parameters in the VRE integration constraints. 
    # Here, we add the parameters for LDES and SDES to the firm capacity constraint
    # and the flexibility constraint. We assume the storage is fully contributing to
    # the firm capacity constraint and the flexibility constraint, i.e., they have 
    # values of 1.0 (which is the same value for stor_ppl):
    
    # firm capacity constraint
    df_res_marg = make_df('relation_total_capacity',
                          relation = "res_marg",
                          node_rel = region,
                          year_rel = years,
                          technology = tech,
                          value = 1.0,
                          unit = "???"
                          )
    
    # flexibility constraint
    df_oper_res = make_df('relation_activity',
                          relation = "oper_res",
                          node_rel = region,
                          year_rel = years,
                          node_loc = region,
                          technology = tech,
                          year_act = years,
                          mode = "M1",
                          value = 1.0,
                          unit = "???"
                          )

    sc.add_set("technology", tech)
    sc.add_par('input',df_input)
    sc.add_par('output',df_output)
    sc.add_par("capacity_factor", df_CF)
    sc.add_par('inv_cost',df_inv_cost)
    sc.add_par('technical_lifetime',df_lifetime)
    sc.add_par('relation_total_capacity',df_res_marg)
    sc.add_par("relation_activity",df_oper_res)

    # curtailment constraint
    if not change_structure:
        df_storage_curtailment = sc_ref.par("relation_activity",
                                            {"node_rel": region,
                                                "relation": ["wind_curtailment_1","wind_curtailment_2","wind_curtailment_3",
                                                             "solar_curtailment_1","solar_curtailment_2","solar_curtailment_3"],
                                                "technology":"stor_ppl",
                                                })
        df_storage_curtailment.replace({"stor_ppl":tech}, inplace=True)

        # get technology impact from PyPSA-Eur scenarios (we need to group the PyPSA-Eur parameters to fit the old structure)
        sdes_solar_lvl1 = df_beta_solar_SDES["0"].loc["1"]
        sdes_solar_lvl2 = df_beta_solar_SDES["0"].loc["2"]
        sdes_solar_lvl3 = df_beta_solar_SDES["0"].loc[["3","4","5","6"]].sum()

        ldes_solar_lvl1 = df_beta_solar_LDES["0"].loc["1"]
        ldes_solar_lvl2 = df_beta_solar_LDES["0"].loc["2"]
        ldes_solar_lvl3 = df_beta_solar_LDES["0"].loc[["3","4","5","6"]].sum()

        ldes_wind_lvl1 = df_beta_wind_LDES.T["0"].loc["1"]
        ldes_wind_lvl2 = df_beta_wind_LDES.T["0"].loc["2"]
        ldes_wind_lvl3 = df_beta_wind_LDES.T["0"].loc[["3","4","5","6"]].sum()

        sdes_wind_lvl1 = df_beta_wind_SDES.T["0"].loc["1"]
        sdes_wind_lvl2 = df_beta_wind_SDES.T["0"].loc["2"]
        sdes_wind_lvl3 = df_beta_wind_SDES.T["0"].loc[["3","4","5","6"]].sum()

        tech_impact_solar = {"SDES":[sdes_solar_lvl1, sdes_solar_lvl2, sdes_solar_lvl3],
                            "LDES":[ldes_solar_lvl1, ldes_solar_lvl2, ldes_solar_lvl3]}

        tech_impact_wind = {"SDES":[sdes_wind_lvl1, sdes_wind_lvl2, sdes_wind_lvl3],
                            "LDES":[ldes_wind_lvl1, ldes_wind_lvl2, ldes_wind_lvl3]}

        for i in [1,2,3]:
            wind_curt_tech = "wind_curtailment_" + str(i)
            index_curt_tech = df_storage_curtailment.query("relation == @wind_curt_tech").index
            df_storage_curtailment.loc[index_curt_tech,"value"] = tech_impact_wind[tech][i-1]

            solar_curt_tech = "solar_curtailment_" + str(i)
            index_curt_tech = df_storage_curtailment.query("relation == @solar_curt_tech").index
            df_storage_curtailment.loc[index_curt_tech,"value"] = tech_impact_solar[tech][i-1]

        sc.add_par("relation_activity",df_storage_curtailment)

def keep_existing_curtailment_rates(sc_ref, renewable, gamma_ij):
    # gamma coefficients in the original representation
    gamma_old = sc_ref.par("input",
                            filters={"technology": [renewable + "_curtailment1",
                                                    renewable + "_curtailment2",
                                                    renewable + "_curtailment3"],
                                        "node_loc": ["R11_WEU"],
                                        "year_act": [2050],
                                        })[["technology","value"]]

    # penetration levels in the original representation
    elec_par = sc_ref.par("relation_activity",
                    filters={"relation": [renewable + "_curtailment_1",
                                            renewable + "_curtailment_2",
                                            renewable + "_curtailment_3"],
                                "node_loc": ["R11_WEU"],
                                "year_rel": [2020],
                                "technology": ["elec_t_d"],
                                })

    # penetration level in PyPSA-Eur
    penetration_lvl_pypsa = {0:0,
                            1:10,
                            2:30,
                            3:40,
                            4:50,
                            5:60,
                            6:70}

    gamma_new = gamma_ij.copy()
    
    gamma_new.columns = ["gamma_coef"]
    gamma_new.loc[:,:] = 0
    solar_index = pd.Index([int(x.split("_")[2].split("s")[1].split("w")[0]) for x in gamma_new.index])
    wind_index = pd.Index([int(x.split("_")[2].split("w")[1].split("s")[0]) for x in gamma_new.index])
    gamma_new["wind_share"] = [penetration_lvl_pypsa[x] for x in wind_index]
    gamma_new["solar_share"] = [penetration_lvl_pypsa[x] for x in solar_index]        

    penetration_lvl_message = (elec_par.value.abs()*100).to_list()
    secondary_index = solar_index if renewable == "wind" else wind_index

    gamma_new_insert = gamma_new.copy()

    for lvl in range(len(penetration_lvl_message)):
        penetration_lvl_message_lvl = penetration_lvl_message[lvl]
        penetration_lvl_message_lvlp1 = penetration_lvl_message[lvl+1] if lvl < len(penetration_lvl_message)-1 else 100

        index_lvl = gamma_new.loc[secondary_index == 0].query(renewable + "_share > @penetration_lvl_message_lvl").query(renewable + "_share < @penetration_lvl_message_lvlp1").index

        if len(index_lvl) > 0:
            index_lvl = index_lvl[0]

        if renewable == "wind":
            gamma_new_insert.loc[index_lvl,"gamma_coef"] = gamma_old.iloc[lvl].value
        else:
            gamma_new_insert.loc[index_lvl,"gamma_coef"] = sum(gamma_old.value) # all bins in the existing MESSAGEix-GLOBIOM for solar PV belong to the same bin in the PyPSA-Eur data

    gamma_new = gamma_new_insert.drop(columns=["solar_share","wind_share"]
                               )
    gamma_new.columns = [0]
    
    return gamma_new

def add_curtailment_techs(renewable_penetration_dict, beta_dict, gamma_ij_wind, gamma_ij_solar, replace_stor_ppl):
    # number of bins (determined from the PyPSA-Eur data):
    bins = list(set(renewable_penetration_dict.values()))
    bins.sort()

    bins_dict = {}
    curt_relation = {} # i = solar, j = wind
    curt_relation_tech = {}
    new_bins = {}
    prefix = "vre" # prefix of relation name
    for i in range(len(bins)): # wind index
        for j in range(len(bins)): # solar index
            
            w = "wind_curtailment_w" + str(i) + "s" + str(j) # index naming in the data achieved from PyPSA-Eur
            s = "solar_curtailment_s" + str(j) + "w" + str(i) # index naming in the data achieved from PyPSA-Eur

            wind_curt_name = "wind_curtailment" + str(i+1) # naming of wind curtailment technology in MESSAGEix-GLOBIOM
            solar_curt_name = "solar_curtailment" + str(j+1) # naming of solar curtailment technology in MESSAGEix-GLOBIOM
            # for naming convenience, we shift the index naming by one such that the first bin starts at index = 1

            if (w not in gamma_ij_wind.index) and (s not in gamma_ij_solar.index): 
                continue
            
            # technology parameters from PyPSA-Eur
            LDES_tech_term = beta_dict["wind","LDES"][w] + beta_dict["solar","LDES"][s]
            SDES_tech_term = beta_dict["wind","SDES"][w] + beta_dict["solar","SDES"][s]
            
            # keeping technology parameters from original MESSAGEix-GLOBIOM representation
            EV_tech_term = beta_dict["wind","elec_trp"][w] + beta_dict["solar","elec_trp"][s]
            h2_tech_term = beta_dict["wind","h2_elec"][w] + beta_dict["solar","h2_elec"][s]
            exp_tech_term = beta_dict["wind","elec_exp_eurasia"][w] + beta_dict["solar","elec_exp_eurasia"][s]
            imp_tech_term = beta_dict["wind","elec_imp_eurasia"][w] + beta_dict["solar","elec_imp_eurasia"][s]

            new_bins[prefix + "_curtailment_w" + str(i+1) + "s" + str(j+1)] = [wind_curt_name, solar_curt_name]
            
            rel_dict = {wind_curt_name:gamma_ij_wind.loc[w].item(),
                        solar_curt_name:gamma_ij_solar.loc[s].item(),
                        "elec_trp":EV_tech_term,
                        "h2_elec":h2_tech_term,
                        "elec_exp_eurasia":exp_tech_term,
                        "elec_imp_eurasia":imp_tech_term,
                        "elec_exp_eur_afr":exp_tech_term,
                        "elec_imp_eur_afr":imp_tech_term,
                        }
            
            if replace_stor_ppl:
                rel_dict.update({"LDES":LDES_tech_term,
                                "SDES":SDES_tech_term})
                
            else:
                stor_ppl_tech_term = beta_dict["wind","stor_ppl"][w] + beta_dict["solar","stor_ppl"][s]
                rel_dict.update({"stor_ppl":stor_ppl_tech_term})

            curt_relation[prefix + "_curtailment_w" + str(i+1) + "s" + str(j+1)] = [rel_dict]
            
            if j == 0:
                curt_relation_tech["wind_curtailment_" + str(i+1)] = wind_curt_name
                bins_dict["wind_curtailment_" + str(i+1)] = bins[i]
            
            if i == 0:
                curt_relation_tech["solar_curtailment_" + str(j+1)] = solar_curt_name
                bins_dict["solar_curtailment_" + str(j+1)] = bins[j]

    curt_relation_tech = {k: curt_relation_tech[k] for k in sorted(curt_relation_tech)}

    # print first five entries of the dictionaries
    print({k: curt_relation[k] for k in list(curt_relation)[:5]})
    print({k: curt_relation_tech[k] for k in list(curt_relation_tech)[:5]})

    return curt_relation_tech, bins_dict, new_bins, curt_relation

def add_curtailment_relations(sc, sc_ref, bins_dict, regions, curt_relation_tech, parname):
    # In this step, we add the parameters representing the base curtailment, i.e., we don't yet 
    # consider the role of curtailment-reducing technologies like storage etc.
    for rel, region in product(sorted(bins_dict.keys()), regions):
        
        # renewable name (wind or solar)
        renewable = rel.split("_")[0]
        # curtailment technology name (e.g., "wind_curtailment_1")
        curtail_tec = curt_relation_tech[rel]

        # Load existing data (and use it later for configuring the new data)
        old = sc_ref.par(parname, {"node_loc": region, 
                                   "relation": renewable + "_curtailment_1"})
        
        # Generate theoretical curtailment bins (with contributor technologies, but
        # these will be removed at the end)
        new = old.copy()
        new.relation = rel
        new.technology.replace({renewable + "_curtailment1":curtail_tec},inplace=True)
        # Edit the % share of wind/solar bins (Notice (-) sign)
        new.loc[new.query("technology == 'elec_t_d'").index,"value"] = -bins_dict[rel]

        # Add theoretical curtailment to the scenario
        sc.add_par(parname, new)

        # Add an upper bound for the theoretical curtailment (needed for new bins)
        bound = new.drop_duplicates(["node_rel", "year_rel", "relation"]).copy()
        bound["value"] = 0
        sc.add_par("relation_upper", bound)

        # Update the data of "input" electricity for this curtailment technology
        inp_old = sc.par("input", {"node_loc": region, 
                            "technology": renewable + "_curtailment1"})
        
        inp_new = inp_old.copy()
        inp_new.technology = curt_relation_tech[rel] # is needed in case a new bin is added
        inp_new.value = 1 # Note that this is different compared to the old approach. The gamma coefficients are added later.
        sc.add_par("input", inp_new)

    print("- New theoretical curtailment relations configured.")

def add_new_bins(sc, new_bins, parname, curt_relation, curt_relation_tech, regions, years):
    sc.add_set("relation", new_bins.keys())

    for rel_new, region in product(sorted(new_bins.keys()), regions):
        # Relevant wind/solar relations
        relations = [x for x, y in curt_relation_tech.items() if y in new_bins[rel_new]]

        old = sc.par(parname, {"node_loc": region, 
                                "relation": relations})
        if old.empty:
            continue

        # Keep only contributor and curtailment technologies
        # This does not need VRE generation and electricity grid, as the curtailment
        # bins as % of the grid were calculated in step (3) and here we use them
        techs = [x for x in set(old["technology"]) if
                not any([y in x for y in ["wind_r", "solar_r", "elec_t_d"]])]
        
        new = old.loc[old["technology"].isin(techs)].copy()

        # Group technology contributions across wind and solar relations. 
        # This is done by taking the sum of contribution values for solar and
        # wind for the considered VRE bin (e.g., if storage has 0.2 for wind, 0.1 for solar,
        # sum(0.2 , 0.1) = 0.3 will be used for this VRE bin)
        new = new.groupby(
            ["node_rel", "node_loc", "mode", "technology", "year_rel", "year_act"]
                        ).sum(numeric_only=True)
        new = new.assign(relation=rel_new, unit="GWa").reset_index()

        # insert value
        new.loc[new["technology"].isin(new_bins[rel_new]), "value"] = 1
        for t in curt_relation[rel_new][0].keys():
            new_ind = new.query("technology == @t").index

            if len(new_ind) > 0:
                new.loc[new_ind,"value"] = curt_relation[rel_new][0][t]
            else:
                df = make_df("relation_activity",
                            relation=rel_new,
                            node_rel="R11_WEU",
                            year_rel=years,
                            node_loc="R11_WEU",
                            technology=t,
                            year_act=years,
                            mode = "M1",
                            value = curt_relation[rel_new][0][t],
                            unit = "???")
            
                sc.add_par("relation_activity", df)

        # Add the new equation for the VRE curtailment bin to the scenario
        sc.add_par(parname, new)

        # Add an upper bound for the new relation
        bound = new.drop_duplicates(["node_rel", "year_rel", "relation"]).copy()
        bound["value"] = 0
        sc.add_par("relation_upper", bound)

        # If treating input electricity at the VRE level (combination of solar-wind bins)
        
        # Relevant VRE technology
        tech_new = ("_").join(rel_new.split("_")[:2]) + rel_new.split("_")[2]
        
        # Add new VRE technology to the scenario
        sc.add_set("technology", tech_new)
        
        # Add VRE technology to this curtailment relation
        vre = new.loc[new["technology"].isin(new_bins[rel_new])]
        
        # Change the sign of VRE curtailment value to (-)
        # VRE curtailment is equal to the unmet curtailment of wind and solar, i.e.:
        # wind_curtail + solar_curtail <= storage (all contributors) + VRE_curtail
        vre = vre.assign(technology=tech_new, 
                        value=-1)
        
        sc.add_par(parname, vre)

        # Add "input" electricity for this VRE technology
        # Load sample data and copy it for each bin
        # if include_vre_input_loss:
        inp = sc.par("input", {"node_loc": regions,
                                "technology": "solar_curtailment1"})
        
        # Add input values for curtailed electricity at each combination of wind and solar
        inp = inp.assign(technology=tech_new, 
                            value=1)
        
        sc.add_par("input", inp)

    print("- New VRE curtailment relations configured.")

def add_solar_wind_share_to_df(beta_tech_solar, beta_tech_wind, gamma_ij_wind, gamma_ij_solar):
    df_beta_solar = beta_tech_solar["LDES"].copy()
    df_beta_solar["ind"] = df_beta_solar.index
    df_beta_solar["ind"] = df_beta_solar.ind.str.split("_",expand=True)[2]
    df_beta_solar["solar"] = df_beta_solar["ind"].str.split("s",expand=True)[1].str.split("w",expand=True)[0]
    df_beta_solar["wind"] = df_beta_solar["ind"].str.split("s",expand=True)[1].str.split("w",expand=True)[1]
    df_beta_solar = df_beta_solar.set_index(["solar","wind"])["0"]
    df_beta_solar = df_beta_solar.unstack().sort_index(ascending=False)
    df_beta_solar_LDES = df_beta_solar.round(3).copy()

    df_beta_solar = beta_tech_solar["SDES"].copy()
    df_beta_solar["ind"] = df_beta_solar.index
    df_beta_solar["ind"] = df_beta_solar.ind.str.split("_",expand=True)[2]
    df_beta_solar["solar"] = df_beta_solar["ind"].str.split("s",expand=True)[1].str.split("w",expand=True)[0]
    df_beta_solar["wind"] = df_beta_solar["ind"].str.split("s",expand=True)[1].str.split("w",expand=True)[1]
    df_beta_solar = df_beta_solar.set_index(["solar","wind"])["0"]
    df_beta_solar = df_beta_solar.unstack().sort_index(ascending=False)
    df_beta_solar_SDES = df_beta_solar.round(3).copy()

    df_beta_wind = beta_tech_wind["LDES"].copy()
    df_beta_wind["ind"] = df_beta_wind.index
    df_beta_wind["ind"] = df_beta_wind.ind.str.split("_",expand=True)[2]
    df_beta_wind["solar"] = df_beta_wind["ind"].str.split("s",expand=True)[1]
    df_beta_wind["wind"] = df_beta_wind["ind"].str.split("s",expand=True)[0].str.split("w",expand=True)[1]
    df_beta_wind = df_beta_wind.set_index(["solar","wind"])["0"]
    df_beta_wind = df_beta_wind.unstack().sort_index(ascending=False)
    df_beta_wind_LDES = df_beta_wind.round(3).copy()

    df_beta_wind = beta_tech_wind["SDES"].copy()
    df_beta_wind["ind"] = df_beta_wind.index
    df_beta_wind["ind"] = df_beta_wind.ind.str.split("_",expand=True)[2]
    df_beta_wind["solar"] = df_beta_wind["ind"].str.split("s",expand=True)[1]
    df_beta_wind["wind"] = df_beta_wind["ind"].str.split("s",expand=True)[0].str.split("w",expand=True)[1]
    df_beta_wind = df_beta_wind.set_index(["solar","wind"])["0"]
    df_beta_wind = df_beta_wind.unstack().sort_index(ascending=False)
    df_beta_wind_SDES = df_beta_wind.round(3).copy()

    df_gamma_wind = gamma_ij_wind.copy()
    df_gamma_wind["ind"] = df_gamma_wind.index
    df_gamma_wind["ind"] = df_gamma_wind.ind.str.split("_",expand=True)[2]
    df_gamma_wind["solar"] = df_gamma_wind["ind"].str.split("s",expand=True)[1]
    df_gamma_wind["wind"] = df_gamma_wind["ind"].str.split("s",expand=True)[0].str.split("w",expand=True)[1]
    df_gamma_wind = df_gamma_wind.set_index(["solar","wind"])["0"]
    df_gamma_wind = df_gamma_wind.unstack().sort_index(ascending=False)
    df_gamma_wind = df_gamma_wind.round(3).copy()

    df_gamma_solar = gamma_ij_solar.copy()
    df_gamma_solar["ind"] = df_gamma_solar.index
    df_gamma_solar["ind"] = df_gamma_solar.ind.str.split("_",expand=True)[2]
    df_gamma_solar["solar"] = df_gamma_solar["ind"].str.split("s",expand=True)[1].str.split("w",expand=True)[0]
    df_gamma_solar["wind"] = df_gamma_solar["ind"].str.split("s",expand=True)[1].str.split("w",expand=True)[1]
    df_gamma_solar = df_gamma_solar.set_index(["solar","wind"])["0"]
    df_gamma_solar = df_gamma_solar.unstack().sort_index(ascending=False)
    df_gamma_solar = df_gamma_solar.round(3).copy()

    return df_beta_solar_LDES, df_beta_solar_SDES, df_beta_wind_LDES, df_beta_wind_SDES, df_gamma_wind, df_gamma_solar
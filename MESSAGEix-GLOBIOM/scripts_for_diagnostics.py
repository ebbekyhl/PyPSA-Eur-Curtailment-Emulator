# -*- coding: utf-8 -*-
"""
Created on 5th of January 2023
Author: Ebbe Kyhl Gøtske

This script contains functions to ...

"""

import pandas as pd
import matplotlib.pyplot as plt

fs = 18
plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['axes.axisbelow'] = True

colors_dict = {"solar":"#f9d002",
                "wind":"#235ebc",
                "VRE": "#5DBB63",
                "gas":'#e0986c',
                "coal":'#545454',
                "bio":'#baa741',
                "csp":'#ffbf2b',
                "oil":'#c9c9c9',
                "solar":"#f9d002",
                "wind":"#235ebc",
                "nuc":'#ff8c00',
                "hydro":'#298c81',
                "geo":"brown",
                "igcc":"grey",
                "syn":"pink",
                "foil":"k",
                "loil":"k",
                "stor":'#ace37f',
                "h2":"magenta",
                "SDES":'#ace37f',
                "LDES":"purple",
                "imp": "#597D35",
                "exp": "#4A2511",
                }


# removing or slacking the four VRE integration constraints in the model

# 1. remove curtailment constraint
def remove_curtailment(scen, df_rel_activity, source=None):
    if source is None:
        df_curtailment = df_rel_activity.loc[df_rel_activity.relation.str.contains("curtailment")]
        tech = [x for x in scen.set("technology") if "curtailment" in x]
        df_curtailment_techs = df_curtailment.loc[df_curtailment.technology.str.contains("curtailment")]
        print("removing all curtailment")
    else:
        df_curtailment = df_rel_activity.loc[df_rel_activity.relation.str.contains(source + "_curtailment")]
        tech = [x for x in scen.set("technology") if source + "_curtailment" in x]
        df_curtailment_techs = df_curtailment.loc[df_curtailment.technology.str.contains("curtailment")]
        print("removing " + source + " curtailment")
        
    scen.remove_par("relation_activity", df_curtailment_techs)

    df_inputs = scen.par('input',{"technology":tech})
    scen.remove_par("input",df_inputs)

    df_inputs.value = 0
    scen.add_par("input", df_inputs)

# 2. remove integration costs
def remove_integration_cost(scen):
    df_var_cost = scen.par("var_cost")
    df_integration_cost = df_var_cost.loc[df_var_cost.technology.str.contains("cv")]
    scen.remove_par("var_cost", df_integration_cost)
    df_integration_cost.value = 0
    scen.add_par("var_cost", df_integration_cost)

# 3. remove firm capacity constraint
def remove_firm_capacity_constraint(scen, df_rel_activity):
    df_res_marg = df_rel_activity.query("relation == 'res_marg'")
    df_res_marg_cv = df_res_marg.loc[df_res_marg.technology.str.contains("cv")]
    scen.remove_par("relation_activity", df_res_marg_cv)
    df_res_marg_cv.value = 1
    scen.add_par("relation_activity", df_res_marg_cv)

# 4. remove flexibility constraint
def remove_flexibility_constraint(scen, df_rel_activity):
    df_oper_res = df_rel_activity.query("relation == 'oper_res'")
    df_oper_res_cv = df_oper_res.loc[df_oper_res.technology.str.contains("cv")]
    scen.remove_par("relation_activity", df_oper_res_cv)
    df_oper_res_cv.value = 0
    scen.add_par("relation_activity", df_oper_res_cv)

# calculate theoretical curtailment
def calculate_theoretical_curtailment(scen, regions, wind_resources, solar_resources, technologies, model_years):
    demand = scen.var("ACT",
                            {"technology":technologies["load"],
                                "node_loc":regions,
                                }).groupby("year_act").sum().lvl

    demand = demand.loc[model_years]

    # theoretical wind curtailment
    theor_wind_curtailment_1 = 0.10*(wind_resources - 0.222*demand) # old input values and penetration bins! 
    theor_wind_curtailment_2 = 0.25*(wind_resources - 0.389*demand) # old input values and penetration bins!
    theor_wind_curtailment_3 = 0.35*(wind_resources - 0.500*demand) # old input values and penetration bins!

    theor_wind_curtailment_1[theor_wind_curtailment_1 < 0] = 0
    theor_wind_curtailment_2[theor_wind_curtailment_2 < 0] = 0
    theor_wind_curtailment_3[theor_wind_curtailment_3 < 0] = 0

    theor_wind_curtailment_sum = theor_wind_curtailment_1 + theor_wind_curtailment_2 + theor_wind_curtailment_3

    # theoretical solar curtailment
    theor_solar_curtailment_1 = 0.15*(solar_resources - 0.144*demand)
    theor_solar_curtailment_2 = 0.25*(solar_resources - 0.200*demand)
    theor_solar_curtailment_3 = 0.35*(solar_resources - 0.278*demand)

    theor_solar_curtailment_1[theor_solar_curtailment_1 < 0] = 0
    theor_solar_curtailment_2[theor_solar_curtailment_2 < 0] = 0
    theor_solar_curtailment_3[theor_solar_curtailment_3 < 0] = 0

    theor_solar_curtailment_sum = theor_solar_curtailment_1 + theor_solar_curtailment_2 + theor_solar_curtailment_3

    return theor_wind_curtailment_sum, theor_solar_curtailment_sum, wind_resources, solar_resources

def calculate_theoretical_curtailment_new(scen, regions, technologies, model_years):

    vre_list = ["vre_curtailment_w" + str(i) + "s" + str(j) for i in range(1,8) for j in range(1,8)]

    df_solar_res = scen.var("ACT",
                            {"node_loc": regions,
                            "technology":technologies["solar"]}).groupby("year_act").sum().lvl

    df_wind_res = scen.var("ACT",
                            {"node_loc": regions,
                            "technology":technologies["wind"]}).groupby("year_act").sum().lvl

    df_vre_curtailment = scen.par("relation_activity",
                                {"node_loc": regions,
                                "year_rel":2050,
                                "relation":vre_list})

    gamma_wind = df_vre_curtailment.loc[df_vre_curtailment.technology.str.contains("wind")][["relation","technology","value"]]
    gamma_wind["solar"] = gamma_wind.relation.str.split("s",expand=True)[1].astype(int)
    gamma_wind["wind"] = gamma_wind.technology.str.split("t",expand=True)[2].astype(int)
    gamma_wind.set_index(["solar","wind"],inplace=True)
    gamma_wind = gamma_wind.value

    gamma_solar = df_vre_curtailment.loc[df_vre_curtailment.technology.str.contains("solar")][["relation","technology","value"]]
    gamma_solar["wind"] = gamma_solar.relation.str.split("w",expand=True)[1].str.split("s",expand=True)[0]
    gamma_solar["solar"] = gamma_solar.relation.str.split("s",expand=True)[1]
    gamma_solar.set_index(["solar","wind"],inplace=True)
    gamma_solar = gamma_solar.value

    df_curtailment = scen.var("ACT",
                            {"node_loc": regions,
                            "year_act":model_years,
                            "technology":technologies["wind curtailed"] + technologies["solar curtailed"]})

    df_curtailment.set_index("technology",inplace=True)

    total_wind_base_curtailment = pd.Series(index=model_years)
    total_solar_base_curtailment = pd.Series(index=model_years)
    for y in model_years:
        df_curtailment_y = df_curtailment.query("year_act == @y")

        wind_th_curtailment = {}
        solar_th_curtailment = {}
        for i,j in gamma_wind.index:
            # print("i: ", i, ",", df_curtailment_y.loc["wind_curtailment" + str(i)].lvl)
            # print("grouped: ", df_curtailment_y.loc["wind_curtailment" + str(i)].lvl.groupby("technology").sum())
            # print("gamma: ", gamma_wind.loc[i,j].unique())
            wind_th_curtailment[i,j] = gamma_wind.loc[i,j].unique()[0]*(df_curtailment_y.loc["wind_curtailment" + str(i)].lvl.groupby("technology").sum())
        
        for i,j in gamma_solar.index:
            solar_th_curtailment[i,j] = gamma_solar.loc[i,j].unique()[0]*(df_curtailment_y.loc["solar_curtailment" + str(i)].lvl.groupby("technology").sum())

        total_wind_base_curtailment.loc[y] = sum(wind_th_curtailment.values())
        total_solar_base_curtailment.loc[y] = sum(solar_th_curtailment.values())

    return total_wind_base_curtailment, total_solar_base_curtailment, df_wind_res, df_solar_res

# calculate contribution from technologies
def calculate_tech_contribution(scen, regions, renewable, model_years, bins):
    tech_contribution = {}
    for i in bins:
        rel = scen.par("relation_activity", {"relation": [renewable + "_curtailment_" + str(i)],
                                            "year_rel": 2050, 
                                            "node_rel": regions[0]}) # we assume the relation parameters are the same for both regions (WEU and EEU)

        rel = rel[~rel.technology.str.contains(renewable)]
        rel = rel[~rel.technology.str.contains("elec_t_d")]
        tec_act = [x for x in set(rel["technology"])]

        act = scen.var("ACT", {"technology": tec_act, 
                            "year_act": model_years,
                            "node_loc": regions}).groupby(["technology","year_act"]).sum().lvl

        cap = scen.var("CAP", {"technology": tec_act, 
                                "year_act": model_years,
                                "node_loc": regions}).groupby(["technology","year_act"]).sum().lvl

        rel = rel.set_index("technology").value

        # Multiply activity of technology with coefficient from curtailment relations
        tech_contribution_i = act.loc[rel.index]*rel
        tech_contribution[i] = tech_contribution_i

        if i == bins[0]:
            tech_contribution_sum = tech_contribution_i
        else:
            tech_is_in = tech_contribution_i[tech_contribution_i.index.isin(tech_contribution_sum.index)].index
            tech_is_not_in = tech_contribution_i[~tech_contribution_i.index.isin(tech_contribution_sum.index)].index
            tech_contribution_sum.loc[tech_is_in] += tech_contribution_i.loc[tech_is_in]
            tech_contribution_sum = tech_contribution_sum.append(tech_contribution_i.loc[tech_is_not_in])  
    
    return tech_contribution_sum, cap, act

def plot_curtailment(tech_contribution_sum, 
                     theor_curtailment_sum,
                     resources,
                     legend=True):
    
    tech_contribution_sum_unstack = tech_contribution_sum.unstack().sum(axis=1).abs()
    tech_contribution_sum = tech_contribution_sum[tech_contribution_sum_unstack[tech_contribution_sum_unstack > 0].index]
    df_plot = pd.DataFrame(index=tech_contribution_sum.index)
    df_plot["tech_contribution"] = tech_contribution_sum
    
    df = pd.DataFrame(theor_curtailment_sum)
    df["technology"] = "base curtailment"
    df["year_act"] = df.index
    df.set_index(["technology","year_act"], inplace=True)
    df.columns = ["tech_contribution"]
    df = df.append(df_plot)

    df_plot = df.unstack().T.loc["tech_contribution"]
    cols = df_plot.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_plot = df_plot[cols]

    negative_contributions = df_plot[df_plot < 0].sum(axis=1)
    positive_contributions = df_plot[df_plot > 0].sum(axis=1)
    net_curtailment = positive_contributions + negative_contributions
    net_curtailment[net_curtailment < 0] = 0

    # plot 
    fig, ax = plt.subplots(figsize=(10,5))
    df_plot.plot(kind="bar", stacked=True, ax=ax)

    ylim_upper = tech_contribution_sum[tech_contribution_sum > 0].unstack().sum().max()
    ylim_lower = tech_contribution_sum[tech_contribution_sum < 0].unstack().sum().min()
    ylim = 1.1*max(abs(ylim_lower),ylim_upper)
    ax.set_ylim(-ylim,ylim)
    # ax.set_ylim(-1100, 1100)
    ax.legend().remove()
    ax.set_xlabel("")

    # plot net curtailment
    ax.axhline(0, color="black", linewidth=0.5,ls="--")
    ax.plot(range(len(net_curtailment.index)), net_curtailment, color="black", linewidth=2, label = "net curtailment")

    # add axis on right side and plot the relative curtailment: 
    # net_curtailment_rel = net_curtailment/resources*100
    # ax2 = ax.twinx()
    # ax2.plot(range(len(net_curtailment.index)), net_curtailment_rel, color="black", linewidth=2, label = "net curtailment")
    # make right axis thick and make ticks bold 
    # ax2.spines['right'].set_linewidth(2)
    # ax2.tick_params(axis='y', which='major', width=2, labelsize=fs-4)
    # ax2.set_ylim(-110,110)
    # ax2.set_ylabel("Net curtailment (%)", fontsize=fs-4)

    if legend:
        fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=2, fontsize=fs)

    return fig, ax 

def calculate_vre_generation(theor_wind_curtailment_sum,
                             theor_solar_curtailment_sum, 
                             tech_contribution_wind_sum, 
                             tech_contribution_solar_sum, 
                             wind_resources,
                             solar_resources,
                             formulation = "old",
                             curtailment = True,
                             ):

    # wind and solar generation
    if formulation == "old":
        # net curtailment
        if curtailment:
            wind_net_curtailment = theor_wind_curtailment_sum + tech_contribution_wind_sum.groupby("year_act").sum()
            wind_net_curtailment[wind_net_curtailment < 0] = 0

            solar_net_curtailment = theor_solar_curtailment_sum + tech_contribution_solar_sum.groupby("year_act").sum()
            solar_net_curtailment[solar_net_curtailment < 0] = 0

        else:
            wind_net_curtailment = 0
            solar_net_curtailment = 0

        wind_generation = wind_resources - wind_net_curtailment
        solar_generation = solar_resources - solar_net_curtailment
        df_wind_generation = pd.DataFrame(wind_generation)
        df_wind_generation["technology"] = "wind"
        df_wind_generation["year_act"] = df_wind_generation.index

        df_wind_generation.set_index(["technology","year_act"], inplace=True)
        df_solar_generation = pd.DataFrame(solar_generation)
        df_solar_generation["technology"] = "solar"
        df_solar_generation["year_act"] = df_solar_generation.index
        df_solar_generation.set_index(["technology","year_act"], inplace=True)

        if curtailment:
            df_wind_generation = df_wind_generation[0]
            df_solar_generation = df_solar_generation[0]
        else:
            df_wind_generation = df_wind_generation.lvl
            df_solar_generation = df_solar_generation.lvl

    else:
        vre_net_curtailment = theor_wind_curtailment_sum + theor_solar_curtailment_sum + tech_contribution_wind_sum.groupby("year_act").sum()
        vre_generation = wind_resources + solar_resources - vre_net_curtailment
        df_vre_generation = pd.DataFrame(vre_generation)
        df_vre_generation["technology"] = "VRE"
        df_vre_generation["year_act"] = df_vre_generation.index
        df_vre_generation.set_index(["technology","year_act"], inplace=True)
        df_vre_generation = df_vre_generation[0]

        # in the new formulation, the net vre curtailment can not be 
        # split in wind and solar curtailment (only, the constraint
        # tells the model how much vre curtailment changes when either 
        # wind or solar capacity is increased). Thus, to calculate
        # the vre generation, we also need to consider the 
        # vre resources on an aggregate manner, Here, we allocate the
        # vre generation to the df_wind_generation. 
        df_wind_generation = df_vre_generation 
        df_solar_generation = 0

    return df_wind_generation, df_solar_generation

def calculate_total_generation(scen, regions, model_years):
    # get generation for all power plants
    commodity = [x for x in scen.set("commodity") if "electr" in x]
    inputs = scen.par("input",{"commodity":commodity})
    outputs = scen.par("output",{"commodity":"electr"})
    ppl = [x for x in set(outputs.technology) if x not in set(inputs.technology)]
    tec_list_series = pd.Series(ppl)

    # drop wind (because they have not been accounted for curtailment)
    tec_list_series = tec_list_series[~tec_list_series.str.contains("wind")]

    # drop solar (because they have not been accounted for curtailment)
    tec_list_series = tec_list_series[~tec_list_series.str.contains("solar")]

    # drop export and import 
    tec_list_series = tec_list_series[~tec_list_series.str.contains("elec_exp")]
    tec_list_series = tec_list_series[~tec_list_series.str.contains("elec_imp")]

    colors_dict_initial = colors_dict.copy()

    for carrier in colors_dict_initial.keys():
        colors_gas = pd.DataFrame(tec_list_series.loc[tec_list_series.str.contains(carrier)])
        colors_gas[carrier] = colors_dict[carrier]
        colors_dict.update(colors_gas.set_index(0).to_dict()[carrier])

    # generation without wind and solar (will be added later)
    df_generation = scen.var("ACT", 
                            {"technology": tec_list_series.tolist(),
                            "node_loc": regions,
                            "year_act": model_years}).groupby(["technology","year_act"]).sum().lvl
    
    # capacity for all power plants
    df_capacity = scen.var("CAP", 
                            {"technology": ppl,
                            "node_loc": regions,
                            "year_act": model_years}).groupby(["technology","year_act"]).sum().lvl
    
    return df_generation, df_capacity, colors_dict

def plot_generation_and_capacity(scen, regions, technologies, model_years, bins, formulation="old", plot=True, curtailment=True):

    solar_resources = scen.var("ACT",
                                {"technology":technologies["solar"],
                                    "node_loc":regions,
                                    }).groupby("year_act").sum().drop(columns=["year_vtg"]).lvl

    wind_resources = scen.var("ACT",
                                {"technology":technologies["wind"],
                                    "node_loc":regions,
                                    }).groupby("year_act").sum().lvl

    solar_resources = solar_resources.loc[model_years]
    wind_resources = wind_resources.loc[model_years]

    if formulation == "new":
        relation1 = relation2 = "vre"
    else:
        relation1 = "wind"
        relation2 = "solar"

    if curtailment:
        if formulation == "old":
            theor_wind_curtailment_sum, theor_solar_curtailment_sum, wind_resources, solar_resources = calculate_theoretical_curtailment(scen, regions, wind_resources, solar_resources, technologies, model_years)
        else:
            theor_wind_curtailment_sum, theor_solar_curtailment_sum, wind_resources, solar_resources = calculate_theoretical_curtailment_new(scen, regions, technologies, model_years)

        tech_contribution_wind_sum, tech_cap_w, tech_act_w = calculate_tech_contribution(scen, regions, relation1, model_years, bins)
        tech_contribution_solar_sum, tech_cap_s, tech_act_s = calculate_tech_contribution(scen, regions, relation2, model_years, bins)
    else:
        theor_wind_curtailment_sum = theor_solar_curtailment_sum = 0
        tech_contribution_wind_sum = tech_contribution_solar_sum = 0

        tec_act = ["stor_ppl","h2_elec"]#,"elec_trp"]
        tech_act_w = scen.var("ACT", {"technology": tec_act, 
                            "year_act": model_years,
                            "node_loc": regions}).groupby(["technology","year_act"]).sum().lvl

        tech_cap_w = scen.var("CAP", {"technology": tec_act, 
                                "year_act": model_years,
                                "node_loc": regions}).groupby(["technology","year_act"]).sum().lvl

    df_generation, df_capacity, colors_dict = calculate_total_generation(scen, regions, model_years)
    
    df_wind_generation, df_solar_generation = calculate_vre_generation(theor_wind_curtailment_sum,
                                                                       theor_solar_curtailment_sum, 
                                                                       tech_contribution_wind_sum, 
                                                                       tech_contribution_solar_sum, 
                                                                       wind_resources,
                                                                       solar_resources,
                                                                       formulation=formulation,
                                                                       curtailment=curtailment)
    
    if formulation == "old":
        df_generation = pd.concat([df_generation,df_wind_generation,df_solar_generation]) # adding wind and solar generation
    else:
        df_generation = pd.concat([df_generation,df_wind_generation])

    df_generation_unstacked = df_generation.unstack().T
    df_capacity = pd.concat([df_capacity,tech_cap_w]) # adding integration support measures 
    df_capacity_unstacked = df_capacity.unstack().T

    # renaming columns
    df_cnames = pd.DataFrame(df_generation_unstacked.columns)
    df_cnames["simple"] = df_cnames.technology.str.split("_",expand=True)[0]
    df_generation_unstacked.columns = df_cnames.simple
    df_generation_unstacked.rename(columns={"igcc":"gas"},inplace=True)
    df_generation_grouped = df_generation_unstacked.groupby(level=0, axis=1).sum()
    df_generation_grouped.drop(columns=["liq","syn"],inplace=True) # dropping liquified fuel production

    # drop columns if they are less than 0.01% of total generation 
    # df_generation_grouped = df_generation_grouped[df_generation_grouped.sum(axis=0)[df_generation_grouped.sum(axis=0) > 0.0001*df_generation_grouped.sum().sum()].index]

    df_cnames = pd.DataFrame(df_capacity_unstacked.columns)
    df_cnames["simple"] = df_cnames.technology.str.split("_",expand=True)[0]
    df_capacity_unstacked.columns = df_cnames.simple
    df_capacity_unstacked.rename(columns={"igcc":"gas"},inplace=True)
    df_capacity_grouped = df_capacity_unstacked.groupby(level=0, axis=1).sum()
    df_capacity_grouped.drop(columns=["liq","syn"],inplace=True) # dropping liquified fuel production

    # drop columns if they are less than 0.01% of total capacity
    df_capacity_grouped = df_capacity_grouped[df_capacity_grouped.sum(axis=0)[df_capacity_grouped.sum(axis=0) > 0.0001*df_capacity_grouped.sum().sum()].index]
    preferred_order = pd.Index(["wind","solar","csp","hydro",
                                "geo","bio","nuc",
                                "gas","coal",
                                "h2","stor","LDES","SDES"])
    new_columns_c = preferred_order.intersection(df_capacity_grouped.columns).append(
                                                df_capacity_grouped.columns.difference(preferred_order)
                                                )
    
    VRE_share = pd.DataFrame()
    # print(df_generation_grouped)
    if formulation == "old":
        VRE_share["wind"] = (df_generation_grouped["wind"])/df_generation_grouped.sum(axis=1)*100
        VRE_share["solar"] = (df_generation_grouped["solar"])/df_generation_grouped.sum(axis=1)*100
    else:
        VRE_share["VRE"] = (df_generation_grouped["VRE"])/df_generation_grouped.sum(axis=1)*100

    if plot:
        fig1, ax1 = plt.subplots(figsize=(6,5))
        df_gen_plot = df_generation_grouped.loc[model_years].copy()
        
        tech_act_unstack = tech_act_w.unstack()
        df_gen_plot["imp"] = tech_act_unstack[tech_act_unstack.index.str.contains("imp")].sum()
        df_gen_plot["exp"] = - tech_act_unstack[tech_act_unstack.index.str.contains("exp")].sum()
        
        plot_renewables = ["wind","solar"] if formulation == "old" else ["VRE"]
        preferred_order = pd.Index(plot_renewables + ["csp","hydro",
                                                    "geo","bio","nuc",
                                                    "gas","coal",
                                                    "h2", "imp","exp"])
        new_columns_g = preferred_order.intersection(df_gen_plot.columns).append(
                                                    df_gen_plot.columns.difference(preferred_order)
                                                    )

        df_gen_plot[new_columns_g].plot.bar(ax=ax1,
                                            stacked=True,
                                            legend=False,
                                            color=[colors_dict[i] for i in new_columns_g])
        ax1.set_xlabel("")
        ax1.set_ylabel("Generation (GWa)")
        ax1.set_ylim(0,df_gen_plot.sum(axis=1).max()*1.1)

        # add legend outside of plot
        fig1.legend(loc="lower center", bbox_to_anchor=(0.55, -0.4), ncol=3, fontsize=fs)
        
        fig2, ax2 = plt.subplots(figsize=(6,5))
        df_capacity_grouped.loc[model_years][new_columns_c].plot.bar(ax=ax2,
                                            stacked=True,
                                            legend=False,
                                            color=[colors_dict[i] for i in new_columns_c])
        ax2.set_xlabel("")
        ax2.set_ylabel("Capacity (GW)")
        # ax2.set_ylim(0,8500)
        ax2.set_ylim(0,df_capacity_grouped.sum(axis=1).max()*1.1)

        # add legend outside of plot
        fig2.legend(loc="lower center", bbox_to_anchor=(0.55, -0.4), ncol=3, fontsize=fs)

        # plot curtailment 
        if curtailment:
            if formulation != "new":
                fig3, ax3 = plot_curtailment(tech_contribution_wind_sum, theor_wind_curtailment_sum, wind_resources, legend=False)
                ax3.set_ylabel("Wind curtailment (GWa)")        
                fig4, ax4 = plot_curtailment(tech_contribution_solar_sum, theor_solar_curtailment_sum, solar_resources, legend=True)
                ax4.set_ylabel("Solar curtailment (GWa)")
            else:
                total_vre_curt = theor_solar_curtailment_sum + theor_wind_curtailment_sum
                ren_resources = solar_resources + wind_resources
                tech_contribution_solar_sum_unstack = tech_contribution_solar_sum.unstack()
                tech_contribution = tech_contribution_solar_sum_unstack[~tech_contribution_solar_sum_unstack.index.str.contains("curtailment")].stack()
                fig3, ax3 = plot_curtailment(tech_contribution, total_vre_curt, ren_resources, legend=True)
                ax3.set_ylabel("Wind curtailment (GWa)")
                fig4 = None
        else:
            fig3, fig4 = None, None

        fig5, ax5 = plt.subplots(figsize=(10,5))
        VRE_share = VRE_share.loc[model_years]
        VRE_share.plot(kind="bar", stacked=True, ax=ax5, color=["#235ebc","#f9d002"])
        ax5.set_xlabel("")
        ax5.set_ylabel("Share of electricity generation (%)")
        ax5.set_ylim(0,100)
        ax5.legend().remove()
        fig5.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=fs)
        # write percentages on top of bars (only at the far top, accounting for the stacked bars)
        for i in range(len(VRE_share.index)):
            ax5.text(i, VRE_share.iloc[i].sum(), str(int(round(VRE_share.iloc[i].sum(),0))) + "%", ha="center", va="bottom", fontsize=fs-4
                    , color="grey")
            
    else:
        fig1, fig2, fig3, fig4, fig5 = None, None, None, None, None

    return fig1, fig2, fig3, fig4, fig5, df_generation_grouped.loc[model_years], df_capacity_grouped.loc[model_years], tech_cap_w, tech_act_w, VRE_share

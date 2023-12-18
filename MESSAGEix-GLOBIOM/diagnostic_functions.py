import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fs = 15
plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['axes.axisbelow'] = True

tech_colors = {'solar':"#f9d002",
               'wind':"#235ebc",}

def plot_gen_share_and_curtailment(scen, df_generation, df_curtailment, curtailment=True):

    act = scen.var("ACT")
    act_solar_techs = act.loc[act.technology.str.contains("solar_res")]
    act_solar_pv = act_solar_techs.loc[act_solar_techs.node_loc.str.contains("EU")][["year_act","lvl","mrg"]].groupby("year_act").sum()
    act_wind_techs = act.loc[act.technology.str.contains("wind_re")]
    act_wind = act_wind_techs.loc[act_wind_techs.node_loc.str.contains("EU")][["year_act","lvl","mrg"]].groupby("year_act").sum()
    
    wind_resources = act_wind.loc[2025:].lvl
    solar_resources = act_solar_pv.loc[2025:].lvl

    if curtailment:
        wind_electricity_generation = wind_resources - df_curtailment["Wind curtailed"]
        solar_electricity_generation = solar_resources - df_curtailment["Solar PV curtailed"]
    else:
        wind_electricity_generation = wind_resources.copy()
        solar_electricity_generation = solar_resources.copy()

    df_wind = wind_electricity_generation.copy()
    df_solar = solar_electricity_generation.copy()

    backup_generation = df_generation.loc[df_generation.variable.str.contains("gen")][["variable","year","value"]].groupby(["year"]).sum().value
    total_generation = backup_generation + df_wind + df_solar

    #df_generation_gbyv = df_generation.groupby(["year","variable"]).sum()
    #df_tot_generation = df_generation_gbyv[df_generation_gbyv > 0].dropna() 
    #df_tot_generation_gby = df_tot_generation.groupby("year").sum()

    df_solar_share = df_solar/total_generation
    df_wind_share = df_wind/total_generation
    df_vre_share = pd.DataFrame()
    df_vre_share["solar"] = df_solar_share
    df_vre_share["wind"] = df_wind_share

    df_vre_generation_abs = pd.DataFrame()
    df_vre_generation_abs["wind"] = df_wind #.value
    df_vre_generation_abs["solar"] = df_solar #.value

    ############################################################### Fig. 1
    fig1,ax1 = plt.subplots()
    df_vre_generation_abs.plot.bar(ax=ax1,
                                   stacked=True,
                                   color=[tech_colors[i] for i in df_vre_generation_abs.columns])
    ax1.set_ylabel("VRE generation (GWa)")

    ############################################################### Fig. 2
    fig2,ax2 = plt.subplots()
    df_vre_share_pct = df_vre_share[["wind","solar"]]*100
    df_vre_share_pct.plot.bar(ax=ax2,
                              stacked=True,
                              color=[tech_colors[i] for i in df_vre_share_pct.columns])

    ax2.set_ylabel("VRE share \n (% of electricity supply)")
    ax2.set_ylim(0,100)
    #ax2.set_xlim(5.5,6.5)

    if curtailment:
        ############################################################### Fig. 3
        wind_curtailment_bins = df_curtailment[["wind_curtailment_1_input",
                                                "wind_curtailment_2_input",
                                                "wind_curtailment_3_input"]]

        solar_curtailment_bins = df_curtailment[["solar_curtailment_1_input",
                                                "solar_curtailment_2_input",
                                                "solar_curtailment_3_input"]]
        
        df_wind_theoretical = wind_resources.copy() 
        df_solar_theoretical = solar_resources.copy()

        fig3,ax3 = plt.subplots()
        wind_curtailment_bins_rel = wind_curtailment_bins.T.div(df_wind_theoretical).T*100
        wind_curtailment_bins_rel.plot.bar(ax=ax3,
                                        stacked=True,
                                        legend=False)
        ax3.set_ylim(0,100)
        ax3.set_ylabel("wind curtailment \n (% of wind resources)")
        fig3.legend(ncol=1,bbox_to_anchor=[0.75, -0.05],prop={"size":fs})
        # fig3.savefig("figures/wind_curtailment_baseline.png",bbox_inches="tight",dpi=300)

        ############################################################### Fig. 4
        fig4,ax4 = plt.subplots()
        solar_curtailment_bins_rel = solar_curtailment_bins.T.div(df_solar_theoretical).T*100
        solar_curtailment_bins_rel.plot.bar(ax=ax4,
                                            stacked=True,
                                            legend=False)

        ax4.set_ylim(0,100)
        ax4.set_ylabel("solar curtailment \n (% of solar resources)")
        fig4.legend(ncol=1,bbox_to_anchor=[0.75, -0.05],prop={"size":fs})
        # fig4.savefig("figures/solar_curtailment_baseline.png",bbox_inches="tight",dpi=300)
    else:
        fig3 = None
        ax3 = None
        fig4 = None
        ax4 = None
    
    act_renewables = pd.DataFrame(index=df_wind.index)
    act_renewables["Wind"] = wind_resources
    act_renewables["Solar PV"] = solar_resources

    curt_renewables = {}
    curt_renewables["Wind"] = wind_curtailment_bins_rel
    curt_renewables["Solar PV"] = solar_curtailment_bins_rel

    #fig1, ax1, fig2, ax2, fig3, ax3, fig4, ax4, df_vre_share_pct, act_renewables
    return fig2, ax2, df_vre_share_pct, act_renewables, curt_renewables

def make_summary(scen):
    commodity = [x for x in scen.set("commodity") if "electr" in x]
    inputs = scen.par("input",{"commodity":commodity})
    outputs = scen.par("output",{"commodity":"electr"})
    ppl = [x for x in set(outputs.technology) if x not in set(inputs.technology)]
    ppl_including_storage = ppl + ["stor_ppl"]
    
    cap = scen.var("CAP")
    cap_filtered = pd.concat([cap.loc[cap.technology.isin(ppl)],
                              cap.loc[cap.technology.str.contains("stor")]]) 
    act = scen.var("ACT")
    act_filtered = act.loc[act.technology.isin(ppl)]
    relation_upper = scen.par("relation_upper")
    demand = scen.par("demand")
    return cap_filtered, cap, act_filtered, act, demand,  ppl_including_storage, relation_upper

def calculate_VRE_penetration(scen_act,scen_cap,regions):
    lst = [scen_act.node_loc[i] in regions for i in scen_act.index]
    activity_region = scen_act.loc[lst]
    
    lst = [scen_cap.node_loc[i] in regions for i in scen_cap.index]
    capacity_region = scen_cap.loc[lst]
    
    wind_activity_aggregate = activity_region.loc[activity_region.technology.str.contains("wind")].groupby("year_act").sum().lvl
    solar_activity_aggregate = activity_region.loc[activity_region.technology.str.contains("solar")].groupby("year_act").sum().lvl
    act_total_aggregate = activity_region.groupby("year_act").sum().lvl.loc[wind_activity_aggregate.index]
    
    wind_capacity_aggregate = capacity_region.loc[capacity_region.technology.str.contains("wind")].groupby("year_act").sum().lvl
    solar_capacity_aggregate = capacity_region.loc[capacity_region.technology.str.contains("solar")].groupby("year_act").sum().lvl
    cap_total_aggregate = capacity_region.groupby("year_act").sum().lvl.loc[wind_activity_aggregate.index]
    
    VRE_rel_penetration = pd.DataFrame()
    VRE_rel_penetration["wind"] = wind_activity_aggregate/act_total_aggregate*100
    VRE_rel_penetration["solar"] = solar_activity_aggregate/act_total_aggregate*100
    VRE_rel_penetration = VRE_rel_penetration.dropna()
    
    VRE_cap = pd.DataFrame()
    VRE_cap["wind"] = wind_capacity_aggregate
    VRE_cap["solar"] = solar_capacity_aggregate
    VRE_cap = VRE_cap.loc[2020:]
    
    VRE_act = pd.DataFrame()
    VRE_act["wind"] = wind_activity_aggregate
    VRE_act["solar"] = solar_activity_aggregate
    VRE_act = VRE_act.loc[2020:]
    
    return VRE_rel_penetration, VRE_cap, VRE_act

def add_solar_potential_line(ax):
    solar_max_PyPSAEur = 4581
    ax.set_ylim(0,11000)
    ax.axhline(solar_max,color="k",ls="--")
    ax.text(12,solar_max+0.01*ax.get_ylim()[1],"Potential in MESSAGEix-GLOBIOM")
    ax.axhline(solar_max_PyPSAEur,color="grey",ls="--")
    ax.text(12,solar_max_PyPSAEur-0.035*ax.get_ylim()[1],"Potential in PyPSA-Eur (corine 1-20,26,31,32)",color='grey')

def add_wind_potential_line(ax):
    ax.set_ylim(0,11000)
    wind_max_PyPSAEur = 9784
    ax.axhline(wind_max,color="k",ls="--")
    ax.text(12,wind_max+0.01*ax.get_ylim()[1],"Potential in MESSAGEix-GLOBIOM")
    ax.axhline(wind_max_PyPSAEur,color="grey",ls="--")
    ax.text(12,wind_max_PyPSAEur+0.01*ax.get_ylim()[1],"Potential in PyPSA-Eur (corine 12-29,31,32,44,255)",color="grey")

def add_capacity_factor_annotation_solar(ax, solar_act_df_years, solar_cap_df_years):
    cap_factor_solar = solar_act_df_years.sum(axis=0)/solar_cap_df_years.sum(axis=0)
    for p in range(len(years)):
        ax.annotate(str(round(cap_factor_solar.iloc[p]*100,1)) + "%",(p-0.4,0.02*ax.get_ylim()[1]+solar_act_df_years.sum(axis=0).iloc[p]))
    ax.set_ylim(0,1250)

def add_capacity_factor_annotation_wind(ax,wind_act_df_years,wind_cap_df_years):
    cap_factor_wind = wind_act_df_years.sum(axis=0)/wind_cap_df_years.sum(axis=0)
    for p in range(len(years)):
        if p % 2 != 0:
            ydelta = 0.1
        else:
            ydelta = 0
        ax.annotate(str(round(cap_factor_wind.iloc[p]*100,1)) + "%",(p-0.4,(0.02+ydelta)*ax.get_ylim()[1]+wind_act_df_years.sum(axis=0).iloc[p]))
    ax.set_ylim(0,1250);
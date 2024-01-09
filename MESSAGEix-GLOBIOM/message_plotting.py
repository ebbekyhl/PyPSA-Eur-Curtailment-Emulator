import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import ixmp as ix
import message_ix
import pyam
from itertools import product

preferred_order = pd.Index([
                            "wind_res_hist_2005",
                            "wind_res_hist_2010",
                            "wind_res_hist_2015",
                            "wind_res_hist_2020",
                            "wind_ppl",
                            "wind_ppf",
                            "wind_res1",
                            "wind_res2",
                            "wind_res3",
                            "wind_res4",
                            "wind_ref1",
                            "wind_ref2",
                            "wind_ref3",
                            "wind_ref4",
                            "solar_res_hist_2010",
                            "solar_res_hist_2015",
                            "solar_res_hist_2020",
                            "solar_pv_ppl",
                            "solar_res1",
                            "solar_res2",
                            "solar_res3",
                            "solar_res4",
                            "solar_res5",
                            "solar_res6",
                            "solar_res7",
                            "solar_res8",
                            "stor_ppl",
                            'solar_curtailment1',
                           'solar_curtailment2',
                           'solar_curtailment3',
                           'wind_curtailment1',
                           'wind_curtailment2',
                           'wind_curtailment3',
                           ])

def retrieve_capacity_and_activity(cap,act,relation_upper,demand, regions, years):
    ###############################################################################################################
    ############################################ CAPACITY #########################################################
    ###############################################################################################################
    capacity = cap
    
    # regional capacity
    lst = [capacity.node_loc[i] in regions for i in capacity.index]
    capacity_region = capacity.loc[lst]

    wind_cap = capacity_region.loc[capacity_region.technology[capacity_region.technology.str.contains("wind")].index].groupby(["technology","year_vtg","year_act"]).sum()
    solar_cap = capacity_region.loc[capacity_region.technology[capacity_region.technology.str.contains("solar")].index].groupby(["technology","year_vtg","year_act"]).sum()
    
    lst = [relation_upper.node_rel[i] in regions for i in relation_upper.index]
    relation_upper_region = relation_upper.loc[lst]
    
    #relation_upper = #pd.concat([relation_upper.query("node_rel == 'R11_WEU'"),relation_upper.query("node_rel == 'R11_EEU'")])

    wind_potentials = relation_upper_region.loc[relation_upper_region.relation[relation_upper_region.relation.str.contains("wind_po")].index]
    solar_potentials = relation_upper_region.loc[relation_upper_region.relation[relation_upper_region.relation.str.contains("solar_po")].index]

    solar_max = solar_potentials.query("year_rel == 2050").value.sum()
    wind_max = wind_potentials.query("year_rel == 2050").value.sum()

    solar_cap_categories = solar_cap.index.get_level_values(0).unique()
    wind_cap_categories = wind_cap.index.get_level_values(0).unique()
    
    ###############################################################################################################
    ############################################ ACTIVITY #########################################################
    ###############################################################################################################
    # Global activity
    activity = act 

    # Regional activity
    lst = [activity.node_loc[i] in regions for i in activity.index]
    activity_region = activity.loc[lst]

    # Wind activity
    wind_act = activity_region.loc[activity_region.technology[activity_region.technology.str.contains("wind")].index].groupby(["technology","year_vtg","year_act"]).sum()

    # Solar activity
    solar_act = activity_region.loc[activity_region.technology[activity_region.technology.str.contains("solar")].index].groupby(["technology","year_vtg","year_act"]).sum()

    wind_act_categories = wind_act.index.get_level_values(0).unique()
    solar_act_categories = solar_act.index.get_level_values(0).unique()

    # ----------------------------------------------------------------------------------------------------
    output_dict = {}
    output_dict = {"years":years,
                   "wind_cap":wind_cap,
                   "solar_cap":solar_cap,
                   "wind_max":wind_max,
                   "solar_max":solar_max,
                   "wind_cap_categories":wind_cap_categories,
                   "solar_cap_categories":solar_cap_categories,
                   "wind_act":wind_act,
                   "solar_act":solar_act,
                   "wind_act_categories":wind_act_categories,
                   "solar_act_categories":solar_act_categories,
                  }
    return output_dict

#def calculate_curtailment():

#    wind_curt_act = activity_region.loc[activity_region.technology[activity_region.technology.str.contains("wind_curt")].index].groupby(["technology","year_vtg","year_act"]).sum()

#    solar_curt_act = activity_region.loc[activity_region.technology[activity_region.technology.str.contains("solar_curt")].index].groupby(["technology","year_vtg","year_act"]).sum()

#    return wind_curt, solar_curt

def plot_settings():
    colors = [(0, 0, 0), (1, 0, 0)] # first color is black, last is red
    N_red = 8
    cm_red = LinearSegmentedColormap.from_list(
            "Custom", colors, N=N_red)
    
    colors = [(0, 0, 0), (1, 0, 0)] # first color is black, last is red
    N_red_2 = 4
    cm_red_2 = LinearSegmentedColormap.from_list(
                "Custom", colors, N=N_red_2)

    colors = [(0, 0, 0), (0, 0, 1)] # first color is black, last is red
    N_blue = 4
    cm_blue = LinearSegmentedColormap.from_list(
            "Custom", colors, N=N_blue)

    colors = [(0, 0, 0), (0, 1, 0)] # first color is black, last is red
    N_green = 6
    cm_green = LinearSegmentedColormap.from_list(
            "Custom", colors, N=N_green)

    red_colors = [cm_red(a) for a in np.arange(N_red)]
    red_colors_2 = [cm_red_2(a) for a in np.arange(N_red_2)]
    blue_colors = [cm_blue(a) for a in np.arange(N_blue)]
    green_colors = [cm_green(a) for a in np.arange(N_green)]

    tech_colors = {'solar_cv1':blue_colors[1],
                   'solar_cv2':blue_colors[2], 
                   'solar_cv3':blue_colors[3],
                   'solar_cv4':'lightblue',
                   'wind_cv1':blue_colors[1],
                   'wind_cv2':blue_colors[2], 
                   'wind_cv3':blue_colors[3], 
                   'wind_cv4':'lightblue', 
                   'solar_curtailment1':red_colors_2[1],
                   'solar_curtailment2':red_colors_2[2], 
                   'solar_curtailment3':red_colors_2[3],
                   'wind_curtailment1':red_colors_2[1],
                   'wind_curtailment2':red_colors_2[2], 
                   'wind_curtailment3':red_colors_2[3], 
                   'solar_pv_ppl':"orange",
                   'wind_ppl':"cyan",
                   'wind_ppf':"darkcyan",
                   'solar_res1':red_colors[0],
                   'solar_res2':red_colors[1],
                   'solar_res3':red_colors[2],
                   'solar_res4':red_colors[3],
                   'solar_res5':red_colors[4],
                   'solar_res6':red_colors[5],
                   'solar_res7':red_colors[6],
                   'solar_res8':red_colors[7],
                   'wind_res1':red_colors[1],
                   'wind_res2':red_colors[2],
                   'wind_res3':red_colors[3],
                   'wind_res4':red_colors[4],
                   'wind_ref1':green_colors[1],
                   'wind_ref2':green_colors[2],
                   'wind_ref3':green_colors[3],
                   'wind_ref4':green_colors[4],
                   'wind_ref5':green_colors[5],
                   'solar_res_hist_2010':"grey", 
                   'solar_res_hist_2015':"grey",
                   'solar_res_hist_2020':"grey",
                   'wind_res_hist_2005':"grey", 
                   'wind_res_hist_2010':"grey", 
                   'wind_res_hist_2015':"grey",
                   'wind_res_hist_2020':"grey",
                   'i_feed':"pink",
                   'i_spec':"grey", 
                   'i_therm':"red", 
                   'non-comm':"k", 
                   'rc_spec':"darkgreen", 
                   'rc_therm':"orange",
                   'transport':"blue",
                   "shipping":"purple",
                  }

    preferred_order = pd.Index(["solar_res_hist_2010",
                                "solar_res_hist_2015",
                                "solar_res_hist_2020",
                                "solar_pv_ppl",
                                "solar_res1",
                                "solar_res2",
                                "solar_res3",
                                "solar_res4",
                                "solar_res5",
                                "solar_res6",
                                "solar_res7",
                                "solar_res8",
                                "wind_res_hist_2005",
                                "wind_res_hist_2010",
                                "wind_res_hist_2015",
                                "wind_res_hist_2020",
                                "wind_ppl",
                                "wind_ppf",
                                "wind_res1",
                                "wind_res2",
                                "wind_res3",
                                "wind_res4",
                                "wind_ref1",
                                "wind_ref2",
                                "wind_ref3",
                                "wind_ref4",
                                'solar_curtailment1',
                               'solar_curtailment2',
                               'solar_curtailment3',
                               'wind_curtailment1',
                               'wind_curtailment2',
                               'wind_curtailment3',
                               ])
    
    ppl_colors = {'bio_ppl':'#baa741',
                 'coal_ppl':'#545454',
                 'coal_ppl_u':"k",
                 'csp_sm1_ppl':'#ffbf2b',
                 'csp_sm3_ppl':'#ffbf2b',
                 'foil_ppl':'#c9c9c9',
                 'loil_ppl':'#c9c9c9',
                 'gas_ppl':'#e0986c',
                 'geo_ppl':'#ba91b1',
                 'solar_pv_ppl':"#f9d002",
                 'stor_ppl':'#ace37f',
                 'wind_ppl':"#235ebc",
                 'oil_ppl':'#c9c9c9',
                 'nuc_lc':'#ff8c00',
                 'nuc_hc':'#ff8c00',
                 'hydro_lc':'#298c81',
                 'hydro_hc':'#298c81',
                 }
    
    return tech_colors, ppl_colors, preferred_order

# def plot_VRE(categories, plot, years,variable_name = ""):
#     tech_colors, ppl_colors, preferred_order = plot_settings()
    
#     #plot.drop(plot.loc[plot.technology == "solar_th_ppl"].index,inplace=True)
#     #plot.drop(plot.loc[plot.index.get_level_values(0) == "solar_th_ppl"].index,inplace=True)
    
#     df_years = pd.DataFrame(index=categories)
#     for y in years:
#         plot_dict = {}
#         for cat in categories:
#             if "curtailment" in variable_name:
#                 plot_dict[cat] = plot.loc[cat].query("year_act == @y").sum().mrg
#             else:
#                 plot_dict[cat] = plot.loc[cat].query("year_act == @y").sum().lvl
#         plot_df = pd.DataFrame.from_dict(plot_dict,orient="index")
#         df_years[y] = plot_df

#     df_years = df_years.loc[~(df_years==0).all(axis=1)]
#     df_years = df_years[df_years.columns.sort_values()]

#     fig,ax = plt.subplots()
#     new_index = preferred_order.intersection(df_years.index).append(
#             df_years.index.difference(preferred_order)
#         )
#     df_years.loc[new_index].T.plot.bar(ax=ax,
#                                        color=[tech_colors[i] for i in new_index],
#                                        stacked=True,
#                                        legend=False)
#     ax.set_ylabel(variable_name)
#     ax.set_xlim(11.5,25.5)
#     fig.legend(ncol=2,bbox_to_anchor=[0.75, 0])
#     return fig, ax, df_years

def plot_demand(demand, regions, years, unit=""):
    tech_colors, ppl_colors, preferred_order = plot_settings()
    
    
    lst = [demand.node[i] in regions for i in demand.index]
    demand_region = demand.loc[lst]
    
    plot = demand_region.set_index("commodity")
    categories = list(set(plot.index))
    
    df_years = pd.DataFrame(index=categories)
    for y in years:
        plot_dict = {}
        for cat in categories:
            plot_dict[cat] = plot.loc[cat].query("year == @y").sum().value
        plot_df = pd.DataFrame.from_dict(plot_dict,orient="index")
        df_years[y] = plot_df

    df_years = df_years.loc[~(df_years==0).all(axis=1)]
    df_years = df_years[df_years.columns.sort_values()]

    fig,ax = plt.subplots()
    new_index = preferred_order.intersection(df_years.index).append(
            df_years.index.difference(preferred_order)
        )
    df_years.loc[new_index].T.plot.bar(ax=ax,
                                       color=[tech_colors[i] for i in new_index],
                                       stacked=True,
                                       legend=False)
    ax.set_ylabel("Demand " + unit)
    ax.set_xlim(11.5,25.5)
    fig.legend(ncol=2,bbox_to_anchor=[0.75, 0])
    return fig, ax, df_years

def plot_ppl_capacity(cap,regions,years,power_plants):
    colors_dict = {}
    power_plants_series = pd.Series(power_plants)
    carrier_colors = {"gas":'#e0986c',
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
                      "stor":'#ace37f',
                     }
    for carrier in carrier_colors.keys():
        colors_gas = pd.DataFrame(power_plants_series.loc[power_plants_series.str.contains(carrier)])
        colors_gas[carrier] = carrier_colors[carrier]
        colors_dict.update(colors_gas.set_index(0).to_dict()[carrier])

    region_list = []
    for region in regions:
        region_list.append(cap.query("node_loc == @region"))
        
    cap_regions = pd.concat(region_list)

    power_plants_cap_df = pd.DataFrame(index=years,
                                       columns=power_plants)
    for ppl in power_plants:
        ppl_series = cap_regions.query("technology == @ppl").groupby(["year_act"]).sum().lvl
        power_plants_cap_df[ppl].loc[ppl_series.index] = ppl_series

        
    tolerance = 1e-4*power_plants_cap_df.sum().sum()
    columns = power_plants_cap_df.sum()[power_plants_cap_df.sum() > tolerance].index
    power_plants_cap_df = power_plants_cap_df[columns]
    
    fig,ax = plt.subplots()
    
    new_columns = preferred_order.intersection(power_plants_cap_df.columns).append(
        power_plants_cap_df.columns.difference(preferred_order)
    )
    
    power_plants_cap_df[new_columns].plot.bar(ax=ax,
                                             stacked=True,
                                             legend=False,
                                             color=[colors_dict[i] for i in new_columns])
    ax.set_xlim(11.5,25.5)
    fig.legend(ncol=2,bbox_to_anchor=[0.75, 0])
    
    return fig, ax, power_plants_cap_df



########## Script from Behnam ###################

# Functions and utilites
def inp_or_out(
    sc,
    node,
    tec_list,
    commodity,
    year,
    direction="output",
    grouping=["node_loc", "technology", "year_vtg", "year_act"],
):
    """
    Groups and sums the data of parameters "input" or "output" based on a
    grouping index.

    Parameters
    ----------
    sc : message_ix.Scenario
    node : list
        Node names for the analysis.
    tec_list : list
        Technologies to be grouped.
    commodity : str
        Commodity to be included in analysis.
    year : int
        Year of the analysis.
    direction : str, optional
        "input" or "output" parameter. The default is "output".
    grouping : list, optional
        The index names for groupby method.
        The default is ["node_loc", "technology", "year_vtg", "year_act"].

    Returns
    -------
    DataFrame
        Sorted and grouped data.

    """
    # Input or output
    df = sc.par(
        direction,
        {
            "technology": tec_list,
            "node_loc": node,
            "year_act": year,
            "commodity": commodity,
        },
    )
    return df.groupby(grouping).sum().sort_index()


def activity(
    sc,
    node,
    tec_list,
    year,
    grouping=["node_loc", "technology", "year_vtg", "year_act"],
):
    # Calculating activity from the results
    act = sc.var("ACT", {"technology": tec_list, "node_loc": node, "year_act": year})
    act = (
        act.groupby(grouping)
        .sum()
        .rename({"lvl": "value"}, axis=1)
        .drop(["mrg"], axis=1)
        .sort_index()
    )
    return act


def capacity(
        sc, node, tec_list, year, grouping=["node_loc", "technology", "year_act"]
        ):
    # Calculating capacity from the results
    cap = sc.var("CAP", {"technology": tec_list, "node_loc": node, "year_act": year})
    cap = (
        cap.groupby(grouping)
        .sum()
        .rename({"lvl": "value"}, axis=1)
        .drop(["mrg"], axis=1)
        .sort_index()
    )
    return cap


def inp_act(sc, node, tec_list, commodity, year):
    # Activity
    act = activity(sc, node, tec_list, year)
    # Input
    inp = inp_or_out(sc, node, tec_list, commodity, year, direction="input")
    # Filling entries with no "input" with act
    return (act * inp).fillna(act)


def out_act(sc, node, tec_list, commodity, year):
    # Activity
    act = activity(sc, node, tec_list, year)
    # Output
    out = inp_or_out(sc, node, tec_list, commodity, year)
    # Filling entries with no "output" with act
    return (act * out).fillna(act)


# Finding the contribution of each technology to a relation
def contribute_relation(
    sc,
    node="R11_CPA",
    year=2050,
    relation="oper_res",
    commodity="electr",
    assertion=False, #True,
):
    # Capacity relations
    df = sc.par(
        "relation_total_capacity",
        {"relation": relation, "year_rel": year, "node_rel": node},
    )

    if not df.empty:
        df = df.rename(
            {"year_rel": "year_act", "node_rel": "node_loc"}, axis=1
        ).set_index(["node_loc", "technology", "year_act"])
        tec_cap = list(set([x[1] for x in df.index]))

        # Calculating capacity from the results
        cap = capacity(sc, node, tec_cap, year)

        # Calculating the contribution from capacity to the relation
        rel_cap = (
            (cap["value"] * df["value"])
            .reset_index()
            .groupby(["year_act", "technology"])
            .sum()
        )
        rel_cap_tot = rel_cap.reset_index().groupby("year_act").sum()
    else:
        rel_cap = pd.DataFrame()

    # Activity relations
    rel = sc.par(
        "relation_activity", {"relation": relation, "year_rel": year, "node_rel": node}
    )

    if relation == "res_marg":
        tec_exclude = []
    else:
        tec_exclude = []
    tec_act = [x for x in set(rel["technology"]) if x not in tec_exclude]

    # Calculating activity considering input/output commodities
    if commodity:
        # Input
        df = sc.par(
            "input",
            {
                "technology": tec_act,
                "node_loc": node,
                "year_act": year,
                "commodity": commodity,
            },
        )
        df = df.loc[df["value"] > 0].copy()
        tec_in = [x for x in tec_act if x in set(df["technology"])]

        # ACT * "input"
        act = inp_act(sc, node, tec_in, commodity, year)

        # Output
        df = sc.par(
            "output",
            {
                "technology": tec_act,
                "node_loc": node,
                "year_act": year,
                "commodity": commodity,
            },
        )
        df = df.loc[df["value"] > 0].copy()
        tec_out = [x for x in tec_act if x in set(df["technology"])]

        # Exclude "elec_t_d" from both input and output (later should be userdefined)
        tec_out = [x for x in tec_out if x not in tec_in]

        act = act.append(out_act(sc, node, tec_out, commodity, year))

        # Remaining technologies with no "input" and "output"
        tec_rem = [x for x in tec_act if x not in tec_in + tec_out]
        act = act.append(activity(sc, node, tec_rem, year))
    else:
        act = activity(sc, node, tec_act, year)

    # Calculating the contribution from activity to the relation
    rel = (
        rel.loc[~rel["technology"].isin(tec_exclude)]
        .set_index(["node_loc", "technology", "year_act"])
        .rename({"lvl": "value"}, axis=1)
    )
    rel_act = (
        (act["value"] * rel["value"])
        .reset_index()
        .groupby(["year_act", "technology"])
        .sum()
        .drop(["year_vtg"], axis=1)
    )
    rel_act_tot = rel_act.reset_index().groupby("year_act").sum()

    # Assert that the total contribution is equal to the needs
    if rel_cap.empty:
        rel_cap_tot = rel_act_tot.copy()
        rel_cap_tot["value"] = 0

    if assertion:
        for i in rel_act_tot.index:
            assert (rel_cap_tot + rel_act_tot).loc[
                i, "value"
            ].round() >= 0.001 * float(rel_act_tot.loc[i, "value"])

    # Calculate the share of each technology to reserve margin
    result = rel_cap.append(rel_act).reset_index().set_index(["year_act"])

    result["share"] = result["value"] / [
        result.loc[result["value"] > 0, "value"][i].sum() for i in result.index
    ]
    return result

def read_and_plot_curtailment_B(scenarios,node,plot_curtailment = True):
    
    idx = ["model", "scenario", "region", "variable", "unit"]

    rels = {
        "Operating reserves (GWa)": ["oper_res", None],
        "Reserve margin (GW)": ["res_marg", None],
    }
    # %% Main calculation part
    results = {}
    figures = {}

    # Loading scenarios and calculating
    for num, scen in enumerate(scenarios.keys()):
        sc = scenarios[scen] 

        model_years = [x for x in sc.set("year") if x >= sc.firstmodelyear and x < 2110]
        for r in rels.keys():
            results[(scen, r)] = contribute_relation(
                sc,
                node=node,
                year=model_years,
                relation=rels[r][0],
                commodity=rels[r][1],
            )

        # Curtailment relations
        curt_list = [x for x in sc.set("relation") if "curtailment" in x]
        for curt in curt_list:
            results[(scen, curt)] = contribute_relation(
                sc,
                node=node,
                year=model_years,
                relation=curt,
                commodity=None,
                assertion=False,
            )

            # print(results[(scen, curt)])

        # Calculating share of PV from total generation
        # List of technologies generating at the secondary level
        tec_list = sc.par("output", {"level": "secondary", "commodity": "electr"})[
            "technology"
        ].unique()
        # List of technologies consuming at the secondary level
        tec_in = sc.par("input", {"level": "secondary", "commodity": "electr"})[
            "technology"
        ].unique()
        # List of flexible technologies
        flex = sc.par("relation_activity", {"relation": "oper_res"})
        tec_flex = flex.loc[flex["value"] > 0, "technology"].unique()
        flex_gen = [x for x in tec_list if x in tec_flex]
        flex_se = [x for x in tec_in if x in tec_flex]
        flex_load = [x for x in tec_flex if x not in flex_gen + flex_se]

        tec_inflex = flex.loc[flex["value"] <= 0, "technology"].unique()
        info = {
            "Solar PV": (["solar_res"], ["solar_curtail"]),
            "Wind": (["wind_res", "wind_ref"], ["wind_curtail"]),
            "Generation": (tec_list, []),
            "Secondary": (tec_in, ["elec_t_d", "stor_ppl"]),
            "Flex gen.": (flex_gen, []),
            "Flex SE": (flex_se, []),
            "Flex load": (flex_load, []),
            "Grid": (["elec_t_d"], []),
            "Storage loss": (["stor_ppl"], []),
            "Load": (["elec_t_d"], []),
        }

        df = pd.DataFrame(index=model_years)
        for tec, data in info.items():

            # Share of main technology
            main = sorted(
                [x for x in set(sc.set("technology")) if any([y in x for y in data[0]])]
            )

            if "Flex" in tec:
                d = contribute_relation(
                    sc,
                    node,
                    model_years,
                    relation="oper_res",
                    commodity="electr",
                    assertion=False,
                )
                d1 = d.loc[d["technology"].isin(data[0])].copy()
                d1["year_vtg"] = 2020  # dummy
            elif tec in ["Generation", "Load"]:
                d1 = out_act(sc, node, main, "electr", model_years).reset_index()
            else:
                d1 = inp_act(sc, node, main, "electr", model_years).reset_index()
            d1["technology"] = tec

            # Share of deduction (curtailment)
            if data[1]:
                deduct = sorted(
                    [x for x in set(sc.set("technology")) if any([y in x for y in data[1]])]
                )
                if "Flex" in tec:
                    d2 = d.loc[d["technology"].isin(deduct)].copy()
                    d2["year_vtg"] = 2020  # dummy
                else:
                    d2 = inp_act(
                        sc, node, deduct, "electr", model_years
                    ).reset_index()
                    d2["technology"] = tec
                    d2["value"] *= -1
            else:
                d2 = pd.DataFrame()
            
            # Deducting curtailment from generation
            df[tec] = (
                pd.concat([d1, d2])
                .groupby(["year_act"])
                .sum()
                .drop(["year_vtg"], axis=1)["value"]
            )

            if tec == "Wind":
                d1_wind = d1
                d2_wind = d2

        # Check balances
        df["Generation"] >= df["Secondary"] - df["Grid"]

        # Calculating the share per year
        df["Grid loss"] = df["Grid"] - df["Load"]
        df["Secondary"] -= df["Flex SE"]
        df["Load"] -= df["Flex load"]
        df["Inflex gen."] = df["Generation"] - df["Solar PV"] - df["Wind"] - df["Flex gen."]

        # Giving different signs to generation (+) vs. consumption (-)
        ord_neg = ["Load", "Flex load", "Secondary", "Flex SE", "Storage loss", "Grid loss"]
        df.loc[:, df.columns.isin(ord_neg)] *= -1

        # Calculating shares
        share = df / df[["Generation"]].values

        # Ordering and saving
        ord_pos = ["Solar PV", "Wind", "Flex gen.", "Inflex gen."]
        df = df[ord_pos + ord_neg]

        # df.columns = pd.MultiIndex.from_product([['value'], df.columns])
        # df = df.stack()
        # df.index.names = ["year", "variable"]
        df = df.stack().reset_index()
        df.columns = ["year", "variable", "value"]
        df["model"] = sc.model
        df["scenario"] = sc.scenario
        df["unit"] = "GWa"
        df["region"] = node

        figures[(scen, "Generation (GWa)")] = pd.pivot_table(
            df, index=idx, columns="year", values="value"
        ).reset_index()
    
    df_generation = df
    
    if plot_curtailment:
        # Visualization and plotting configuration
        # Input values for curtailment electricity (please check for each region)
        inp_curt = {
            "wind_curtailment_1": 0.10,
            "wind_curtailment_2": 0.25,
            "wind_curtailment_3": 0.35,
            "solar_curtailment_1": 0.15,
            "solar_curtailment_2": 0.25,
            "solar_curtailment_3": 0.35,
        }

        rename = {
            "Coal": ["coal_adv", "coal_ppl", "coal_ppl_u", "igcc"],
            "Coal w CCS": ["coal_adv_ccs", "igcc_ccs"],
            "Gas": ["gas_ppl", "gas_cc"],
            "Gas w CCS": ["gas_cc_ccs"],
            "Gas CT": ["gas_ct"],
            "Oil": ["foil_ppl", "loil_ppl", "loil_cc"],
            "Nuclear": ["nuc_lc", "nuc_hc"],
            "Hydro": ["hydro_lc", "hydro_hc"],
            "Biomass": ["bio_istig", "bio_ppl"],
            "Biomass w CCS": ["bio_istig_ccs"],
            "Geothermal": ["geo_ppl"],
            "CSP": [
                x
                for x in set(sc.set("technology"))
                if any([y in x for y in ["csp_sm1", "csp_sm3", "solar_th_ppl"]])
            ],
            "Solar PV": [
                x
                for x in set(sc.set("technology"))
                if any([y in x for y in ["solar_cv", "solar_res"]])
            ],
            "Wind": [
                x
                for x in set(sc.set("technology"))
                if any([y in x for y in ["wind_cv", "wind_res", "wind_ref"]])
            ],
            "Wind curtailed": [x for x in set(sc.set("technology")) if "wind_curt" in x],
            "Solar PV curtailed": [x for x in set(sc.set("technology")) if "solar_curt" in x],
            "Storage": ["stor_ppl"],
            "Export": [x for x in set(sc.set("technology")) if "elec_exp" in x],
            "Import": [x for x in set(sc.set("technology")) if "elec_imp" in x],
            "E-mobility": ["elec_trp"],
            "Electrolysis": ["h2_elec"],
            "Fuel cell": ["h2_fc_I", "h2_fc_RC"],
            "DAC": [x for x in set(sc.set("technology")) if "dac_" in x],
            "Load": ["elec_t_d"],
        }

        ren = {}
        for x, y in rename.items():
            for z in y:
                ren[z] = x

        # Dealing with curtailment in more detail (for each VRE step)
        stor_tecs_input = {"stor_ppl": 1.25, "h2_elec": 1.25}
        stor_tecs_loss = {"stor_ppl": 0.25}

        for scen in scenarios:
            df = pd.DataFrame(index=model_years)
            other = pd.DataFrame(index=model_years)
            for curt in curt_list:
                d = results[(scen, curt)]

                # 1. Estimating theoretical curtailment
                if "wind" in curt:
                    tec_list = rename["Wind"] + ["elec_t_d"]
                else:
                    tec_list = rename["Solar PV"] + ["elec_t_d"]

                # Loading the data of resource technologies
                cm = d.loc[d["technology"].isin(tec_list)]  # curtailment techs

                # Calculating theoretical curtailment (positive value) for each year
                df[curt] = cm.reset_index().groupby(["year_act"]).sum().drop(["share"], axis=1)
                df = df[df > 0].copy()

                # 2. Calculating the role of each technology for wind and solar curtailment
                # Aggregating all
                role = d.loc[~d["technology"].isin(tec_list)]  # other techs
                other[curt] = (
                    role.reset_index().groupby(["year_act"]).sum().drop(["share"], axis=1)
                )

            # Calculating aggregates
            wind_rel = [x for x in df.columns if "wind" in x]
            pv_rel = [x for x in df.columns if "solar" in x]
            df["Wind"] = df.loc[:, df.columns.isin(wind_rel)].sum(axis=1)
            df["Solar PV"] = df.loc[:, df.columns.isin(pv_rel)].sum(axis=1)

            # Multiplying by electricity input of each curtailment category
            for col, value in inp_curt.items():
                df[col + "_input"] = df[col] * value
            wind_rel = [x for x in df.columns if "wind" in x and "input" in x]
            pv_rel = [x for x in df.columns if "solar" in x and "input" in x]
            df["Wind curtailed"] = df.loc[:, df.columns.isin(wind_rel)].sum(axis=1)
            df["Solar PV curtailed"] = df.loc[:, df.columns.isin(pv_rel)].sum(axis=1)

            # Estimating storage contribution (by activity)
            for tec, inp_loss in stor_tecs_loss.items():
                df[ren[tec]] = d.loc[d["technology"] == tec, "value"]
                df[ren[tec] + "_loss"] = d.loc[d["technology"] == tec, "value"] * inp_loss
            # Estimating storage contribution (by charging when curtailment happens)
            for tec, inp_el in stor_tecs_input.items():
                df[ren[tec] + "_input"] = d.loc[d["technology"] == tec, "value"] * inp_el
        
        df_curtailment = df
    else:
        df_curtailment = 0
    
    return results, df_curtailment, df_generation, d1_wind, d2_wind

def plot_Behnam_script(scenarios,node,plot_curtailment = True):

    idx = ["model", "scenario", "region", "variable", "unit"]

    rels = {
        "Operating reserves (GWa)": ["oper_res", None],
        "Reserve margin (GW)": ["res_marg", None],
    }

    results = {}
    figures = {}

    # Loading scenarios and calculating
    for num, scen in enumerate(scenarios.keys()):
        sc = scenarios[scen] 

        model_years = [x for x in sc.set("year") if x >= sc.firstmodelyear and x < 2110]
        for r in rels.keys():
            results[(scen, r)] = contribute_relation(
                sc,
                node=node,
                year=model_years,
                relation=rels[r][0],
                commodity=rels[r][1],
            )

        # Curtailment relations
        curt_list = [x for x in sc.set("relation") if "curtailment" in x]
        for curt in curt_list:
            results[(scen, curt)] = contribute_relation(
                sc,
                node=node,
                year=model_years,
                relation=curt,
                commodity=None,
                assertion=False,
            )

        # Calculating share of PV from total generation
        # List of technologies generating at the secondary level
        tec_list = sc.par("output", {"level": "secondary", "commodity": "electr"})[
            "technology"
        ].unique()
        # List of technologies consuming at the secondary level
        tec_in = sc.par("input", {"level": "secondary", "commodity": "electr"})[
            "technology"
        ].unique()
        # List of flexible technologies
        flex = sc.par("relation_activity", {"relation": "oper_res"})
        tec_flex = flex.loc[flex["value"] > 0, "technology"].unique()
        flex_gen = [x for x in tec_list if x in tec_flex]
        flex_se = [x for x in tec_in if x in tec_flex]
        flex_load = [x for x in tec_flex if x not in flex_gen + flex_se]

        tec_inflex = flex.loc[flex["value"] <= 0, "technology"].unique()
        info = {
            "Solar PV": (["solar_res"], ["solar_curtail"]),
            "Wind": (["wind_res", "wind_ref"], ["wind_curtail"]),
            "Generation": (tec_list, []),
            "Secondary": (tec_in, ["elec_t_d", "stor_ppl"]),
            "Flex gen.": (flex_gen, []),
            "Flex SE": (flex_se, []),
            "Flex load": (flex_load, []),
            "Grid": (["elec_t_d"], []),
            "Storage loss": (["stor_ppl"], []),
            "Load": (["elec_t_d"], []),
        }

        df = pd.DataFrame(index=model_years)
        for tec, data in info.items():

            # Share of main technology
            main = sorted(
                [x for x in set(sc.set("technology")) if any([y in x for y in data[0]])]
            )

            if "Flex" in tec:
                d = contribute_relation(
                    sc,
                    node,
                    model_years,
                    relation="oper_res",
                    commodity="electr",
                    assertion=False,
                )
                d1 = d.loc[d["technology"].isin(data[0])].copy()
                d1["year_vtg"] = 2020  # dummy
            elif tec in ["Generation", "Load"]:
                d1 = out_act(sc, node, main, "electr", model_years).reset_index()
            else:
                d1 = inp_act(sc, node, main, "electr", model_years).reset_index()
            d1["technology"] = tec

            # Share of deduction (curtailment)
            if data[1]:
                deduct = sorted(
                    [x for x in set(sc.set("technology")) if any([y in x for y in data[1]])]
                )
                if "Flex" in tec:
                    d2 = d.loc[d["technology"].isin(deduct)].copy()
                    d2["year_vtg"] = 2020  # dummy
                else:
                    d2 = inp_act(
                        sc, node, deduct, "electr", model_years
                    ).reset_index()
                    d2["technology"] = tec
                    d2["value"] *= -1
            else:
                d2 = pd.DataFrame()
            # Deducting curtailment from generation
            df[tec] = (
                pd.concat([d1, d2])
                .groupby(["year_act"])
                .sum()
                .drop(["year_vtg"], axis=1)["value"]
            )

            if tec == "Wind":
                d1_wind = d1
                d2_wind = d2

        # Check balances
        df["Generation"] >= df["Secondary"] - df["Grid"]

        # Calculating the share per year
        df["Grid loss"] = df["Grid"] - df["Load"]
        df["Secondary"] -= df["Flex SE"]
        df["Load"] -= df["Flex load"]
        df["Inflex gen."] = df["Generation"] - df["Solar PV"] - df["Wind"] - df["Flex gen."]

        # Giving different signs to generation (+) vs. consumption (-)
        ord_neg = ["Load", "Flex load", "Secondary", "Flex SE", "Storage loss", "Grid loss"]
        df.loc[:, df.columns.isin(ord_neg)] *= -1

        # Calculating shares
        share = df / df[["Generation"]].values

        # Ordering and saving
        ord_pos = ["Solar PV", "Wind", "Flex gen.", "Inflex gen."]
        df = df[ord_pos + ord_neg]

        # df.columns = pd.MultiIndex.from_product([['value'], df.columns])
        # df = df.stack()
        # df.index.names = ["year", "variable"]
        df = df.stack().reset_index()
        df.columns = ["year", "variable", "value"]
        df["model"] = sc.model
        df["scenario"] = sc.scenario
        df["unit"] = "GWa"
        df["region"] = node

        figures[(scen, "Generation (GWa)")] = pd.pivot_table(
            df, index=idx, columns="year", values="value"
        ).reset_index()

    df_generation = df

    if plot_curtailment:
        # Visualization and plotting configuration
        # Input values for curtailment electricity (please check for each region)
        inp_curt = {
            "wind_curtailment_1": 0.10,
            "wind_curtailment_2": 0.25,
            "wind_curtailment_3": 0.35,
            "solar_curtailment_1": 0.15,
            "solar_curtailment_2": 0.25,
            "solar_curtailment_3": 0.35,
        }

        rename = {
            "Coal": ["coal_adv", "coal_ppl", "coal_ppl_u", "igcc"],
            "Coal w CCS": ["coal_adv_ccs", "igcc_ccs"],
            "Gas": ["gas_ppl", "gas_cc"],
            "Gas w CCS": ["gas_cc_ccs"],
            "Gas CT": ["gas_ct"],
            "Oil": ["foil_ppl", "loil_ppl", "loil_cc"],
            "Nuclear": ["nuc_lc", "nuc_hc"],
            "Hydro": ["hydro_lc", "hydro_hc"],
            "Biomass": ["bio_istig", "bio_ppl"],
            "Biomass w CCS": ["bio_istig_ccs"],
            "Geothermal": ["geo_ppl"],
            "CSP": [
                x
                for x in set(sc.set("technology"))
                if any([y in x for y in ["csp_sm1", "csp_sm3", "solar_th_ppl"]])
            ],
            "Solar PV": [
                x
                for x in set(sc.set("technology"))
                if any([y in x for y in ["solar_cv", "solar_res"]])
            ],
            "Wind": [
                x
                for x in set(sc.set("technology"))
                if any([y in x for y in ["wind_cv", "wind_res", "wind_ref"]])
            ],
            "Wind curtailed": [x for x in set(sc.set("technology")) if "wind_curt" in x],
            "Solar PV curtailed": [x for x in set(sc.set("technology")) if "solar_curt" in x],
            "Storage": ["stor_ppl"],
            "Export": [x for x in set(sc.set("technology")) if "elec_exp" in x],
            "Import": [x for x in set(sc.set("technology")) if "elec_imp" in x],
            "E-mobility": ["elec_trp"],
            "Electrolysis": ["h2_elec"],
            "Fuel cell": ["h2_fc_I", "h2_fc_RC"],
            "DAC": [x for x in set(sc.set("technology")) if "dac_" in x],
            "Load": ["elec_t_d"],
        }

        ren = {}
        for x, y in rename.items():
            for z in y:
                ren[z] = x

        # Dealing with curtailment in more detail (for each VRE step)
        stor_tecs_input = {"stor_ppl": 1.25, "h2_elec": 1.25}
        stor_tecs_loss = {"stor_ppl": 0.25}

        for scen in scenarios:
            df = pd.DataFrame(index=model_years)
            other = pd.DataFrame(index=model_years)
            for curt in curt_list:
                d = results[(scen, curt)]

                # 1. Estimating theoretical curtailment
                if "wind" in curt:
                    tec_list = rename["Wind"] + ["elec_t_d"]
                else:
                    tec_list = rename["Solar PV"] + ["elec_t_d"]

                # Loading the data of resource technologies
                cm = d.loc[d["technology"].isin(tec_list)]  # curtailment techs

                # Calculating theoretical curtailment (positive value) for each year
                df[curt] = cm.reset_index().groupby(["year_act"]).sum().drop(["share"], axis=1)
                df = df[df > 0].copy()

                # 2. Calculating the role of each technology for wind and solar curtailment
                # Aggregating all
                role = d.loc[~d["technology"].isin(tec_list)]  # other techs
                other[curt] = (
                    role.reset_index().groupby(["year_act"]).sum().drop(["share"], axis=1)
                )

            # Calculating aggregates
            wind_rel = [x for x in df.columns if "wind" in x]
            pv_rel = [x for x in df.columns if "solar" in x]
            df["Wind"] = df.loc[:, df.columns.isin(wind_rel)].sum(axis=1)
            df["Solar PV"] = df.loc[:, df.columns.isin(pv_rel)].sum(axis=1)

            # Multiplying by electricity input of each curtailment category
            for col, value in inp_curt.items():
                df[col + "_input"] = df[col] * value
            wind_rel = [x for x in df.columns if "wind" in x and "input" in x]
            pv_rel = [x for x in df.columns if "solar" in x and "input" in x]
            df["Wind curtailed"] = df.loc[:, df.columns.isin(wind_rel)].sum(axis=1)
            df["Solar PV curtailed"] = df.loc[:, df.columns.isin(pv_rel)].sum(axis=1)

            # Estimating storage contribution (by activity)
            for tec, inp_loss in stor_tecs_loss.items():
                df[ren[tec]] = d.loc[d["technology"] == tec, "value"]
                df[ren[tec] + "_loss"] = d.loc[d["technology"] == tec, "value"] * inp_loss
            # Estimating storage contribution (by charging when curtailment happens)
            for tec, inp_el in stor_tecs_input.items():
                df[ren[tec] + "_input"] = d.loc[d["technology"] == tec, "value"] * inp_el

        df_curtailment = df
    else:
        df_curtailment = 0

    return results, df_curtailment, df_generation, d1_wind, d2_wind

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

def plot_gen_share_and_curtailment_2(df_generation, df_curtailment):
    tech_colors = {"solar":"#ffcc00","wind":"#235ebc"}
    fs = 15

    df_solar = df_generation.query("variable == 'Solar PV'")
    df_solar = df_solar.set_index("year")
    df_wind = df_generation.query("variable == 'Wind'")
    df_wind = df_wind.set_index("year")

    df_generation_gbyv = df_generation.groupby(["year","variable"]).sum()
    df_tot_generation = df_generation_gbyv[df_generation_gbyv > 0].dropna() 
    df_tot_generation_gby = df_tot_generation.groupby("year").sum()

    df_solar_share = df_solar.value/df_tot_generation_gby.value
    df_wind_share = df_wind.value/df_tot_generation_gby.value
    df_vre_share = pd.DataFrame()
    df_vre_share["solar"] = df_solar_share
    df_vre_share["wind"] = df_wind_share

    df_vre_generation_abs = pd.DataFrame()
    df_vre_generation_abs["wind"] = df_wind.value
    df_vre_generation_abs["solar"] = df_solar.value

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

    wind_curtailment_bins = df_curtailment[["wind_curtailment_1_input",
                                            "wind_curtailment_2_input",
                                            "wind_curtailment_3_input"]]

    solar_curtailment_bins = df_curtailment[["solar_curtailment_1_input",
                                             "solar_curtailment_2_input",
                                             "solar_curtailment_3_input"]]
    
    df_wind_theoretical = df_wind.value + wind_curtailment_bins.sum(axis=1)
    df_solar_theoretical = df_solar.value + solar_curtailment_bins.sum(axis=1)

    ############################################################### Fig. 3
    fig3,ax3 = plt.subplots()
    wind_curtailment_bins_rel = wind_curtailment_bins.T.div(df_wind_theoretical).T*100
    wind_curtailment_bins_rel.plot.bar(ax=ax3,
                                       stacked=True,
                                      legend=False)
    ax3.set_ylim(0,100)
    ax3.set_ylabel("wind curtailment \n (% of wind theo. generation)")
    fig3.legend(ncol=1,bbox_to_anchor=[0.75, -0.05],prop={"size":fs})
    # fig3.savefig("figures/wind_curtailment_baseline.png",bbox_inches="tight",dpi=300)

    ############################################################### Fig. 4
    fig4,ax4 = plt.subplots()
    solar_curtailment_bins_rel = solar_curtailment_bins.T.div(df_solar_theoretical).T*100
    solar_curtailment_bins_rel.plot.bar(ax=ax4,
                                        stacked=True,
                                        legend=False)

    ax4.set_ylim(0,100)
    ax4.set_ylabel("solar curtailment \n (% of solar theo. generation)")
    fig4.legend(ncol=1,bbox_to_anchor=[0.75, -0.05],prop={"size":fs})
    # fig4.savefig("figures/solar_curtailment_baseline.png",bbox_inches="tight",dpi=300)
    
    return fig1, ax1, fig2, ax2, fig3, ax3, fig4, ax4, df_vre_share_pct

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
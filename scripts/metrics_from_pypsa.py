# -*- coding: utf-8 -*-
"""
Created on 29th of November 2023
Author: Ebbe Kyhl Gøtske

This script calculates the metrics from the PyPSA-Eur scenario. 
The metrics are calculated for each scenario and saved as a .csv file.

The metrics include the following:
    - curtailment
    - storage dispatch
    - backup generation
    - endogenous demand
    - renewable penetration
    - CO2 emissions
    - costs
    - wind and solar PV capacity (GW)
    - wind and solar PV capacity factor (%)
    - wind and solar PV generation relative to the electricity demand (%)
    - wind and solar PV resources relative to the electricity demand (%)
    - wind and solar PV curtailment (TWh)
    - wind and solar PV curtailment relative to resources (%)
    - SDES and LDES storage dispatch (TWh)
    - SDES and LDES storage dispatch relative to the electricity demand (%)
    - SDES and LDES storage energy capacity (TWh)
    - SDES and LDES storage discharge capacity (GW)
    - transmission volume (TWkm)
    - transmission peak load ratio (GW/GW)
    - system cost (billion EUR)
    - capacity value of wind and solar (%)
    - backup capacity (GW)
    - backup capacity factor low (GW/GW)
    - backup capacity factor high (GW/GW)
    - total demand (TWh)
"""
import numpy as np
import os
import glob
import pypsa
import pandas as pd
from vresutils.costdata import annuity
import matplotlib.dates as mdates
from _helpers import override_component_attrs
import warnings
warnings.filterwarnings('ignore')
overrides = override_component_attrs("override_component_attrs")
locator = mdates.DayLocator()  # every month
fmt = mdates.DateFormatter('%b-%d')

def calculate_curtailment(n,denominator_category="load"):
    total_inflexible_load = n.loads_t.p[n.loads.query('carrier == "electricity"').index].sum(axis=1).sum()
    total_wind_generation_actual = n.generators_t.p[n.generators.index[n.generators.index.str.contains('wind')]].sum().sum()
    total_wind_generation_theoretical = (n.generators_t.p_max_pu[n.generators.index[n.generators.index.str.contains('wind')]]*n.generators.loc[n.generators.index[n.generators.index.str.contains('wind')]].p_nom_opt).sum().sum()
    total_solar_generation_actual = n.generators_t.p[pd.concat([n.generators.query("carrier == 'solar'"),n.generators.query("carrier == 'solar rooftop'")]).index].sum().sum()
    total_solar_generation_theoretical = (n.generators_t.p_max_pu[pd.concat([n.generators.query("carrier == 'solar'"),n.generators.query("carrier == 'solar rooftop'")]).index]*n.generators.loc[pd.concat([n.generators.query("carrier == 'solar'"),n.generators.query("carrier == 'solar rooftop'")]).index].p_nom_opt).sum().sum()
    
    wind_abs_curt = []
    generator_total_curtailment_percentage_wind = []
    solar_abs_curt = []
    generator_total_curtailment_percentage_solar = []
    
    generators = ['solar','solar rooftop','onwind','offwind-ac','offwind-dc']
    for generator in generators:
        generator_index = n.generators.query('carrier == @generator').index
        capacity = n.generators.loc[generator_index].p_nom_opt
        
        nodal_production = n.generators_t.p[generator_index].sum()
        nodal_potential = (n.generators_t.p_max_pu[generator_index]*n.generators.query('carrier == @generator').p_nom_opt).sum()

        if "wind" in generator:
            denominator_dict = {"load":total_inflexible_load,
                                "gen_actual":total_wind_generation_actual,
                                "gen_theoretical":total_wind_generation_theoretical}
    
            denominator = denominator_dict[denominator_category]
            
            if nodal_potential.sum() > 0:
                abs_diff = nodal_potential.sum() - nodal_production.sum()
                wind_abs_curt.append(abs_diff)
                generator_total_curtailment_percentage_wind.append(((abs_diff)/denominator*100).round(3))
            else:
                wind_abs_curt.append(0)
                generator_total_curtailment_percentage_wind.append(0)
                
        elif "solar" in generator:
            denominator_dict = {"load":total_inflexible_load,
                                "gen_actual":total_solar_generation_actual,
                                "gen_theoretical":total_solar_generation_theoretical}
    
            denominator = denominator_dict[denominator_category]
        
            if nodal_potential.sum() > 0:
                abs_diff = nodal_potential.sum() - nodal_production.sum()
                solar_abs_curt.append(abs_diff)
                generator_total_curtailment_percentage_solar.append(((abs_diff)/denominator*100).round(3))
            else:
                solar_abs_curt.append(0)
                generator_total_curtailment_percentage_solar.append(0)
            
    return solar_abs_curt, generator_total_curtailment_percentage_solar, wind_abs_curt, generator_total_curtailment_percentage_wind

def calculate_storage_dispatch(n):
    #tot_load = n.loads_t.p_set[n.loads.query('carrier == "electricity"').index].sum().sum()
    
    bat_discharge = -n.links_t.p1[n.links.index[n.links.index.str.contains('battery discharge')]] # discharged electricity
    LDES_discharge = -n.links_t.p1[n.links.query('carrier == "LDES discharger"').index]
    H2_discharge = -n.links_t.p1[n.links.query('carrier == "H2 Fuel Cell"').index] # discharged electricity
    
    wind_generators = n.generators.index[n.generators.index.str.contains('wind')]
    solar_generators = pd.concat([n.generators.query('carrier == "solar"'),n.generators.query('carrier == "solar rooftop"')]).index
    VRE_generation = n.generators_t.p[wind_generators].sum().sum() + n.generators_t.p[solar_generators].sum().sum()
    
    # SDES
    storage_dispatch_sdes = bat_discharge.sum().sum()
        
    # LDES
    storage_dispatch_ldes = LDES_discharge.sum().sum()
    
    # H2
    storage_dispatch_H2 = H2_discharge.sum().sum()
    
    return storage_dispatch_sdes, storage_dispatch_ldes, storage_dispatch_H2

def calculate_backup_generation(n):
    links_wo_transmission = n.links.drop(n.links.query("carrier == 'DC'").index)
    electricity_buses = list(n.buses.query('carrier == "AC"').index) + list(
        n.buses.query('carrier == "low voltage"').index
        )
    
    buses = n.links.columns[n.links.columns.str.contains("bus")]
    backup_capacity = {}
    backup_capacity_factor_low = {}
    backup_capacity_factor_high = {}
    for bus in ["bus1"]: #buses:
        if bus != "bus0":
            boolean_elec_demand_via_links = [
                links_wo_transmission.bus1[i] in electricity_buses
                for i in range(len(links_wo_transmission.bus1))
                ]
            
            boolean_elec_demand_via_links_series = pd.Series(boolean_elec_demand_via_links)
            links = links_wo_transmission.iloc[
                        boolean_elec_demand_via_links_series[boolean_elec_demand_via_links_series ].index
                        ]

            # Drop batteries, LDES, and distribution
            links = links.drop(
                                links.index[links.index.str.contains("battery")]
                              )

            links = links.drop(
                                links.index[links.index.str.contains("LDES")]
                              )

            links = links.drop(
                                links.index[links.index.str.contains("distribution")]
                              )
            
            links = links.drop(
                                links.index[links.index.str.contains("V2G")]
                              )

            # Calculate technology-aggregated backup capacities
            bus_no = bus[-1]
            if bus_no == "1":
                cap = n.links.loc[links.index].p_nom_opt*n.links.loc[links.index].efficiency
            else:
                cap = n.links.loc[links.index].p_nom_opt*eval("n.links.loc[links.index].efficiency" + bus_no)

            cap_grouped = cap.groupby(links.carrier).sum()
            cap_grouped[cap_grouped < 1] = np.nan # drop technologies with zero capacities
            tech_capacity_factor = -eval("n.links_t.p" + bus_no)[links.index].groupby(links.carrier,axis=1).sum().sum()/(cap_grouped*len(n.snapshots))*100

            backup_capacity[bus] = cap_grouped/1e3 # GW
            backup_capacity_factor_low[bus] = tech_capacity_factor.min() # % min average capacity factor
            backup_capacity_factor_high[bus] = tech_capacity_factor.max() # % max average capacity factor
    
    df_backup_capacity = round(pd.DataFrame.from_dict(backup_capacity).sum().item(),2) #.dropna()
    
    df_backup_capacity_factor_low = round(backup_capacity_factor_low["bus1"],2)
    df_backup_capacity_factor_high = round(backup_capacity_factor_high["bus1"],2)
    
    return df_backup_capacity, df_backup_capacity_factor_low, df_backup_capacity_factor_high

def calculate_endogenous_demand(n):
    links_wo_transmission = n.links.drop(n.links.query("carrier == 'DC'").index)
    electricity_buses = list(n.buses.query('carrier == "AC"').index) + list(
        n.buses.query('carrier == "low voltage"').index
    )
    boolean_elec_demand_via_links = [
        links_wo_transmission.bus0[i] in electricity_buses
        for i in range(len(links_wo_transmission.bus0))
    ]
    boolean_elec_demand_via_links_series = pd.Series(boolean_elec_demand_via_links)
    elec_demand_via_links = links_wo_transmission.iloc[
        boolean_elec_demand_via_links_series[boolean_elec_demand_via_links_series].index
    ]

    # Drop batteries
    elec_demand_via_links = elec_demand_via_links.drop(
        elec_demand_via_links.index[elec_demand_via_links.index.str.contains("battery")]
    )

    # Drop LDES
    elec_demand_via_links = elec_demand_via_links.drop(
        elec_demand_via_links.index[elec_demand_via_links.index.str.contains("LDES")]
    )

    # Drop distribution links
    elec_demand_via_links = elec_demand_via_links.drop(
        elec_demand_via_links.index[
            elec_demand_via_links.index.str.contains("distribution")
        ]
    )
    
    endogenous_demand = n.links_t.p0[elec_demand_via_links.index]

    return endogenous_demand

def calculate_renewable_penetration(n):
    
    # exogenous demand
    exo_demand = load = (n.loads_t.p_set[n.loads.query("carrier == 'electricity'").index].sum().sum() + 
        (n.loads.query("carrier == 'industry electricity'").p_set*len(n.snapshots)).sum()
    )
    
    # endogenous demand
    endo_demand_i = calculate_endogenous_demand(n)
    endo_demand = endo_demand_i.sum().sum()
    
    # total electricity demand
    tot_load = exo_demand + endo_demand
    
    # solar generation
    solar_generators = pd.concat([n.generators.query('carrier == "solar"'),n.generators.query('carrier == "solar rooftop"')]).index
    solar_generation_capacity = n.generators.loc[solar_generators].p_nom_opt
    solar_generation = n.generators_t.p[solar_generators]
    solar_cap_factor = (solar_generation.sum()/(solar_generation_capacity*len(n.snapshots))).mean()    
    solar_potential = (n.generators_t.p_max_pu[solar_generators]*n.generators.p_nom_opt.loc[solar_generators]).sum().sum()
    
    solar_share = solar_generation.sum().sum()/tot_load
    solar_theo_share = solar_potential/tot_load
    
    # wind generation
    wind_generators = n.generators.index[n.generators.index.str.contains('wind')]
    wind_generation_capacity = n.generators.loc[wind_generators].p_nom_opt
    wind_generation = n.generators_t.p[wind_generators]
    wind_cap_factor = (wind_generation.sum()/(wind_generation_capacity*len(n.snapshots))).mean() 
    wind_potential = (n.generators_t.p_max_pu[wind_generators]*n.generators.p_nom_opt.loc[wind_generators]).sum().sum()
    
    wind_share = wind_generation.sum().sum()/tot_load
    wind_theo_share = wind_potential/tot_load
    
    solar_share = round(solar_share*100,1)
    wind_share = round(wind_share*100,1)
    solar_theo_share = round(solar_theo_share*100,1)
    wind_theo_share = round(wind_theo_share*100,1)
    solar_cap_factor = round(solar_cap_factor*100,1)
    wind_cap_factor = round(wind_cap_factor*100,1)
    # residual_share = round((100 - solar_share - wind_share),1)
    
    return solar_share, wind_share, solar_theo_share, wind_theo_share, solar_cap_factor, wind_cap_factor #, residual_share

def calculate_co2_emissions(n):
    ###############################################################
    ######################## CO2 emissions ########################
    ###############################################################
    
    outputs = [
                "summary",
              ]

    columns = pd.MultiIndex.from_tuples(
        networks_dict.keys(), names=["design_year","weather_year"]
    )
    
    tres_factor = 8760/len(n.snapshots)

    df = {}

    for output in outputs:
        df[output] = pd.DataFrame(columns=columns, dtype=float)
    
    # CO2 emittors and capturing facilities from power, heat and fuel production
    co2_emittors = n.links.query('bus2 == "co2 atmosphere"') # links going from fuel buses (e.g., gas, coal, lignite etc.) to "CO2 atmosphere" bus
    co2_emittors = co2_emittors.query('efficiency2 != 0') # excluding links with no CO2 emissions (e.g., nuclear)
    co2_t = -n.links_t.p2[co2_emittors.index]*tres_factor

    co2_t_renamed = co2_t.rename(columns=co2_emittors.carrier.to_dict())
    co2_t_grouped = co2_t_renamed.groupby(by=co2_t_renamed.columns,axis=1).sum().sum()
    for i in range(len(co2_t_grouped.index)):
        if 'gas boiler' not in co2_t_grouped.index[i]:
            co2_t_i = round(co2_t_grouped.iloc[i]/1e6,sig_dig)
            df.loc['co2 emissions ' + co2_t_grouped.index[i],label] = co2_t_i

    co2_t_gas_boiler = round(co2_t_grouped.loc[co2_t_grouped.index[co2_t_grouped.index.str.contains('gas boiler')]].sum()/1e6,sig_dig)

    df.loc['co2 emissions gas boiler',label] = co2_t_gas_boiler
    ###############################################################

    # CO2 emissions and capturing from chp plants
    chp = n.links.query('bus3 == "co2 atmosphere"') # links going from CHP fuel buses (e.g., biomass or natural gas) to "CO2 atmosphere" bus
    co2_chp_t = -n.links_t.p3[chp.index]*tres_factor
    # NB! Biomass w.o. CC has no emissions. For this reason, Biomass w. CC has negative emissions.

    co2_chp_t_renamed = co2_chp_t.rename(columns=chp.carrier.to_dict())
    co2_chp_t_renamed_grouped = co2_chp_t_renamed.groupby(by=co2_chp_t_renamed .columns,axis=1).sum().sum()

    co2_chp_t_renamed_grouped_gas = round(co2_chp_t_renamed_grouped.loc[co2_chp_t_renamed_grouped.index[co2_chp_t_renamed_grouped.index.str.contains('gas CHP')]].sum()/1e6,sig_dig)
    co2_chp_t_renamed_grouped_biomass = round(co2_chp_t_renamed_grouped.loc[co2_chp_t_renamed_grouped.index[co2_chp_t_renamed_grouped.index.str.contains('solid biomass CHP')]].sum()/1e6,sig_dig)

    df.loc['co2 emissions gas CHP',label] = co2_chp_t_renamed_grouped_gas
    df.loc['co2 emissions biomass CHP',label] = co2_chp_t_renamed_grouped_biomass 
    ###############################################################

    # process emissions
    co2_process = n.links.query('bus1 == "co2 atmosphere"').index # links going from "process emissions" to "CO2 atmosphere" bus
    co2_process_t = -n.links_t.p1[co2_process]*tres_factor
    # process emissions have CC which captures 90 % of emissions. Here, we only consider the 10 % being emitted.
    # to include the 90% capture in the balance, call: -n.links_t.p2["EU process emissions CC"]
    co2_process_t_sum = round(co2_process_t.sum().sum()/1e6,sig_dig)

    df.loc['co2 emissions process', label] = co2_process_t_sum 
    ###############################################################

    # load emissions (e.g., land transport or agriculture)
    loads_co2 = n.loads # .query('bus == "co2 atmosphere"')
    load_emissions_index = loads_co2.index[loads_co2.index.str.contains('emissions')]
    load_emissions = n.loads.loc[load_emissions_index]
    load_emissions_t = -n.loads_t.p[load_emissions_index]*tres_factor

    load_emissions_t_sum_oil = round(load_emissions_t['oil emissions'].sum()/1e6,sig_dig)
    load_emissions_t_sum_agriculture = round(load_emissions_t['agriculture machinery oil emissions'].sum()/1e6,sig_dig)

    df.loc['co2 emissions oil load', label] = load_emissions_t_sum_oil
    df.loc['co2 emissions agriculture machinery', label] = load_emissions_t_sum_agriculture
    ###############################################################

    # direct air capture
    dac = n.links.index[n.links.index.str.contains('DAC')] # links going from "CO2 atmosphere" to "CO2 stored" (sequestration)
    co2_dac_t = -n.links_t.p1[dac]*tres_factor 
    co2_dac_t_sum = -round(co2_dac_t.sum().sum()/1e6,sig_dig) # i.e., negative emissions

    df.loc['co2 emissions dac', label] = co2_dac_t_sum
    ###############################################################

    # CO2 balance
    co2_tot = df.loc[df.index[df.index.str.contains('co2 emissions')],label].sum().sum()
    df.loc['net emissions', label] = round(co2_tot,sig_dig)
    
    return df

def prepare_costs(nyears):
    
    fill_values = {"FOM": 0,
                    "VOM": 0,
                    "efficiency": 1,
                    "fuel": 0,
                    "investment": 0,
                    "lifetime": 25,
                    "CO2 intensity": 0,
                    "discount rate": 0.07}
    
    # set all asset costs and other parameters
    costs = pd.read_csv("costs_2030.csv", index_col=[0, 1]).sort_index()

    # correct units to MW and EUR
    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3

    # min_count=1 is important to generate NaNs which are then filled by fillna
    costs = (
        costs.loc[:, "value"].unstack(level=1).groupby("technology").sum(min_count=1)
    )

    costs = costs.fillna(fill_values)

    def annuity_factor(v):
        return annuity(v["lifetime"], v["discount rate"]) + v["FOM"] / 100

    costs["fixed"] = [
        annuity_factor(v) * v["investment"] * nyears for i, v in costs.iterrows()
    ]

    return costs


def calculate_costs(n, label, costs):
    opt_name = {"Store": "e", "Line": "s", "Transformer": "s"}

    for c in n.iterate_components(
        n.branch_components | n.controllable_one_port_components ^ {"Load"}
    ):
        capital_costs = c.df.capital_cost * c.df[opt_name.get(c.name, "p") + "_nom_opt"]
        capital_costs_grouped = capital_costs.groupby(c.df.carrier).sum()

        capital_costs_grouped = pd.concat([capital_costs_grouped], keys=["capital"])
        capital_costs_grouped = pd.concat([capital_costs_grouped], keys=[c.list_name])

        costs = costs.reindex(capital_costs_grouped.index.union(costs.index))

        costs.loc[capital_costs_grouped.index, label] = capital_costs_grouped

        if c.name == "Link":
            p = c.pnl.p0.multiply(n.snapshot_weightings.generators, axis=0).sum()
        elif c.name == "Line":
            continue
        elif c.name == "StorageUnit":
            p_all = c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0)
            p_all[p_all < 0.0] = 0.0
            p = p_all.sum()
        else:
            p = c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0).sum()

        # correct sequestration cost
        if c.name == "Store":
            items = c.df.index[
                (c.df.carrier == "co2 stored") & (c.df.marginal_cost <= -100.0)
            ]
            c.df.loc[items, "marginal_cost"] = -20.0

        marginal_costs = p * c.df.marginal_cost

        marginal_costs_grouped = marginal_costs.groupby(c.df.carrier).sum()

        marginal_costs_grouped = pd.concat([marginal_costs_grouped], keys=["marginal"])
        marginal_costs_grouped = pd.concat([marginal_costs_grouped], keys=[c.list_name])

        costs = costs.reindex(marginal_costs_grouped.index.union(costs.index))

        costs.loc[marginal_costs_grouped.index, label] = marginal_costs_grouped

    return costs

RDIR = "calculated_metrics/"
path = 'networks/'

# scenarios 

scens = [#"new_base",
         #"new_flipped_merit_order"
         #"new_base_co2_lim"
         "new_LDES_co2_lim",
         #"new_SDES",
         #"new_SDES_LDES"
         #"new_LDES_co2_lim",
         #"new_SDES_co2_lim",
         #"new_SDES_LDES_co2_lim"
         #"new_transport",
         #"new_transport_co2_lim",
         #"new_heating_demand",
         #"new_heating_demand_co2_lim"
        ]

# scenarios with existing hydropower facilities: "1H_w_hydro", "1H_w_co2_lim_w_hydro",
denominator = "gen_theoretical" # normalization of curtailment w.r.t. either theoretical or actual generation, or pick "load" as units of demand
for scen in scens:
    if not os.path.isdir(RDIR + scen):
        os.mkdir(RDIR + scen)
        print("creating new directory for", scen)
    else:
        print(scen, " already exists. Overwriting existing files!")
        
    network_names = glob.glob(path + scen + "/*.nc")
    no_networks = len(network_names)
        
    percentage_storage_sdes = {}
    percentage_storage_ldes = {}
    wind_cap = {}
    solar_cap = {}
    wind_cap_factor = {}
    solar_cap_factor = {}
    DK_wind_cap = {}
    ES_wind_cap = {}
    DK_solar_cap = {}
    ES_solar_cap = {}
    solar_curt = {}
    solar_abs_curt = {}
    wind_curt = {}
    wind_abs_curt = {}
    wind_share = {}
    solar_share = {}
    wind_theo_share = {}
    solar_theo_share = {}
    VRE_share = {}
    ren_curt = {}
    abs_storage_sdes = {}
    abs_storage_ldes = {}
    LDES_discharge_capacity = {}
    LDES_energy_capacity = {}
    SDES_discharge_capacity = {}
    SDES_energy_capacity = {}
    trans_vol = {}
    system_efficiency = {}
    system_cost = {}
    transmission_peak_load_ratio = {}
    CV_wind = {}
    CV_solar = {}
    backup_capacity = {}
    backup_capacity_factor_low = {}
    backup_capacity_factor_high = {}
    transmission_peak_load_ratio = {}
    total_demand = {}

    df = {}
    df["cost"] = pd.DataFrame(columns=[""], dtype=float)
    
    for j in range(no_networks):
        n = pypsa.Network(network_names[j],override_component_attrs=overrides)
        try:
            n.objective
        except:
            continue
        opts = network_names[j].split('-')

        shares = {}
        for o in opts:
            if "share" in o:
                tech = o.split('+')[0][5:]
                shares[tech] = float(o.split('+')[1][0:3])
                
        exogenous_demand = (n.loads_t.p_set[n.loads.query("carrier == 'electricity'").index].sum().sum() + 
        (n.loads.query("carrier == 'industry electricity'").p_set*len(n.snapshots)).sum()
        )
        endogenous_demand = calculate_endogenous_demand(n)
        total_demand_j = exogenous_demand + endogenous_demand.sum().sum()
        total_demand[shares["solar"],shares["wind"]] = round(total_demand_j/1e6,3) # TWh
        
        peak_load = (exogenous_demand + endogenous_demand.sum(axis=1)).max()/1e3 #GW
        
        # Calculate penetration levels (actual generation)
        solar_share_j, wind_share_j, solar_theo_share_j, wind_theo_share_j, solar_cap_factor_j, wind_cap_factor_j = calculate_renewable_penetration(n)
        
        solar_share[shares["solar"],shares["wind"]] = round(solar_share_j,1)
        wind_share[shares["solar"],shares["wind"]] = round(wind_share_j,1)
        solar_theo_share[shares["solar"],shares["wind"]] = round(solar_theo_share_j,1)
        wind_theo_share[shares["solar"],shares["wind"]] = round(wind_theo_share_j,1)
        solar_cap_factor[shares["solar"],shares["wind"]] = round(solar_cap_factor_j,1)
        wind_cap_factor[shares["solar"],shares["wind"]] = round(wind_cap_factor_j,1)
        VRE_share[shares["solar"],shares["wind"]] = round(solar_share_j + wind_share_j,1)

        # Calculate capacities (in GW)        
        solar_cap[shares["solar"],shares["wind"]] = round(n.generators.loc[pd.concat([n.generators.query("carrier == 'solar'"),n.generators.query("carrier == 'solar rooftop'")]).index].p_nom_opt.sum().sum()/1e3,1)
        wind_cap[shares["solar"],shares["wind"]] = round(n.generators[n.generators.index.str.contains("wind")].p_nom_opt.sum()/1e3,1)
        
        # calculate nodal wind capacity in DK and ES (in GW)
        DK_wind = n.generators.loc[n.generators.index[n.generators.index.str.contains("DK")][n.generators.index[n.generators.index.str.contains("DK")].str.contains("wind")]].p_nom_opt.sum()/1e3
        DK_wind_cap[shares["solar"],shares["wind"]] = round(DK_wind,1)
        ES_wind = n.generators.loc[n.generators.index[n.generators.index.str.contains("ES")][n.generators.index[n.generators.index.str.contains("ES")].str.contains("wind")]].p_nom_opt.sum()/1e3
        ES_wind_cap[shares["solar"],shares["wind"]] = round(ES_wind,1)
        
        # calculate nodal solar capacity in DK and ES (in GW)
        DK_solar = n.generators.loc[n.generators.index[n.generators.index.str.contains("DK")][n.generators.index[n.generators.index.str.contains("DK")].str.contains("solar")]].p_nom_opt.sum()/1e3
        DK_solar_cap[shares["solar"],shares["wind"]] = round(DK_solar,1)
        ES_solar = n.generators.loc[n.generators.index[n.generators.index.str.contains("ES")][n.generators.index[n.generators.index.str.contains("ES")].str.contains("solar")]].p_nom_opt.sum()/1e3
        ES_solar_cap[shares["solar"],shares["wind"]] = round(ES_solar,1)
        
        # Calculate curtailment
        solar_abs_curt_j, solar_curt_j, wind_abs_curt_j, wind_curt_j = calculate_curtailment(n,denominator_category=denominator)
        solar_abs_curt[shares["solar"],shares["wind"]] = round(sum(solar_abs_curt_j),1)
        wind_abs_curt[shares["solar"],shares["wind"]] = round(sum(wind_abs_curt_j),1)
        solar_curt[shares["solar"],shares["wind"]] = round(sum(solar_curt_j),1)
        wind_curt[shares["solar"],shares["wind"]] = round(sum(wind_curt_j),1)
        
        solar_share_j = solar_share[shares["solar"],shares["wind"]]/100
        wind_share_j = wind_share[shares["solar"],shares["wind"]]/100
        solar_curt_j_norm = sum(solar_curt_j)*solar_share_j
        wind_curt_j_norm = sum(wind_curt_j)*wind_share_j
        if denominator == "gen_theoretical":
            ren_curt_j = (solar_curt_j_norm + wind_curt_j_norm)/(solar_share_j+wind_share_j)
            #print("curtailment relative to theor. generation")
        else:
            ren_curt_j = solar_curt_j + wind_curt_j
            #print("curtailment relative to electricity demand")
        
        ren_curt[shares["solar"],shares["wind"]] = round(ren_curt_j,1) # round(sum(solar_curt_j_norm + wind_curt_j_norm),1)

        # Calculate storage share
        storage_sdes_j, storage_ldes_j, storage_H2_j = calculate_storage_dispatch(n)
        
        percentage_storage_sdes_j = storage_sdes_j/total_demand_j
        percentage_storage_ldes_j = storage_ldes_j/total_demand_j
        percentage_storage_H2_j = storage_H2_j/total_demand_j
        
        abs_storage_sdes[shares["solar"],shares["wind"]] = round(storage_sdes_j,1)
        abs_storage_ldes[shares["solar"],shares["wind"]] = round(storage_ldes_j + storage_H2_j,1)
        
        percentage_storage_sdes[shares["solar"],shares["wind"]] = round(percentage_storage_sdes_j,1)
        percentage_storage_ldes[shares["solar"],shares["wind"]] = round(percentage_storage_ldes_j + percentage_storage_H2_j,1)
    
        # Calculate H2 storage discharge capacity
        H2_discharge_capacity_j = n.links.query("carrier == 'H2 Fuel Cell'").p_nom_opt.sum().sum()/1e3 # convert to GW
        #H2_discharge_capacity[shares["solar"],shares["wind"]] = round(H2_discharge_capacity_j,1)

        # Calculate H2 storage energy capacity
        H2_energy_capacity_j = n.stores.query("carrier == 'H2'").e_nom_opt.sum().sum()/1e6 # convert to TWh
        #H2_energy_capacity[shares["solar"],shares["wind"]] = round(H2_energy_capacity_j,1)
    
        # Calculate LDES storage discharge capacity
        LDES_discharge_capacity_j = n.links.query("carrier == 'LDES discharger'").p_nom_opt.sum().sum()/1e3 # convert to GW
        LDES_discharge_capacity[shares["solar"],shares["wind"]] = round(LDES_discharge_capacity_j + H2_discharge_capacity_j,1)

        # Calculate LDES storage energy capacity
        LDES_energy_capacity_j = n.stores.query("carrier == 'LDES'").e_nom_opt.sum().sum()/1e6 # convert to TWh
        LDES_energy_capacity[shares["solar"],shares["wind"]] = round(LDES_energy_capacity_j + H2_energy_capacity_j,1)
        
        # Calculate SDES discharge power capacity
        SDES_discharge_capacity_j = n.links[n.links.index.str.contains("battery")].p_nom_opt.sum().sum()/1e3 # convert to GW
        SDES_discharge_capacity[shares["solar"],shares["wind"]] = round(SDES_discharge_capacity_j,1)

        # Calculate SDES energy capacity
        SDES_energy_capacity_j = n.stores[n.stores.index.str.contains("battery")].e_nom_opt.sum().sum()/1e6 # convert to TWh
        SDES_energy_capacity[shares["solar"],shares["wind"]] = round(SDES_energy_capacity_j,1)
        
        # Calculate volume of electricity transmission lines
        DC_vol = (n.links.query("carrier == 'DC'").length*n.links.query("carrier == 'DC'").p_nom_opt).sum()/1e6 # convert to TWkm
        AC_vol = (n.lines.query("carrier == 'AC'").length*n.lines.query("carrier == 'AC'").s_nom_opt).sum()/1e6 # convert to TWkm
        trans_vol_j = DC_vol + AC_vol
        trans_vol[shares["solar"],shares["wind"]] = round(trans_vol_j,1)
        
        # Calculate system efficiency
        electricity_demand = n.loads_t.p[n.loads[n.loads.carrier.str.contains("electricity")].index].sum().sum()
        links_wo_transmission = n.links.drop(n.links.query("carrier == 'DC'").index)
        electricity_buses = list(n.buses.query('carrier == "AC"').index) + list(n.buses.query('carrier == "low voltage"').index)
        boolean_index_generators = [n.generators.bus[i] in electricity_buses for i in range(len(n.generators.bus))]
        boolean_index_links = [n.links.bus1[i] in electricity_buses for i in range(len(links_wo_transmission.bus1))]
        s_g = pd.Series(boolean_index_generators)
        s_l = pd.Series(boolean_index_links)
        electricity_generators_g = n.generators.iloc[s_g[s_g].index]
        electricity_generators_l = links_wo_transmission.iloc[s_l[s_l].index]
        electricity_generation = n.generators_t.p[electricity_generators_g.index].sum().sum() + (-n.links_t.p1[electricity_generators_l.index].sum().sum())
        system_efficiency_j = electricity_demand/electricity_generation*100
        system_efficiency[shares["solar"],shares["wind"]] = round(system_efficiency_j,1)

        # Calculate system cost
        system_cost[shares["solar"],shares["wind"]] = round(calculate_costs(n, "", df["cost"]).sum().item()/1e9,1)
        
        # Calculate transmission peak load ratio
        transmission_capacity = (n.lines.s_nom_opt.sum() + n.links.query("carrier == 'DC'").p_nom_opt.sum())/1e3 # GW
        transmission_peak_load_ratio[shares["solar"],shares["wind"]] = round(transmission_capacity/peak_load,3)
        
        # Capacity value
        load_peak_hour = (n.loads_t.p.sum(axis=1) + endogenous_demand.sum(axis=1)).idxmax()
        wind_generators = n.generators.index[n.generators.index.str.contains('wind')]
        solar_generators = pd.concat([n.generators.query("carrier == 'solar'"),n.generators.query("carrier == 'solar rooftop'")]).index
        CV_wind_j = (n.generators_t.p_max_pu[wind_generators].loc[load_peak_hour]*n.generators.loc[wind_generators].p_nom_opt).sum()/n.generators.loc[wind_generators].p_nom_opt.sum()
        CV_solar_j = (n.generators_t.p_max_pu[solar_generators].loc[load_peak_hour]*n.generators.loc[solar_generators].p_nom_opt).sum()/n.generators.loc[solar_generators].p_nom_opt.sum()
        CV_wind[shares["solar"],shares["wind"]] = round(CV_wind_j,1)
        CV_solar[shares["solar"],shares["wind"]] = round(CV_solar_j,1)

        # Backup capacity
        df_backup_capacity, df_backup_capacity_factor_low, df_backup_capacity_factor_high= calculate_backup_generation(n)
        backup_capacity[shares["solar"],shares["wind"]] = df_backup_capacity
        backup_capacity_factor_low[shares["solar"],shares["wind"]] = df_backup_capacity_factor_low
        backup_capacity_factor_high[shares["solar"],shares["wind"]] = df_backup_capacity_factor_high

        # Check for unintended storage cycling (is computationally heavy)
        #percentage_distortion_j = calculate_storage_distortion(n)
        #percentage_distortion[shares["solar"],shares["wind"]] = round(percentage_distortion_j,1)

        print(j)

    variables_dict = {
                    # Capacities 
                      "wind_cap":wind_cap,
                      "solar_cap":solar_cap,                      
                      "wind_capacity_factor":wind_cap_factor,
                      "solar_capacity_factor":solar_cap_factor,
                      "DK_wind_cap":DK_wind_cap,
                      "DK_solar_cap":DK_solar_cap,
                      "ES_wind_cap":ES_wind_cap,
                      "ES_solar_cap":ES_solar_cap,
                      "backup_capacity":backup_capacity,
                      "backup_capacity_factor_low":backup_capacity_factor_low, # "flexible" generation
                      "backup_capacity_factor_high":backup_capacity_factor_high, # "inflexible" generation
                      
                    # electricity shares
                      "wind_share":wind_share,
                      "solar_share":solar_share,
                      "wind_theo_share":wind_theo_share,
                      "solar_theo_share":solar_theo_share,
                      "VRE_share":VRE_share,
                      "VRE_curt":ren_curt,
                    
                    # curtailment 
                      "wind_curt":wind_curt,
                      "solar_curt":solar_curt,
                      "wind_absolute_curt":wind_abs_curt,
                      "solar_absolute_curt":solar_abs_curt,
                      
                    # Storage (ldes and sdes)
                      "SDES_absolute_dispatch":abs_storage_sdes,
                      "LDES_absolute_dispatch":abs_storage_ldes,
                      "SDES_dispatch":percentage_storage_sdes,
                      "SDES_energy_cap":SDES_energy_capacity,
                      "SDES_discharge_cap":SDES_discharge_capacity,
                      "LDES_dispatch":percentage_storage_ldes,                     
                      "LDES_energy_cap":LDES_energy_capacity,
                      "LDES_discharge_cap":LDES_discharge_capacity,
                      
                    # transmission volume
                      "transmission_volume":trans_vol,
                      "transmission_peak_load_ratio":transmission_peak_load_ratio,

                    # system cost
                      "system_cost":system_cost,
                      
                    # firm capacity
                      "cv_wind":CV_wind,
                      "cv_solar":CV_solar,
                      
                    # energy demand
                      "total_demand":total_demand,
                     }

    for key in variables_dict.keys():
        df_name = " ".join(key.split("_"))
        df = pd.DataFrame(variables_dict[key].values(),index=variables_dict[key].keys(),columns=[df_name])
        df.index.set_names(["solar", "wind"],inplace=True)
        df.to_csv(RDIR + scen + "/" + key + ".csv")
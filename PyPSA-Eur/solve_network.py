# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

"""
Solves optimal operation and capacity for a network with the option to
iteratively optimize while updating line reactances.

This script is used for optimizing the electrical network as well as the
sector coupled network.

Description
-----------

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning
is provided in the
`documentation of PyPSA <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#linear-optimal-power-flow>`_.

The optimization is based on the :func:`network.optimize` function.
Additionally, some extra constraints specified in :mod:`solve_network` are added.

.. note::

    The rules ``solve_elec_networks`` and ``solve_sector_networks`` run
    the workflow for all scenarios in the configuration file (``scenario:``)
    based on the rule :mod:`solve_network`.
"""
import logging
import re
import os
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from _helpers import (
    configure_logging,
    override_component_attrs,
    update_config_with_sector_opts,
)
from vresutils.benchmark import memory_logger

logger = logging.getLogger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)


def add_land_use_constraint(n, config):
    if "m" in snakemake.wildcards.clusters:
        _add_land_use_constraint_m(n, config)
    else:
        _add_land_use_constraint(n, config)


def _add_land_use_constraint(n, config):
    # warning: this will miss existing offwind which is not classed AC-DC and has carrier 'offwind'

    for carrier in ["solar", "onwind", "offwind-ac", "offwind-dc"]:
        ext_i = (n.generators.carrier == carrier) & ~n.generators.p_nom_extendable
        existing = (
            n.generators.loc[ext_i, "p_nom"]
            .groupby(n.generators.bus.map(n.buses.location))
            .sum()
        )
        existing.index += " " + carrier + "-" + snakemake.wildcards.planning_horizons
        n.generators.loc[existing.index, "p_nom_max"] -= existing

    # check if existing capacities are larger than technical potential
    existing_large = n.generators[
        n.generators["p_nom_min"] > n.generators["p_nom_max"]
    ].index
    if len(existing_large):
        logger.warning(
            f"Existing capacities larger than technical potential for {existing_large},\
                       adjust technical potential to existing capacities"
        )
        n.generators.loc[existing_large, "p_nom_max"] = n.generators.loc[
            existing_large, "p_nom_min"
        ]

    n.generators.p_nom_max.clip(lower=0, inplace=True)


def _add_land_use_constraint_m(n, config):
    # if generators clustering is lower than network clustering, land_use accounting is at generators clusters

    planning_horizons = config["scenario"]["planning_horizons"]
    grouping_years = config["existing_capacities"]["grouping_years"]
    current_horizon = snakemake.wildcards.planning_horizons

    for carrier in ["solar", "onwind", "offwind-ac", "offwind-dc"]:
        existing = n.generators.loc[n.generators.carrier == carrier, "p_nom"]
        ind = list(
            set(
                [
                    i.split(sep=" ")[0] + " " + i.split(sep=" ")[1]
                    for i in existing.index
                ]
            )
        )

        previous_years = [
            str(y)
            for y in planning_horizons + grouping_years
            if y < int(snakemake.wildcards.planning_horizons)
        ]

        for p_year in previous_years:
            ind2 = [
                i for i in ind if i + " " + carrier + "-" + p_year in existing.index
            ]
            sel_current = [i + " " + carrier + "-" + current_horizon for i in ind2]
            sel_p_year = [i + " " + carrier + "-" + p_year for i in ind2]
            n.generators.loc[sel_current, "p_nom_max"] -= existing.loc[
                sel_p_year
            ].rename(lambda x: x[:-4] + current_horizon)

    n.generators.p_nom_max.clip(lower=0, inplace=True)


def add_co2_sequestration_limit(n, limit=200):
    """
    Add a global constraint on the amount of Mt CO2 that can be sequestered.
    """
    n.carriers.loc["co2 stored", "co2_absorptions"] = -1
    n.carriers.co2_absorptions = n.carriers.co2_absorptions.fillna(0)

    limit = limit * 1e6
    for o in opts:
        if "seq" not in o:
            continue
        limit = float(o[o.find("seq") + 3 :]) * 1e6
        break

    n.add(
        "GlobalConstraint",
        "co2_sequestration_limit",
        sense="<=",
        constant=limit,
        type="primary_energy",
        carrier_attribute="co2_absorptions",
    )


def prepare_network(n, solve_opts=None, config=None):
    if "clip_p_max_pu" in solve_opts:
        for df in (
            n.generators_t.p_max_pu,
            n.generators_t.p_min_pu,  # TODO: check if this can be removed
            n.storage_units_t.inflow,
        ):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    if solve_opts.get("load_shedding"):
        # intersect between macroeconomic and surveybased willingness to pay
        # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
        # TODO: retrieve color and nice name from config
        n.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")
        buses_i = n.buses.query("carrier == 'AC'").index
        if not np.isscalar(load_shedding):
            # TODO: do not scale via sign attribute (use Eur/MWh instead of Eur/kWh)
            load_shedding = 1e2  # Eur/kWh

        n.madd(
            "Generator",
            buses_i,
            " load",
            bus=n.buses.index,
            carrier="load",
            sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=load_shedding,  # Eur/kWh
            p_nom=1e9,  # kW
        )

    if solve_opts.get("noisy_costs"):
        for t in n.iterate_components():
            # if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if "marginal_cost" in t.df:
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (
                    np.random.random(len(t.df)) - 0.5
                )

        for t in n.iterate_components(["Line", "Link"]):
            t.df["capital_cost"] += (
                1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)
            ) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    if config["foresight"] == "myopic":
        add_land_use_constraint(n, config)

    if n.stores.carrier.eq("co2 stored").any():
        limit = config["sector"].get("co2_sequestration_potential", 200)
        add_co2_sequestration_limit(n, limit=limit)

    return n


def add_CCL_constraints(n, config):
    """
    Add CCL (country & carrier limit) constraint to the network.

    Add minimum and maximum levels of generator nominal capacity per carrier
    for individual countries. Opts and path for agg_p_nom_minmax.csv must be defined
    in config.yaml. Default file is available at data/agg_p_nom_minmax.csv.

    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Example
    -------
    scenario:
        opts: [Co2L-CCL-24H]
    electricity:
        agg_p_nom_limits: data/agg_p_nom_minmax.csv
    """
    agg_p_nom_minmax = pd.read_csv(
        config["electricity"]["agg_p_nom_limits"], index_col=[0, 1]
    )
    logger.info("Adding generation capacity constraints per carrier and country")
    p_nom = n.model["Generator-p_nom"]

    gens = n.generators.query("p_nom_extendable").rename_axis(index="Generator-ext")
    grouper = [gens.bus.map(n.buses.country), gens.carrier]
    grouper = xr.DataArray(pd.MultiIndex.from_arrays(grouper), dims=["Generator-ext"])
    lhs = p_nom.groupby(grouper).sum().rename(bus="country")

    minimum = xr.DataArray(agg_p_nom_minmax["min"].dropna()).rename(dim_0="group")
    index = minimum.indexes["group"].intersection(lhs.indexes["group"])
    if not index.empty:
        n.model.add_constraints(
            lhs.sel(group=index) >= minimum.loc[index], name="agg_p_nom_min"
        )

    maximum = xr.DataArray(agg_p_nom_minmax["max"].dropna()).rename(dim_0="group")
    index = maximum.indexes["group"].intersection(lhs.indexes["group"])
    if not index.empty:
        n.model.add_constraints(
            lhs.sel(group=index) <= maximum.loc[index], name="agg_p_nom_max"
        )


def add_EQ_constraints(n, o, scaling=1e-1):
    """
    Add equity constraints to the network.

    Currently this is only implemented for the electricity sector only.

    Opts must be specified in the config.yaml.

    Parameters
    ----------
    n : pypsa.Network
    o : str

    Example
    -------
    scenario:
        opts: [Co2L-EQ0.7-24H]

    Require each country or node to on average produce a minimal share
    of its total electricity consumption itself. Example: EQ0.7c demands each country
    to produce on average at least 70% of its consumption; EQ0.7 demands
    each node to produce on average at least 70% of its consumption.
    """
    # TODO: Generalize to cover myopic and other sectors?
    float_regex = "[0-9]*\.?[0-9]+"
    level = float(re.findall(float_regex, o)[0])
    if o[-1] == "c":
        ggrouper = n.generators.bus.map(n.buses.country).to_xarray()
        lgrouper = n.loads.bus.map(n.buses.country).to_xarray()
        sgrouper = n.storage_units.bus.map(n.buses.country).to_xarray()
    else:
        ggrouper = n.generators.bus.to_xarray()
        lgrouper = n.loads.bus.to_xarray()
        sgrouper = n.storage_units.bus.to_xarray()
    load = (
        n.snapshot_weightings.generators
        @ n.loads_t.p_set.groupby(lgrouper, axis=1).sum()
    )
    inflow = (
        n.snapshot_weightings.stores
        @ n.storage_units_t.inflow.groupby(sgrouper, axis=1).sum()
    )
    inflow = inflow.reindex(load.index).fillna(0.0)
    rhs = scaling * (level * load - inflow)
    p = n.model["Generator-p"]
    lhs_gen = (
        (p * (n.snapshot_weightings.generators * scaling))
        .groupby(ggrouper)
        .sum()
        .sum("snapshot")
    )
    # TODO: double check that this is really needed, why do have to subtract the spillage
    if not n.storage_units_t.inflow.empty:
        spillage = n.model["StorageUnit-spill"]
        lhs_spill = (
            (spillage * (-n.snapshot_weightings.stores * scaling))
            .groupby(sgrouper)
            .sum()
            .sum("snapshot")
        )
        lhs = lhs_gen + lhs_spill
    else:
        lhs = lhs_gen
    n.model.add_constraints(lhs >= rhs, name="equity_min")


def add_BAU_constraints(n, config):
    """
    Add a per-carrier minimal overall capacity.

    BAU_mincapacities and opts must be adjusted in the config.yaml.

    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Example
    -------
    scenario:
        opts: [Co2L-BAU-24H]
    electricity:
        BAU_mincapacities:
            solar: 0
            onwind: 0
            OCGT: 100000
            offwind-ac: 0
            offwind-dc: 0
    Which sets minimum expansion across all nodes e.g. in Europe to 100GW.
    OCGT bus 1 + OCGT bus 2 + ... > 100000
    """
    mincaps = pd.Series(config["electricity"]["BAU_mincapacities"])
    p_nom = n.model["Generator-p_nom"]
    ext_i = n.generators.query("p_nom_extendable")
    ext_carrier_i = xr.DataArray(ext_i.carrier.rename_axis("Generator-ext"))
    lhs = p_nom.groupby(ext_carrier_i).sum()
    index = mincaps.index.intersection(lhs.indexes["carrier"])
    rhs = mincaps[index].rename_axis("carrier")
    n.model.add_constraints(lhs >= rhs, name="bau_mincaps")


# TODO: think about removing or make per country
def add_SAFE_constraints(n, config):
    """
    Add a capacity reserve margin of a certain fraction above the peak demand.
    Renewable generators and storage do not contribute. Ignores network.

    Parameters
    ----------
        n : pypsa.Network
        config : dict

    Example
    -------
    config.yaml requires to specify opts:

    scenario:
        opts: [Co2L-SAFE-24H]
    electricity:
        SAFE_reservemargin: 0.1
    Which sets a reserve margin of 10% above the peak demand.
    """
    peakdemand = n.loads_t.p_set.sum(axis=1).max()
    margin = 1.0 + config["electricity"]["SAFE_reservemargin"]
    reserve_margin = peakdemand * margin
    # TODO: do not take this from the plotting config!
    conv_techs = config["plotting"]["conv_techs"]
    ext_gens_i = n.generators.query("carrier in @conv_techs & p_nom_extendable").index
    p_nom = n.model["Generator-p_nom"].loc[ext_gens_i]
    lhs = p_nom.sum()
    exist_conv_caps = n.generators.query(
        "~p_nom_extendable & carrier in @conv_techs"
    ).p_nom.sum()
    rhs = reserve_margin - exist_conv_caps
    n.model.add_constraints(lhs >= rhs, name="safe_mintotalcap")


def add_operational_reserve_margin(n, sns, config):
    """
    Build reserve margin constraints based on the formulation given in
    https://genxproject.github.io/GenX/dev/core/#Reserves.

    Parameters
    ----------
        n : pypsa.Network
        sns: pd.DatetimeIndex
        config : dict

    Example:
    --------
    config.yaml requires to specify operational_reserve:
    operational_reserve: # like https://genxproject.github.io/GenX/dev/core/#Reserves
        activate: true
        epsilon_load: 0.02 # percentage of load at each snapshot
        epsilon_vres: 0.02 # percentage of VRES at each snapshot
        contingency: 400000 # MW
    """
    reserve_config = config["electricity"]["operational_reserve"]
    EPSILON_LOAD = reserve_config["epsilon_load"]
    EPSILON_VRES = reserve_config["epsilon_vres"]
    CONTINGENCY = reserve_config["contingency"]

    # Reserve Variables
    n.model.add_variables(
        0, np.inf, coords=[sns, n.generators.index], name="Generator-r"
    )
    reserve = n.model["Generator-r"]
    lhs = reserve.sum("Generator")

    # Share of extendable renewable capacities
    ext_i = n.generators.query("p_nom_extendable").index
    vres_i = n.generators_t.p_max_pu.columns
    if not ext_i.empty and not vres_i.empty:
        capacity_factor = n.generators_t.p_max_pu[vres_i.intersection(ext_i)]
        p_nom_vres = (
            n.model["Generator-p_nom"]
            .loc[vres_i.intersection(ext_i)]
            .rename({"Generator-ext": "Generator"})
        )
        lhs = lhs + (p_nom_vres * (-EPSILON_VRES * capacity_factor)).sum()

    # Total demand per t
    demand = n.loads_t.p_set.sum(axis=1)

    # VRES potential of non extendable generators
    capacity_factor = n.generators_t.p_max_pu[vres_i.difference(ext_i)]
    renewable_capacity = n.generators.p_nom[vres_i.difference(ext_i)]
    potential = (capacity_factor * renewable_capacity).sum(axis=1)

    # Right-hand-side
    rhs = EPSILON_LOAD * demand + EPSILON_VRES * potential + CONTINGENCY

    n.model.add_constraints(lhs >= rhs, name="reserve_margin")

    reserve = n.model["Generator-r"]

    lhs = n.model.constraints["Generator-fix-p-upper"].lhs
    lhs = lhs + reserve.loc[:, lhs.coords["Generator-fix"]].drop("Generator")
    rhs = n.model.constraints["Generator-fix-p-upper"].rhs
    n.model.add_constraints(lhs <= rhs, name="Generator-fix-p-upper-reserve")

    lhs = n.model.constraints["Generator-ext-p-upper"].lhs
    lhs = lhs + reserve.loc[:, lhs.coords["Generator-ext"]].drop("Generator")
    rhs = n.model.constraints["Generator-ext-p-upper"].rhs
    n.model.add_constraints(lhs >= rhs, name="Generator-ext-p-upper-reserve")


def add_battery_constraints(n):
    """
    Add constraint ensuring that charger = discharger, i.e.
    1 * charger_size - efficiency * discharger_size = 0
    """
    if not n.links.p_nom_extendable.any():
        return

    discharger_bool = n.links.index.str.contains("battery discharger")
    charger_bool = n.links.index.str.contains("battery charger")

    dischargers_ext = n.links[discharger_bool].query("p_nom_extendable").index
    chargers_ext = n.links[charger_bool].query("p_nom_extendable").index

    eff = n.links.efficiency[dischargers_ext].values
    lhs = (
        n.model["Link-p_nom"].loc[chargers_ext]
        - n.model["Link-p_nom"].loc[dischargers_ext] * eff
    )

    n.model.add_constraints(lhs == 0, name="Link-charger_ratio")

def add_renewable_potential_target(n,tech,share,sectors=""):
    # Exogenous electricity load
    load = (n.loads_t.p_set[n.loads.query("carrier == 'electricity'").index].sum().sum() + 
        (n.loads.query("carrier == 'industry electricity'").p_set*len(n.snapshots)).sum()
    )

    # Additional electricity demand from sector-coupling
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

    if tech == "wind":
        wind_generators = n.generators.index[n.generators.index.str.contains("wind")]
        lhs_wind_i = []
        for i in range(len(wind_generators)):
            lhs_wind_i.append(
                n.model.variables["Generator-p_nom"][wind_generators[i]]
                * n.generators_t.p_max_pu[wind_generators[i]].sum()
            )
        wind_lhs = sum(lhs_wind_i)
        print("Adding capacity of wind eq. to ", share, " of elec. load before curtailment")

        if len(sectors) == 0:
            n.model.add_constraints(
                wind_lhs == share * load, name="wind potential share equality constraint"
            )
        else:
            elec_demand_via_links_sum = []
            for i in range(len(elec_demand_via_links)):
                elec_demand_via_links_sum.append(
                    n.model.variables["Link-p"]
                    .sel(Link=elec_demand_via_links.iloc[i].name)
                    .sum()
                )

            rhs_links = sum(elec_demand_via_links_sum)
            n.model.add_constraints(
                wind_lhs.to_linexpr() - share * rhs_links == share * load,
                name="wind potential share equality constraint",
            )

    if tech == "solar":
        solar_generators = pd.concat(
            [
                n.generators.query("carrier == 'solar'"),
                n.generators.query("carrier == 'solar rooftop'"),
            ]
        ).index
        lhs_solar_i = []
        for i in range(len(solar_generators)):
            lhs_solar_i.append(
                n.model.variables["Generator-p_nom"][solar_generators[i]]
                * n.generators_t.p_max_pu[solar_generators[i]].sum()
            )
        solar_lhs = sum(lhs_solar_i)
        print(
            "Adding capacity of solar eq. to ", share, " of elec. load before curtailment"
        )

        if len(sectors) == 0:
            n.model.add_constraints(
                solar_lhs == share * load, name="solar potential share equality constraint"
            )
        else:
            elec_demand_via_links_sum = []
            for i in range(len(elec_demand_via_links)):
                elec_demand_via_links_sum.append(
                    n.model.variables["Link-p"]
                    .sel(Link=elec_demand_via_links.iloc[i].name)
                    .sum()
                )

            rhs_links = sum(elec_demand_via_links_sum)
            n.model.add_constraints(
                solar_lhs.to_linexpr() - share * rhs_links == share * load,
                name="solar potential share equality constraint",
            )


def add_renewable_generation_target(n,solarshare,windshare):
    """
    Add constraint that enforces a minimum VRE penetration
    in percentage of total electricity load
    """
    load = n.loads_t.p_set[n.loads.query('carrier == "electricity"').index].sum().sum()
    generators = n.generators.carrier.to_xarray()

    wind_lhs1 = n.model.variables["Generator-p"].groupby(generators).sum().sel(carrier='onwind').sum()  
    wind_lhs2 = n.model.variables["Generator-p"].groupby(generators).sum().sel(carrier='offwind-dc').sum()
    wind_lhs3 = n.model.variables["Generator-p"].groupby(generators).sum().sel(carrier='offwind-ac').sum()
    wind_lhs = wind_lhs1 + wind_lhs2 + wind_lhs3

    solar_lhs1 = n.model.variables["Generator-p"].groupby(generators).sum().sel(carrier='solar').sum()
    solar_lhs2 = n.model.variables["Generator-p"].groupby(generators).sum().sel(carrier='solar rooftop').sum()
    solar_lhs = solar_lhs1 + solar_lhs2

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # To avoid storage distortion, one way is to include storage losses in the constraint.
    # battery_charger = n.links.index[n.links.index.str.contains('battery charger')]
    # battery_discharger = n.links.index[n.links.index.str.contains('battery discharger')]
    # eta_bat = n.links.loc[battery_discharger].efficiency.iloc[0]
    # storage_losses_1 = (n.model.variables["Link-p"].sel(Link=battery_charger) - eta_bat*n.model.variables["Link-p"].sel(Link=battery_discharger)).sum()

    # H2_charger = n.links.index[n.links.index.str.contains('H2 Electrolysis')]
    # H2_discharger = n.links.index[n.links.index.str.contains('H2 Fuel Cell')]
    # eta_h2 = n.links.loc[H2_discharger].efficiency.iloc[0]
    # storage_losses_2 = (n.model.variables["Link-p"].sel(Link=H2_charger) - eta_h2*n.model.variables["Link-p"].sel(Link=H2_discharger)).sum()

    # storage_losses = storage_losses_1 + storage_losses_2
    
    # wind_rhs1 = windshare*load
    # wind_rhs2 = windshare*storage_losses
    
    # solar_rhs1 = solarshare*load
    # solar_rhs2 = solarshare*storage_losses

    # wind_rhs = wind_rhs1 + wind_rhs2
    # solar_rhs = solar_rhs1 + solar_rhs2
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    n.model.add_constraints(wind_lhs >= windshare*load, name="wind minimum generation share constraint")
    n.model.add_constraints(solar_lhs >= solarshare*load, name="solar minimum generation share constraint")

def add_chp_constraints(n):
    electric = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("electric")
    )
    heat = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("heat")
    )

    electric_ext = n.links[electric].query("p_nom_extendable").index
    heat_ext = n.links[heat].query("p_nom_extendable").index

    electric_fix = n.links[electric].query("~p_nom_extendable").index
    heat_fix = n.links[heat].query("~p_nom_extendable").index

    p = n.model["Link-p"]  # dimension: [time, link]

    # output ratio between heat and electricity and top_iso_fuel_line for extendable
    if not electric_ext.empty:
        p_nom = n.model["Link-p_nom"]

        lhs = (
            p_nom.loc[electric_ext]
            * (n.links.p_nom_ratio * n.links.efficiency)[electric_ext].values
            - p_nom.loc[heat_ext] * n.links.efficiency[heat_ext].values
        )
        n.model.add_constraints(lhs == 0, name="chplink-fix_p_nom_ratio")

        rename = {"Link-ext": "Link"}
        lhs = (
            p.loc[:, electric_ext]
            + p.loc[:, heat_ext]
            - p_nom.rename(rename).loc[electric_ext]
        )
        n.model.add_constraints(lhs <= 0, name="chplink-top_iso_fuel_line_ext")

    # top_iso_fuel_line for fixed
    if not electric_fix.empty:
        lhs = p.loc[:, electric_fix] + p.loc[:, heat_fix]
        rhs = n.links.p_nom[electric_fix]
        n.model.add_constraints(lhs <= rhs, name="chplink-top_iso_fuel_line_fix")

    # back-pressure
    if not electric.empty:
        lhs = (
            p.loc[:, heat] * (n.links.efficiency[heat] * n.links.c_b[electric].values)
            - p.loc[:, electric] * n.links.efficiency[electric]
        )
        n.model.add_constraints(lhs <= rhs, name="chplink-backpressure")


def add_pipe_retrofit_constraint(n):
    """
    Add constraint for retrofitting existing CH4 pipelines to H2 pipelines.
    """
    gas_pipes_i = n.links.query("carrier == 'gas pipeline' and p_nom_extendable").index
    h2_retrofitted_i = n.links.query(
        "carrier == 'H2 pipeline retrofitted' and p_nom_extendable"
    ).index

    if h2_retrofitted_i.empty or gas_pipes_i.empty:
        return

    p_nom = n.model["Link-p_nom"]

    CH4_per_H2 = 1 / n.config["sector"]["H2_retrofit_capacity_per_CH4"]
    lhs = p_nom.loc[gas_pipes_i] + CH4_per_H2 * p_nom.loc[h2_retrofitted_i]
    rhs = n.links.p_nom[gas_pipes_i].rename_axis("Link-ext")

    n.model.add_constraints(lhs == rhs, name="Link-pipe_retrofit")


def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to
    ``pypsa.optimization.optimize``.

    If you want to enforce additional custom constraints, this is a good
    location to add them. The arguments ``opts`` and
    ``snakemake.config`` are expected to be attached to the network.
    """
    opts = n.opts
    config = n.config
    if "BAU" in opts and n.generators.p_nom_extendable.any():
        add_BAU_constraints(n, config)
    if "SAFE" in opts and n.generators.p_nom_extendable.any():
        add_SAFE_constraints(n, config)
    if "CCL" in opts and n.generators.p_nom_extendable.any():
        add_CCL_constraints(n, config)
    reserve = config["electricity"].get("operational_reserve", {})
    if reserve.get("activate"):
        add_operational_reserve_margin(n, snapshots, config)
    print("opts: ", opts)
    
    sectors_list = []
    for o in opts:
        if o in ["T","H","I"]:
            sectors_list.append(o)

    sectors = "-".join(sectors_list)

    for o in opts:
        print("opts[o]: ", o)
        if "EQ" in o:
            add_EQ_constraints(n, o)
        if "share" in o:
            share = float(o.split("+")[1])
            tech = o.split("+")[0][5:]
            add_renewable_potential_target(n,tech,share,sectors)

    add_battery_constraints(n)
    add_pipe_retrofit_constraint(n)

def solve_network(n, config, opts="",tmpdir='/tmp/', **kwargs):
    set_of_options = config["solving"]["solver"]["options"]
    solver_options = (
        config["solving"]["solver_options"][set_of_options] if set_of_options else {}
    )
    solver_name = config["solving"]["solver"]["name"]
    cf_solving = config["solving"]["options"]
    track_iterations = cf_solving.get("track_iterations", False)
    min_iterations = cf_solving.get("min_iterations", 4)
    max_iterations = cf_solving.get("max_iterations", 6)

    # add to network for extra_functionality
    n.config = config
    n.opts = opts

    skip_iterations = cf_solving.get("skip_iterations", False)
    if not n.lines.s_nom_extendable.any():
        skip_iterations = True
        logger.info("No expandable lines found. Skipping iterative solving.")

    if skip_iterations:
        status, condition = n.optimize(
            solver_name=solver_name,
            model_kwargs={"solver_dir":tmpdir},
            extra_functionality=extra_functionality,
            **solver_options,
            **kwargs,
        )
    else:
        status, condition = n.optimize.optimize_transmission_expansion_iteratively(
            solver_name=solver_name,
            track_iterations=track_iterations,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            extra_functionality=extra_functionality,
            **solver_options,
            **kwargs,
        )

    if status != "ok":
        logger.warning(
            f"Solving status '{status}' with termination condition '{condition}'"
        )
    if "infeasible" in condition:
        raise RuntimeError("Solving status 'infeasible'")

    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_sector_network",
            configfiles="test/config.overnight.yaml",
            simpl="",
            opts="",
            clusters="5",
            ll="v1.5",
            sector_opts="CO2L0-24H-T-H-B-I-A-solar+p3-dist1",
            planning_horizons="2030",
        )
    configure_logging(snakemake)
    if "sector_opts" in snakemake.wildcards.keys():
        update_config_with_sector_opts(
            snakemake.config, snakemake.wildcards.sector_opts
        )

    tmpdir_scratch = '/scratch/' + os.environ['SLURM_JOB_ID']
    if tmpdir_scratch is not None:
        from pathlib import Path
        Path(tmpdir_scratch).mkdir(parents=True, exist_ok=True)

    opts = snakemake.wildcards.opts
    if "sector_opts" in snakemake.wildcards.keys():
        opts += "-" + snakemake.wildcards.sector_opts
    opts = [o for o in opts.split("-") if o != ""]

    print("opts: ", opts)

    solve_opts = snakemake.config["solving"]["options"]

    np.random.seed(solve_opts.get("seed", 123))

    fn = getattr(snakemake.log, "memory", None)
    with memory_logger(filename=fn, interval=30.0) as mem:
        if "overrides" in snakemake.input.keys():
            overrides = override_component_attrs(snakemake.input.overrides)
            n = pypsa.Network(
                snakemake.input.network, override_component_attrs=overrides
            )
        else:
            n = pypsa.Network(snakemake.input.network)

        n = prepare_network(n, solve_opts, config=snakemake.config)

        n = solve_network(
            n, 
            config=snakemake.config, 
            opts=opts, 
            tmpdir=tmpdir_scratch,
            log_fn=snakemake.log.solver
        )

        n.model.to_netcdf(snakemake.output[0][:-3]+'_linopy.nc')

        n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
        n.export_to_netcdf(snakemake.output[0])

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))

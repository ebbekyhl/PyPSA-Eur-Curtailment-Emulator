# -*- coding: utf-8 -*-
"""
Created on 29th of November 2023
Author: Ebbe Kyhl Gøtske

This script contains functions for plotting the results 
from the scenarios obtained in PyPSA-Eur.
"""
import pandas as pd
import glob
from matplotlib import pyplot as plt, colors
import cartopy.crs as ccrs
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle, Ellipse
import numpy as np
from itertools import product

fs = 15
plt.style.use('seaborn-ticks')
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['axes.axisbelow'] = True

var_name = {"wind share":"wind share \n (% of electricity demand)",
             "solar share":"solar share \n (% of electricity demand)",
             "wind capacity factor":"wind capacity factor \n (%)",
             "solar capacity factor":"solar capacity factor \n (%)",
             "wind theo share":"wind theo share \n (% of electricity demand)",
             "solar theo share":"solar theo share \n (% of electricity demand)",
             "wind cap":"wind capacity (GW)",
             "solar cap":"solar capacity (GW)",
             "wind curt":"curtailment of " + r"$\bf{wind}$" + " \n (% of wind resources)",
             "solar curt":"curtailment of " + r"$\bf{solar}$" +  " \n (% of solar resources)",
             "SDES dispatch":"SDES dispatch \n (% of electricity demand)",
             "SDES energy cap":"SDES energy capacity (TWh)",
             "SDES discharge cap":"SDES capacity \n (GW)",
             "LDES dispatch":"LDES dispatch \n (% of electricity demand)",                  
             "LDES energy cap":"LDES capacity (TWh)",
             "LDES discharge cap":"LDES capacity (GW)",
             "H2 dispatch":"H2 Fuel cell dispatch \n (% of electricity demand)",
             "H2 energy cap":"H2 storage capacity (TWh)",
             "H2 discharge cap":"H2 Fuel cell capacity (GW)",
             "transmission volume":"transmission volume (TWkm)",
             "system efficiency":"system efficiency (%)",
             "system cost":"system cost (bEUR)",
             "backup capacity":"backup capacity [GW]",
             "cv wind":"capacity value wind [-]",
             "cv solar":"capacity value solar [-]",
             "backup capacity factor low":"backup capacity factor low [%]",
             "backup capacity factor high":"backup capacity factor high [%]",
             "VRE share":"VRE share \n (% of electricity demand)",
             "VRE curt":"VRE curtailment \n (% of resources)",
             "DK wind cap":"DK wind capacity [GW]",
             "DK solar cap":"DK solar capacity [GW]",
             "ES wind cap":"ES wind capacity [GW]",
             "ES solar cap":"ES solar capacity [GW]",
             "total demand": "total electricity demand [TWh]",
            }

variables = {"wind share":"solar ",
             "solar share":"wind ",
             "wind capacity factor":"solar ",
             "solar capacity factor":"wind ",
             "wind theo share":"solar ",
             "solar theo share":"wind ",
             "VRE share":"solar ",
             "VRE curt":"solar ",
             "wind cap":"solar ",
             "solar cap":"wind ",
             "wind curt":"solar ",
             "wind absolute curt": "solar",
             "solar curt":"wind ",
             "solar absolute curt":"wind",
             "SDES dispatch":"wind ",
             "SDES energy cap":"wind ",
             "SDES discharge cap":"wind ",
             "LDES dispatch":"solar ",                     
             "LDES energy cap":"solar ",
             "LDES discharge cap":"solar ",
             "H2 dispatch":"solar ",                     
             "H2 energy cap":"solar ",
             "H2 discharge cap":"solar ",
             "cv wind":"solar",
             "cv solar":"wind",
             "transmission volume":"wind ",
             "system efficiency":"solar ",
             "system cost":"solar ",
             "backup capacity":"solar ",
             "backup capacity factor low":"solar ",
             "backup capacity factor high":"solar ",
             "DK wind cap":"solar ",
             "ES wind cap":"solar ",
             "DK solar cap":"solar ",
             "ES solar cap":"solar ",
             "total demand":"wind ",
            }

colors_dict = {"wind share":"Greens",
             "solar share":"Greens",
             "wind cap":"Greens",
             "solar cap":"Greens",
             "wind curt":"Reds",
             "solar curt":"Reds",
             "wind absolute curt":"Reds",
             "solar absolute curt":"Reds",
             "SDES dispatch":"Blues",
             "SDES energy cap":"Blues",
             "SDES discharge cap":"Blues",
             "LDES dispatch":"Purples",     
             "LDES energy cap":"Purples", 
             "LDES discharge cap":"Purples", 
             "H2 dispatch":"Purples", 
             "H2 energy cap":"Purples", 
             "H2 discharge cap":"Purples", 
             "transmission volume":"Greens",
             "system efficiency":"Oranges",
             "system cost":"Oranges",
             "VRE share":"Greens",
             "VRE curt":"Reds",
             "backup capacity":"Reds",
              "backup capacity factor high":"Reds",
              "backup capacity factor low":"Reds",
             "cv wind":"Greens",
             "cv solar":"Greens",
            }

labels = {"wind share":"wind share \n (% of electricity demand)",
         "solar share":"solar share \n (% of electricity demand)",
         "wind cap":"wind capacity (GW)",
         "solar cap":"solar capacity (GW)",
         "wind curt":"curtailment of wind \n (% of resources)",
         "solar curt":"curtailment of solar \n (% of resources)",
         "SDES dispatch":"SDES dispatch \n (% of electricity demand)",
         "SDES energy cap":"SDES energy capacity (TWh)",
         "SDES discharge cap":"SDES capacity \n (GW)",
         "LDES dispatch":"LDES dispatch \n (% of electricity demand)",                  
         "LDES energy cap":"LDES capacity (TWh)",
         "LDES discharge cap":"LDES capacity (GW)",
         "H2 dispatch":"H2 Fuel cell dispatch \n (% of electricity demand)",
         "H2 energy cap":"H2 storage capacity (TWh)",
         "H2 discharge cap":"H2 Fuel cell capacity (GW)",
         "transmission volume":"transmission volume (TWkm)",
         "system efficiency":"system efficiency (%)",
         "system cost":"system cost (bEUR)",
         "VRE share":"VRE share \n (% of electricity demand)",
         "VRE curt":"VRE curtailment \n (% of electricity demand)",
         "backup capacity":"Backup capacity [GW]",
         "backup capacity factor high":"Backup capacity factor high [%]",
          "backup capacity factor low":"Backup capacity factor low [%]",
          "cv wind":"capacity value wind [-]",
         "cv solar":"capacity value solar [-]",
        }

def mesh_color_2(scens,variables,RDIR,vmin,vmax, norm_w_demand=False, equidistant=False):
    
    variables_read = variables["read"]
    variables_plot = variables["plot"]
    df = read_and_plot_scenarios(scens,variables_read,RDIR,plot=False)
    figs = {}
    for j in range(len(variables_plot)):

        if variables_plot[j] not in colors_dict.keys():
            colors_dict.update({variables_plot[j]:"Greens"})

        if variables_plot[j] not in labels.keys():
            labels.update({variables_plot[j]:variables_plot[j]})
        
        for scen in scens:
            df_plot = df[scen][variables_plot[j]]

            if norm_w_demand:
                df_demand = df[scen]["total demand"]
                df_plot = df_plot/(df_demand*1e6)*100 # curtailment is in units of MWh and demand in units of TWh
                label = "el. demand"
            else:
                label = "resources"

            fig, ax, cb = meshcolor(df_plot,
                                    vmax=vmax,
                                    vmin=vmin,
                                    colormap = colors_dict[variables_plot[j]], 
                                    shading = "nearest",
                                    label=labels[variables_plot[j]],
                                    equidistant=equidistant)
            
            ax.set_xlabel("wind share (% of " + label + ")")
            ax.set_ylabel("solar share (% of " + label + ")")

            cb.set_label(variables_plot[j][0:5] + " curtailment (% of " + label + ")")
            # add cell values to the plot
            df_plot_2D = df_plot.unstack()

            if not equidistant:
                df_plot_2D.index = pd.Index([0,1,2,3,4,5,6])
                df_plot_2D.columns = pd.Index([0,1,2,3,4,5,6])
            else:
                df_plot_2D.index = pd.Index([0,2,3,4,5,6,8])
                df_plot_2D.columns = pd.Index([0,2,3,4,5,6,8])

            for i in range(len(df_plot_2D.index)):
                for k in range(len(df_plot_2D.columns)):
                    c = df_plot_2D.iloc[i,k]
                    if c != '' and c != np.nan and c > 0:
                        ax.text(df_plot_2D.columns[k], 
                                df_plot_2D.index[i], 
                                "{:.1f}".format(c), 
                                va='center', ha='center')

            figs[scen,j] = fig

    return df, figs


def plot_scenario_effects(df,var,test,base_case,ylim="",include_message=True):    
    
    test_effect = df[test][var] - df[base_case][var]

    cmap = plt.cm.Greens
    norm = colors.Normalize(vmin=-0.5, vmax=1)

    ###########################################################################################
    fig, ax = plt.subplots()
    ax.set_title(test)
    if "solar" in var or "SDES" in var  or "transmission" in var:
        test_effect = test_effect.reorder_levels(['wind','solar']).sort_index()

    lvls = list(test_effect.index.unique(level=0))
    for lvl in lvls:
        test_effect.loc[lvl].plot(ax=ax,color=cmap(norm(lvl)),marker="o",lw=1,label=variables[var] + str(int(100*lvl)) + "%")
    
    leg_x_coord = 1.15
    
    if var == "solar curt" and "SDES" in test and include_message == True:
        ax.plot([0.1,0.3,0.4,0.5,0.6,0.7,0.9],[0,-16,-19,-20,-20,-19,-17],"k--",label="MESSAGEix-GLOBIOM \n w. storage determined \n in PyPSA",marker="o",lw=2)
        leg_x_coord = 1.3
        
    if var == "wind curt" and "LDES" in test and include_message == True:
        ax.plot([0.1,0.3,0.4,0.5,0.6,0.7,0.9],[0,0,-1,-1,-1,-1,-1],"k--",label="MESSAGEix-GLOBIOM \n w. storage determined \n in PyPSA",marker="o",lw=2)
        leg_x_coord = 1.3
        
    ax.set_xlabel(ax.get_xlabel() + " resources (% of electricity demand)")
    ax.set_xticks(lvls)
    ax.set_xticklabels((np.array(lvls)*100).astype(int))
    ax.set_ylabel("Change in " + var_name[var])        

    if type(ylim) == list:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(ax.get_ylim())
    
    ax.grid()
    fig.legend(prop={"size":12},ncol=1,bbox_to_anchor=(leg_x_coord, 0.9),frameon=True)

def read_and_plot_scenarios(scens,
                            variables_input="",
                            ylim="",
                            plot_message_curt=True,
                            plot_optimum_mix=True,
                            directory="calculated_metrics",
                            plot=True,
                            savefig=False,
                            title=True):

    dic_lst = [{i:{}} for i in scens]
    
    variable_plot_dic = {}
    for d in dic_lst:
        variable_plot_dic.update(d)

    df_dict = {}
    scen_count = 0
    # lvls = [0.1,0.2,0.3,0.5,0.7,0.8,0.9]

    cmap = plt.cm.Greens
    norm = colors.Normalize(vmin=-0.5, vmax=1)

    if type(variables_input) == str:
        variables_input = variables.copy() # if variables not specified, then use the dict "variables" defined in the top of this script
        
    for scen in scens:
        ###########################################################################################
        # Read all .csv files
        csv_names = glob.glob(directory + "/" + scen + "/*.csv") #'./*.txt')
        
        csvs = [pd.read_csv(csv_names[i]) for i in range(len(csv_names))]
        for i in range(len(csvs)):
            csvs[i].set_index(["solar","wind"],inplace=True)
            
        # save .csv data into the dictionary of dataframes"df_dict"  
        df_temp = pd.concat(csvs,axis=1)
        df_dict[scen] = df_temp[variables_input]

        df_wind_index = df_dict[scen]
        df_solar_index = df_dict[scen].reorder_levels(['wind','solar']).sort_index()
        
        ###########################################################################################
        # Plot all variables
        if plot:
            for var in variables_input.keys():
                
                if var not in variables.keys():
                    variables.update({var:"solar"})
                    
                if var not in var_name:
                    var_name.update({var:var})
                
                fig, ax = plt.subplots()
                if title:
                    ax.set_title(scen)
                if "solar" in var or "SDES" in var: # or "sector" in scen:
                    df_index = df_solar_index.copy()
                    lvls = list(df_solar_index.index.unique(level=0))
                else:
                    df_index = df_wind_index.copy()
                    lvls = list(df_wind_index.index.unique(level=0))

                lvl_count = 0
                for lvl in lvls:
                    variable_plot_dic[scen][str(lvl)] = df_index[var].loc[lvl]
                    variable_plot_dic[scen][str(lvl)].plot(ax=ax,color=cmap(norm(lvl)),marker="o",lw=1,label=variables_input[var] + str(int(100*lvl)) + "%")

                    lvl_count += 1

                scen_count += 1

                if type(ylim) == str:
                    ax.set_ylim([0,1.1*ax.get_ylim()[1]])
                else:
                    ax.set_ylim([0,ylim])
                    
                ax.set_xlabel(ax.get_xlabel() + " resources (% of electricity demand)")
                ax.set_xticks(lvls)
                ax.set_xticklabels((np.array(lvls)*100).astype(int))
                ax.set_ylabel(var_name[var])        

                leg_x_coord = 1.15
                if var == "wind curt" and plot_message_curt:
                    if "LDES" not in scen:
                        ax.plot([0.1,0.3,0.4,0.5,0.6,0.7,0.9],[0,3,5,11,21,28,37],"k--",label="MESSAGEix-GLOBIOM",marker="o",lw=2)
                    else:
                        ax.plot([0.1,0.3,0.4,0.5,0.6,0.7,0.9],[0,2,4,10,20,27,36],"k--",label="MESSAGEix-GLOBIOM \n w. storage determined \n in PyPSA",marker="o",lw=2)
                    
                    if plot_optimum_mix:
                        ax.plot([0.527],[12.9],color="r",marker="X",label="optimum mix \n (5% CO2 cap)",alpha=0.5)
                        ax.axvline(0.527,color="red",alpha=0.5,lw=0.5)
                        ax.axhline(12.9,color="r",alpha=0.5,lw=0.5)
                    ax.set_ylabel("Curtailment of wind \n (% of wind generation)")
                    if title:
                        ax.set_title(scen)
                    leg_x_coord = 1.3
                    ax.set_ylim([0,50])

                elif var == "solar curt" and plot_message_curt:
                    
                    if "SDES" not in scen:
                        ax.plot([0.1,0.3,0.4,0.5,0.6,0.7,0.9],[0,19,33,41,47,51,56],"k--",label="MESSAGEix-GLOBIOM",marker="o",lw=2)
                    else:
                        ax.plot([0.1,0.3,0.4,0.5,0.6,0.7,0.9],[0,3,14,21,27,32,39],"k--",label="MESSAGEix-GLOBIOM \n w. storage determined \n in PyPSA",marker="o",lw=2)
                        #ax.plot([0.1,0.3,0.4,0.5,0.6,0.7,0.9],[0,0,6,19,29,35,44],"k--",label="MESSAGEix-GLOBIOM \n 10% storage",marker="o",lw=2)
                    ax.set_ylim([0,75])    
                    
                    if plot_optimum_mix:
                        ax.axvline(0.354,color="red",alpha=0.5,lw=0.5)
                        ax.axhline(5.1,color="r",alpha=0.5,lw=0.5)
                        ax.plot([0.354],[5.1],color="r",marker="X",label="optimum mix \n (5% CO2 cap)",alpha=0.5)
                    ax.set_ylabel("Curtailment of solar \n (% of solar generation)")
                    leg_x_coord = 1.3
                
                ax.grid()

                fig.legend(prop={"size":12},ncol=1,bbox_to_anchor=(leg_x_coord, 0.9),frameon=True)
                if savefig:
                    fig.savefig("figures/" + scen + "_" + var +  ".png",
                                bbox_inches="tight",
                                dpi=300)
    return df_dict

def plot_single_scenario(df,var,scenario,ylim="",color_str="",legend=True,plot_bins=False,no_bins=3):
    lvls = list(df.index.unique(level=0)) # [0.1,0.3,0.5,0.7,0.9]
    cmap = plt.cm.Greens
    norm = colors.Normalize(vmin=-0.5, vmax=1)
    
    data_by_lvl_dict = {}
    
    fig, ax = plt.subplots()
    ax.set_title(scenario)
    if "solar" in var or "SDES" in var or "transmission" in var:
        df = df.reorder_levels(['wind','solar']).sort_index()

    lvl_count = 0
    for lvl in lvls:
        label_i = variables[var] + str(int(100*lvl)) + "%" if var in variables else "solar " + str(int(100*lvl)) + "%"
        data_by_lvl = df.loc[lvl]
        data_by_lvl_dict[lvl] = data_by_lvl
        
        if len(color_str) == 0:      
            colors_mapped = cmap(norm(lvl))
        else:
            colors_mapped = color_str
            
        data_by_lvl.plot(ax=ax,
                         color=colors_mapped,
                         marker="o",
                         lw=1,
                         label=label_i,
                         legend=False)
            
        lvl_count += 1

    if type(ylim) == float and ylim > 0:
        ax.set_ylim([0,ylim])
    elif type(ylim) == float and ylim < 0:
        ax.set_ylim([ylim,0])
    else:
        ax.set_ylim([0,1.1*ax.get_ylim()[1]])
    
    ax.set_xlabel(ax.get_xlabel() + " resources (% of electricity demand)")
    ax.set_xticks(lvls)
    ax.set_xticklabels((np.array(lvls)*100).astype(int))
    ylabel = var_name[var] if var in var_name else var
    ax.set_ylabel(ylabel)        

    ax.grid()
    
    leg_x_coord = 1.15
    
    if legend:
        fig.legend(prop={"size":12},ncol=1,bbox_to_anchor=(leg_x_coord, 0.9),frameon=True)

    if plot_bins:
        ax.margins(0, 0)
        gamma_lower = ax.get_ylim()[0]
        gamma_upper = ax.get_ylim()[1]
        ax.margins(0.05, 0.05)
        gamma_range = gamma_upper - gamma_lower
        delta = gamma_range/no_bins
        bin_borders = np.arange(gamma_lower,gamma_upper+delta,delta)
        for i in range(no_bins):
            bin_i = [bin_borders[i],bin_borders[i+1]]
            xlims = ax.get_xlim()
            ax.fill_between(x=[xlims[0],xlims[1]],y1=[bin_i[1],bin_i[1]],y2=[bin_i[0],bin_i[0]],color="red",alpha=0.2+0.6/no_bins*i,lw=0)
            ax.set_xlim(xlims)
    else:
        bin_borders = 0
    
    return fig, ax, data_by_lvl_dict,bin_borders

def cartesian_product(l1, l2):
   return list(product(l1, l2))

def meshcolor(plot_variable,vmin=0, vmax="",colormap = "Reds", shading = "nearest",label="", equidistant=False):
    # shading options:
    # - 'nearest' # No interpolation or averaging
    # - 'flat' # The color represent the average of the corner values
    # - 'gouraud'

    if equidistant:
        index_lvl_0 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] #list(plot_variable.index.unique(level=0))
    else:
        index_lvl_0 = list(plot_variable.index.unique(level=0))

    df_all = pd.DataFrame(index = pd.Index(cartesian_product(index_lvl_0,index_lvl_0)),
                          columns=["var"],
                          data=np.nan)
    df_all.index.set_names(["solar","wind"],inplace=True)
    df_all.loc[plot_variable.index,"var"] = plot_variable
    df_plotting = df_all
    nrows,ncols = len(list(df_plotting .index.levels[0])), len(list(df_plotting .index.levels[1]))
    Z = df_plotting.values.reshape(nrows,ncols)
    masked_array = np.ma.array(Z, mask=np.isnan(Z))
    cmap = eval("plt.cm." + colormap)
    cmap.set_bad('lightgrey',1)
    
    x = np.arange(ncols) 
    y = np.arange(nrows)
    fig, ax = plt.subplots()
    if vmin == "min":
        vmin_plot = plot_variable.min()
    else:
        vmin_plot = vmin
        
    if type(vmax) != str:
        vmax_plot = vmax
    else:
        vmax_plot = plot_variable.max()
        
    im = ax.pcolormesh(x, y, masked_array, vmin=vmin_plot, vmax=vmax_plot,shading=shading, cmap=colormap,zorder=0)
    cb = fig.colorbar(im, ax=ax)

    ax.set_xlabel("wind constraint (%)")
    ax.set_ylabel("solar constraint (%)")
    ax.set_yticks(np.arange(nrows))
    ax.set_xticks(np.arange(ncols))

    ax.set_xticklabels(list((np.array(index_lvl_0)*100).astype(int)))
    ax.set_yticklabels(list((np.array(index_lvl_0)*100).astype(int)))

    cb.set_label(label)

    return fig, ax, cb

def assign_location(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        ifind = pd.Series(c.df.index.str.find(" ", start=4), c.df.index)
        for i in ifind.value_counts().index:
            # these have already been assigned defaults
            if i == -1: continue
            names = ifind.index[ifind == i]
            c.df.loc[names, 'location'] = names.str[:i]
            
def assign_carriers(n):
    if "carrier" not in n.lines:
        n.lines["carrier"] = "AC"

def rename_techs(label):

    prefix_to_remove = [
        "residential ",
        "services ",
        "urban ",
        "rural ",
        "central ",
        "decentral "
    ]

    rename_if_contains = [
        #"CHP CC",
        #"gas boiler",
        #"biogas",
        "solar thermal",
        #"air heat pump",
        #"ground heat pump",
        "resistive heater",
        "Fischer-Tropsch"
    ]

    rename_if_contains_dict = {
        "solar": "solar PV",
        "solar rooftop": "solar PV",
        "CHP CC":"heat-power CC",
        "CHP":"CHP",
        #"nuclear":"nuclear",
        "water tanks": "hot water storage",
        "retrofitting": "building retrofitting",
        "H2 Electrolysis": "H2 charging",
        "H2 Fuel Cell": "H2",
        "H2 pipeline": "H2",
        "battery": "battery storage",
        "biogas": "biomass",
        "biomass": "biomass",
        "air heat pump": "heat pump",
        "ground heat pump": "heat pump",
        "gas": "gas",
        "process emissions CC": "CO2 capture",
        "DAC":"CO2 capture",
    }

    rename = {
        "heat-power CC":"CHP CC",
        "solar rooftop": "solar PV",
        "solar": "solar PV",
        "Sabatier": "gas", #methanation
        "offwind": "wind",
        "offwind-ac": "wind",
        "offwind-dc": "wind",
        "onwind": "wind",
        "ror": "hydroelectricity",
        "hydro": "hydroelectricity",
        "PHS": "hydroelectricity",
        "co2 Store": "CO2 capture",
        "co2 stored": "CO2 sequestration",
        "AC": "transmission lines",
        "DC": "transmission lines",
        "B2B": "transmission lines",
    }

    for ptr in prefix_to_remove:
        if label[:len(ptr)] == ptr:
            label = label[len(ptr):]

    for rif in rename_if_contains:
        if rif in label:
            label = rif

    for old,new in rename_if_contains_dict.items():
        if old in label:
            label = new

    for old,new in rename.items():
        if old == label:
            label = new
    return label

def rename_techs_tyndp(label):
    label = rename_techs(label)
    rename_if_contains_dict = {"H2 charging":'H2'}
    for old,new in rename_if_contains_dict.items():
        if old in label:
            label = new
            
    return label


def worst_best_week(network,country,case):
    n = network.copy()

    renewables_p = pd.DataFrame(index=pd.date_range('1/1/2013','1/1/2014',freq='3h')[0:-1])
    generators = ['offwind-ac','offwind-dc','onwind','solar','solar rooftop']
    for generator in generators:
        renewables_index = n.generators.query('carrier == @generator').index
        if country != 'EU':
            renewables_index = renewables_index[renewables_index.str.contains(country)]
        renewables_p[generator] = n.generators_t.p[renewables_index].sum(axis=1).values

    idx_worst_energy = renewables_p.sum(axis=1).groupby(pd.Grouper(freq='7d')).sum().iloc[0:-1].idxmin()
    idx_best_energy = renewables_p.sum(axis=1).groupby(pd.Grouper(freq='7d')).sum().iloc[0:-1].idxmax()

    worst_indices = pd.date_range(idx_worst_energy-pd.Timedelta('4d'),idx_worst_energy+pd.Timedelta('5d'),freq='3h')
    best_indices = pd.date_range(idx_best_energy-pd.Timedelta('4d'),idx_best_energy+pd.Timedelta('5d'),freq='3h')

    worst_indices = worst_indices[worst_indices < pd.to_datetime('31/12/2013 21:00:00')]
    worst_indices = worst_indices[worst_indices > pd.to_datetime('1/1/2013 00:00:00')]
    best_indices = best_indices[best_indices < pd.to_datetime('31/12/2013 21:00:00')]
    best_indices = best_indices[best_indices > pd.to_datetime('1/1/2013 00:00:00')]

    if case == 'worst':
        dstart = worst_indices[0]
        dend = worst_indices[-1]

    if case == 'best':
        dstart = best_indices[0]
        dend = best_indices[-1]

    return dstart,dend


def rename_low_voltage_techs(label):
    rename_if_contains = ['home battery',
                          'BEV',
                          'V2G',
                          'heat pump',
                          'resistive heater',
                          ]
    
    for rif in rename_if_contains:
        if rif in label:
            label = rif
            
    rename_if_contains_dict = {'electricity distribution grid':'High voltage electricity provision'}
    
    for old,new in rename_if_contains_dict.items():
        if old in label:
            label = new
    
    return label

def split_el_distribution_grid(supply,country,network):
    n = network.copy()
    
    low_voltage_consumers = n.links.bus0[n.links.bus0.str.endswith('low voltage')].index
    low_voltage_providers = n.links.bus1[n.links.bus1.str.endswith('low voltage')].index
    domestic_consumers = n.loads.query('carrier == "electricity"').index
    industry_consumers = n.loads.query('carrier == "industry electricity"').index
        
    if country != 'EU':
        low_voltage_consumers = low_voltage_consumers[low_voltage_consumers.str.contains(country)]
        low_voltage_providers = low_voltage_providers[low_voltage_providers.str.contains(country)]
        domestic_consumers = domestic_consumers[domestic_consumers.str.contains(country)]
        industry_consumers = industry_consumers[industry_consumers.str.contains(country)]
    
    # Consumption (negative):
    lv_consumption = -n.links_t.p0[low_voltage_consumers].groupby(rename_low_voltage_techs,axis=1).sum() # From low voltage grid to battery
    domestic_consumption = -n.loads_t.p[domestic_consumers].sum(axis=1)
    industry_consumption = -n.loads_t.p[industry_consumers].sum(axis=1)
    
    # Provision (positive)
    lv_provision = n.links_t.p0[low_voltage_providers]
    lv_provision = lv_provision[lv_provision.columns[~lv_provision.columns.str.endswith('grid')]].groupby(rename_low_voltage_techs,axis=1).sum() # From appliance to low voltage grid
    solar_rooftop = n.generators_t.p[n.generators.query('carrier == "solar rooftop"').index]
    
    if country != 'EU':
        solar_rooftop = solar_rooftop[solar_rooftop.columns[solar_rooftop.columns.str.contains(country)]]
    solar_rooftop = solar_rooftop.sum(axis=1)

    try:
        supply.drop(columns='electricity distribution grid',inplace=True)
    except:
        supply = supply
        
    supply['domestic demand'] = domestic_consumption
    supply['industry demand'] = industry_consumption
    supply['solar rooftop'] = solar_rooftop
    
    for i in lv_consumption.columns:
        supply[i] = lv_consumption[i]
        
    for i in lv_provision.columns:
        supply[i] = lv_provision[i]
    
    return supply

def plot_series(network, country, dstart, dend, tech_colors, rolling_freq = 24, carrier="AC"):

    n = network.copy()
    assign_location(n)
    assign_carriers(n)

    buses = n.buses.index[n.buses.carrier.str.contains(carrier)]
    if country != 'EU':
        buses = buses[buses.str.contains(country)]

    supply = pd.DataFrame(index=n.snapshots)
    for c in n.iterate_components(n.branch_components):
        
        n_port = 4 if c.name=='Link' else 2
        for i in range(n_port):
            supply = pd.concat((supply,
                                (-1) * c.pnl["p" + str(i)].loc[:,
                                                               c.df.index[c.df["bus" + str(i)].isin(buses)]].groupby(c.df.carrier,
                                                                                                                     axis=1).sum()),
                               axis=1)   

    for c in n.iterate_components(n.one_port_components):
        comps = c.df.index[c.df.bus.isin(buses)]
        
        nom_opt_columns = c.df.columns[c.df.columns.str.contains("nom_opt")]
        if len(nom_opt_columns) > 0:
            if c.df[nom_opt_columns].sum().item() == 0:
                continue
        
        if country != 'EU':
            comps = comps[comps.str.contains(country)]
  
        supply = pd.concat((supply, ((c.pnl["p"].loc[:, comps]).multiply(
            c.df.loc[comps, "sign"])).groupby(c.df.carrier, axis=1).sum()), axis=1)

    supply = split_el_distribution_grid(supply,country,n)

    supply = supply.groupby(rename_techs_tyndp, axis=1).sum()

    both = supply.columns[(supply < 0.0).any() & (supply > 0.0).any()]

    positive_supply = supply[both]
    negative_supply = supply[both]

    positive_supply[positive_supply < 0.0] = 0.0
    negative_supply[negative_supply > 0.0] = 0.0

    supply[both] = positive_supply

    suffix = " charging"

    negative_supply.columns = negative_supply.columns + suffix

    supply = pd.concat((supply, negative_supply), axis=1)

    # 14-21.2 for flaute
    # 19-26.1 for flaute

    start = dstart #pd.Timestamp('2013-01-01')
    stop = dend #pd.Timestamp('2013-12-31')
    
    # start = pd.Timestamp('2013-01-01')
    # stop = pd.Timestamp('2013-12-31')

    threshold = 10e3

    to_drop = supply.columns[(abs(supply) < threshold).all()]

    if len(to_drop) != 0: 
        #print("dropping", to_drop)
        supply.drop(columns=to_drop, inplace=True)

    supply.index.name = None

    supply = supply / 1e3

    supply.rename(columns={"electricity": "electric demand",
                           "heat": "heat demand"},
                  inplace=True)
    supply.columns = supply.columns.str.replace("residential ", "")
    supply.columns = supply.columns.str.replace("services ", "")
    supply.columns = supply.columns.str.replace("urban decentral ", "decentral ")

    preferred_order = pd.Index(["domestic demand",
                                "industry demand",
                                "heat pump",
                                "resistive heater",
                                "BEV",
                                "H2 charging",
                                "nuclear",
                                "hydroelectricity",
                                "wind",
                                "solar PV",
                                "solar rooftop",
                                "CHP",
                                "CHP CC",
                                "biomass",
                                "gas",
                                "home battery",
                                "battery",
                                "V2G",
                                "H2"
                                "solar thermal",
                                "Fischer-Tropsch",
                                "CO2 capture",
                                "CO2 sequestration",
                            ])

    supply =  supply.groupby(supply.columns, axis=1).sum()
    
    f = int(8760/len(n.snapshots))
    supply.index = pd.date_range('1/1/2013','1/1/2014',freq=str(f) + 'h')[0:-1]

    fig, ax = plt.subplots()
    fig.set_size_inches((8, 5))

    supply_temp = supply.rename(columns={'battery storage charging':'battery charging',
                                         'battery storage':'battery',
                                         })
    supply = supply_temp.groupby(by=supply_temp.columns,axis=1).sum()

    new_columns = (preferred_order.intersection(supply.columns)
                   .append(supply.columns.difference(preferred_order)))
    
    (supply.loc[start:stop,new_columns].rolling(rolling_freq).mean()
     .plot(ax=ax, kind="area", stacked=True, linewidth=0.,legend=False,
           color=[tech_colors[i.replace(suffix,"")] for i in new_columns]))

    handles, labels = ax.get_legend_handles_labels()

    handles.reverse()
    labels.reverse()

    new_handles = []
    new_labels = []

    for i, item in enumerate(labels):
        if item == 'H2 charging' and 'H2' not in labels:
            new_handles.append(handles[i])
            new_labels.append('H2')
            
        if "charging" not in item:
            new_handles.append(handles[i])
            new_labels.append(labels[i])

    # fig.legend(new_handles, new_labels,loc='lower center', bbox_to_anchor=(0.65, -0.5), prop={'size':15},ncol=3)
    ax.legend(new_handles, new_labels, ncol=3, loc="upper left", frameon=False)
    ax.set_xlim([start, stop])
    
    if country != 'EU':
        ax.set_ylim([-1.2*supply[supply > 0].sum(axis=1).max(), 1.2*supply[supply > 0].sum(axis=1).max()])
    
    else:
        ax.set_ylim([-500,500])
        
    ax.grid(True)
    ax.set_ylabel("Power [GW]")
    fig.tight_layout()

    return fig,supply

def make_handler_map_to_scale_circles_as_in(ax, dont_resize_actively=False):
    fig = ax.get_figure()

    def axes2pt():
        return np.diff(ax.transData.transform([(0, 0), (1, 1)]), axis=0)[
            0] * (72. / fig.dpi)

    ellipses = []
    if not dont_resize_actively:
        def update_width_height(event):
            dist = axes2pt()
            for e, radius in ellipses:
                e.width, e.height = 2. * radius * dist
        fig.canvas.mpl_connect('resize_event', update_width_height)
        ax.callbacks.connect('xlim_changed', update_width_height)
        ax.callbacks.connect('ylim_changed', update_width_height)

    def legend_circle_handler(legend, orig_handle, xdescent, ydescent,
                              width, height, fontsize):
        w, h = 2. * orig_handle.get_radius() * axes2pt()
        e = Ellipse(xy=(0.5 * width - 0.5 * xdescent, 0.5 *
                        height - 0.5 * ydescent), width=w, height=w)
        ellipses.append((e, orig_handle.get_radius()))
        return e
    return {Circle: HandlerPatch(patch_func=legend_circle_handler)}

def make_legend_circles_for(sizes, scale=1.0, **kw):
    return [Circle((0, 0), radius=(s / scale)**0.5, **kw) for s in sizes]

def plot_map(network, tech_colors, threshold=10,components=["links", "generators"],
             bus_size_factor=15e4, transmission=False):
   
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    
    n = network.copy()
    
    assign_location(n)
    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)
    # costs = pd.DataFrame(index=n.buses.index)
    capacity = pd.DataFrame(index=n.buses.index)
    for comp in components:
        df_c = getattr(n, comp)
        if len(df_c) == 0:
            continue # Some countries might not have e.g. storage_units
        df_c["nice_group"] = df_c.carrier.map(rename_techs_tyndp)
        attr = "e_nom_opt" if comp == "stores" else "p_nom_opt"
        capacity_c = ((df_c[attr])
                    .groupby([df_c.location, df_c.nice_group]).sum()
                    .unstack().fillna(0.))
        
        if comp == 'generators':
            capacity_c = capacity_c[['solar PV','wind','hydroelectricity']]
            
        elif comp == 'links':
            capacity_c = capacity_c[['OCGT','CCGT','CHP','CHP CC','coal','coal CC','nuclear']]
            
        # costs_c = ((df_c.capital_cost * df_c[attr])
        #            .groupby([df_c.location, df_c.nice_group]).sum()
        #            .unstack().fillna(0.))
        # costs = pd.concat([costs, costs_c], axis=1)
        capacity = pd.concat([capacity, capacity_c], axis=1)
    plot = capacity.groupby(capacity.columns, axis=1).sum() #costs.groupby(costs.columns, axis=1).sum()
    try:
        plot.drop(index=['H2 pipeline',''],inplace=True)
    except:
        print('No H2 pipeline to drop')
    # plot.drop(columns=['electricity distribution grid'],inplace=True) # 'transmission lines'
    plot.drop(columns=plot.sum().loc[plot.sum() < threshold].index,inplace=True)
    technologies = plot.columns
    plot.drop(list(plot.columns[(plot == 0.).all()]), axis=1, inplace=True)
    
    preferred_order = pd.Index(["domestic demand",
                            "industry demand",
                            "heat pump",
                            "resistive heater",
                            "BEV",
                            "H2 charging",
                            "nuclear",
                            "hydroelectricity",
                            "wind",
                            "solar PV",
                            "solar rooftop",
                            "CHP",
                            "CHP CC",
                            "biomass",
                            "gas",
                            "home battery",
                            "battery",
                            "V2G",
                            "H2"
                            "solar thermal",
                            "Fischer-Tropsch",
                            "CO2 capture",
                            "CO2 sequestration",
                        ])
    
    new_columns = ((preferred_order & plot.columns)
                   .append(plot.columns.difference(preferred_order)))
    plot = plot[new_columns]
    for item in new_columns:
        if item not in tech_colors:
            print("Warning!",item,"not in config/plotting/tech_colors")
    plot = plot.stack()  # .sort_index()
    # hack because impossible to drop buses...
    if 'stores' in components:
        n.buses.loc["EU gas", ["x", "y"]] = n.buses.loc["DE0 0", ["x", "y"]]
    to_drop = plot.index.levels[0] ^ n.buses.index
    if len(to_drop) != 0:
        # print("dropping non-buses", to_drop)
        plot.drop(to_drop, level=0, inplace=True, axis=0)
    # make sure they are removed from index
    plot.index = pd.MultiIndex.from_tuples(plot.index.values)
    # PDF has minimum width, so set these to zero
    line_lower_threshold = 500.
    line_upper_threshold = 2e4
    linewidth_factor = 2e3
    ac_color = "gray"
    dc_color = "m"
    links = n.links #[n.links.carrier == 'DC']
    lines = n.lines
    line_widths = lines.s_nom_opt - lines.s_nom_min
    link_widths = links.p_nom_opt - links.p_nom_min
    if transmission:
        line_widths = lines.s_nom_opt
        link_widths = links.p_nom_opt
        # linewidth_factor = 2e3
        line_lower_threshold = 0.
    line_widths[line_widths < line_lower_threshold] = 0.
    link_widths[link_widths < line_lower_threshold] = 0.
    line_widths[line_widths > line_upper_threshold] = line_upper_threshold
    link_widths[link_widths > line_upper_threshold] = line_upper_threshold
    
    fig.set_size_inches(16, 12)
    n.plot(bus_sizes=plot / bus_size_factor,
           bus_colors=tech_colors,
           line_colors=ac_color,
           link_colors=dc_color,
           line_widths=line_widths / linewidth_factor,
           link_widths=link_widths / linewidth_factor,
           ax=ax,  boundaries=(-10, 30, 34, 70),
           color_geomap={'ocean': 'white', 'land': "whitesmoke"})
    for i in technologies:
        ax.plot([0,0],[1,1],label=i,color=tech_colors[i],lw=5)
    fig.legend(bbox_to_anchor=(1.01, 0.6), frameon=False,prop={'size':18})
    # fig.suptitle('Installed power capacities and transmission lines',y=0.92,fontsize=15)
    
    handles = make_legend_circles_for(
        [1e5,1e4], scale=bus_size_factor, facecolor="grey")
    labels = ["    {} GW".format(s) for s in (100,10)]
    l1 = ax.legend(handles, labels,
                   loc="upper left", bbox_to_anchor=(0.01, 0.98),
                   labelspacing=2,
                   frameon=False,
                   title='Generation capacity',
                   fontsize=15,
                   title_fontsize = 15,
                   handler_map=make_handler_map_to_scale_circles_as_in(ax))
    ax.add_artist(l1)
    handles = []
    labels = []
    for s in (20, 10):
        handles.append(plt.Line2D([0], [0], color=ac_color,
                                  linewidth=s * 1e3 / linewidth_factor))
        labels.append("{} GW".format(s))
    l2 = ax.legend(handles, labels,
                    loc="upper left", bbox_to_anchor=(0.2, 0.98),
                    frameon=False,
                    fontsize=15,
                    title_fontsize = 15,
                    labelspacing=2, handletextpad=1.5,
                    title='    Transmission reinforcement')
    ax.add_artist(l2)

    return fig
    # fig.savefig(snakemake.output.map, transparent=True,
    #             bbox_inches="tight")

def make_comparison(df, renewable, base, tech, tech_act, ax1, ax2, level=0.1):
    
    base_curtailment_all_lvls = df[base][renewable + " curt"]
    tech_curtailment_all_lvls = df[tech][renewable + " curt"]

    if renewable == "solar":
        base_curtailment = base_curtailment_all_lvls.reorder_levels([1,0]).sort_index().loc[level]
        tech_curtailment = tech_curtailment_all_lvls.reorder_levels([1,0]).sort_index().loc[level]
    else:
        base_curtailment = base_curtailment_all_lvls.loc[level]
        tech_curtailment = tech_curtailment_all_lvls.loc[level]
    
    df_comparison = pd.DataFrame(index=base_curtailment.index)
    df_comparison["base"] = base_curtailment
    df_comparison["tech"] = tech_curtailment
    df_comparison["act_tech"] = tech_act.loc[level]
    df_comparison.plot(ax = ax1)

    df_comparison["diff"] = (df_comparison["base"] - df_comparison["tech"])/df_comparison["base"]*100
    df_comparison["diff"].plot(ax = ax2)

    return df_comparison
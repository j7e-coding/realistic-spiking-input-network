#!/usr/bin/env python

import json
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import run
import src.logging as log

plt.switch_backend("Agg")
with open("default.json") as config_file:
    default_config = json.load(config_file)

pool = None


def vary_calculate(args):
    x, variable_name, plot_name, config, run_no, max_runs = args
    log.logger.info(f"### ({run_no}/{max_runs}) VARYING {variable_name}={x}")
    if not config:
        config = {}
    config.update({variable_name: x})
    plot_name = f"{plot_name}{x}.svg"
    _run = log.ex.run(
        config_updates=config,
        meta_info={
            "comment": f"varying {variable_name}={x}",
            # f"previous varying {variable_name} coherences": list(
            #     results["coherence"].tolist()
            # ),
            f"coherence vs {variable_name} plot": plot_name,
            # f"previous varying {variable_name} firing rates": list(
            #     results["firing rate"].tolist()
            # ),
            f"firing rate vs {variable_name} plot": plot_name,
            # f"previous varying {variable_name} firing rate stds": list(
            #     results["firing rate std"].tolist()
            # ),
            f"firing rate std vs {variable_name} plot": plot_name,
        }
    )
    plt.close("all")
    return _run.result


def calculate_config(args):
    config_name, config = args
    log.logger.info(f"### CALCULATING config {config_name}")
    _run = log.ex.run(
        config_updates=config,
        meta_info={
            "comment": f"running config {config_name}",
        }
    )
    plt.close("all")
    return _run.result


def vary(
    variable_name: str,
    arange,
    plot_comment="",
    variable_unit: str = "",
    x_ticks=None,
    repeats=None,
    config=None,
    plot=True,
    vertical_line=None,
) -> pd.DataFrame:
    if repeats is None:
        repeats = default_config["repeats"]
    index = np.repeat(arange, repeats)
    info_name = (
        f"coherence/firing rate/firing rate std "
        f"vs {variable_name} {plot_comment}"
    )
    results_folder = default_config["results_folder"]
    plot_name = (
        f"{results_folder}/results_vs"
        f"_{variable_name}_{plot_comment}"
    )

    map_results = pool.map(
        vary_calculate,
        (
            (float(x), variable_name, plot_name, config, r, len(index))
            for r, x in enumerate(index)
        ),
        chunksize=1,
    )
    results_frame = pd.DataFrame(map_results, index=index)
    results_frame.index.name = variable_name

    results: pd.DataFrame = results_frame.groupby(level=0).mean()
    results_std: pd.DataFrame = (
        results_frame
        .groupby(level=0)
        .std()
        .add_suffix("_std")
    )
    results.index = arange
    results_std.index = arange
    results = pd.concat(
        [results, results_std],
        axis=1,
    )
    log.logger.info(
        f"{info_name}: \n"
        f"{results}"
    )
    with open("default.json", "r") as file:
        config_print = json.load(file)
    config_print.update(config)
    with open(plot_name + ".json", "w") as file:
        json.dump(config_print, fp=file, indent=4, sort_keys=True)
    if plot:
        plot_results_frame(
            results=results,
            title=info_name,
            x_label=f"{variable_name} ({variable_unit})",
            x_ticks=x_ticks,
            plot_name=plot_name,
            vertical_line=vertical_line,
        )
    return results


def plot_results_frame(
        results,
        title,
        x_label,
        plot_name,
        x_ticks=None,
        vertical_line=None,
):
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, dpi=1200)
    # fig.suptitle(title)
    results.plot(
        y=[
            "coherence",
            "coherence_layer_0_ex",
            "coherence_layer_0_inh",
            "coherence_layer_1_ex",
            "coherence_layer_1_inh",
        ],
        ax=axes[0],
    )
    axes[0].set_ylabel("coherence (1)")
    legend0 = axes[0].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    results.plot(
        y=[
            "firing_rate",
            "firing_rate_neuron_std",
            "firing_rate_layer_0_ex",
            "firing_rate_layer_0_inh",
            "firing_rate_layer_1_ex",
            "firing_rate_layer_1_inh",
        ],
        ax=axes[1],
    )
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("f (Hz)")
    legend1 = axes[1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    frequency_normalized = (
            results["firing_rate_neuron_std"]
            / results["firing_rate"]
    )
    axes[2].plot(results.index, frequency_normalized)
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel("f_σ / f_µ (Hz)")
    if vertical_line is not None:
        for axis in axes:
            axis.axvline(vertical_line, linestyle="--")
    if x_ticks is not None:
        for axis in axes:
            axis.set_xticks(x_ticks, minor=False)
    fig.savefig(
        plot_name + ".svg",
        dpi=600,
        additional_artists=[legend0, legend1],
        bbox_inches="tight",
    )
    plt.close("all")
    results.to_csv(plot_name + ".csv")


def vary_configs(
        configs,
        config_names,
        variable_name,
        plot_comment,
        arange,
        repeats=None,
):
    coherences = pd.DataFrame()
    frequencies = pd.DataFrame()

    for config, config_name in zip(configs, config_names):
        results = vary(
            variable_name=variable_name,
            arange=arange,
            plot_comment=f"{plot_comment} ({config_name})",
            variable_unit="1",
            config=config,
            plot=False,
            repeats=repeats,
        )
        if results.empty:
            coherences.index = results.index
            frequencies.index = results.index
        coherences[config_name] = results["coherence"]
        frequencies[config_name] = results["firing_rate"]
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, dpi=1200)
    fig.suptitle(f"variation of {variable_name}")
    coherences.plot(ax=axes[0])
    axes[0].set_xlabel(variable_name)
    axes[0].set_ylabel("coherence")
    legend0 = axes[0].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    frequencies.plot(ax=axes[1])
    axes[1].set_xlabel(variable_name)
    axes[1].set_ylabel("frequency (Hz)")
    legend1 = axes[1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    results_folder = default_config["results_folder"]
    plot_name = (
        f"{results_folder}/results_vs"
        f"_{variable_name} ({plot_comment}).svg"
    )
    fig.savefig(
        plot_name,
        dpi=600,
        additional_artists=[legend0, legend1],
        bbox_inches="tight",
    )
    plt.close("all")
    coherences.to_csv(plot_name + "_coherences.csv")
    frequencies.to_csv(plot_name + "_frequencies.csv")


def run_configs(
        configs,
        config_names,
        plot_comment,
        repeats=None,
        plot=True,
        vertical_line=None,
        x_label=f"config",
        x=None,
        x_ticks=None,
        single_tread=False,
):
    if repeats is None:
        repeats = default_config["repeats"]
    info_name = (
        f"coherence/firing rate/firing rate std "
        f"vs config {plot_comment}"
    )
    results_folder = default_config["results_folder"]
    plot_name = (
        f"{results_folder}/results_vs"
        f"_config_{plot_comment}"
    )
    index = list(range(len(configs)))*repeats
    if single_tread:
        map_results = [
            calculate_config(args)
            for args in zip(
                config_names*repeats,
                configs*repeats,
            )
        ]  # for debugging only
    else:
        map_results = pool.map(calculate_config, zip(
            config_names*repeats,
            configs*repeats,
        ))
        # TODO: ^^use config_runner()

    results = pd.DataFrame(map_results, index=index)
    results.index.name = "config"

    results: pd.DataFrame = results.groupby(level=0).mean()
    if x is not None:
        results.index = x
    log.logger.info(
        f"{info_name}: \n"
        f"{results}"
    )
    if plot:
        plot_results_frame(
            results=results,
            title=info_name,
            x_label=x_label,
            x_ticks=x_ticks,
            plot_name=plot_name,
            vertical_line=vertical_line,
        )
    return results


def config_runner(arg):
    config, info, comment = arg
    log.logger.info(info)
    run_ = log.ex.run(
        config_updates=config,
        meta_info={"comment": comment}
    )
    return run_.result


def run_wb():
    calculate_config(("default_WB_model", {
        "input_inhibitory_neurons": 1,
        "output_inhibitory_neurons": 0,
        "i_mu_inhibitory_input": 0.3,
        "i_sigma_inhibitory_input": 0.0,
        "plot_everything": True,
    }))


def vary_i_mu_1a():
    """
    Verify results in Wang, Buzsáki. 1996. Figure 1a.
    """
    vary(
        variable_name="i_mu_inhibitory_input",
        arange=np.arange(0.0, 21, 1),
        plot_comment="W-B-plot 1A",
        variable_unit="µA/cm²",
        config={"input_inhibitory_neurons": 1, "output_inhibitory_neurons": 0}
    )


def vary_i_mu_1b():
    vary(
        variable_name="i_mu_inhibitory_input",
        arange=np.arange(0.0, 1.01, 0.05),
        plot_comment="W-B-plot 1B",
        variable_unit="µA/cm²",
        config={
            "output_inhibitory_neurons": 0,
            "input_inhibitory_neurons": 100,
            "i_sigma_inhibitory_input": 0.03,
            "m_syn_input": 0,
        }
    )


def vary_phi_3():
    phis = [5, 3.33, 2]
    i_mus = [1, 1.2, 1.4]
    w_b_figures = ["A", "B", "C"]
    configs = []
    for phi, i_mu, w_b_figure in zip(phis, i_mus, w_b_figures):
        config = {
            "output_inhibitory_neurons": 0,
            "phi": phi,
            "i_mu_inhibitory_input": i_mu,
            "plot_everything": True,
        }
        info = f"varying phi: {phi} and i_mu_inhibitory: {i_mu}"
        comment = f"varying phi (W-B-plot 3{w_b_figure})"
        arg = (config, info, comment)
        configs.append(arg)
        # configs.extend([arg for _ in range(repeats)])
    pool.map(config_runner, configs)


def vary_e_syn_4a():
    v_ahp = -67
    config = {"output_inhibitory_neurons": 0}
    vary(
        variable_name="e_syn_inhibitory",
        arange=np.arange(-90, -10, 3),
        plot_comment="W-B-plot 4A",
        variable_unit="mV",
        vertical_line=v_ahp,
        config=config,
    )


def excitatory_network_4bc():
    log.logger.info("running excitatory network")
    log.ex.run(
        config_updates={
            "output_inhibitory_neurons": 0,
            "g_syn_layer0_in_from_in": 0.5,
            # g_syn_in_in = capacity / tau_syn; tau_syn = 2m, capacity=1
            "e_syn_inhibitory": -50.0,
            "i_mu_inhibitory_input": 0.1,
            # "spike_threshold": -40.0,
        },
        meta_info={
            "comment": "simple excitatory network (W-B-plot 4BC)"
        }
    )


def vary_i_sigma_5ab():
    config = {"output_inhibitory_neurons": 0}
    vary(
        variable_name="i_sigma_inhibitory_input",
        arange=np.arange(0.0, 0.31, 0.01),
        # range_step=0.05,
        plot_comment="W-B-plot 5AB",
        variable_unit="µA/cm²",
        config=config,
        plot=True,
    )


def vary_n_async_6bc():
    ns = [100, 200, 300, 400, 500]
    repeats = default_config["repeats"]
    configs = []
    for n in ns:
        config = {
            "output_inhibitory_neurons": 0,
            "input_inhibitory_neurons": n,
            "i_sigma_inhibitory_input": 0.1,
            "plot_everything": True,
        }
        info = f"varying n for async network: {n}"
        comment = f"varying n={n} for async network (W-B-plot 6BC)"
        arg = (config, info, comment)
        configs.append(arg)
        # configs.extend([arg for _ in range(repeats)])
    pool = multiprocessing.Pool()
    results_list = pool.map(config_runner, configs)
    results = pd.DataFrame(results_list, index=ns)


def vary_m_syn_7a():
    configs = [
        {
            "output_inhibitory_neurons": 0,
            # "i_sigma_inhibitory_input": 0.0,  # TODO: run this with noise
            # "plot_everything": True,
        },
        {
            "output_inhibitory_neurons": 0,
            # "i_sigma_inhibitory_input": 0.0,
            "g_syn_layer0_in_from_in": 0.05,
        },
        {
            "output_inhibitory_neurons": 0,
            # "i_sigma_inhibitory_input": 0.0,
            "i_mu_inhibitory_input": 3,
        },
        # {
        #     "output_inhibitory_neurons": 0,
        #     "i_sigma_inhibitory_input": 0.0,
        #     "i_mu_inhibitory_input": 3,
        #     "g_syn_layer0_in_from_in": 0.05,
        # },
    ]
    config_names = [
        "Iµ=1, g_syn_layer0_in_from_in=0.1",
        "Iµ=1, g_syn_layer0_in_from_in=0.05",
        "Iµ=3, g_syn_layer0_in_from_in=0.1",
        # "Iµ=3, g_syn_layer0_in_from_in=0.05",
    ]
    vary_configs(
        configs=configs,
        config_names=config_names,
        variable_name="m_syn_input",
        plot_comment="W-B-plot 7A",
        arange=np.arange(0.0, 1.01, 0.05),
        repeats=10,
    )


def vary_m_syn_7b():
    configs = [
        {"input_inhibitory_neurons": 100, "output_inhibitory_neurons": 0, },
        {"input_inhibitory_neurons": 200, "output_inhibitory_neurons": 0, },
        {"input_inhibitory_neurons": 500, "output_inhibitory_neurons": 0, },
        {"input_inhibitory_neurons": 1000, "output_inhibitory_neurons": 0, },
    ]
    config_names = [
        "N=100",
        "N=200",
        "N=500",
        "N=1000",
    ]
    vary_configs(
        configs=configs,
        config_names=config_names,
        variable_name="m_syn_input",
        plot_comment="W-B-plot 7B",
        arange=np.arange(0.0, 201, 20),
    )


def vary_tau_syn_10a():
    """
    tau_syn = 1/beta_inhibitory
    beta_inhibitory = 1/tau_syn
    """
    configs = [
        {
            "output_inhibitory_neurons": 0,
            "i_sigma_inhibitory_input": 0.0,
            "m_syn_input": 100,
        },
        {
            "output_inhibitory_neurons": 0,
            "i_sigma_inhibitory_input": 0.03,
            "m_syn_input": 100,
        },
        {
            "output_inhibitory_neurons": 0,
            "i_sigma_inhibitory_input": 0.0,
            "m_syn_input": 60,
        },
        {
            "output_inhibitory_neurons": 0,
            "i_sigma_inhibitory_input": 0.03,
            "m_syn_input": 60,
        },
    ]
    config_names = [
        "Iσ=0, m_syn_input=100",
        "Iσ=0.03, m_syn_input=100",
        "Iσ=0, m_syn_input=60",
        "Iσ=0.03, m_syn_input=60",
    ]
    tau_range = np.arange(1, 21, 1)
    # beta_range = 1/tau_range
    vary_configs(
        configs=configs,
        config_names=config_names,
        variable_name="tau_syn_falling_inhibitory",
        plot_comment="W-B-plot 10A",
        arange=tau_range,
    )


def vary_tau_syn_11a():
    config = {
        "output_inhibitory_neurons": 0,
        "m_syn_input": 60,
        "i_sigma_inhibitory_input": 0.03,
        "tau_is_inverse_frequency": True,
        "tau_multiplier": 0.1,
    }
    configs = [
        dict(config, i_mu_inhibitory_input=1),
        dict(config, i_mu_inhibitory_input=2),
        dict(config, i_mu_inhibitory_input=3),
    ]
    config_names = [
        "Iµ=1",
        "Iµ=2",
        "Iµ=3",
    ]
    vary_configs(
        configs=configs,
        config_names=config_names,
        variable_name="tau_syn_falling_inhibitory",
        plot_comment="W-B-plot 11A",
        arange=np.arange(0.0, 20.1, 3),
    )


def vary_i_mu_12a():
    config = {
        "output_inhibitory_neurons": 0,
        "m_syn_input": 60,
        "i_sigma_inhibitory_input": 0.03,
        "tau_is_inverse_frequency": True,
    }
    vary(
        variable_name="i_mu_inhibitory_input",
        arange=np.arange(0, 3.1, 0.3),
        plot_comment="W-B-plot 12a",
        variable_unit="µA/cm²",
        repeats=1,
        config=config,
    )


def vary_g_syn_12b():
    config = {
        "output_inhibitory_neurons": 0,
        "m_syn_input": 60,
        "i_sigma_inhibitory_input": 0.03,
        "tau_is_inverse_frequency": True,
    }
    configs = [
        dict(config, i_mu_inhibitory_input=1),
        dict(config, i_mu_inhibitory_input=2),
        dict(config, i_mu_inhibitory_input=3),
    ]
    config_names = [
        "i_µ=1",
        "i_µ=2",
        "i_µ=3",
    ]
    vary_configs(
        configs=configs,
        config_names=config_names,
        variable_name="g_syn_layer0_in_from_in",
        plot_comment="W-B-plot 12B",
        arange=np.arange(0.0, 0.31, 0.03),
    )


def ing_dual_layer_sync():
    config = {}
    config.update(
        # input_excitatory_neurons=80,
        # input_inhibitory_neurons=20,
        # output_excitatory_neurons=80,
        # output_inhibitory_neurons=20,
        # m_syn_input=1,  # 0.6 # async: 0
        # m_syn_output=1,  # 0.6
        # m_syn_inter_ex_ex=0.1,  # 0.1
        # m_syn_inter_ex_in=0.0,  # 0
        # m_syn_inter_in_ex=0.04,  # 0.04
        # m_syn_inter_in_in=0.0,  # 0
        # g_syn_layer0_ex_from_ex=0.0,  # 0.0
        # g_syn_layer0_in_from_in=0.1,  # 0.1
        # g_syn_layer0_ex_from_in=1.0,  # 1.0
        # g_syn_layer0_in_from_ex=0.5,  # 0.5
        # g_syn_layer1_ex_from_ex=0.0,  # 0.0
        # g_syn_layer1_in_from_in=0.1,  # 0.1
        # g_syn_layer1_ex_from_in=1.0,  # 1.0
        # g_syn_layer1_in_from_ex=0.5,  # 0.5
        # g_syn_inter_ex_from_ex=30.0,  # 1
        # g_syn_inter_ex_from_in=0.0,  # 0
        # g_syn_inter_in_from_ex=30.0,  # 1
        # g_syn_inter_in_from_in=0.0,  # 0
        # i_mu_excitatory_input=0.7,  # 0.7 # async: 0.05
        # i_mu_inhibitory_input=1.1,  # 0.5 # async: 0.4
        # for 27Hz async(m_syn=0) firing: inhib: 0.4
        # i_mu_excitatory_output=0.7,  # 0.7
        # i_mu_inhibitory_output=1.0,  # 0.5
        # i_sigma_excitatory_input=0.0,  # 0.2 # async: 0.2
        # i_sigma_inhibitory_input=0.2,  # 0.2 # async: 0.2
        # i_sigma_excitatory_output=0.0,  # 0.2
        # i_sigma_inhibitory_output=0.08,  # 0.2
        stim_layer_diff=20,
        # stim_spike_max_excitatory_input=0,  # 0.05
        # stim_spike_max_inhibitory_input=0,  # 0.02
        # stim_spike_max_excitatory_output=0,  # 0.05
        # stim_spike_max_inhibitory_output=0,  # 0.02
        # f_stim_excitatory_input=100,  # 10 # async: 0
        # f_stim_inhibitory_input=0,  # 10
        # f_stim_excitatory_output=0,  # 10
        # f_stim_inhibitory_output=0,  # 10
        # tau_is_inverse_frequency=True,
        # tau_default=1,
        plot_everything=True,
    )
    log.ex.run(
        config_updates=config,
        meta_info={
            "comment": "ing_dual_layer_sync",
        }
    )


def ing_dual_layer_async():
    config = {}
    config.update(
        # input_excitatory_neurons=80,
        # input_inhibitory_neurons=20,
        # output_excitatory_neurons=80,
        # output_inhibitory_neurons=20,
        # m_syn_input=1,  # 0.6 # async: 0
        # m_syn_output=1,  # 0.6
        # m_syn_inter_ex_ex=0.1,  # 0.1
        # m_syn_inter_ex_in=0.0,  # 0
        # m_syn_inter_in_ex=0.04,  # 0.04
        # m_syn_inter_in_in=0.0,  # 0
        # g_syn_layer0_ex_from_ex=0.0,  # 0.0
        # g_syn_layer0_in_from_in=0.1,  # 0.1
        # g_syn_layer0_ex_from_in=1.0,  # 1.0
        # g_syn_layer0_in_from_ex=0.5,  # 0.5
        # g_syn_layer1_ex_from_ex=0.0,  # 0.0
        # g_syn_layer1_in_from_in=0.1,  # 0.1
        # g_syn_layer1_ex_from_in=1.0,  # 1.0
        # g_syn_layer1_in_from_ex=0.5,  # 0.5
        # g_syn_inter_ex_from_ex=30.0,  # 1
        # g_syn_inter_ex_from_in=0.0,  # 0
        # g_syn_inter_in_from_ex=30.0,  # 1
        # g_syn_inter_in_from_in=0.0,  # 0
        # i_mu_excitatory_input=0.7,  # 0.7 # async: 0.05
        i_mu_inhibitory_input=1.1,  # 0.5 # async: 0.4
        # for 27Hz async(m_syn=0) firing: inhib: 0.4
        # i_mu_excitatory_output=0.7,  # 0.7
        # i_mu_inhibitory_output=1.0,  # 0.5
        # i_sigma_excitatory_input=0.0,  # 0.2 # async: 0.2
        i_sigma_inhibitory_input=0.2,  # 0.2 # async: 0.2
        # i_sigma_excitatory_output=0.0,  # 0.2
        # i_sigma_inhibitory_output=0.08,  # 0.2
        stim_layer_diff=20,
        # stim_spike_max_excitatory_input=0,  # 0.05
        # stim_spike_max_inhibitory_input=0,  # 0.02
        # stim_spike_max_excitatory_output=0,  # 0.05
        # stim_spike_max_inhibitory_output=0,  # 0.02
        # f_stim_excitatory_input=100,  # 10 # async: 0
        # f_stim_inhibitory_input=0,  # 10
        # f_stim_excitatory_output=0,  # 10
        # f_stim_inhibitory_output=0,  # 10
        # tau_is_inverse_frequency=True,
        # tau_default=1,
        plot_everything=True,
    )
    log.ex.run(
        config_updates=config,
        meta_info={
            "comment": "ing_dual_layer_async",
        }
    )


def vary_output_i_sigma():
    config = {}
    config.update(
        # input_excitatory_neurons=80,
        # input_inhibitory_neurons=20,
        # output_excitatory_neurons=80,
        # output_inhibitory_neurons=20,
        # m_syn_input=1,  # 0.6 # async: 0
        # m_syn_output=1,  # 0.6
        # m_syn_inter_ex_ex=0.1,  # 0.1
        # m_syn_inter_ex_in=0.0,  # 0
        # m_syn_inter_in_ex=0.04,  # 0.04
        # m_syn_inter_in_in=0.0,  # 0
        # g_syn_layer0_ex_from_ex=0.0,  # 0.0
        # g_syn_layer0_in_from_in=0.1,  # 0.1
        # g_syn_layer0_ex_from_in=1.0,  # 1.0
        # g_syn_layer0_in_from_ex=0.5,  # 0.5
        # g_syn_layer1_ex_from_ex=0.0,  # 0.0
        # g_syn_layer1_in_from_in=0.1,  # 0.1
        # g_syn_layer1_ex_from_in=1.0,  # 1.0
        # g_syn_layer1_in_from_ex=0.5,  # 0.5
        # g_syn_inter_ex_from_ex=30.0,  # 1
        # g_syn_inter_ex_from_in=0.0,  # 0
        # g_syn_inter_in_from_ex=30.0,  # 1
        # g_syn_inter_in_from_in=0.0,  # 0
        # i_mu_excitatory_input=0.7,  # 0.7 # async: 0.05
        # i_mu_inhibitory_input=1.1,  # 0.5 # async: 0.4
        # for 27Hz async(m_syn=0) firing: inhib: 0.4
        # i_mu_excitatory_output=0.7,  # 0.7
        # i_mu_inhibitory_output=1.0,  # 0.5
        # i_sigma_excitatory_input=0.0,  # 0.2 # async: 0.2
        # i_sigma_inhibitory_input=0.2,  # 0.2 # async: 0.2
        # i_sigma_excitatory_output=0.0,  # 0.2
        # i_sigma_inhibitory_output=0.1,  # 0.2
        stim_layer_diff=20,
        # stim_spike_max_excitatory_input=0,  # 0.05
        # stim_spike_max_inhibitory_input=0,  # 0.02
        # stim_spike_max_excitatory_output=0,  # 0.05
        # stim_spike_max_inhibitory_output=0,  # 0.02
        # f_stim_excitatory_input=100,  # 10 # async: 0
        # f_stim_inhibitory_input=0,  # 10
        # f_stim_excitatory_output=0,  # 10
        # f_stim_inhibitory_output=0,  # 10
        # tau_is_inverse_frequency=True,
        # tau_default=1,
        plot_everything=False,
    )
    # manual_range = np.concatenate([
    #     np.linspace(0.0, 0.2, num=25)[:-1],
    #     np.linspace(0.2, 5.0, num=15)
    # ])
    vary(
        variable_name="i_sigma_inhibitory_output",
        arange=np.linspace(0.0, 0.5, num=50),
        plot_comment="vary i_sigma_inhibitory_output ING",
        variable_unit="µA/cm²",
        repeats=10,
        config=config,
    )


def vary_output_i_sigma_async():
    config = {}
    config.update(
        # input_excitatory_neurons=80,
        # input_inhibitory_neurons=20,
        # output_excitatory_neurons=80,
        # output_inhibitory_neurons=20,
        # m_syn_input=1,  # 0.6 # async: 0
        # m_syn_output=1,  # 0.6
        # m_syn_inter_ex_ex=0.1,  # 0.1
        # m_syn_inter_ex_in=0.0,  # 0
        # m_syn_inter_in_ex=0.04,  # 0.04
        # m_syn_inter_in_in=0.0,  # 0
        # g_syn_layer0_ex_from_ex=0.0,  # 0.0
        # g_syn_layer0_in_from_in=0.1,  # 0.1
        # g_syn_layer0_ex_from_in=1.0,  # 1.0
        # g_syn_layer0_in_from_ex=0.5,  # 0.5
        # g_syn_layer1_ex_from_ex=0.0,  # 0.0
        # g_syn_layer1_in_from_in=0.1,  # 0.1
        # g_syn_layer1_ex_from_in=1.0,  # 1.0
        # g_syn_layer1_in_from_ex=0.5,  # 0.5
        # g_syn_inter_ex_from_ex=30.0,  # 1
        # g_syn_inter_ex_from_in=0.0,  # 0
        # g_syn_inter_in_from_ex=30.0,  # 1
        # g_syn_inter_in_from_in=0.0,  # 0
        # i_mu_excitatory_input=0.7,  # 0.7 # async: 0.05
        i_mu_inhibitory_input=1.1,  # 0.5 # async: 0.4
        # for 27Hz async(m_syn=0) firing: inhib: 0.4
        # i_mu_excitatory_output=0.7,  # 0.7
        # i_mu_inhibitory_output=1.0,  # 0.5
        # i_sigma_excitatory_input=0.0,  # 0.2 # async: 0.2
        i_sigma_inhibitory_input=0.2,  # 0.2 # async: 0.2
        # i_sigma_excitatory_output=0.0,  # 0.2
        # i_sigma_inhibitory_output=0.1,  # 0.2
        stim_layer_diff=20,
        # stim_spike_max_excitatory_input=0,  # 0.05
        # stim_spike_max_inhibitory_input=0,  # 0.02
        # stim_spike_max_excitatory_output=0,  # 0.05
        # stim_spike_max_inhibitory_output=0,  # 0.02
        # f_stim_excitatory_input=100,  # 10 # async: 0
        # f_stim_inhibitory_input=0,  # 10
        # f_stim_excitatory_output=0,  # 10
        # f_stim_inhibitory_output=0,  # 10
        # tau_is_inverse_frequency=True,
        # tau_default=2,
        plot_everything=False,
    )
    vary(
        variable_name="i_sigma_inhibitory_output",
        arange=np.linspace(0.0, 0.5, num=20),
        plot_comment="vary i_sigma_inhibitory_output ING async",
        variable_unit="µA/cm²",
        repeats=10,
        config=config,
    )


def vary_output_i_mu():
    config = {}
    config.update(
        # input_excitatory_neurons=80,
        # input_inhibitory_neurons=20,
        # output_excitatory_neurons=80,
        # output_inhibitory_neurons=20,
        # m_syn_input=1,  # 0.6 # async: 0
        # m_syn_output=1,  # 0.6
        # m_syn_inter_ex_ex=0.1,  # 0.1
        # m_syn_inter_ex_in=0.0,  # 0
        # m_syn_inter_in_ex=0.04,  # 0.04
        # m_syn_inter_in_in=0.0,  # 0
        # g_syn_layer0_ex_from_ex=0.0,  # 0.0
        # g_syn_layer0_in_from_in=0.1,  # 0.1
        # g_syn_layer0_ex_from_in=1.0,  # 1.0
        # g_syn_layer0_in_from_ex=0.5,  # 0.5
        # g_syn_layer1_ex_from_ex=0.0,  # 0.0
        # g_syn_layer1_in_from_in=0.1,  # 0.1
        # g_syn_layer1_ex_from_in=1.0,  # 1.0
        # g_syn_layer1_in_from_ex=0.5,  # 0.5
        # g_syn_inter_ex_from_ex=30.0,  # 1
        # g_syn_inter_ex_from_in=0.0,  # 0
        # g_syn_inter_in_from_ex=30.0,  # 1
        # g_syn_inter_in_from_in=0.0,  # 0
        # i_mu_excitatory_input=0.7,  # 0.7 # async: 0.05
        # i_mu_inhibitory_input=1.0,  # 0.5 # async: 0.4
        # for 27Hz async(m_syn=0) firing: inhib: 0.4
        # i_mu_excitatory_output=0.7,  # 0.7
        # i_mu_inhibitory_output=1.0,  # 0.5
        # i_sigma_excitatory_input=0.0,  # 0.2 # async: 0.2
        # i_sigma_inhibitory_input=0.1,  # 0.2 # async: 0.2
        # i_sigma_excitatory_output=0.0,  # 0.2
        # i_sigma_inhibitory_output=0.1,  # 0.2
        stim_layer_diff=20,
        # stim_spike_max_excitatory_input=0,  # 0.05
        # stim_spike_max_inhibitory_input=0,  # 0.02
        # stim_spike_max_excitatory_output=0,  # 0.05
        # stim_spike_max_inhibitory_output=0,  # 0.02
        # f_stim_excitatory_input=100,  # 10 # async: 0
        # f_stim_inhibitory_input=0,  # 10
        # f_stim_excitatory_output=0,  # 10
        # f_stim_inhibitory_output=0,  # 10
        # tau_is_inverse_frequency=True,
        # tau_default=2,
        plot_everything=False,
    )
    vary(
        variable_name="i_mu_inhibitory_output",
        arange=np.linspace(0.0, 5.0, num=25),
        plot_comment="vary i_µ_inhibitory_output ING",
        variable_unit="µA/cm²",
        repeats=10,
        config=config,
    )


def vary_output_i_mu_async():
    config = {}
    config.update(
        # input_excitatory_neurons=80,
        # input_inhibitory_neurons=20,
        # output_excitatory_neurons=80,
        # output_inhibitory_neurons=20,
        # m_syn_input=1,  # 0.6 # async: 0
        # m_syn_output=1,  # 0.6
        # m_syn_inter_ex_ex=0.1,  # 0.1
        # m_syn_inter_ex_in=0.0,  # 0
        # m_syn_inter_in_ex=0.04,  # 0.04
        # m_syn_inter_in_in=0.0,  # 0
        # g_syn_layer0_ex_from_ex=0.0,  # 0.0
        # g_syn_layer0_in_from_in=0.1,  # 0.1
        # g_syn_layer0_ex_from_in=1.0,  # 1.0
        # g_syn_layer0_in_from_ex=0.5,  # 0.5
        # g_syn_layer1_ex_from_ex=0.0,  # 0.0
        # g_syn_layer1_in_from_in=0.1,  # 0.1
        # g_syn_layer1_ex_from_in=1.0,  # 1.0
        # g_syn_layer1_in_from_ex=0.5,  # 0.5
        # g_syn_inter_ex_from_ex=30.0,  # 1
        # g_syn_inter_ex_from_in=0.0,  # 0
        # g_syn_inter_in_from_ex=30.0,  # 1
        # g_syn_inter_in_from_in=0.0,  # 0
        # i_mu_excitatory_input=0.7,  # 0.7 # async: 0.05
        i_mu_inhibitory_input=1.1,  # 0.5 # async: 0.4
        # for 27Hz async(m_syn=0) firing: inhib: 0.4
        # i_mu_excitatory_output=0.7,  # 0.7
        # i_mu_inhibitory_output=1.0,  # 0.5
        # i_sigma_excitatory_input=0.0,  # 0.2 # async: 0.2
        i_sigma_inhibitory_input=0.2,  # 0.2 # async: 0.2
        # i_sigma_excitatory_output=0.0,  # 0.2
        # i_sigma_inhibitory_output=0.1,  # 0.2
        stim_layer_diff=20,
        # stim_spike_max_excitatory_input=0,  # 0.05
        # stim_spike_max_inhibitory_input=0,  # 0.02
        # stim_spike_max_excitatory_output=0,  # 0.05
        # stim_spike_max_inhibitory_output=0,  # 0.02
        # f_stim_excitatory_input=100,  # 10 # async: 0
        # f_stim_inhibitory_input=0,  # 10
        # f_stim_excitatory_output=0,  # 10
        # f_stim_inhibitory_output=0,  # 10
        # tau_is_inverse_frequency=True,
        # tau_default=2,
        plot_everything=False,
    )
    vary(
        variable_name="i_mu_inhibitory_output",
        arange=np.linspace(0.0, 5.0, num=25),
        plot_comment="vary i_µ_inhibitory_output ING async",
        variable_unit="µA/cm²",
        repeats=10,
        config=config,
    )


def vary_input_i_mu():
    config = {}
    config.update(
        # input_excitatory_neurons=80,
        # input_inhibitory_neurons=20,
        # output_excitatory_neurons=80,
        # output_inhibitory_neurons=20,
        # m_syn_input=1,  # 0.6 # async: 0
        # m_syn_output=1,  # 0.6
        # m_syn_inter_ex_ex=0.1,  # 0.1
        # m_syn_inter_ex_in=0.0,  # 0
        # m_syn_inter_in_ex=0.04,  # 0.04
        # m_syn_inter_in_in=0.0,  # 0
        # g_syn_layer0_ex_from_ex=0.0,  # 0.0
        # g_syn_layer0_in_from_in=0.1,  # 0.1
        # g_syn_layer0_ex_from_in=1.0,  # 1.0
        # g_syn_layer0_in_from_ex=0.5,  # 0.5
        # g_syn_layer1_ex_from_ex=0.0,  # 0.0
        # g_syn_layer1_in_from_in=0.1,  # 0.1
        # g_syn_layer1_ex_from_in=1.0,  # 1.0
        # g_syn_layer1_in_from_ex=0.5,  # 0.5
        # g_syn_inter_ex_from_ex=30.0,  # 1
        # g_syn_inter_ex_from_in=0.0,  # 0
        # g_syn_inter_in_from_ex=30.0,  # 1
        # g_syn_inter_in_from_in=0.0,  # 0
        # i_mu_excitatory_input=0.7,  # 0.7 # async: 0.05
        # i_mu_inhibitory_input=1.0,  # 0.5 # async: 0.4
        # for 27Hz async(m_syn=0) firing: inhib: 0.4
        # i_mu_excitatory_output=0.7,  # 0.7
        # i_mu_inhibitory_output=1.0,  # 0.5
        # i_sigma_excitatory_input=0.0,  # 0.2 # async: 0.2
        i_sigma_inhibitory_input=0.0,  # 0.2 # async: 0.2
        # i_sigma_excitatory_output=0.0,  # 0.2
        i_sigma_inhibitory_output=0.0,  # 0.2
        stim_layer_diff=20,
        # stim_spike_max_excitatory_input=0,  # 0.05
        # stim_spike_max_inhibitory_input=0,  # 0.02
        # stim_spike_max_excitatory_output=0,  # 0.05
        # stim_spike_max_inhibitory_output=0,  # 0.02
        # f_stim_excitatory_input=100,  # 10 # async: 0
        # f_stim_inhibitory_input=0,  # 10
        # f_stim_excitatory_output=0,  # 10
        # f_stim_inhibitory_output=0,  # 10
        tau_is_inverse_frequency=False,
        # tau_default=2,
        plot_everything=False,
    )
    manual_range = np.concatenate([
        np.linspace(0.0, 1.5, num=25)[:-1],
        np.linspace(1.5, 5.0, num=15)
    ])
    vary(
        variable_name="i_mu_inhibitory_input",
        arange=manual_range,
        plot_comment="vary i_µ_inhibitory_input ING",
        variable_unit="µA/cm²",
        repeats=10,
        config=config,
    )


def vary_output_m_syn_sync():
    config = {}
    config.update(
        # input_excitatory_neurons=80,
        # input_inhibitory_neurons=20,
        # output_excitatory_neurons=80,
        # output_inhibitory_neurons=20,
        # m_syn_input=1,  # 0.6 # async: 0
        # m_syn_output=1,  # 0.6
        # m_syn_inter_ex_ex=0.1,  # 0.1
        # m_syn_inter_ex_in=0.0,  # 0
        # m_syn_inter_in_ex=0.04,  # 0.04
        # m_syn_inter_in_in=0.0,  # 0
        # g_syn_layer0_ex_from_ex=0.0,  # 0.0
        # g_syn_layer0_in_from_in=0.1,  # 0.1
        # g_syn_layer0_ex_from_in=1.0,  # 1.0
        # g_syn_layer0_in_from_ex=0.5,  # 0.5
        # g_syn_layer1_ex_from_ex=0.0,  # 0.0
        # g_syn_layer1_in_from_in=0.1,  # 0.1
        # g_syn_layer1_ex_from_in=1.0,  # 1.0
        # g_syn_layer1_in_from_ex=0.5,  # 0.5
        # g_syn_inter_ex_from_ex=30.0,  # 1
        # g_syn_inter_ex_from_in=0.0,  # 0
        # g_syn_inter_in_from_ex=30.0,  # 1
        # g_syn_inter_in_from_in=0.0,  # 0
        # i_mu_excitatory_input=0.7,  # 0.7 # async: 0.05
        # i_mu_inhibitory_input=1.1,  # 0.5 # async: 0.4
        # for 27Hz async(m_syn=0) firing: inhib: 0.4
        # i_mu_excitatory_output=0.7,  # 0.7
        # i_mu_inhibitory_output=1.0,  # 0.5
        # i_sigma_excitatory_input=0.0,  # 0.2 # async: 0.2
        # i_sigma_inhibitory_input=0.2,  # 0.2 # async: 0.2
        # i_sigma_excitatory_output=0.0,  # 0.2
        # i_sigma_inhibitory_output=0.1,  # 0.2
        stim_layer_diff=20,
        # stim_spike_max_excitatory_input=0,  # 0.05
        # stim_spike_max_inhibitory_input=0,  # 0.02
        # stim_spike_max_excitatory_output=0,  # 0.05
        # stim_spike_max_inhibitory_output=0,  # 0.02
        # f_stim_excitatory_input=100,  # 10 # async: 0
        # f_stim_inhibitory_input=0,  # 10
        # f_stim_excitatory_output=0,  # 10
        # f_stim_inhibitory_output=0,  # 10
        # tau_is_inverse_frequency=True,
        # tau_default=2,
        plot_everything=False,
    )
    manual_range = np.concatenate([
        # np.linspace(0.0, 0.8, num=16)[:-1],
        np.linspace(0.0, 1.0, num=20),
    ])
    vary(
        variable_name="m_syn_output",
        arange=manual_range,
        plot_comment="vary m_syn_output ING",
        variable_unit="1",
        repeats=10,
        config=config,
    )


def vary_output_m_syn_async():
    config = {}
    config.update(
        # input_excitatory_neurons=80,
        # input_inhibitory_neurons=20,
        # output_excitatory_neurons=80,
        # output_inhibitory_neurons=20,
        # m_syn_input=1,  # 0.6 # async: 0
        # m_syn_output=1,  # 0.6
        # m_syn_inter_ex_ex=0.1,  # 0.1
        # m_syn_inter_ex_in=0.0,  # 0
        # m_syn_inter_in_ex=0.04,  # 0.04
        # m_syn_inter_in_in=0.0,  # 0
        # g_syn_layer0_ex_from_ex=0.0,  # 0.0
        # g_syn_layer0_in_from_in=0.1,  # 0.1
        # g_syn_layer0_ex_from_in=1.0,  # 1.0
        # g_syn_layer0_in_from_ex=0.5,  # 0.5
        # g_syn_layer1_ex_from_ex=0.0,  # 0.0
        # g_syn_layer1_in_from_in=0.1,  # 0.1
        # g_syn_layer1_ex_from_in=1.0,  # 1.0
        # g_syn_layer1_in_from_ex=0.5,  # 0.5
        # g_syn_inter_ex_from_ex=30.0,  # 1
        # g_syn_inter_ex_from_in=0.0,  # 0
        # g_syn_inter_in_from_ex=30.0,  # 1
        # g_syn_inter_in_from_in=0.0,  # 0
        # i_mu_excitatory_input=0.7,  # 0.7 # async: 0.05
        i_mu_inhibitory_input=1.1,  # 0.5 # async: 0.4
        # for 27Hz async(m_syn=0) firing: inhib: 0.4
        # i_mu_excitatory_output=0.7,  # 0.7
        # i_mu_inhibitory_output=1.0,  # 0.5
        # i_sigma_excitatory_input=0.0,  # 0.2 # async: 0.2
        i_sigma_inhibitory_input=0.2,  # 0.2 # async: 0.2
        # i_sigma_excitatory_output=0.0,  # 0.2
        # i_sigma_inhibitory_output=0.1,  # 0.2
        stim_layer_diff=20,
        # stim_spike_max_excitatory_input=0,  # 0.05
        # stim_spike_max_inhibitory_input=0,  # 0.02
        # stim_spike_max_excitatory_output=0,  # 0.05
        # stim_spike_max_inhibitory_output=0,  # 0.02
        # f_stim_excitatory_input=100,  # 10 # async: 0
        # f_stim_inhibitory_input=0,  # 10
        # f_stim_excitatory_output=0,  # 10
        # f_stim_inhibitory_output=0,  # 10
        # tau_is_inverse_frequency=True,
        # tau_default=2,
        plot_everything=False,
    )
    vary(
        variable_name="m_syn_output",
        arange=np.linspace(0.0, 1.0, num=20),
        plot_comment="vary m_syn_output ING async",
        variable_unit="1",
        repeats=10,
        config=config,
    )


def vary_interlayer_connections():
    config = {}
    config.update(
        # input_inhibitory_neurons=100,
        # output_inhibitory_neurons=100,
        # g_syn_inter_in_from_in=0.1,
        # m_syn_inter_in_in=0.2,
        stim_layer_diff=20,
        # i_mu_inhibitory_input=0,
        # i_sigma_inhibitory_input=0,
        # m_syn_input=1,
        # tau_default=2,
        plot_everything=False,
    )
    manual_range = np.concatenate([
        # np.linspace(0.0, 0.2, num=20)[:-1],
        np.linspace(0.0, 1.0, num=20),
    ])
    vary(
        variable_name="m_syn_inter_in_in",
        arange=manual_range,
        plot_comment="vary m_syn_inter ING",
        variable_unit="1",
        repeats=10,
        config=config,
    )


def vary_interlayer_connection_strength(**kwargs):
    config = {}
    config.update(
        input_inhibitory_neurons=100,
        output_inhibitory_neurons=100,
        # g_syn_inter_in_from_in=0.05,
        stim_layer_diff=20,
        # i_mu_inhibitory_input=0,
        # i_sigma_inhibitory_input=0,
        # i_mu_inhibitory_output=1,
        # m_syn_input=1,
        # tau_default=5,
        plot_everything=True,
        **kwargs
    )
    manual_range = np.concatenate([
        # np.linspace(0.0, 0.2, num=5)[:-1],
        # np.linspace(0.08, 0.12, num=10)[:-1],
        # np.linspace(0.12, 0.2, num=5)[:-1],
        np.linspace(0.0, 1.0, num=21)[:-1],
        # np.linspace(1.0, 5.0, num=6),
        # np.linspace(10.0, 30.0, num=9),

        # np.linspace(0.0, 1.0, num=20),
    ])
    m_syn_inter_in_in = kwargs.get("m_syn_inter_in_in", "default")
    i_mu_inhibitory_output = kwargs.get("i_mu_inhibitory_output", "default")
    vary(
        variable_name="g_syn_inter_in_from_in",
        arange=manual_range,
        plot_comment=(
            f"vary interconnect strength "
            f"m_syn={m_syn_inter_in_in} "
            f"i_mu_1={i_mu_inhibitory_output}"
        ),
        variable_unit="mS/cm²",
        repeats=1,
        config=config,
    )


def run_bartos_2007():
    """
    Parameters based on https://www.researchgate.net/publication/6620407_Synaptic_mechanisms_of_synchronized_gamma_oscillations_in_inhibitory_interneuron_networks
    """
    config = {
        "delta_t": 0.01,
        "input_excitatory_neurons": 80,
        "input_inhibitory_neurons": 20,
        "m_syn_input": 60,
        # "m_syn_output": 60,
        # "general_interconnect": 250,
        "g_syn_ex_ex": 0.1,
        "g_syn_in_in": 0.1,
        "e_syn_excitatory": -55.0,
        "e_syn_inhibitory": -55.0,
        # "spike_threshold": -40.0,
        "tau_syn_falling_inhibitory": 2.0,
        "tau_syn_falling_excitatory": 2.0,
        # "delay": 1.0,
        "delay_excitatory": 1.0,
        "delay_inhibitory": 1.0,
        "i_mu_inhibitory_input": 1.0,
        "i_mu_excitatory_input": 1.0,
        "i_sigma_excitatory_input": 0.03,
        "i_sigma_inhibitory_input": 0.03,
        "tau_is_inverse_frequency": True,
    }
    log.ex.run(
        config_updates=config,
        meta_info={
            "comment": "basic_bartos_params",
        }
    )


def run_tort_2007():
    """
    Based on https://pubmed.ncbi.nlm.nih.gov/17679692/
    """
    config = {
        "input_excitatory_neurons": 80,
        "input_inhibitory_neurons": 20,
        "m_syn_input": 100,
        # "m_syn_output": 60,
        # "general_interconnect": 250,
        "g_syn_ex_ex": 0.1,
        "g_syn_in_in": 0.1,
        "e_syn_excitatory": 0.0,
        "e_syn_inhibitory": -80.0,
        "tau_syn_rising_inhibitory": 0.07,
        "tau_syn_rising_excitatory": 0.05,
        "tau_syn_falling_inhibitory": 9.1,
        "tau_syn_falling_excitatory": 5.3,
        "delay_excitatory": 0.0,
        "delay_inhibitory": 0.0,
        "i_mu_inhibitory_input": 1.0,
        "i_mu_excitatory_input": 1.0,
        "i_sigma_excitatory_input": 0.0,
        "i_sigma_inhibitory_input": 0.0,
    }
    log.ex.run(
        config_updates=config,
        meta_info={
            "comment": "basic_tort_params",
        }
    )


def run_ping_test():
    config = {
        "delta_t": 0.05,
        "input_excitatory_neurons": 80,
        "input_inhibitory_neurons": 20,
        "m_syn_input": 60,
        # "m_syn_output": 60,
        # "general_interconnect": 250,
        "g_syn_ex_ex": 0.1,
        "g_syn_in_in": 0.1,
        "g_syn_ex_in": 0.1,
        "g_syn_in_ex": 0.1,
        "e_syn_excitatory": -55.0,
        "e_syn_inhibitory": -75.0,
        # "spike_threshold": -40.0,
        "tau_syn_falling_inhibitory": 2.0,
        "tau_syn_falling_excitatory": 2.0,
        # "delay": 1.0,
        "delay_excitatory": 1.0,
        "delay_inhibitory": 1.0,
        "i_mu_inhibitory_input": 1.0,
        "i_mu_excitatory_input": 1.0,
        "i_sigma_excitatory_input": 0.03,
        "i_sigma_inhibitory_input": 0.03,
        "tau_is_inverse_frequency": True,
    }
    log.ex.run(
        config_updates=config,
        meta_info={
            "comment": "ping_test_e_syn_inh_-75_no_delay",
        }
    )


def vary_g_syn_ping():
    config = {
        "delta_t": 0.05,
        "input_excitatory_neurons": 80,
        "input_inhibitory_neurons": 20,
        "m_syn_input": 60,
        # "m_syn_output": 60,
        # "general_interconnect": 250,
        "g_syn_ex_ex": 0.1,
        "g_syn_in_in": 0.1,
        "e_syn_excitatory": -55.0,
        "e_syn_inhibitory": -75.0,
        # "spike_threshold": -40.0,
        "tau_syn_falling_inhibitory": 2.0,
        "tau_syn_falling_excitatory": 2.0,
        # "delay": 1.0,
        "delay_excitatory": 1.0,
        "delay_inhibitory": 1.0,
        "i_mu_inhibitory_input": 1.0,
        "i_mu_excitatory_input": 1.0,
        "i_sigma_excitatory_input": 0.03,
        "i_sigma_inhibitory_input": 0.03,
        "tau_is_inverse_frequency": True,
    }
    configs = [
        dict(config, i_mu_inhibitory=1, i_mu_excitatory=1),
        dict(config, i_mu_inhibitory=2, i_mu_excitatory=2),
        dict(config, i_mu_inhibitory=3, i_mu_excitatory=3),
    ]
    config_names = [
        "i_µ=1",
        "i_µ=2",
        "i_µ=3",
    ]
    vary_configs(
        configs=configs,
        config_names=config_names,
        variable_name="g_syn_in_in",
        plot_comment="ping-plot vary_g_syn",
        arange=np.arange(0.0, 1.0, 0.05),
    )


config_bek = {
    "delta_t": 0.05,
    "input_excitatory_neurons": 80,
    "input_inhibitory_neurons": 20,
    "output_excitatory_neurons": 80,
    "output_inhibitory_neurons": 20,
    "m_syn_input": 1,
    "m_syn_output": 1,
    "m_syn_inter_ex_ex": 0.1,  # 0.1
    "m_syn_inter_ex_in": 0.0,  # 0
    "m_syn_inter_in_ex": 0.04,  # 0.04
    "m_syn_inter_in_in": 0.0,  # 0
    "g_na": 100.0,
    "g_k": 80.0,
    "g_l": 0.1,
    "g_syn_layer0_ex_from_ex": 0.0,  # 0.0
    "g_syn_layer0_in_from_in": 0.1,  # 0.1
    "g_syn_layer0_ex_from_in": 1.0,  # 1.0
    "g_syn_layer0_in_from_ex": 0.5,  # 0.5
    "g_syn_layer1_ex_from_ex": 0.0,  # 0.0
    "g_syn_layer1_in_from_in": 0.1,  # 0.1
    "g_syn_layer1_ex_from_in": 1.0,  # 1.0
    "g_syn_layer1_in_from_ex": 0.5,  # 0.5
    "g_syn_inter_ex_from_ex": 30.0,  # 1
    "g_syn_inter_ex_from_in": 0.0,  # 0
    "g_syn_inter_in_from_ex": 30.0,  # 1
    "g_syn_inter_in_from_in": 0.0,  # 0
    "e_na": 50.0,
    "e_k": -100.0,
    "e_l": -67.0,
    "e_syn_excitatory": 0.0,
    "e_syn_inhibitory": 0.0,
    # "e_syn_inhibitory": -80.0,
    "v_init_inhibitory_min": -80,
    "v_init_inhibitory_range": 20,
    # "spike_threshold": -55.0,
    "tau_syn_rising_excitatory": 0.2,
    "tau_syn_rising_inhibitory": 0.2,
    # "tau_syn_rising_inhibitory": 0.5,
    "tau_syn_falling_excitatory": 2.0,
    "tau_syn_falling_inhibitory": 2.0,
    # "tau_syn_falling_inhibitory": 10.0,
    # "delay": 1.0,
    "delay_excitatory": 0.0,
    "delay_inhibitory": 0.0,
    "i_mu_excitatory_input": 0.7,  # 0.7
    "i_mu_inhibitory_input": 0.5,  # 0.5
    "i_mu_excitatory_output": 0.7,  # 0.7
    "i_mu_inhibitory_output": 0.5,  # 0.5
    "i_sigma_excitatory_input": 0.02,  # 0.02
    "i_sigma_inhibitory_input": 0.02,  # 0.02
    "i_sigma_excitatory_output": 0.02,  # 0.02
    "i_sigma_inhibitory_output": 0.02,  # 0.02
    "f_stim_excitatory_input": 100,  # 10
    "f_stim_inhibitory_input": 0,  # 10
    "f_stim_excitatory_output": 0,  # 10
    "f_stim_inhibitory_output": 0,  # 10
    "stim_spike_max_excitatory_input": 0.05,
    "stim_spike_max_inhibitory_input": 0.02,
    "stim_spike_max_excitatory_output": 0.05,
    "stim_spike_max_inhibitory_output": 0.02,
    "tau_is_inverse_frequency": False,
    "tau_default": 1,
    "method": "bek",
}


def run_bek_2005_1a():
    config = config_bek.copy()
    config.update(
        output_excitatory_neurons=1,
        output_inhibitory_neurons=1,
        # i_mu_excitatory_input=0.7,  # 0.7 # async: 0.05
        # i_mu_inhibitory_input=0.5,  # 0.5 # async: 0.4
        # i_sigma_excitatory_input=0.05,  # 0.2 # async: 0.2
        # i_sigma_inhibitory_input=0.05,  # 0.2 # async: 0.2
        # f_stim_excitatory_input=100,  # 10 # async: 0
        # f_stim_inhibitory_input=0,  # 10
    )
    log.ex.run(
        config_updates=config,
        meta_info={
            "comment": "bek_1a",
        }
    )


def run_bek_2005_1c():
    config = config_bek.copy()
    config.update(
        # g_syn_layer0_ex_from_ex=0.0,  # 0.0
        g_syn_layer0_in_from_in=0.0,  # 0.1
        # g_syn_layer0_ex_from_in=1.0,  # 1.0
        # g_syn_layer0_in_from_ex=0.5,  # 0.5
        output_excitatory_neurons=1,
        output_inhibitory_neurons=1,
        # i_mu_excitatory_input=0.7,  # 0.7 # async: 0.05
        # i_mu_inhibitory_input=0.5,  # 0.5 # async: 0.4
        # i_sigma_excitatory_input=0.05,  # 0.2 # async: 0.2
        # i_sigma_inhibitory_input=0.05,  # 0.2 # async: 0.2
        # f_stim_excitatory_input=100,  # 10 # async: 0
        # f_stim_inhibitory_input=0,  # 10
    )
    log.ex.run(
        config_updates=config,
        meta_info={
            "comment": "bek_1c",
        }
    )


def vary_g_syn_ii_bek():
    config = config_bek.copy()
    vary(
        variable_name="g_syn_in_in",
        arange=np.arange(0, 0.25, 0.02),
        plot_comment="vary g_syn_in_in",
        variable_unit="1",
        repeats=3,
        config=config,
    )


def try_dual_layer_ping():
    config = config_bek.copy()
    config.update(
        # input_excitatory_neurons=80,
        # input_inhibitory_neurons=20,
        # output_excitatory_neurons=80,
        # output_inhibitory_neurons=20,
        # m_syn_input=1,  # 0.6 # async: 0
        # m_syn_output=1,  # 0.6
        # m_syn_inter_ex_ex=0.1,  # 0.1
        # m_syn_inter_ex_in=0.0,  # 0
        # m_syn_inter_in_ex=0.04,  # 0.04
        # m_syn_inter_in_in=0.0,  # 0
        # g_syn_layer0_ex_from_ex=0.0,  # 0.0
        # g_syn_layer0_in_from_in=0.1,  # 0.1
        # g_syn_layer0_ex_from_in=1.0,  # 1.0
        # g_syn_layer0_in_from_ex=0.5,  # 0.5
        # g_syn_layer1_ex_from_ex=0.0,  # 0.0
        # g_syn_layer1_in_from_in=0.1,  # 0.1
        # g_syn_layer1_ex_from_in=1.0,  # 1.0
        # g_syn_layer1_in_from_ex=0.5,  # 0.5
        # g_syn_inter_ex_from_ex=30.0,  # 1
        # g_syn_inter_ex_from_in=0.0,  # 0
        # g_syn_inter_in_from_ex=30.0,  # 1
        # g_syn_inter_in_from_in=0.0,  # 0
        # i_mu_excitatory_input=0.7,  # 0.7 # async: 0.05
        # i_mu_inhibitory_input=0.5,  # 0.5 # async: 0.4
        # for 27Hz async(m_syn=0) firing: inhib: 0.4
        # i_mu_excitatory_output=0.7,  # 0.7
        # i_mu_inhibitory_output=0.5,  # 0.5
        # i_sigma_excitatory_input=0.0,  # 0.2 # async: 0.2
        # i_sigma_inhibitory_input=0.0,  # 0.2 # async: 0.2
        # i_sigma_excitatory_output=0.0,  # 0.2
        # i_sigma_inhibitory_output=0.0,  # 0.2
        # stim_spike_max_excitatory_input=0,  # 0.05
        # stim_spike_max_inhibitory_input=0,  # 0.02
        # stim_spike_max_excitatory_output=0,  # 0.05
        # stim_spike_max_inhibitory_output=0,  # 0.02
        # f_stim_excitatory_input=100,  # 10 # async: 0
        # f_stim_inhibitory_input=0,  # 10
        # f_stim_excitatory_output=0,  # 10
        # f_stim_inhibitory_output=0,  # 10
        plot_everything=True,
    )
    log.ex.run(
        config_updates=config,
        meta_info={
            "comment": "ping_dual_layer_try",
        }
    )


def dual_layer_ping_default():
    config = config_bek.copy()
    config.update(
        # input_excitatory_neurons=80,
        # input_inhibitory_neurons=20,
        # output_excitatory_neurons=80,
        # output_inhibitory_neurons=20,
        # m_syn_input=1,  # 0.6 # async: 0
        # m_syn_output=1,  # 0.6
        # m_syn_inter_ex_ex=0.1,  # 0.1
        # m_syn_inter_ex_in=0.0,  # 0
        # m_syn_inter_in_ex=0.04,  # 0.04
        # m_syn_inter_in_in=0.0,  # 0
        # g_syn_layer0_ex_from_ex=0.0,  # 0.0
        # g_syn_layer0_in_from_in=0.1,  # 0.1
        # g_syn_layer0_ex_from_in=1.0,  # 1.0
        # g_syn_layer0_in_from_ex=0.5,  # 0.5
        # g_syn_layer1_ex_from_ex=0.0,  # 0.0
        # g_syn_layer1_in_from_in=0.1,  # 0.1
        # g_syn_layer1_ex_from_in=1.0,  # 1.0
        # g_syn_layer1_in_from_ex=0.5,  # 0.5
        # g_syn_inter_ex_from_ex=30.0,  # 1
        # g_syn_inter_ex_from_in=0.0,  # 0
        # g_syn_inter_in_from_ex=30.0,  # 1
        # g_syn_inter_in_from_in=0.0,  # 0
        # i_mu_excitatory_input=0.7,  # 0.7 # async: 0.05
        # i_mu_inhibitory_input=0.5,  # 0.5 # async: 0.4
        # for 27Hz async(m_syn=0) firing: inhib: 0.4
        # i_mu_excitatory_output=0.7,  # 0.7
        # i_mu_inhibitory_output=0.5,  # 0.5
        # i_sigma_excitatory_input=0.0,  # 0.2 # async: 0.2
        # i_sigma_inhibitory_input=0.0,  # 0.2 # async: 0.2
        # i_sigma_excitatory_output=0.0,  # 0.2
        # i_sigma_inhibitory_output=0.0,  # 0.2
        # stim_spike_max_excitatory_input=0,  # 0.05
        # stim_spike_max_inhibitory_input=0,  # 0.02
        # stim_spike_max_excitatory_output=0,  # 0.05
        # stim_spike_max_inhibitory_output=0,  # 0.02
        # f_stim_excitatory_input=100,  # 10 # async: 0
        # f_stim_inhibitory_input=0,  # 10
        # f_stim_excitatory_output=0,  # 10
        # f_stim_inhibitory_output=0,  # 10
        plot_everything=True,
    )
    log.ex.run(
        config_updates=config,
        meta_info={
            "comment": "ping_dual_layer_default",
        }
    )


def dual_layer_ping_async(**kwargs):
    config = config_bek.copy()
    config.update(
        # input_excitatory_neurons=80,
        # input_inhibitory_neurons=20,
        # output_excitatory_neurons=80,
        # output_inhibitory_neurons=20,
        m_syn_input=0,  # 0.6 # async: 0
        # m_syn_output=1,  # 0.6
        # m_syn_inter_ex_ex=0.1,  # 0.1
        # m_syn_inter_ex_in=0.0,  # 0
        # m_syn_inter_in_ex=0.04,  # 0.04
        # m_syn_inter_in_in=0.0,  # 0
        # g_syn_layer0_ex_from_ex=0.0,  # 0.0
        # g_syn_layer0_in_from_in=0.1,  # 0.1
        # g_syn_layer0_ex_from_in=1.0,  # 1.0
        # g_syn_layer0_in_from_ex=0.5,  # 0.5
        # g_syn_layer1_ex_from_ex=0.0,  # 0.0
        # g_syn_layer1_in_from_in=0.1,  # 0.1
        # g_syn_layer1_ex_from_in=1.0,  # 1.0
        # g_syn_layer1_in_from_ex=0.5,  # 0.5
        # g_syn_inter_ex_from_ex=30.0,  # 1
        # g_syn_inter_ex_from_in=0.0,  # 0
        # g_syn_inter_in_from_ex=30.0,  # 1
        # g_syn_inter_in_from_in=0.0,  # 0
        i_mu_excitatory_input=0.045,  # 0.7 # async: 0.05
        i_mu_inhibitory_input=0.5,  # 0.5 # async: 0.4
        # for 27Hz async(m_syn=0) firing: inhib: 0.4
        # i_mu_excitatory_output=0.7,  # 0.7
        # i_mu_inhibitory_output=0.5,  # 0.5
        i_sigma_excitatory_input=0.2,  # 0.2 # async: 0.2
        i_sigma_inhibitory_input=0.2,  # 0.2 # async: 0.2
        # i_sigma_excitatory_output=0.0,  # 0.2
        # i_sigma_inhibitory_output=0.0,  # 0.2
        # stim_spike_max_excitatory_input=0,  # 0.05
        # stim_spike_max_inhibitory_input=0,  # 0.02
        # stim_spike_max_excitatory_output=0,  # 0.05
        # stim_spike_max_inhibitory_output=0,  # 0.02
        f_stim_excitatory_input=0,  # 10 # async: 0
        # f_stim_inhibitory_input=0,  # 10
        # f_stim_excitatory_output=0,  # 10
        # f_stim_inhibitory_output=0,  # 10
        plot_everything=True,
        **kwargs
    )
    log.ex.run(
        config_updates=config,
        meta_info={
            "comment": "ping_dual_layer_async",
        }
    )


def vary_i_mu_bek(repeats=None):
    config = config_bek.copy()
    config.update(
        # input_excitatory_neurons=160,
        # input_inhibitory_neurons=40,
        # output_excitatory_neurons=0,
        # output_inhibitory_neurons=0,
        # m_syn_input=0.2,
        # i_sigma_excitatory_input=0,
        # i_sigma_inhibitory_input=0,
        # stim_spike_max_excitatory_input=0,
        # stim_spike_max_inhibitory_input=0,
        # plot_everything=True,
    )
    repeats = 10
    steps = 16
    i_mus_ex = np.linspace(0.0, 1.4, num=steps)
    i_mus_in = np.linspace(0.0, 1.0, num=steps)
    configs = []
    config_names = []
    for i_mu_ex, i_mu_in in zip(i_mus_ex, i_mus_in):
        config_ = config.copy()
        config_.update(
            i_mu_excitatory_input=i_mu_ex,
            i_mu_inhibitory_input=i_mu_in,
        )
        configs.append(config_)
        config_names.append(
            f"varying i_mu(ex_{'{:4.3f}'.format(i_mu_ex)})")
    run_configs(
        configs=configs,
        config_names=config_names,
        plot_comment=f"varying i_mu_ex and i_mu_inh",
        repeats=repeats,
        x_label="config for i_mu",
    )


def vary_i_sigma_bek():
    config = config_bek.copy()
    config.update(
        # input_excitatory_neurons=80,
        # input_inhibitory_neurons=20,
        # output_excitatory_neurons=0,
        # output_inhibitory_neurons=0,
        # m_syn_input=1.0,
        # i_sigma_excitatory_input=0,
        # i_sigma_inhibitory_input=0,
        # stim_spike_max_excitatory_input=0,
        # stim_spike_max_inhibitory_input=0,
        # stim_spike_max_excitatory_output=0,
        # stim_spike_max_inhibitory_output=0,
        # plot_everything=True,
        plot_everything=True,
        tau_is_inverse_frequency=True,
        tau_multiplier=0.1,
    )
    repeats = 10
    manual_range = np.concatenate([
        np.linspace(0.0, 0.2, num=10)[:-1],
        np.linspace(0.2, 0.5, num=10),
    ])
    # manual_range = np.linspace(0.0, 0.5, num=steps)
    i_sigmas_ex = manual_range
    i_sigmas_in = manual_range
    i_mus_ex = -i_sigmas_ex / 2 + 0.7
    i_mus_in = -i_sigmas_in / 2 + 0.5
    configs = []
    config_names = []
    for i_sigma_ex, i_sigma_in, i_mu_ex, i_mu_in in zip(
            i_sigmas_ex, i_sigmas_in,
            i_mus_ex, i_mus_in,
    ):
        config_ = config.copy()
        config_.update(
            i_mu_excitatory_input=i_mu_ex,
            i_mu_inhibitory_input=i_mu_in,
            i_sigma_excitatory_input=i_sigma_ex,
            i_sigma_inhibitory_input=i_sigma_in,
            i_mu_excitatory_output=i_mu_ex,
            i_mu_inhibitory_output=i_mu_in,
            i_sigma_excitatory_output=i_sigma_ex,
            i_sigma_inhibitory_output=i_sigma_in,
            m_syn_output=0.9,
        )
        configs.append(config_)
        config_names.append(
            f"varying i_sigma(ex_{'{:4.3f}'.format(i_sigma_ex)})")
    run_configs(
        configs=configs,
        config_names=config_names,
        plot_comment=f"varying i_sigma_ex and i_sigma_inh",
        repeats=repeats,
        x_label="config for i_sigma",
        x=i_sigmas_ex,
    )


def vary_m_syn_bek():
    config = config_bek.copy()
    config.update(
        # input_excitatory_neurons=80,
        # input_inhibitory_neurons=20,
        # output_excitatory_neurons=0,
        # output_inhibitory_neurons=0,
        # i_sigma_excitatory_input=0.6,
        # i_sigma_inhibitory_input=0.6,
        # i_mu_excitatory_input=0.4,
        # i_mu_inhibitory_input=0.2,
        # stim_spike_max_excitatory_input=0,
        # stim_spike_max_inhibitory_input=0,
        plot_everything=False,
        # tau_is_inverse_frequency=False,
        # tau_multiplier=0.1,
    )
    x_range = np.concatenate([
        # np.arange(0.0, 0.8, 0.05),
        np.arange(0.0, 1.05, 0.05),
    ])
    vary(
        variable_name="m_syn_output",
        arange=x_range,
        plot_comment="vary m_syn output PING",
        variable_unit="1",
        repeats=10,
        config=config,
    )


def vary_f_noise_bek_exc():
    config = config_bek.copy()
    config.update(
        # output_excitatory_neurons=0,
        # output_inhibitory_neurons=0,
        # plot_everything=True,
    )
    repeats = 10
    fs_ex = np.concatenate([
        np.linspace(0.0, 500, num=20)[:-1],
        np.linspace(500, 2000, num=10),
    ])
    configs = []
    config_names = []
    for f_ex in fs_ex:
        config_ = config.copy()
        config_.update(
            # f_stim_excitatory_input=f_ex,
            # f_stim_inhibitory_input=0,
            # f_stim_inhibitory_output=f_ex,
            # stim_spike_max_excitatory_input=0,
            # stim_spike_max_inhibitory_input=0,
        )
        configs.append(config_)
        config_names.append(
            f"varying f_noise(ex_{'{:4.3f}'.format(f_ex)})")
    run_configs(
        configs=configs,
        config_names=config_names,
        plot_comment=f"varying f noise",
        repeats=repeats,
        x_label="config for f_noise",
        x=fs_ex,
    )


def vary_f_noise_bek_inh():
    config = config_bek.copy()
    config.update(
        output_excitatory_neurons=0,
        output_inhibitory_neurons=0,
        # plot_everything=True,
    )
    repeats = 10
    fs_in = np.concatenate([
        np.linspace(0.0, 10, num=20)[:-1],
        np.linspace(10, 100, num=10),
    ])
    configs = []
    config_names = []
    for f_in in fs_in:
        config_ = config.copy()
        config_.update(
            f_stim_excitatory_input=100,
            f_stim_inhibitory_input=f_in,
        )
        configs.append(config_)
        config_names.append(
            f"varying f_noise(in_{'{:4.3f}'.format(f_in)})")
    run_configs(
        configs=configs,
        config_names=config_names,
        plot_comment=f"varying f noise inh",
        repeats=repeats,
        x_label="config for f_noise inh",
        x=fs_in,
    )


def vary_f_noise_bek_exc2():
    config = config_bek.copy()
    config.update(
        # input_excitatory_neurons=80,
        # input_inhibitory_neurons=20,
        # output_excitatory_neurons=0,
        # output_inhibitory_neurons=0,
        # i_sigma_excitatory_input=0.6,
        # i_sigma_inhibitory_input=0.6,
        # i_mu_excitatory_input=0.4,
        # i_mu_inhibitory_input=0.2,
        # stim_spike_max_excitatory_input=0,
        # stim_spike_max_inhibitory_input=0,
        plot_everything=False,
        # tau_is_inverse_frequency=False,
        # tau_multiplier=0.1,
    )
    fs_ex = np.concatenate([
        np.linspace(0.0, 500, num=20)[:-1],
        np.linspace(500, 2000, num=10),
    ])
    vary(
        variable_name="f_stim_excitatory_output",
        arange=fs_ex,
        plot_comment="PING vary stim frequency excitatory output",
        variable_unit="Hz",
        repeats=10,
        config=config,
    )


def vary_interlayer_connections_ex_ex_ping():
    config = config_bek.copy()
    config.update(
        # input_excitatory_neurons=80,
        # input_inhibitory_neurons=20,
        # output_excitatory_neurons=0,
        # output_inhibitory_neurons=0,
        # i_sigma_excitatory_input=0.6,
        # i_sigma_inhibitory_input=0.6,
        # i_mu_excitatory_input=0.4,
        # i_mu_inhibitory_input=0.2,
        # stim_spike_max_excitatory_input=0,
        # stim_spike_max_inhibitory_input=0,
        plot_everything=False,
        # tau_is_inverse_frequency=False,
        # tau_multiplier=0.1,
    )
    x_range = np.concatenate([
        # np.arange(0.0, 0.8, 0.05),
        np.arange(0.0, 1.05, 0.05),
    ])
    vary(
        variable_name="m_syn_inter_ex_ex",
        arange=x_range,
        plot_comment="vary m_syn_inter_ex_ex PING",
        variable_unit="1",
        repeats=10,
        config=config,
    )


def vary_interlayer_connections_ex_in_ping():
    config = config_bek.copy()
    config.update(
        # input_excitatory_neurons=80,
        # input_inhibitory_neurons=20,
        # output_excitatory_neurons=0,
        # output_inhibitory_neurons=0,
        # i_sigma_excitatory_input=0.6,
        # i_sigma_inhibitory_input=0.6,
        # i_mu_excitatory_input=0.4,
        # i_mu_inhibitory_input=0.2,
        # stim_spike_max_excitatory_input=0,
        # stim_spike_max_inhibitory_input=0,
        plot_everything=False,
        # tau_is_inverse_frequency=False,
        # tau_multiplier=0.1,
    )
    x_range = np.concatenate([
        # np.arange(0.0, 0.8, 0.05),
        np.arange(0.0, 1.05, 0.05),
    ])
    vary(
        variable_name="m_syn_inter_ex_in",
        arange=x_range,
        plot_comment="vary m_syn_inter_ex_in PING",
        variable_unit="1",
        repeats=10,
        config=config,
    )


def vary_interlayer_connections_ping():
    config = config_bek.copy()
    config.update(
        # output_excitatory_neurons=0,
        # output_inhibitory_neurons=0,
        # plot_everything=True,
    )
    repeats = 10
    m_syns = np.concatenate([
        # np.linspace(0.0, 10, num=10)[:-1],
        np.linspace(0.0, 1.0, num=20),
    ])
    configs = []
    config_names = []
    for m_syn in m_syns:
        config_ = config.copy()
        config_.update(
            m_syn_inter_ex_ex=m_syn,
            m_syn_inter_in_ex=m_syn,
            # f_stim_excitatory_input=f_ex,
            # f_stim_inhibitory_input=0,
            # f_stim_inhibitory_output=f_ex,
            # stim_spike_max_excitatory_input=0,
            # stim_spike_max_inhibitory_input=0,
        )
        configs.append(config_)
        config_names.append(
            f"varying m_syn_inter({'{:4.3f}'.format(m_syn)}) ping")
    run_configs(
        configs=configs,
        config_names=config_names,
        plot_comment=f"varying m_syn_inter ping",
        repeats=repeats,
        x_label="inter layer connections",
        x=m_syns,
    )


def vary_interlayer_connection_strength_ping():
    config = config_bek.copy()
    config.update(
        # output_excitatory_neurons=0,
        # output_inhibitory_neurons=0,
        # plot_everything=True,
    )
    repeats = 10
    g_syns = np.concatenate([
        np.linspace(0.0, 10, num=10)[:-1],
        np.linspace(10, 50, num=10),
    ])
    configs = []
    config_names = []
    for g_syn in g_syns:
        config_ = config.copy()
        config_.update(
            g_syn_inter_ex_from_ex=g_syn,
            g_syn_inter_in_from_ex=g_syn,
            # f_stim_excitatory_input=f_ex,
            # f_stim_inhibitory_input=0,
            # f_stim_inhibitory_output=f_ex,
            # stim_spike_max_excitatory_input=0,
            # stim_spike_max_inhibitory_input=0,
        )
        configs.append(config_)
        config_names.append(
            f"varying g_syn_inter({'{:4.3f}'.format(g_syn)}) ping")
    run_configs(
        configs=configs,
        config_names=config_names,
        plot_comment=f"varying g_syn_inter ping",
        repeats=repeats,
        x_label="inter layer connection strength (mS/cm²)",
        x=g_syns,
    )


def vary_f_noise_bek_inh2():
    config = config_bek.copy()
    config.update(
        # input_excitatory_neurons=80,
        # input_inhibitory_neurons=20,
        # output_excitatory_neurons=0,
        # output_inhibitory_neurons=0,
        # i_sigma_excitatory_input=0.6,
        # i_sigma_inhibitory_input=0.6,
        # i_mu_excitatory_input=0.4,
        # i_mu_inhibitory_input=0.2,
        # stim_spike_max_excitatory_input=0,
        # stim_spike_max_inhibitory_input=0,
        plot_everything=False,
        # tau_is_inverse_frequency=False,
        # tau_multiplier=0.1,
    )
    fs_ex = np.concatenate([
        np.linspace(0.0, 500, num=20)[:-1],
        np.linspace(500, 2000, num=10),
    ])
    vary(
        variable_name="f_stim_inhibitory_output",
        arange=fs_ex,
        plot_comment="PING vary stim frequency inhibitory output",
        variable_unit="Hz",
        repeats=10,
        config=config,
    )


def main():
    global pool
    pool = multiprocessing.Pool(processes=8)

    # run_wb()
    # vary_i_mu_1a()
    # vary_i_mu_1b()
    # vary_phi_3()
    # vary_e_syn_4a()
    # excitatory_network_4bc()
    # vary_i_sigma_5ab()
    # vary_n_async_6bc()
    # vary_m_syn_7a()
    # vary_m_syn_7b()
    # vary_tau_syn_10a()
    # vary_tau_syn_11a()
    # vary_i_mu_12a()
    # vary_g_syn_12b()

    ing_dual_layer_sync()
    ing_dual_layer_async()
    # vary_output_i_sigma()
    # vary_output_i_sigma_async()
    # vary_output_i_mu()
    # vary_output_i_mu_async()
    # vary_input_i_mu()
    # vary_output_m_syn_sync()
    # vary_output_m_syn_async()
    # vary_interlayer_connections()
    #
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.05,
    #     i_mu_inhibitory_output=1,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.05,
    #     i_mu_inhibitory_output=2,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.05,
    #     i_mu_inhibitory_output=3,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.1,
    #     i_mu_inhibitory_output=1,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.1,
    #     i_mu_inhibitory_output=2,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.1,
    #     i_mu_inhibitory_output=3,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.1,
    #     i_mu_inhibitory_output=4,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.1,
    #     i_mu_inhibitory_output=5,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.2,
    #     i_mu_inhibitory_output=1,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.2,
    #     i_mu_inhibitory_output=2,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.2,
    #     i_mu_inhibitory_output=3,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.2,
    #     i_mu_inhibitory_output=4,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.2,
    #     i_mu_inhibitory_output=5,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.3,
    #     i_mu_inhibitory_output=1,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.3,
    #     i_mu_inhibitory_output=2,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.3,
    #     i_mu_inhibitory_output=3,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.4,
    #     i_mu_inhibitory_output=1,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.4,
    #     i_mu_inhibitory_output=2,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.4,
    #     i_mu_inhibitory_output=3,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.5,
    #     i_mu_inhibitory_output=1,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.5,
    #     i_mu_inhibitory_output=2,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.5,
    #     i_mu_inhibitory_output=3,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.6,
    #     i_mu_inhibitory_output=1,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.6,
    #     i_mu_inhibitory_output=2,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.6,
    #     i_mu_inhibitory_output=3,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.7,
    #     i_mu_inhibitory_output=1,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.7,
    #     i_mu_inhibitory_output=2,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.7,
    #     i_mu_inhibitory_output=3,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.8,
    #     i_mu_inhibitory_output=1,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.8,
    #     i_mu_inhibitory_output=2,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.8,
    #     i_mu_inhibitory_output=3,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.9,
    #     i_mu_inhibitory_output=1,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.9,
    #     i_mu_inhibitory_output=2,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=0.9,
    #     i_mu_inhibitory_output=3,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=1.0,
    #     i_mu_inhibitory_output=0.5,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=1.0,
    #     i_mu_inhibitory_output=1,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=1.0,
    #     i_mu_inhibitory_output=2,
    # )
    # vary_interlayer_connection_strength(
    #     m_syn_inter_in_in=1.0,
    #     i_mu_inhibitory_output=3,
    # )

    # run_bartos_2007()
    # run_tort_2007()
    # run_ping_test()
    # vary_g_syn_ping()
    # run_bek_2005_1a()
    # run_bek_2005_1c()
    # vary_g_syn_ii_bek()
    # try_dual_layer_ping()
    dual_layer_ping_default()
    dual_layer_ping_async()
    # dual_layer_ping_async(i_mu_excitatory_input=0.2)
    # dual_layer_ping_async(i_mu_excitatory_input=0.1)
    # dual_layer_ping_async(i_mu_excitatory_input=0.08)
    # dual_layer_ping_async(i_mu_excitatory_input=0.07)
    # dual_layer_ping_async(i_mu_excitatory_input=0.06)
    # dual_layer_ping_async(i_mu_excitatory_input=0.05)
    # dual_layer_ping_async(i_mu_excitatory_input=0.05)
    # dual_layer_ping_async(i_mu_excitatory_input=0.05)
    # vary_i_mu_bek()
    # vary_i_sigma_bek()
    # vary_m_syn_bek()
    # vary_f_noise_bek_exc2()
    # vary_f_noise_bek_inh2()
    # vary_f_noise_bek_exc()
    # vary_f_noise_bek_inh()
    # vary_interlayer_connections_ex_ex_ping()
    # vary_interlayer_connections_ex_in_ping()
    # vary_interlayer_connections_ping()
    # vary_interlayer_connection_strength_ping()

    pool.close()


if __name__ == "__main__":
    profile = False
    if profile:
        import cProfile
        import sys

        profile_name = "results/profile.txt"
        cProfile.run(
            "sys.exit(main())",
            profile_name,
        )
        import pstats
        from pstats import SortKey

        p = pstats.Stats(profile_name)
        p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(40)
    else:
        main()

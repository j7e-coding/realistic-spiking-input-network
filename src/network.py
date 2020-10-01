import abc
import dataclasses as dc
import datetime as dt
import gc
import json
import os
import time
import typing as t
from math import ceil

import matplotlib.pyplot as plt
import memory_profiler
import numexpr as ne
import numpy as np
import pandas as pd
import scipy.signal as scipy_signal
from tqdm import tqdm

import src.logging as log

plt.switch_backend("Agg")
np.seterr(all='raise')


def pyplot_concurrency_workaround(func, tries=100, sleep=0.001):
    """
    workaround for pyplot "glyph/load_char" error
    happens if pyplot is used concurrently.
    Usage: put around saveimg calls, i.e.:
    pyplot_concurrency_workaround(lambda: plt.saveimg(file))
    """
    for _ in range(tries):
        try:
            func()
        except RuntimeError as e:
            if "load glyph" in str(e) or "load charcode" in str(e):
                time.sleep(sleep)
                continue
            else:
                raise
        return
    raise RuntimeError("could not save plots, even with handling concurrency.")


@dc.dataclass
class NetworkResults:
    coherence: float = None
    coherence_by_tau: t.Dict = dc.field(default_factory=lambda: dict())
    layer_coherences: t.List[t.List[t.Dict]] = dc.field(
        default_factory=lambda: [[dict(), dict()], [dict(), dict()]])
    # ^^ list of excit/inhib pairs for each layer,
    # each element is a dict of taus
    coherence_e_0: float = None
    coherence_i_0: float = None
    coherence_e_1: float = None
    coherence_i_1: float = None
    frequency: float = None
    frequency_std: float = None
    layer_frequencies: t.List[t.List[float]] = dc.field(
        default_factory=lambda: list())
    # ^^ list of excit/inhib pairs for each layer


class Method:
    def __init__(self, network: "Network"):
        self.network = network

    @abc.abstractmethod
    def compute_syn_current(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def k_v(self, v, m, h, n, syn):
        raise NotImplementedError()

    def k_n(self, v, n):
        return self.network.phi * (
                self.alpha_n(v) * (1. - n)
                - self.beta_n(v) * n
        )

    @abc.abstractmethod
    def k_h(self, v, n, h):
        raise NotImplementedError()

    @abc.abstractmethod
    def k_s(self, v, s):
        raise NotImplementedError()

    def k_m(self, v):
        return (
            self.alpha_m(v)
            / (self.alpha_m(v) + self.beta_m(v))
        )

    def init_m(self, v_rest):
        return self.k_m(v_rest)

    def init_n(self, v_rest):
        return (
            self.alpha_n(v_rest)
            / (
                self.alpha_n(v_rest)
                + self.beta_n(v_rest)
            )
        )

    def init_h(self, v_rest):
        return (
            self.alpha_h(v_rest)
            / (
                self.alpha_h(v_rest)
                + self.beta_h(v_rest)
            )
        )

    @abc.abstractmethod
    def alpha_m(self, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def beta_m(self, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def alpha_n(self, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def beta_n(self, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def alpha_h(self, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def beta_h(self, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def post_calc_k_h(self, h, h1, h2, h3, h4, v, n):
        raise NotImplementedError()

    @abc.abstractmethod
    def generate_poisson_spikes(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def generate_applied_currents(self):
        raise NotImplementedError()


class WangBuzsaki(Method):
    """
    Equations and parameters from "Gamma Oscillation by Synaptic
     Inhibition in a Hippocampal Interneuronal Network Model"; Wang, Buzsaki; 1996.
     """
    def compute_syn_current(self):
        # use BEKs version for now:
        BEK.compute_syn_current(self)

    def __compute_syn_current_wb(self):
        v_reshape = np.repeat(
            self.network.v.reshape((-1, 1)),
            self.network.neurons,
            axis=1,
        )
        inputs = (
                self.network.g_syn
                * self.network.s.T
                * (v_reshape - self.network.e_syn)
        )
        self.network.i_syn = np.nansum(
            inputs,
            axis=1,
        )
        # TODO: noise and interlayer synapses

    def k_v(
        self,
        v, m, h, n, syn,
    ):
        i_na = (self.network.g_na*(m**3)*h*(v-self.network.e_na))
        i_k = (self.network.g_k*(n**4)*(v-self.network.e_k))
        i_l = (self.network.g_l*(v-self.network.e_l))
        result = (
            self.network.stim  # I_app
            - i_na
            - i_k
            - i_l
            - self.network.i_syn0
        ) / self.network.capacity
        self.network.k_v_print_debug(
            v, m, h, n, syn, result, i_na, i_k, i_l)
        return result

    def k_h(self, v, n, h):
        return self.network.phi * (
            self.alpha_h(v) * (1. - h)
            - self.beta_h(v) * h
        )

    def k_s(self, v, s):
        alpha = self.network.alpha
        beta = self.network.beta
        return ne.evaluate(
            "alpha * (1. - s) / (1 + exp(-0.5*v)) - beta * s",
        )
        # return (
        #     self.alpha
        #     * (self._f(v) * (1. - s))
        #     - self.beta * s
        # )

    def alpha_h(self, x):
        return 0.07 * np.exp(-1 * (x + 58) / 20)

    def beta_h(self, x):
        return 1 / (np.exp(-0.1 * (x + 28)) + 1)

    def alpha_n(self, x):
        return -0.01 * (x + 34) / (np.exp(-0.1 * (x + 34)) - 1)

    def beta_n(self, x):
        return 0.125 * np.exp(-1 * (x + 44) / 80)

    def alpha_m(self, x):
        return -0.1 * (x + 35) / (np.exp(-0.1 * (x + 35)) - 1)

    def beta_m(self, x):
        return 4 * np.exp(-1 * (x + 60) / 18)

    def post_calc_k_h(self, h, h1, h2, h3, h4, v, n):
        h += (h1 + 2. * h2 + 2. * h3 + h4) / 6.
        return h

    def generate_poisson_spikes(self):
        # TODO: make poisson spike available to ING
        self.network.poisson_input = np.zeros((
            self.network.neurons, self.network.steps))

    def generate_applied_currents(self):
        # TODO: update wb model with const + poisson noise?
        self.network.applied_currents[
            0:self.network.input_neurons
        ] = np.random.normal(
            self.network.i_mu_inhibitory_input,
            self.network.i_sigma_inhibitory_input,
            size=(self.network.input_neurons,),
        )
        self.network.applied_currents[
            self.network.input_neurons:
        ] = np.random.normal(
            self.network.i_mu_inhibitory_output,
            self.network.i_sigma_inhibitory_output,
            size=(self.network.output_neurons,),
        )


class BEK(Method):
    """
    Equations and parameters from "Gamma oscillations mediate 
    stimulus competition and attentional selection in a cortical 
    network model"; Christoph Börgers, Steven Epstein, and Nancy J. Kopell; 2005.
    """
    def compute_syn_current(self):
        e_syn = self.network.e_syn
        v = self.network.v
        g_syn_noise = self.network.g_syn_noise
        s_noise = self.network.s_noise
        intra = np.nansum(
            self.network.synaptic_signs * self.network.g_syn * self.network.s,
            axis=1,
        )
        diff = (e_syn - v)
        # self.network.i_syn = ne.evaluate(
        #     "(intra + g_syn_noise * s_noise) * (e_syn - v)"
        # )
        self.network.i_syn = (intra + g_syn_noise * s_noise) * diff
        if self.network.debug_active:
            debug = self.network.debug
            debug_columns = self.network.debug_columns
            step_count = self.network.step_count
            debug[
                debug_columns["i_syn_noise0"],
                step_count,
            ] = (g_syn_noise * s_noise * diff)[0]
            debug[
                debug_columns["i_syn_noise1"],
                step_count,
            ] = (g_syn_noise * s_noise * diff)[1]
            debug[
                debug_columns["s_noise0"],
                step_count,
            ] = s_noise[0]
            debug[
                debug_columns["s_noise1"],
                step_count,
            ] = s_noise[1]

    def k_v(
        self,
        v, m, h, n, syn,
    ):
        h = self.k_h(v, n, h)
        i_na = self.network.g_na*(m**3)*h*(self.network.e_na-v)
        i_k = self.network.g_k*(n**4)*(self.network.e_k-v)
        i_l = self.network.g_l*(self.network.e_l-v)
        result = (
            i_na
            + i_k
            + i_l
            + self.network.stim  # I_app
            + self.network.i_syn0
        ) / self.network.capacity
        result_lower_bounds_maks = result < self.network.min_v
        result_upper_bounds_mask = result > self.network.max_v
        result[result_lower_bounds_maks] = self.network.min_v
        result[result_upper_bounds_mask] = self.network.max_v
        self.network.k_v_print_debug(
            v, m, h, n, syn, result, i_na, i_k, i_l)
        return result

    def k_h(self, v, n, h):
        return np.maximum(1.0 - 1.25 * n, 0.0)

    def k_s(self, v, s):
        alpha = self.network.alpha
        beta = self.network.beta
        return ne.evaluate(
            "(1 + tanh(v / 10)) / 2 * alpha * (1. - s)- beta * s",
        )
        # return (
        #     (
        #         1
        #         + np.tanh(v / 10)
        #     ) / 2
        #     * self.alpha
        #     * (1. - s)
        #     - self.beta * s
        # )

    def init_h(self, v_rest):
        n = super().init_n(v_rest=v_rest)
        return self.k_h(v=None, n=n, h=None)

    def alpha_h(self, x):
        # does not exist for BEK
        raise NotImplementedError()

    def beta_h(self, x):
        # does not exist for BEK
        raise NotImplementedError()

    def alpha_n(self, x):
        return 0.032 * (52 + x) / (1 - np.exp(-0.2 * (x + 52)))

    def beta_n(self, x):
        return 0.5 * np.exp(-0.025 * (57 + x))

    def alpha_m(self, x):
        return 0.32 * (54 + x) / (1 - np.exp(-0.25 * (x + 54)))

    def beta_m(self, x):
        return (
            0.28 * (27 + x)
            / (
                np.exp(0.2 * (x + 27)) - 1
                # np.exp(0.2 * (27 + x) - 1)  # older paper
            )
        )

    def post_calc_k_h(self, h, h1, h2, h3, h4, v, n):
        return self.k_h(v=v, n=n, h=h)

    def generate_poisson_spikes(self):
        delta_t = self.network.delta_t
        steps = self.network.steps
        excitatory = self.network.excitatory
        layer = self.network.layer
        # generate poisson streams for every cell:
        f_input_ex = self.network.f_stim_excitatory_input / 1_000 * delta_t
        f_input_inh = self.network.f_stim_inhibitory_input / 1_000 * delta_t
        f_output_ex = self.network.f_stim_excitatory_output / 1_000 * delta_t
        f_output_inh = self.network.f_stim_inhibitory_output / 1_000 * delta_t
        poisson_stream_input_ex = np.random.poisson(
            lam=f_input_ex,
            size=(self.network.input_excitatory_neurons, steps),
        )
        poisson_stream_input_inh = np.random.poisson(
            lam=f_input_inh,
            size=(self.network.input_inhibitory_neurons, steps),
        )
        poisson_stream_output_ex = np.random.poisson(
            lam=f_output_ex,
            size=(self.network.output_excitatory_neurons, steps),
        )
        poisson_stream_output_inh = np.random.poisson(
            lam=f_output_inh,
            size=(self.network.output_inhibitory_neurons, steps),
        )
        self.network.poisson_spikes = np.zeros(
            (self.network.neurons, steps), dtype=int)
        self.network.poisson_spikes[
            excitatory & (layer == 0),
            :
        ] = poisson_stream_input_ex
        self.network.poisson_spikes[
            ~excitatory & (layer == 0),
            :
        ] = poisson_stream_input_inh
        self.network.poisson_spikes[
            excitatory & (layer == 1),
            :
        ] = poisson_stream_output_ex
        self.network.poisson_spikes[
            ~excitatory & (layer == 1),
            :
        ] = poisson_stream_output_inh

    def generate_applied_currents(self):
        net = self.network
        # input:
        net.applied_currents[0:net.input_excitatory_neurons] = (
            net.i_mu_excitatory_input
            + net.i_sigma_excitatory_input
            * np.random.random((net.input_excitatory_neurons,))
        )
        net.applied_currents[
            net.input_excitatory_neurons:net.input_neurons
        ] = (
            net.i_mu_inhibitory_input
            + net.i_sigma_inhibitory_input
            * np.random.random((net.input_inhibitory_neurons,))
        )
        # output
        net.applied_currents[
            net.input_neurons
            :net.input_neurons+net.output_excitatory_neurons
        ] = (
                net.i_mu_excitatory_output
                + net.i_sigma_excitatory_output
                * np.random.random((net.output_excitatory_neurons,))
        )
        net.applied_currents[
            net.input_neurons+net.output_excitatory_neurons:
        ] = (
                net.i_mu_inhibitory_output
                + net.i_sigma_inhibitory_output
                * np.random.random((net.output_inhibitory_neurons,))
        )


class Network:
    @log.ex.capture
    def __init__(
            self,
            input_inhibitory_neurons,
            output_inhibitory_neurons,
            input_excitatory_neurons,
            output_excitatory_neurons,
            interlayer_connection_strength,
            t_max,
            delta_t,
            delay,
            delay_excitatory,
            delay_inhibitory,
            spike_threshold,
            epsilon_t,
            duration,
            i_mu_inhibitory_input,
            i_mu_excitatory_input,
            i_mu_inhibitory_output,
            i_mu_excitatory_output,
            i_sigma_inhibitory_input,
            i_sigma_excitatory_input,
            i_sigma_inhibitory_output,
            i_sigma_excitatory_output,
            f_stim_excitatory_input,
            f_stim_inhibitory_input,
            f_stim_excitatory_output,
            f_stim_inhibitory_output,
            stim_spike_max_excitatory_input,
            stim_spike_max_inhibitory_input,
            stim_spike_max_excitatory_output,
            stim_spike_max_inhibitory_output,
            decay_poisson_spike,
            stim_start,
            stim_start_std,
            stim_layer_diff,
            v_rest,
            v_init_inhibitory_min,
            v_init_inhibitory_range,
            min_v,
            max_v,
            g_na, g_k, g_l,
            g_syn_layer0_in_from_in,
            g_syn_layer0_ex_from_ex,
            g_syn_layer0_ex_from_in,
            g_syn_layer0_in_from_ex,
            g_syn_layer1_in_from_in,
            g_syn_layer1_ex_from_ex,
            g_syn_layer1_ex_from_in,
            g_syn_layer1_in_from_ex,
            g_syn_inter_ex_from_ex,
            g_syn_inter_ex_from_in,
            g_syn_inter_in_from_ex,
            g_syn_inter_in_from_in,
            e_na, e_k, e_l,
            e_syn_inhibitory, e_syn_excitatory,
            capacity,
            phi,
            m_syn_input,
            m_syn_output,
            m_syn_inter_ex_ex,
            m_syn_inter_ex_in,
            m_syn_inter_in_ex,
            m_syn_inter_in_in,
            alpha_excitatory,
            alpha_inhibitory,
            beta_excitatory,
            beta_inhibitory,
            results_folder,
            sample_window,
            plot_everything,
            tau_is_inverse_frequency,
            tau_multiplier,
            tau_default,
            method,
            debug_active,
            tau_syn_falling_inhibitory=None,
            tau_syn_falling_excitatory=None,
            tau_syn_rising_inhibitory=None,
            tau_syn_rising_excitatory=None,
    ):
        """
        renamed s to open_ion_channels
        renamed spike to spike_train
        information of pre is in synapses
        """
        self.g_na = g_na
        self.g_k = g_k
        self.g_l = g_l
        self.g_syn_layer0_ex_from_ex = g_syn_layer0_ex_from_ex
        self.g_syn_layer0_in_from_in = g_syn_layer0_in_from_in
        self.g_syn_layer0_ex_from_in = g_syn_layer0_ex_from_in
        self.g_syn_layer0_in_from_ex = g_syn_layer0_in_from_ex
        self.g_syn_layer1_ex_from_ex = g_syn_layer1_ex_from_ex
        self.g_syn_layer1_in_from_in = g_syn_layer1_in_from_in
        self.g_syn_layer1_ex_from_in = g_syn_layer1_ex_from_in
        self.g_syn_layer1_in_from_ex = g_syn_layer1_in_from_ex
        self.g_syn_inter_ex_from_ex = g_syn_inter_ex_from_ex
        self.g_syn_inter_ex_from_in = g_syn_inter_ex_from_in
        self.g_syn_inter_in_from_ex = g_syn_inter_in_from_ex
        self.g_syn_inter_in_from_in = g_syn_inter_in_from_in
        self.e_na = e_na
        self.e_k = e_k
        self.e_l = e_l
        self.e_syn_excitatory = e_syn_excitatory
        self.e_syn_inhibitory = e_syn_inhibitory
        self.capacity = capacity
        self.phi = phi
        self.v_init_inhibitory_min = v_init_inhibitory_min
        self.v_init_inhibitory_range = v_init_inhibitory_range
        self.min_v = min_v
        self.max_v = max_v
        self.m_syn_input = m_syn_input
        self.m_syn_output = m_syn_output
        self.m_syn_inter_ex_ex = m_syn_inter_ex_ex
        self.m_syn_inter_ex_in = m_syn_inter_ex_in
        self.m_syn_inter_in_ex = m_syn_inter_in_ex
        self.m_syn_inter_in_in = m_syn_inter_in_in
        self.interlayer_connection_strength = interlayer_connection_strength
        self.i_mu_excitatory_input = i_mu_excitatory_input
        self.i_mu_inhibitory_input = i_mu_inhibitory_input
        self.i_mu_excitatory_output = i_mu_excitatory_output
        self.i_mu_inhibitory_output = i_mu_inhibitory_output
        self.i_sigma_excitatory_input = i_sigma_excitatory_input
        self.i_sigma_inhibitory_input = i_sigma_inhibitory_input
        self.i_sigma_excitatory_output = i_sigma_excitatory_output
        self.i_sigma_inhibitory_output = i_sigma_inhibitory_output
        self.f_stim_excitatory_input = f_stim_excitatory_input
        self.f_stim_inhibitory_input = f_stim_inhibitory_input
        self.f_stim_excitatory_output = f_stim_excitatory_output
        self.f_stim_inhibitory_output = f_stim_inhibitory_output
        self.stim_spike_max_excitatory_input = stim_spike_max_excitatory_input
        self.stim_spike_max_inhibitory_input = stim_spike_max_inhibitory_input
        self.stim_spike_max_excitatory_output = stim_spike_max_excitatory_output
        self.stim_spike_max_inhibitory_output = stim_spike_max_inhibitory_output
        self.decay_poisson_spike = decay_poisson_spike
        self.delta_t = delta_t
        self.duration = duration
        self.t_max = t_max
        self.delay = delay
        self.delay_excitatory = delay_excitatory
        self.delay_inhibitory = delay_inhibitory
        self.spike_threshold = spike_threshold
        self.epsilon_t = epsilon_t
        self.alpha_excitatory = alpha_excitatory
        self.alpha_inhibitory = alpha_inhibitory
        self.beta_excitatory = beta_excitatory
        self.beta_inhibitory = beta_inhibitory
        self.results_folder = results_folder
        self.run_id = log.ex.current_run._id
        self.sample_window = sample_window
        self.plot_everything = plot_everything
        self.tau_is_inverse_frequency = tau_is_inverse_frequency
        self.tau_multiplier = tau_multiplier
        self.tau_default = tau_default
        methods = {
            "wb": WangBuzsaki,
            "bek": BEK,
        }
        try:
            self.method = methods[method](network=self)
        except KeyError:
            raise RuntimeError(f"Unknown method: {method}")
        self.debug_active = debug_active
        if tau_syn_rising_excitatory:
            self.alpha_excitatory = 1 / tau_syn_rising_excitatory
        if tau_syn_rising_inhibitory:
            self.alpha_inhibitory = 1 / tau_syn_rising_inhibitory
        if tau_syn_falling_excitatory:
            self.beta_excitatory = 1 / tau_syn_falling_excitatory
        if tau_syn_falling_inhibitory:
            self.beta_inhibitory = 1 / tau_syn_falling_inhibitory
        self.input_inhibitory_neurons = input_inhibitory_neurons
        self.output_inhibitory_neurons = output_inhibitory_neurons
        self.input_excitatory_neurons = input_excitatory_neurons
        self.output_excitatory_neurons = output_excitatory_neurons
        self.input_neurons = (
                input_excitatory_neurons
                + input_inhibitory_neurons
        )
        self.output_neurons = (
            output_excitatory_neurons
            + output_inhibitory_neurons
        )
        self.neurons = (
            self.input_neurons
            + self.output_neurons
        )
        self.steps = int(ceil(t_max / delta_t))
        log.logger.debug(f"Max number of time steps: {self.steps}")
        self.step_count = 0
        self.time_now = 0.0
        self.spike_trains_np = np.full((self.neurons, self.steps), False)

        self.old_time = 0.0
        self.old_time_inhibitory = np.zeros((self.input_inhibitory_neurons,))
        self.v = (
            self.v_init_inhibitory_min
            + self.v_init_inhibitory_range
            * np.random.random((self.neurons,))
        )
        self.m = np.full((self.neurons, ), self.method.init_m(v_rest=self.v))
        self.n = np.full((self.neurons, ), self.method.init_n(v_rest=self.v))
        self.h = np.full((self.neurons, ), self.method.init_h(v_rest=self.v))
        self.k1 = None
        self.k2 = None
        self.k3 = None
        self.is_on = np.full((self.neurons, ), True)
        self.i_syn = np.full((self.neurons, ), 0.0)
        self.i_syn0 = np.full((self.neurons, ), 0.0)
        self.stim = np.full(self.neurons, np.nan)
        self.stim_time = (
            stim_start
            + stim_start_std * np.random.random((self.neurons,))
        )
        self.stim_time[self.input_neurons:] += stim_layer_diff
        ex_input = np.full((self.input_neurons, ), False)
        ex_input[:self.input_excitatory_neurons] = True
        ex_output = np.full((self.output_neurons,), False)
        ex_output[:self.output_excitatory_neurons] = True
        self.excitatory = np.concatenate((ex_input, ex_output))
        self.layer = np.full(self.neurons, 0)
        self.layer[self.input_neurons:] = 1
        self.synapse_probability = self.__calculate_synapse_probabilities()
        self.random_synapses = np.random.random((self.neurons, self.neurons))
        self.synapses = self.random_synapses < self.synapse_probability
        """^^self.synapse[from, to]"""
        np.fill_diagonal(self.synapses, False)
        synapses_per_neuron = (
            self
            .synapses
            .sum(axis=0)
            .mean()
        )
        log.logger.debug(
            f"Number of synapses per neuron: {synapses_per_neuron}")
        if self.output_neurons:
            self.synapses_per_output_neuron = (
                self
                .synapses
                [:, self.input_neurons:]
                .sum(axis=0)
                .mean()
            )
        else:
            self.synapses_per_output_neuron = 0
        # open ion channels, s[from, to]: # TODO: shouldn't this be reversed?
        self.s = np.full((self.neurons, self.neurons), np.nan)
        self.s[self.synapses] = 0.0
        self.syn_drive = np.zeros((self.steps,))
        # synaptic gating variable for the poisson noise stream:
        self.s_noise = np.full((self.neurons, ), 0.0)
        self.membrane_potential_history = np.full(
            (self.neurons, self.steps),
            np.nan,
        )
        self.syn_current_history = np.full(
            (self.neurons, self.steps),
            np.nan,
        )
        self.beta = np.full((self.neurons, ), self.beta_excitatory)
        self.beta[~self.excitatory] = self.beta_inhibitory
        self.alpha = np.full((self.neurons, ), self.alpha_excitatory)
        self.alpha[~self.excitatory] = self.alpha_inhibitory
        self.e_syn = np.full((self.neurons, ), self.e_syn_excitatory)
        self.e_syn[~self.excitatory] = self.e_syn_inhibitory
        self. g_syn = self.__init_g_syn()
        # g_syn for noise:
        self.g_syn_noise = self.__init_g_syn_noise()
        # index after which to include steps in coherence/... calcs:
        self.sample_start_index = int(
            (self.t_max - self.sample_window) / self.delta_t)

        self.applied_currents = np.zeros((self.neurons,))
        self.method.generate_applied_currents()
        self.poisson_input = np.zeros((self.neurons, self.steps))
        self.poisson_spikes = np.zeros((self.neurons, self.steps))
        self.method.generate_poisson_spikes()
        self.synaptic_signs = np.full((self.neurons,), 1)
        self.synaptic_signs[~self.excitatory] = -1
        debug_names = [
            "v0",
            "v1",
            "k_v",
            "i_syn0",
            "i_syn1",
            "s01",
            "s10",
            "spikes_e",
            "spikes_i",
            "i_syn_noise0",
            "i_syn_noise1",
            "s_noise0",
            "s_noise1",
            "i_na",
            "i_k",
            "i_l",
            "i_stim",
            "m",
            "n",
            "h",
            "diff_na",
            "diff_k",
            "diff_l",
        ]
        self.debug_columns = dict(zip(debug_names, range(len(debug_names))))
        self.debug_print_columns = [
            # "v",
            # "k_v",
            "s01",
            "s10",
            "i_syn0",
            "spikes_e",
            "spikes_i",
        ]
        self.debug_print_styles = [
            '-',
            '-',
            '-',
            'rx',
            'bx',
        ]
        self.debug = None
        if self.debug_active:
            self.debug = np.full((len(self.debug_columns), self.steps), np.nan)

    def __init_g_syn(self):
        g_syn = np.full((self.neurons, self.neurons), np.nan)
        excitatory_0_divisor = self.input_excitatory_neurons
        if not excitatory_0_divisor:
            excitatory_0_divisor = 1
        excitatory_1_divisor = self.output_excitatory_neurons
        if not excitatory_1_divisor:
            excitatory_1_divisor = 1
        inhibitory_1_divisor = self.output_inhibitory_neurons
        if not inhibitory_1_divisor:
            inhibitory_1_divisor = 1
        # set up g_syn for intra-layer connections:
        # ex getting input from ex:
        g_syn[self.two_dim_mask(
            layer1=0, layer2=0,
            type1="ex", type2="ex",
        )] = self.g_syn_layer0_ex_from_ex / excitatory_0_divisor
        g_syn[self.two_dim_mask(
            layer1=1, layer2=1,
            type1="ex", type2="ex",
        )] = self.g_syn_layer1_ex_from_ex / excitatory_1_divisor
        # inh getting input from inh:
        g_syn[self.two_dim_mask(
            layer1=0, layer2=0,
            type1="in", type2="in",
        )] = self.g_syn_layer0_in_from_in / self.input_inhibitory_neurons
        g_syn[self.two_dim_mask(
            layer1=1, layer2=1,
            type1="in", type2="in",
        )] = self.g_syn_layer1_in_from_in / inhibitory_1_divisor
        # ex getting input from inh
        g_syn[self.two_dim_mask(
            layer1=0, layer2=0,
            type1="ex", type2="in",
        )] = self.g_syn_layer0_ex_from_in / self.input_inhibitory_neurons
        g_syn[self.two_dim_mask(
            layer1=1, layer2=1,
            type1="ex", type2="in",
        )] = self.g_syn_layer1_ex_from_in / inhibitory_1_divisor
        # inh getting input from ex:
        g_syn[self.two_dim_mask(
            layer1=0, layer2=0,
            type1="in", type2="ex",
        )] = self.g_syn_layer0_in_from_ex / excitatory_0_divisor
        g_syn[self.two_dim_mask(
            layer1=1, layer2=1,
            type1="in", type2="ex",
        )] = self.g_syn_layer1_in_from_ex / excitatory_1_divisor
        # set connections from layer 1 to layer 0 to zero:
        g_syn[self.two_dim_layer_mask(0, 1)] = 0
        # set up g_syn for inter-layer connections:
        g_syn[self.two_dim_mask(
            layer1=1, layer2=0,
            type1="ex", type2="ex",
        )] = self.g_syn_inter_ex_from_ex / excitatory_0_divisor
        g_syn[self.two_dim_mask(
            layer1=1, layer2=0,
            type1="in", type2="in",
        )] = self.g_syn_inter_in_from_in / self.input_inhibitory_neurons
        g_syn[self.two_dim_mask(
            layer1=1, layer2=0,
            type1="ex", type2="in",
        )] = self.g_syn_inter_ex_from_in / self.input_inhibitory_neurons
        g_syn[self.two_dim_mask(
            layer1=1, layer2=0,
            type1="in", type2="ex",
        )] = self.g_syn_inter_in_from_ex / excitatory_0_divisor
        np.fill_diagonal(g_syn, 0)
        return g_syn

    def __init_g_syn_noise(self):
        g_syn_noise = np.full((self.neurons,), np.nan)
        g_syn_noise[
            self.excitatory
            & (self.layer == 0)
            ] = self.stim_spike_max_excitatory_input
        g_syn_noise[
            (~self.excitatory)
            & (self.layer == 0)
            ] = self.stim_spike_max_inhibitory_input
        g_syn_noise[
            self.excitatory
            & (self.layer == 1)
            ] = self.stim_spike_max_excitatory_output
        g_syn_noise[
            (~self.excitatory)
            & (self.layer == 1)
            ] = self.stim_spike_max_inhibitory_output
        return g_syn_noise

    def __calculate_synapse_probabilities(self) -> np.ndarray:
        """
        order:
            input, output
                excitatory, inhibitory
        """
        probabilities = np.zeros((self.neurons, self.neurons))
        # input_internal:
        probabilities[self.two_dim_mask(
            layer1=0,
            layer2=0,
        )] = self.m_syn_input
        # output internal:
        if self.output_neurons:
            probabilities[self.two_dim_mask(
                layer1=1,
                layer2=1,
            )] = self.m_syn_output
        # from input to output:
        probabilities[self.two_dim_mask(
            layer1=1,
            layer2=0,
            type1="ex",
            type2="ex",
        )] = self.m_syn_inter_ex_ex
        probabilities[self.two_dim_mask(
            layer1=1,
            layer2=0,
            type1="ex",
            type2="in",
        )] = self.m_syn_inter_ex_in
        probabilities[self.two_dim_mask(
            layer1=1,
            layer2=0,
            type1="in",
            type2="ex",
        )] = self.m_syn_inter_in_ex
        probabilities[self.two_dim_mask(
            layer1=1,
            layer2=0,
            type1="in",
            type2="in",
        )] = self.m_syn_inter_in_in
        np.fill_diagonal(probabilities, 0.0)
        return probabilities

    def single_step(self):
        self.time_now = self.step_count * self.delta_t
        stim = np.full((self.neurons,), 0.0)
        mask = self.stim_time <= self.time_now
        mask &= (
            (self.duration + self.stim_time)
            >= self.time_now
        )
        # self.generate_applied_currents()
        random_values = self.applied_currents[mask]
        stim[mask] = random_values
        # stim[:self.input_neurons] += self.poisson_input[:, self.step_count]
        self.stim = stim
        self.compute_currents()
        self.calculate_syn_drive()
        self.print_summary()
        self.step_count += 1

    def compute_currents(self):
        # log.logger.debug("computing currents")

        # syn = 1
        # log.logger.debug("computing membrane potentials")
        self.calculate_variables()

        # log.logger.debug("computing spikes")
        spike_mask = self.v >= self.spike_threshold
        spike_mask &= self.is_on.astype(bool)
        self.spike_trains_np[spike_mask, self.step_count] = True
        if self.debug_active:
            self.debug[
                self.debug_columns["spikes_e"],
                self.step_count,
            ] = self.spike_trains_np[self.excitatory, self.step_count].sum()
            self.debug[
                self.debug_columns["spikes_i"],
                self.step_count,
            ] = self.spike_trains_np[~self.excitatory, self.step_count].sum()
        self.is_on[spike_mask] = False
        after_spike_mask = self.v < self.spike_threshold
        self.is_on[after_spike_mask] = True

        # log.logger.debug("computing open ion channels")
        membrane_potential_input = self.compute_v_input()

        self.calculate_synaptic_gating_variables(
            v_pre_synaptic=membrane_potential_input)
        # s = np.nan_to_num(self.s)
        # assert (
        #     (s >= 0.0)
        #     & (s <= 1.0)
        # ).all(), "open ion channels out of bounds"
        self.method.compute_syn_current()

        self.membrane_potential_history[:, self.step_count] = self.v
        self.syn_current_history[:, self.step_count] = self.i_syn
        if (
            np.abs(self.time_now - (self.old_time + self.delay))
            <= self.delta_t + self.epsilon_t
        ):
            self.old_time = self.time_now
            self.i_syn0 = self.i_syn
        self.check_and_set_delay(
            delay=self.delay_excitatory, mask=self.excitatory)
        self.check_and_set_delay(
            delay=self.delay_inhibitory, mask=~self.excitatory)
        # assert (
        #     (self.v >= -250.0)
        #     & (self.v <= 200.0)
        # ).all(), f"v out of bounds (min:{self.v.min()}, max: {self.v.max()})"

        # log.logger.debug("done computing currents")

    def check_and_set_delay(self, delay, mask):
        if 0 < delay <= self.time_now:
            delay_index_offset = int(delay // self.delta_t)
            self.i_syn0[mask] = self.syn_current_history[
                mask,
                self.step_count - delay_index_offset,
            ]

    def compute_v_input(self):
        # s[from, to]
        index = np.where(self.synapses)
        # index_from = index[0]
        index_to = index[1]

        membrane_potential_input = np.zeros((self.neurons, self.neurons))
        membrane_potential_input[index] = self.v[index_to]
        return membrane_potential_input

    def calculate_variables(self):
        """
        Solve differential equation using 4th-order Runge-Kutta.
        """
        # make variables local for better performance:
        k_v = self.method.k_v
        k_h = self.method.k_h
        k_n = self.method.k_n
        k_m = self.method.k_m
        delta_t = self.delta_t
        v = self.v
        m = self.m
        n = self.n
        h = self.h

        syn = self.synaptic_signs

        v[v < self.min_v] = self.min_v
        v[v > self.max_v] = self.max_v

        k1 = delta_t * k_v(
            v=v,
            m=m,
            n=n,
            h=h,
            syn=syn,
        )
        h1 = delta_t * k_h(v=v, n=n, h=h)
        n1 = delta_t * k_n(v=v, n=n)

        k2 = delta_t * k_v(
            v=v + (.5 * k1),
            m=m,
            h=h + (.5 * h1),
            n=n + (.5 * n1),
            syn=syn,
        )
        h2 = delta_t * k_h(
            v=v + .5 * k1,
            n=n + .5 * n1,
            h=h + .5 * h1,
        )
        n2 = delta_t * k_n(
            v=v + .5 * k1,
            n=n + .5 * n1,
        )

        k3 = delta_t * k_v(
            v=v + .5 * k2,
            m=m,
            h=h + .5 * h2,
            n=n + .5 * n2,
            syn=syn,
        )
        h3 = delta_t * k_h(
            v=v + .5 * k2,
            n=n + .5 * n2,
            h=h + .5 * h2,
        )
        n3 = delta_t * k_n(
            v=v + .5 * k2,
            n=n + .5 * n2,
        )

        k4 = delta_t * k_v(
            v=v + k3,
            m=m,
            h=h + h3,
            n=n + n3,
            syn=syn,
        )
        h4 = delta_t * k_h(
            v=v + k3,
            n=n + n3,
            h=h + h3,
        )
        n4 = delta_t * k_n(
            v=v + k3,
            n=n + n3,
        )

        h = self.method.post_calc_k_h(
            h=h, h1=h1, h2=h2, h3=h3, h4=h4,
            v=v, n=n,
        )
        m = k_m(v)
        v += (k1 + 2. * k2 + 2. * k3 + k4) / 6.
        n += (n1 + 2. * n2 + 2. * n3 + n4) / 6.
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.v = v
        self.m = m
        self.n = n
        self.h = h

    def calculate_synaptic_gating_variables(self, v_pre_synaptic):
        k_s = self.method.k_s
        self.s = self.calculate_synaptic_gating_variable(
            v_pre_synaptic=v_pre_synaptic,
            s=self.s,
            k_s=k_s,
        )
        spike_mask = self.poisson_spikes[:, self.step_count].astype(bool)
        self.s_noise[spike_mask] = 1.0
        self.s_noise = self.calculate_synaptic_gating_variable(
            v_pre_synaptic=v_pre_synaptic,
            s=self.s_noise,
            k_s=self.k_s_noise,
        )

        if self.debug_active:
            self.debug[
                self.debug_columns["s01"],
                self.step_count,
            ] = self.s[0, 1]
            self.debug[
                self.debug_columns["s10"],
                self.step_count,
            ] = self.s[1, 0]

    def calculate_synaptic_gating_variable(self, v_pre_synaptic, s, k_s):
        delta_t = self.delta_t
        v = v_pre_synaptic
        s1 = delta_t * k_s(
            v=v,
            s=s,
        )
        s2 = delta_t * k_s(
            v=v + .5 * self.k1,
            s=s + (.5 * s1),
        )
        s3 = delta_t * k_s(
            v=v + .5 * self.k2,
            s=s + (.5 * s2),
        )
        s4 = delta_t * k_s(
            v=v + self.k3,
            s=s + s3,
        )
        # return ne.evaluate("s + ((s1 + 2. * s2 + 2. * s3 + s4) / 6.)")
        return s + ((s1 + 2. * s2 + 2. * s3 + s4) / 6.)

    def k_v_print_debug(
            self, v, m, h, n, syn, result, i_na, i_k, i_l):
        if self.debug_active:
            debug = self.debug
            debug_columns = self.debug_columns
            step_count = self.step_count
            debug[
                debug_columns["v0"],
                step_count,
            ] = v[0]
            debug[
                debug_columns["v1"],
                step_count,
            ] = v[1]
            debug[
                debug_columns["k_v"],
                step_count,
            ] = result[0]
            debug[
                debug_columns["i_syn0"],
                step_count,
            ] = self.i_syn0[0]
            debug[
                debug_columns["i_syn1"],
                step_count,
            ] = self.i_syn0[1]
            debug[
                debug_columns["i_na"],
                step_count,
            ] = i_na[0]
            debug[
                debug_columns["i_k"],
                step_count,
            ] = i_k[0]
            debug[
                debug_columns["i_l"],
                step_count,
            ] = i_l[0]
            debug[
                debug_columns["i_stim"],
                step_count,
            ] = self.stim[0]
            debug[
                debug_columns["m"],
                step_count,
            ] = m[0]
            debug[
                debug_columns["n"],
                step_count,
            ] = n[0]
            debug[
                debug_columns["h"],
                step_count,
            ] = h[0]
            debug[
                debug_columns["diff_na"],
                step_count,
            ] = (self.e_na-v)[0]
            debug[
                debug_columns["diff_k"],
                step_count,
            ] = (self.e_k-v)[0]
            debug[
                debug_columns["diff_l"],
                step_count,
            ] = (self.e_l-v)[0]
            self.debug = debug

    def k_s_input(self, v, s):
        raise NotImplementedError()

    def k_s_noise(self, v, s):
        """
        calculate synaptic gating variables s for noise streams.
        Noise is a poisson stream in self.poisson_spikes.
        The rest of the dynamics should ensure an exponential decay.
        Args:
            v: the current membrane potential
            s: the current state of the (noise) gating variable

        Returns:
            the change in s
        """
        return (
            # self.alpha
            # * (1.0 - s)
            - self.beta * s
        )

    @staticmethod
    def _f(x):
        return (
            1/(1 + np.exp(-0.5*x))
        )

    @staticmethod
    def _f_wb(x):
        return (
            1 / (1 + np.exp(-0.5 * x))
        )

    @staticmethod
    def two_dim_type_mask(type_mask_1, type_mask_2):
        """
        Creates a 2-dimensional bool mask from 2 1-d masks in the two directions.
        """
        return np.dot(
            type_mask_1.reshape((-1, 1)),
            type_mask_2.reshape((1, -1)),
        )

    def two_dim_layer_mask(self, layer_1: int, layer_2: int):
        """
        Creates a 2-dimensional bool mask given the two layers to connect.
        """
        return (
            (self.layer == layer_1).reshape(-1, 1)
            * (self.layer == layer_2).reshape(1, -1)
        )

    def two_dim_mask(
            self,
            layer1,
            layer2,
            type1=None,
            type2=None,
    ):
        """
        Creates a 2-dimensional bool mask from 2 1-d masks
        in the two directions and the indices of the layers to connect.
        the type_masks can be bool arrays or "ex" for self.excitatory,
        "in" for ~self.excitatory or "both" == None for all True.
        """
        def check_type(mask_):
            if mask_ == "ex":
                return self.excitatory
            elif mask_ == "in":
                return ~self.excitatory
            elif mask_ == "both" or mask_ is None:
                return np.full_like(self.excitatory, True)
            else:
                return mask_

        layer_mask = self.two_dim_layer_mask(layer_1=layer1, layer_2=layer2)
        type1 = check_type(type1)
        type2 = check_type(type2)
        type_mask = self.two_dim_type_mask(
            type_mask_1=type1,
            type_mask_2=type2,
        )
        return layer_mask & type_mask

    def generate_applied_currents(self):
        # TODO: update wb model with const + poisson noise?
        if self.method == "wb":
            self.applied_currents[0:self.input_neurons] = np.random.normal(
                self.i_mu_inhibitory_input,
                self.i_sigma_inhibitory_input,
                size=(self.input_neurons,),
            )
            self.applied_currents[self.input_neurons:] = np.random.normal(
                self.i_mu_inhibitory_output,
                self.i_sigma_inhibitory_output,
                size=(self.output_neurons,),
            )
        elif self.method == "bek":
            # input:
            self.applied_currents[0:self.input_excitatory_neurons] = (
                self.i_mu_excitatory_input
                + self.i_sigma_excitatory_input
                * np.random.random((self.input_excitatory_neurons,))
            )
            self.applied_currents[
                self.input_excitatory_neurons:self.input_neurons
            ] = (
                self.i_mu_inhibitory_input
                + self.i_sigma_inhibitory_input
                * np.random.random((self.input_inhibitory_neurons,))
            )
            # output
            self.applied_currents[
                self.input_neurons
                :self.input_neurons+self.output_excitatory_neurons
            ] = (
                    self.i_mu_excitatory_output
                    + self.i_sigma_excitatory_output
                    * np.random.random((self.output_excitatory_neurons,))
            )
            self.applied_currents[
                self.input_neurons+self.output_excitatory_neurons:
            ] = (
                    self.i_mu_inhibitory_output
                    + self.i_sigma_inhibitory_output
                    * np.random.random((self.output_inhibitory_neurons,))
            )
        else:
            raise RuntimeError(f"Unknown method: {self.method}")

    def calculate_spikes_binned(
            self,
            tau,
            sample_window=None,
            tau_max=50,  # 20Hz
    ):
        log.logger.debug("binning the spikes...")
        if tau < self.delta_t:
            log.logger.warn(f"tau({tau}) smaller than delta_t({self.delta_t})")
            tau = self.delta_t
        elif tau > tau_max:
            log.logger.warn(f"tau({tau}) bigger than tau_max({tau_max})")
            tau = tau_max
        if sample_window is None:
            bins_start = self.sample_start_index
        else:
            bins_start = int((self.t_max - sample_window) / self.delta_t)
        steps_per_tau = tau / self.delta_t
        bins = np.arange(bins_start + steps_per_tau, self.steps, steps_per_tau)
        bin_count = len(bins) + 1
        steps = np.arange(bins_start, self.steps)
        try:
            time_bins = np.digitize(steps, bins)
        except ValueError as e:
            raise ValueError(
                f"{e} "
                f"steps: {steps}"
                f", bins:{bins}"
                f", tau:{tau}"
            )
        spikes_binned = np.zeros((self.neurons, bin_count))
        spike_trains = self.spike_trains_np[:, bins_start:]
        for bin_ in range(bin_count):
            spikes_binned[:, bin_] = (
                spike_trains[:, time_bins == bin_]
            ).any(axis=1).astype(int)
        return spikes_binned[:, 1:]

    @staticmethod
    def coherence_from_index(
        index_i, index_j,
        spikes_binned,
        spike_sums,
        activate_surrounding_bins=None,
    ):
        # i = [0, 0, 0, ..., 1, 1, 1, ...]:
        # j = [0, 1, 2, ..., 0, 1, 2, ...]:
        index_mask = index_i < index_j
        # this ^^ is the upper triangular matrix with diagonal=0
        spikes_binned_masked = spikes_binned.copy()
        if activate_surrounding_bins is not None:
            spikes_pd = pd.DataFrame(spikes_binned.T.astype(float))
            spikes_mask = (
                spikes_pd
                .rolling(
                    window=int(activate_surrounding_bins),
                    min_periods=0,
                    center=True,
                )
                .max()
                .values
                .T
            )
            spikes_mask = np.nan_to_num(spikes_mask)
            spikes_mask[spikes_mask < 1] = np.nan
            spikes_binned_masked *= spikes_mask

        index_i_masked = index_i[index_mask]
        index_j_masked = index_j[index_mask]
        correlations = (
            spikes_binned_masked[index_i_masked]
            * spikes_binned_masked[index_j_masked]
        )
        kappa_numerator = np.nansum(correlations, axis=1)  # sum over bins
        spikes_binned_masked[np.isnan(spikes_binned_masked)] = -1
        miss_matrix = (
            spikes_binned_masked[index_i_masked]
            * spikes_binned_masked[index_j_masked]
        )
        misses = np.sum(miss_matrix < 0, axis=1)
        kappa_denominator_squared = (
            spike_sums[index_i_masked]
            * spike_sums[index_j_masked]
            - misses
        )
        kappa_denominator_squared[kappa_denominator_squared < 0] = 0
        kappa_denominator = np.sqrt(kappa_denominator_squared)
        kappa_denominator[kappa_denominator == 0] = np.nan
        with np.errstate(invalid='ignore', divide='ignore'):
            kappa = kappa_numerator / kappa_denominator
        kappa_not_nan = kappa[~np.isnan(kappa)]
        if kappa_not_nan.size == 0:
            return np.nan
        return np.nanmean(kappa)  # mean over (i, j) pairs

    def coherence_from_range(
            self,
            start: int, end: int,
            spikes_binned,
            spike_sums,
            activate_surrounding_bins=None,
    ):
        length = end - start
        index_i = np.repeat(np.arange(start, end), length)
        index_j = np.tile(np.arange(start, end), length)
        return self.coherence_from_index(
            index_i, index_j,
            spikes_binned=spikes_binned,
            spike_sums=spike_sums,
            activate_surrounding_bins=activate_surrounding_bins,
        )

    # @memory_profiler.profile
    def calculate_coherence(
        self, results: NetworkResults, tau,
    ) -> NetworkResults:
        """
        X(l), Y(l) = 0 or 1: spike trains; l=1, ..., K
        coherence:
        kappa_ij(tau) = sum_l=1^K(X(l)Y(l) /
                              sqrt(sum_l=1^K(X(l)) * sum_l=1^K(Y(l)))
        population coherence: average over many i, j
        see: Wang-Buszáki, 1996

        Args:
            results: the network results so far
            tau: tau = T / K = bin size

        Returns:
            the network results with coherences added
        """

        # log.logger.info("calculating the coherence measure...")
        spikes_binned = self.calculate_spikes_binned(tau=tau)
        spike_sums: np.ndarray = spikes_binned.sum(axis=1)  # sum(X(l)) over l
        bin_window = self.sample_window / spikes_binned.shape[1]

        results.coherence_by_tau[tau] = self.coherence_from_range(
            start=0, end=self.neurons,
            spikes_binned=spikes_binned,
            spike_sums=spike_sums,
        )
        frequency_layer0_inh = results.layer_frequencies[0][1]
        # if results.layer_frequencies[0][1]:
        #     wavelength_layer0_inhib = 1000 / results.layer_frequencies[0][1]
        # else:
        #     wavelength_layer0_inhib = 1000 / 1
        frequency_layer0_exc = results.layer_frequencies[0][0]
        if not frequency_layer0_exc:
            frequency_layer0_exc = 1
        # bin_window_layer0_exc = wavelength_layer0_inhib / bin_window
        results.layer_coherences[0][0][tau] = self.coherence_from_range(
            start=0,
            end=self.input_excitatory_neurons,
            spikes_binned=spikes_binned,
            spike_sums=spike_sums,
            # activate_surrounding_bins=bin_window_layer0_exc,
            # ^^ use inhibitory frequency here
        ) * frequency_layer0_inh / frequency_layer0_exc
        results.layer_coherences[0][1][tau] = self.coherence_from_range(
            start=self.input_excitatory_neurons,
            end=self.input_neurons,
            spikes_binned=spikes_binned,
            spike_sums=spike_sums,
        )
        # if results.layer_frequencies[1][1]:
        #     wavelength_layer1_inhib = 1000 / results.layer_frequencies[1][1]
        # else:
        #     wavelength_layer1_inhib = 1000 / 1
        # bin_window_laye1_exc = wavelength_layer1_inhib / bin_window
        frequency_layer1_inh = results.layer_frequencies[1][1]
        frequency_layer1_exc = results.layer_frequencies[1][0]
        if not frequency_layer1_exc:
            frequency_layer1_exc = 1
        results.layer_coherences[1][0][tau] = self.coherence_from_range(
            start=self.input_neurons,
            end=self.input_neurons + self.output_excitatory_neurons,
            spikes_binned=spikes_binned,
            spike_sums=spike_sums,
            # activate_surrounding_bins=bin_window_laye1_exc,
            # ^^ use inhibitory frequency here
        ) * frequency_layer1_inh / frequency_layer1_exc
        results.layer_coherences[1][1][tau] = self.coherence_from_range(
            start=self.input_neurons + self.output_excitatory_neurons,
            end=self.neurons,
            spikes_binned=spikes_binned,
            spike_sums=spike_sums,
        )
        return results

    def calculate_frequency(self, results: NetworkResults) -> NetworkResults:
        """

        Args:
            results: The network results so far

        Returns:
            The results, with the frequencies added
        """
        frequencies_per_neuron = (
            self
            .spike_trains_np
            [:, self.sample_start_index:]
            .sum(axis=1)
            / self.sample_window
            * 1_000
        )
        results.frequency = np.mean(frequencies_per_neuron)
        results.frequency_std = np.std(frequencies_per_neuron)
        input_layer_mask = np.arange(self.neurons) < self.input_neurons
        input_frequencies = [0, 0]
        if self.input_excitatory_neurons:
            input_frequencies[0] = np.mean(
                frequencies_per_neuron[input_layer_mask & self.excitatory])
        if self.input_inhibitory_neurons:
            input_frequencies[1] = np.mean(
                frequencies_per_neuron[input_layer_mask & ~self.excitatory])
        output_frequencies = [0, 0]
        if self.output_excitatory_neurons:
            output_frequencies[0] = np.mean(
                frequencies_per_neuron[~input_layer_mask & self.excitatory])
        if self.output_inhibitory_neurons:
            output_frequencies[1] = np.mean(
                frequencies_per_neuron[~input_layer_mask & ~self.excitatory])

        results.layer_frequencies = [
            input_frequencies,
            output_frequencies,
        ]
        return results

    def calculate_syn_drive(self):
        step = self.step_count
        ion_channel_mean = np.nanmean(self.s)
        self.syn_drive[step] = ion_channel_mean
        return self.syn_drive

    @property
    def comment(self):
        _run = log.ex.current_run
        if "comment" in _run.meta_info:
            comment = _run.meta_info["comment"]
        else:
            comment = ""
        return comment

    def make_filename(self, filename):
        comment = self.comment
        return f"{self.results_folder}/{self.run_id}_{comment}_{filename}"

    def print_summary(self, filename="results/Istim.dat"):
        pass

    def print_analysis(self):
        pass

    def print_syn_drive(self, filename="synaptic_drive",):
        syn_drive = self.syn_drive[self.sample_start_index:]
        sample_steps = self.steps - self.sample_start_index
        sample_frequencies, power_spectrum = scipy_signal.welch(
            x=syn_drive,
            fs=1000/self.delta_t,
            # scaling="spectrum",
            nperseg=sample_steps,
            # detrend=False,
        )
        sample_frequencies = sample_frequencies[sample_frequencies <= 500]
        power_spectrum = power_spectrum[:len(sample_frequencies)]
        time_data = np.arange(0, self.sample_window, self.delta_t)
        fig, axes = plt.subplots(
            ncols=1,
            nrows=2,
            figsize=(25, 10),
            dpi=600,
        )
        ax_syn_drive = axes[0]
        ax_powerspectrum = axes[1]
        ax_syn_drive.plot(time_data, syn_drive)
        ax_powerspectrum.plot(
            sample_frequencies,
            power_spectrum,
        )
        fig.suptitle(f"synaptic drive s(t) for {self.comment}")
        ax_syn_drive.set_xlabel("T (ms)")
        ax_syn_drive.set_ylabel("synaptic drive s(t) (mV)")
        ax_powerspectrum.set_xlabel("frequency (Hz)")  # TODO: is it Hz?
        ax_powerspectrum.set_ylabel("power spectrum of s(t)")
        filename_svg = self.make_filename(filename=filename + ".svg")
        pyplot_concurrency_workaround(lambda: fig.savefig(filename_svg))
        fig.clear()
        plt.close("all")
        syn_drive_pd = pd.DataFrame(
            {"synaptic drive s(t) (mV)": syn_drive},
            index=time_data,
        )
        syn_drive_pd.index.name = "T (ms)"
        syn_drive_pd.to_csv(self.make_filename(
            filename=filename + "_syn_drive.csv"))
        power_spectrum_pd = pd.DataFrame(
            {"power spectrum of s(t)": power_spectrum},
            index=sample_frequencies,
        )
        power_spectrum_pd.index.name = "frequency (Hz)"
        power_spectrum_pd.to_csv(self.make_filename(
            filename=filename + "_power_spectrum.csv"))

    # @memory_profiler.profile
    def print_coherences(
            self,
            results: NetworkResults,
            tau_result=1,
            tau_max=50,
    ) -> NetworkResults:
        _run = log.ex.current_run
        comment = self.comment
        step = 1
        if self.tau_is_inverse_frequency:
            if results.layer_frequencies[0][1]:
                tau_e_0 = (
                    self.tau_multiplier
                    * 1000
                    / results.layer_frequencies[0][1]
                )  # excitatory firing is too sparse to use
                tau_i_0 = (
                    self.tau_multiplier
                    * 1000
                    / results.layer_frequencies[0][1]
                )
            else:
                tau_e_0 = tau_result
                tau_i_0 = tau_result
            if results.layer_frequencies[1][1]:
                tau_e_1 = (
                        self.tau_multiplier
                        * 1000
                        / results.layer_frequencies[1][1]
                )  # excitatory firing is too sparse to use
                tau_i_1 = (
                    self.tau_multiplier
                    * 1000
                    / results.layer_frequencies[1][1]
                )
            else:
                tau_e_1 = tau_result
                tau_i_1 = tau_result
        else:
            tau_e_0 = tau_result
            tau_i_0 = tau_result
            tau_e_1 = tau_result
            tau_i_1 = tau_result

        if self.plot_everything:
            index_tau = np.arange(1, tau_max, step)
            index_tau = np.concatenate((
                [tau_result],
                [tau_e_0],
                [tau_i_0],
                [tau_e_1],
                [tau_i_1],
                index_tau,
            ))
            for tau in index_tau:
                if tau <= 0:
                    tau = 1
                results = self.calculate_coherence(results=results, tau=tau)
                _run.log_scalar(
                    "coherence", results.coherence_by_tau[tau], tau)
            results.coherence = results.coherence_by_tau[tau_result]
            results.coherence_e_0 = results.layer_coherences[0][1][tau_e_0]
            results.coherence_i_0 = results.layer_coherences[0][0][tau_i_0]
            results.coherence_e_1 = results.layer_coherences[1][0][tau_e_1]
            results.coherence_i_1 = results.layer_coherences[1][1][tau_i_1]
            plt.figure()
            plt.plot(
                list(results.coherence_by_tau.keys()),
                list(results.coherence_by_tau.values()),
                label="coherence",
            )
            plt.plot(
                list(results.layer_coherences[0][0].keys()),
                list(results.layer_coherences[0][0].values()),
                label="coherence exitatory layer 0",
            )
            plt.plot(
                list(results.layer_coherences[0][1].keys()),
                list(results.layer_coherences[0][1].values()),
                label="coherence inhibitory layer 0",
            )
            plt.plot(
                list(results.layer_coherences[1][0].keys()),
                list(results.layer_coherences[1][0].values()),
                label="coherence exitatory layer 1",
            )
            plt.plot(
                list(results.layer_coherences[1][1].keys()),
                list(results.layer_coherences[1][1].values()),
                label="coherence inhibitory layer 1",
            )
            plt.title(f"coherence vs tau: ({comment})")
            legend = plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
            plt.xlabel("tau/ms")
            plt.ylabel("coherence")
            filename = self.make_filename("coherences.svg")
            pyplot_concurrency_workaround(lambda: plt.savefig(
                filename,
                additional_artists=[legend],
                bbox_inches="tight",
            ))
            plt.close()
        else:
            for tau_ in (
                tau_result,
                tau_e_0,
                tau_i_0,
                tau_e_1,
                tau_i_1,
            ):
                results = self.calculate_coherence(results=results, tau=tau_)
            results.coherence = results.coherence_by_tau[tau_result]
            results.coherence_e_0 = results.layer_coherences[0][1][tau_e_0]
            results.coherence_i_0 = results.layer_coherences[0][0][tau_i_0]
            results.coherence_e_1 = results.layer_coherences[1][0][tau_e_1]
            results.coherence_i_1 = results.layer_coherences[1][1][tau_i_1]
        return results

    def print_membrane_potential(
            self,
            filename="membrane_potentials",
    ):
        filename = self.make_filename(filename)
        # noinspection PyTypeChecker
        # np.savetxt(
        #     filename + ".csv.gz",
        #     self.membrane_potential_history,
        #     delimiter=",",
        # )
        # do not store this ^^ in an artifact: it is too big; 500Mb for n=1000
        time_data = np.arange(0, self.t_max, self.delta_t)
        log.logger.debug(f"time_data:{time_data}")
        log.logger.debug(f"v:{self.membrane_potential_history[0, :]}")
        fig, axes = plt.subplots(
            ncols=1,
            nrows=2,
            figsize=(25, 10),
            dpi=600,
        )
        ax_v = axes[0]
        ax_isyn = axes[1]
        neurons_to_plot = 3 if self.neurons >= 6 else int(np.ceil(self.neurons / 2))
        # only plot the first and last few. more gets crowded.
        for neuron in (
            list(range(neurons_to_plot))
            + list(range(-neurons_to_plot, 0))
        ):
            v = np.clip(  # clip to have a plot even if bad sim results in inf
                (
                    self
                    .membrane_potential_history
                    [neuron, :]
                    .reshape((-1,))
                ),
                a_min=-200,
                a_max=100,
            )
            syn = np.clip(
                self.syn_current_history[neuron, :].reshape((-1,)),
                a_min=-200,
                a_max=100,
            )
            ax_v.plot(time_data, v)
            ax_isyn.plot(time_data, syn)
        ax_v.set_ylabel("membrane potential/mV")
        ax_isyn.set_ylabel("syn current (µA/cm²)")
        ax_isyn.set_xlabel("T (ms)")
        ax_v.axhline(-52, linestyle=":")
        ax_v.axhline(self.e_syn_inhibitory, linestyle="-.")
        ax_v.axhline(self.e_syn_excitatory, linestyle="-.")
        pyplot_concurrency_workaround(lambda: fig.savefig(filename + ".svg"))
        fig.clear()
        plt.close("all")
        log.ex.add_artifact(filename=filename + ".svg")
        log.ex.info["membrane_potential_plot"] = (
            f"{os.getcwd()}/{filename}.svg")

    def print_rastergram(self, filename="rastergram.png"):
        filename = self.make_filename(filename)
        tau = 1.0
        spikes = (
            self
            .calculate_spikes_binned(
                tau=tau,
                sample_window=self.t_max,
            )
            .astype(bool)
            [:, ]
        )

        bins = spikes.shape[1]
        times = np.tile(
            np.arange(
                tau,
                self.t_max,
                tau,
            ),
            (self.neurons, 1),
        )
        times[~spikes] = np.nan
        neurons = (
            np
            .arange(self.neurons, dtype=float)
            .reshape((-1, 1))
        )
        neurons = np.repeat(
            neurons,
            bins,
            axis=1,
        )
        neurons[~spikes] = np.nan
        plt.figure()
        # TODO: make separate axes/plots for input and output
        plt.plot(
            times[self.excitatory, :].flatten(),
            neurons[self.excitatory, :].flatten(),
            "b.",
            label="excitatory",
        )
        plt.plot(
            times[~self.excitatory, :].flatten(),
            neurons[~self.excitatory, :].flatten(),
            "r.",
            label="inhibitory",
        )
        plt.axhline(y=self.input_neurons-0.25, color="black")
        plt.minorticks_on()
        plt.gca().grid(
            which='major', linestyle='-', linewidth='0.5', color='grey')
        plt.gca().grid(
            which='minor', linestyle=':', linewidth='0.5', color='grey')

        plt.gca().invert_yaxis()
        plt.title(f"rastergram")

        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.xlabel("t/ms")
        plt.ylabel("neuron")
        pyplot_concurrency_workaround(
            lambda: plt.savefig(filename, bbox_inches="tight"))
        plt.close()
        log.ex.add_artifact(filename=filename)
        log.ex.info["rastergram"] = f"{os.getcwd()}/{filename}"

    def print_firing_rates(self, results):
        results = self.calculate_frequency(results)
        return results


def print_step_info(network, time_started, display_fraction=2000):
    run_time = dt.datetime.now() - time_started
    if network.step_count > 0:
        eta = (
            run_time
            / network.step_count
            * (network.steps - network.step_count)
        )
    else:
        eta = "NaT"
    info_string = (
        f"starting step "
        f"{network.step_count}/{network.steps}"
        f" eta: {eta}"
    )
    if (network.step_count % display_fraction) == 0:
        log.logger.info(info_string)
    elif network.step_count == 1:
        log.logger.info(info_string)
    else:
        log.logger.debug(info_string)


# @memory_profiler.profile
def simulate(use_tqdm=True):
    network = Network()
    result = {}
    net_results = NetworkResults()
    if use_tqdm:
        step_range = tqdm(range(network.steps))
    else:
        step_range = range(network.steps)
    for _ in step_range:
        network.single_step()
    net_results = network.print_firing_rates(net_results)
    tau = network.tau_default
    if network.tau_is_inverse_frequency and net_results.frequency != 0:
        tau = network.tau_multiplier * 1000 / net_results.frequency
        log.logger.info(f"Set tau to {tau}.")
    result["tau"] = tau
    net_results = network.print_coherences(results=net_results, tau_result=tau)
    result["coherence"] = net_results.coherence
    if network.tau_is_inverse_frequency:
        if network.input_excitatory_neurons:
            result["coherence_layer_0_ex"] = net_results.coherence_e_0
        else:
            result["coherence_layer_0_ex"] = np.nan
        if network.input_inhibitory_neurons:
            result["coherence_layer_0_inh"] = net_results.coherence_i_0
        else:
            result["coherence_layer_0_inh"] = np.nan
        if network.output_excitatory_neurons:
            result["coherence_layer_1_ex"] = net_results.coherence_e_1
        else:
            result["coherence_layer_1_ex"] = np.nan
        if network.output_inhibitory_neurons:
            result["coherence_layer_1_inh"] = net_results.coherence_i_1
        else:
            result["coherence_layer_1_inh"] = np.nan
    else:
        if network.input_excitatory_neurons:
            result["coherence_layer_0_ex"] = net_results.layer_coherences[0][0][tau]
        else:
            result["coherence_layer_0_ex"] = np.nan
        if network.input_inhibitory_neurons:
            result["coherence_layer_0_inh"] = net_results.layer_coherences[0][1][tau]
        else:
            result["coherence_layer_0_inh"] = np.nan
        if network.output_excitatory_neurons:
            result["coherence_layer_1_ex"] = net_results.layer_coherences[1][0][tau]
        else:
            result["coherence_layer_1_ex"] = np.nan
        if network.output_inhibitory_neurons:
            result["coherence_layer_1_inh"] = net_results.layer_coherences[1][1][tau]
        else:
            result["coherence_layer_1_inh"] = np.nan
    result["firing_rate"] = net_results.frequency
    result["firing_rate_neuron_std"] = net_results.frequency_std
    result["firing_rate_layer_0_ex"] = net_results.layer_frequencies[0][0]
    result["firing_rate_layer_0_inh"] = net_results.layer_frequencies[0][1]
    result["firing_rate_layer_1_ex"] = net_results.layer_frequencies[1][0]
    result["firing_rate_layer_1_inh"] = net_results.layer_frequencies[1][1]
    result_print = {
        "result": result,
        "config": log.ex.current_run.config,
    }
    if network.plot_everything:
        network.print_analysis()
        network.print_membrane_potential()
        network.print_rastergram()
        network.print_syn_drive()
        with open(network.make_filename("results.json"), "w") as file:
            json.dump(result_print, file, indent=4, sort_keys=True)
        if network.debug_active:
            plt.figure()
            debug_df = pd.DataFrame(
                network.debug.T,
                columns=network.debug_columns.keys(),
            )
            debug_df[["spikes_e", "spikes_i"]].replace(0, np.nan)
            (
                debug_df
                [network.debug_print_columns]
                .clip(-100, 50)
                .plot(style=network.debug_print_styles)  # (logy=True)
            )
            plt.savefig(network.make_filename("debug.png"))
            plt.close()
            debug_df.to_csv(network.make_filename("debug.csv"))
        # network.spike_trains_np.tofile(
        #     network.make_filename("spike_trains"))
        # network.spike_trains_np.tofile(
        #     network.make_filename("spike_trains.csv"),
        #     sep=",",
        # )
        # network.membrane_potential_history.tofile(
        #     network.make_filename("membrane_potentials"))
    return result

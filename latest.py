from collections import namedtuple
import numpy as np
from scipy.special import gammaln, digamma
import logging
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

logging.basicConfig(level=logging.ERROR, force=True)
Config = namedtuple("Config", ["truncation_level", "sigma_c", "sigma_x", "kappa", "trainset_size", "data_dim"])

BetaDist = namedtuple("BetaDist", ["alpha", "beta"])
GaussianDist = namedtuple("GaussianDist", ["mean", "stddev"])
CategoricalDist = namedtuple("CategoricalDist", ["headprobs", "headsum", "stabilizer"])


class VDPState:
    def __init__(
        self, 
        *, 
        config: Config, 
        logger: logging.Logger,
        qc=None,
        qv=None,
        pc=None,
        pv=None,
    ):
        self.config = config
        self.qc = qc if qc else self.get_qc_initial()
        self.qv = qv if qv else self.get_qv_initial()
        self.pc = pc if pc else self.get_pc()
        self.pv = pv if pv else self.get_pv()
        self.logger = logger

    def get_pc(self):
        return GaussianDist(mean=0.0, stddev=self.config.sigma_c)

    def get_qc_initial(self):
        mu = np.random.normal(
            loc=0.0, 
            scale=self.config.sigma_c, 
            size=[self.config.truncation_level, self.config.data_dim],
        )
        sigma = np.repeat(np.array([self.config.sigma_c]), repeats=self.config.truncation_level, axis=0)
        return GaussianDist(mean=mu, stddev=sigma)

    def get_pv(self):
        return BetaDist(alpha=1.0, beta=1.0)

    def get_qv_initial(self):
        T = self.config.truncation_level
        # sample a hyperprior on mean: alpha / (alpha + beta)
        mean = np.random.beta(1.1, 1.1, size=[T]) # [T]
        # sample a hyperprior on concentration: (alpha + beta)
        conc = np.random.pareto(1.5, size=[T])    # [T]
        # convert to (alpha, beta) variational params
        alpha = mean * conc
        beta = conc - alpha
        return BetaDist(alpha=alpha, beta=beta)

    def update_qz(self, *, xs_minibatch, qc, qv):
        sx = self.config.sigma_x
        
        line_11 = digamma(qv.alpha) - digamma(qv.alpha + qv.beta)  # [T]
        line_12 = digamma(qv.beta) - digamma(qv.alpha + qv.beta)  # [T]
        D = xs_minibatch.shape[-1]
        trace_term = (D * qc.stddev ** 2) / (sx ** 2)
        line_13 = np.einsum('ld,md->ml', qc.mean / (sx ** 2), xs_minibatch) + \
            -0.5 * (np.einsum('ld,ld->l', qc.mean / (sx ** 2), qc.mean) + trace_term)[None, ...] # [N, T]
        S_n_i = (
            line_11[None,...] + 
            np.cumsum(np.concatenate([np.array([0]), line_12[:-1]], axis=0), axis=0)[None, ...] + 
            line_13
        ) # [M, T]
        stabilizer = np.max(S_n_i, axis=-1)                # [M]
        exp_S_n_i = np.exp(S_n_i - stabilizer[..., None])  # [M, T]
        exp_S_n_headsum = np.sum(exp_S_n_i, axis=-1)       # [M]
        q_zn_head = exp_S_n_i / exp_S_n_headsum[..., None]  # [M, T]
        return CategoricalDist(headprobs=q_zn_head, headsum=exp_S_n_headsum, stabilizer=stabilizer)
    
    def update_qv(self, *, qz, qv, pv):
        kappa = self.config.kappa
        ratio = self.config.trainset_size / qz.headprobs.shape[0]

        qv_nu_1 = (pv.alpha - 1) + ratio * np.sum(qz.headprobs, axis=0) # [T]
        # now compute sum_j={i+1}^T
        # i=1 -> sum i=2 ... i=T
        # ...
        # i=T-2 -> sum i=T-1 ... i=T
        # i=T-1 -> sum i=T
        # i=T -> 0
        N = qz.headprobs.shape[0]
        chop = qz.headprobs[:, 1:]
        flip = chop[:, ::-1]
        pad = np.pad(flip, ((0, 0), (1, 0)), mode='constant')
        cumulative = np.cumsum(pad, axis=-1)
        unflip = cumulative[:, ::-1]  # [M, T]
        qv_nu_2 = (pv.beta - 1) + ratio * np.sum(unflip, axis=0)  # [T]
        
        qv_nu_1 = kappa * qv_nu_1 + (1 - kappa) * (qv.alpha - 1)
        qv_nu_2 = kappa * qv_nu_2 + (1 - kappa) * (qv.beta - 1)
        return BetaDist(alpha=qv_nu_1 + 1, beta=qv_nu_2 + 1)

    def update_qc(self, *, xs_minibatch, qz, qc):
        sx = self.config.sigma_x
        sc = self.config.sigma_c
        kappa = self.config.kappa
        ratio = self.config.trainset_size / xs_minibatch.shape[0]

        qc_gamma_1 = (sx ** -2) * ratio * np.einsum('mt,md->td', qz.headprobs, xs_minibatch)
        qc_gamma_2 = (sc ** -2) + (sx ** -2) * ratio * np.sum(qz.headprobs, axis=0)
        
        qc_gamma_1 = kappa * qc_gamma_1 + (1 - kappa) * (qc.mean * qc.stddev[..., None] ** -2)
        qc_gamma_2 = kappa * qc_gamma_2 + (1 - kappa) * (qc.stddev ** -2)

        return GaussianDist(
            mean=qc_gamma_1 / qc_gamma_2[..., None],
            stddev=qc_gamma_2 ** -0.5,
        )

    def get_total_beta_kl_diverence(self, *, qv, pv):
        log_beta_q = gammaln(qv.alpha + qv.beta) - gammaln(qv.alpha) - gammaln(qv.beta)
        log_beta_p = gammaln(pv.alpha + pv.beta) - gammaln(pv.alpha) - gammaln(pv.beta)
        term_normalization = log_beta_q - log_beta_p

        term_expectation = (
            (qv.alpha - pv.alpha) * (digamma(qv.alpha) - digamma(qv.alpha + qv.beta)) +
            (qv.beta - pv.beta) * (digamma(qv.beta) - digamma(qv.alpha + qv.beta))
        )

        return np.sum(term_normalization + term_expectation, axis=0)

    def get_total_gaussian_kl_divergence(self, *, qc, pc):
        qc_mu = qc.mean
        qc_sigma = np.repeat(qc.stddev[..., None], repeats=qc.mean.shape[-1], axis=1)
        pc_mu = np.full_like(qc_mu, fill_value=pc.mean)
        pc_sigma = np.full_like(qc_mu, fill_value=pc.stddev)

        term1 = np.log(pc_sigma / qc_sigma)
        term2 = (qc_sigma * qc_sigma) / (2.0 * pc_sigma * pc_sigma)
        term3 = ((qc_mu - pc_mu) * (qc_mu - pc_mu)) / (2.0 * pc_sigma * pc_sigma)
        term4 = np.full_like(qc_mu, fill_value=-0.5)
        kls = np.sum(term1 + term2 + term3 + term4, axis=-1)  # [T]
        return np.sum(kls, axis=0) # []

    def get_elbo_normalized(self, *, xs_eval):
        qc = self.qc
        qv = self.qv
        pc = self.pc
        pv = self.pv

        N = self.config.trainset_size
        M = xs_eval.shape[0]
        D = self.config.data_dim
        assert D == xs_eval.shape[1]
        ratio = N / M

        kl_gauss = self.get_total_gaussian_kl_divergence(qc=qc, pc=pc) / (N * D)
        self.logger.info(f"kl_gauss: {kl_gauss}")

        kl_beta = self.get_total_beta_kl_diverence(qv=qv, pv=pv) / (N * D)
        self.logger.info(f"kl_beta: {kl_beta}")

        qz = self.update_qz(xs_minibatch=xs_eval, qc=qc, qv=qv)
        assert qz.headsum.shape == (M,)
        assert qz.stabilizer.shape == (M,)
        sn_infsum = qz.headsum
        stabilizer = qz.stabilizer
        lastterm = ratio * -np.sum(stabilizer + np.log(sn_infsum), axis=0) / (N * D)
        self.logger.info(f"lastterm: {lastterm}")

        free_energy = kl_beta + kl_gauss + lastterm
        self.logger.info(f"free_energy: {free_energy}")

        elbo = -free_energy
        self.logger.info(f"elbo: {elbo}")
        return elbo

    def get_mean_stick_lengths(self):
        qv = self.qv
        means = qv.alpha / (qv.alpha + qv.beta)  # [T]
        minus = 1 - means  # [T]
        minus_prod = np.cumprod(minus, axis=0)  # [T]
        minus_prod = np.pad(minus_prod[0:-1], ((1, 0)), mode='constant', constant_values=1.0)
        return means * minus_prod

    def run_vi_update(self, *, xs_minibatch):
        qz = self.update_qz(xs_minibatch=xs_minibatch, qc=self.qc, qv=self.qv)
        qc = self.update_qc(xs_minibatch=xs_minibatch, qz=qz, qc=self.qc)
        qv = self.update_qv(qz=qz, qv=self.qv, pv=self.pv)
        return VDPState(
            config=self.config,
            logger=self.logger,
            qc=qc,
            qv=qv,
            pc=self.pc,
            pv=self.pv,
        )


def get_dataset(*, config):
    xs0 = np.random.normal(loc=0.5, scale=0.05, size=[config.trainset_size // 2, config.data_dim])
    xs1 = np.random.normal(loc=-0.5, scale=0.05, size=[config.trainset_size // 2, config.data_dim])
    xs = np.concatenate([xs0, xs1], axis=0)
    return xs


@st.cache_data
def vi_loop(*, config, minibatch_size, opt_iters, xs_train, streamlit_info=False):
    state = VDPState(config=config, logger=logging.getLogger(__name__))
    elbo = state.get_elbo_normalized(xs_eval=xs_train)
    states = [state]
    elbos = [elbo]
    for _ in range(opt_iters):
        batch_indices = np.random.choice(config.trainset_size, size=minibatch_size, replace=False)
        state = state.run_vi_update(xs_minibatch=xs_train[batch_indices])
        elbo = state.get_elbo_normalized(xs_eval=xs_train)
        print(elbo)
        states.append(state)
        elbos.append(elbo)
    
    if not streamlit_info:
        return dict(states=states, elbos=elbos, streamlit_df=None)

    dfs = []
    for i in range(config.truncation_level):
        df = pd.DataFrame({
            "timestep": range(opt_iters + 1), 
            "x": [state.qc.mean[i][0] for state in states], 
            "y": [state.qc.mean[i][1] for state in states],
            "weight": [state.get_mean_stick_lengths()[i] for state in states],
            "size": [state.qc.stddev[i] * 10_000 for state in states]
        })
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return dict(states=states, elbos=elbos, streamlit_df=df)


def streamlit_plot(*, xs, df, view_iter):
    st.write("Legend: radius proportional to stddev, opacity to mixture weight")
    filtered_df = df[df["timestep"] == view_iter]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered_df["x"],
            y=filtered_df["y"],
            mode="markers",
            name="Clusters",
            marker=dict(
                size=filtered_df["size"], 
                sizemode="diameter",                 
                sizemin=4,
                opacity=filtered_df["weight"],
            )
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs[:, 0],
            y=xs[:, 1],
            mode="markers",
            name="Datapoints",
            opacity=0.2,
            marker=dict(color="green"),
        )
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    np.random.seed(42)

    st.set_page_config(page_title="DPGMM variational inference in 2D", layout="centered")
    st.title("DPGMM variational inference in 2D")
    truncation_level = st.select_slider("Truncation Level", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], value=10)
    minibatch_size = st.select_slider("SVI Minibatch Size", options=[20, 200, 2000], value=2000)
    kappa = st.number_input("SVI Learning Rate", value=0.001, format="%.4f")
    opt_iters = 100
    view_iter = st.slider("SVI Time Step", min_value=0, max_value=opt_iters, value=0, step=1)
    
    config = Config(
        truncation_level=truncation_level,
        sigma_c=1.0,
        sigma_x=0.05,
        trainset_size=2000,
        data_dim=2,
        kappa=kappa,
    )
    xs = get_dataset(config=config)
    ret = vi_loop(
        config=config, 
        minibatch_size=minibatch_size, 
        opt_iters=opt_iters,
        xs_train=xs,
        streamlit_info=True,
    )
    streamlit_plot(xs=xs, df=ret['streamlit_df'], view_iter=view_iter)

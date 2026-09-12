import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from algos import vi_loop, Config


def get_dataset(*, config):
    xs0 = np.random.normal(loc=0.0, scale=0.05, size=[config.trainset_size // 3, config.data_dim])
    xs1 = np.random.normal(loc=0.0, scale=0.05, size=[config.trainset_size // 3, config.data_dim])
    xs2 = np.random.normal(loc=0.0, scale=0.05, size=[config.trainset_size // 3, config.data_dim])
    xs0[:, 0] += 1
    xs1[:, 1] += 1
    xs2[:, 2] += 1
    xs = np.concatenate([xs0, xs1, xs2], axis=0)
    return xs


@st.cache_data
def streamlit_info(*, config, minibatch_size, opt_iters, xs_train):
    # run vi_loop inside so everything is cached, but avoid decorating vi_loop itself
    # so there is no dependence on streamlit unless the user wants it
    states, elbos = vi_loop(
        config=config, 
        minibatch_size=minibatch_size, 
        opt_iters=opt_iters, 
        xs_train=xs_train,
    )
    dfs = []
    for i in range(config.truncation_level):
        df = pd.DataFrame({
            "timestep": range(opt_iters + 1), 
            "cluster_id": [i for _ in range(opt_iters + 1)],
            "x": [state.qc.mean[i][0] for state in states], 
            "y": [state.qc.mean[i][1] for state in states],
            "z": [state.qc.mean[i][2] for state in states],
            "opacity": [state.get_mean_stick_lengths()[i] for state in states],
            "size": [state.qc.stddev[i] * 10000 for state in states],
        })
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return dict(states=states, elbos=elbos, streamlit_df=df)


def streamlit_plot(*, config, xs, df, view_iter):
    filtered_df = df[df["timestep"] == view_iter]
    filtered_df = filtered_df.reset_index(drop=True)
    fig = go.Figure()
    for i in range(config.truncation_level):
        cluster_df = filtered_df[filtered_df["cluster_id"] == i]
        cluster_df = cluster_df.reset_index(drop=True)
        fig.add_trace(
            go.Scatter3d(
                x=cluster_df["x"],
                y=cluster_df["y"],
                z=cluster_df["z"],
                showlegend=(i == 0),
                mode="markers",
                name="Clusters",
                marker=dict(
                    size=cluster_df["size"], 
                    sizemode="diameter",                 
                    sizemin=4,
                    opacity=cluster_df["opacity"].iloc[0],
                    color="blue",
                )
            )
        )
    fig.add_trace(
        go.Scatter3d(
            x=xs[:, 0],
            y=xs[:, 1],
            z=xs[:, 2],
            mode="markers",
            name="Datapoints",
            opacity=0.2,
            marker=dict(color="green"),
        )
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    np.random.seed(42)

    st.set_page_config(page_title="DPGMM variational inference in 3D", layout="centered")
    st.title("DPGMM variational inference in 3D")
    truncation_level = st.select_slider("Truncation Level", options=[1, 2, 3, 4, 5, 6, 7], value=7)
    minibatch_size = st.select_slider("SVI Minibatch Size", options=[30, 300, 3000], value=3000)
    kappa = st.number_input("SVI Learning Rate", value=0.001, format="%.4f")
    opt_iters = 100
    view_iter = st.slider("SVI Time Step", min_value=0, max_value=opt_iters, value=0, step=1)
    st.write("Legend: radius proportional to stddev, opacity to mixture weight")

    config = Config(
        truncation_level=truncation_level,
        sigma_c=1.0,
        sigma_x=0.05,
        trainset_size=3000,
        data_dim=3,
        kappa=kappa,
    )
    xs = get_dataset(config=config)
    ret = streamlit_info(
        config=config, 
        minibatch_size=minibatch_size, 
        opt_iters=opt_iters,
        xs_train=xs,
    )
    streamlit_plot(config=config, xs=xs, df=ret['streamlit_df'], view_iter=view_iter)

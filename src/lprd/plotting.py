from lprd import options, helper_fns
import pathlib as plib
import plotly.express as px
import logging
import numpy as np
import typing
log_module = logging.getLogger(__name__)


def visualize_noise_cov(noise_cov: typing.Union[int, np.ndarray], opts: options.Config, name: str = ""):
    if isinstance(noise_cov, int):
        return
    if name:
        name += "_"
    fig = px.imshow(np.abs(noise_cov))
    file_name = plib.Path(opts.output_path).absolute()
    file_name = file_name.joinpath("plots").joinpath(f"{name}noise_cov").with_suffix(".html")
    log_module.info(f"\t- writing plot file: {file_name}")
    fig.write_html(file_name.as_posix())


def visualize_mag_phase_slice(k_slice_mag_phase_xy: np.ndarray, opts: options.Config, name=""):
    # plot
    text = ["log mag", "phase"]
    fig = px.imshow(
        k_slice_mag_phase_xy, facet_col=0, labels={"facet_col": "img"}, facet_col_spacing=0.05, title=name
    )
    # set facet titles
    for i, txt in enumerate(text):
        fig.layout.annotations[i]['text'] = txt
    # hide ticklabels
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    # update traces to use different coloraxis
    for i, t in enumerate(fig.data):
        t.update(coloraxis=f"coloraxis{i + 1}")

    # update layout --> plot colorbar for each trace, define colorscale
    fig.update_layout(
        coloraxis={
            "colorbar": {
                "x": 0.475,
                "y": 0.5,
                "len": 1,
                'title': 'log mag'
            },
            "colorscale": "Inferno"
        },
        coloraxis2={
            "colorbar": {
                "x": 1,
                "y": 0.5,
                "len": 1,
                'title': 'Phase'

            },
            "colorscale": 'Viridis'
        }
    )
    file_name = plib.Path(opts.output_path).absolute()
    file_name = file_name.joinpath("plots").joinpath(f"{name}_img").with_suffix(".html")
    log_module.info(f"\t- writing plot file: {file_name}")
    fig.write_html(file_name)


def visualize_k_space(k_data_xyz_ch_t: np.ndarray, opts: options.Config, name="", log_mag: bool = True):
    shape = k_data_xyz_ch_t.shape
    # choose mid slice
    id_mid_slice = int(shape[2] / 2)
    # choose first channel
    id_channel = 0
    # choose echo of first nonzero occurence
    mag = np.abs(k_data_xyz_ch_t[:, :, id_mid_slice, id_channel, :])
    id_echo = np.nonzero(mag)[-1][0]
    mag = mag[:, :, id_echo]
    plot_k_data = np.zeros((2, *shape[:2]), dtype=float)
    # plot magnitude logarithmically for better visualization
    if log_mag:
        plot_k_data[0][mag > 1e-9] = np.log(mag[mag > 1e-9])
    else:
        plot_k_data[0] = mag
    plot_k_data[1] = np.angle(k_data_xyz_ch_t[:, :, id_mid_slice, id_channel, id_echo])
    visualize_mag_phase_slice(plot_k_data, opts=opts, name=name)


def visualize_naive_recon(k_data_xyz_ch_t: np.ndarray, opts: options.Config, name=""):
    shape = k_data_xyz_ch_t.shape
    # choose mid slice
    id_mid_slice = int(shape[2] / 2)
    # choose first 3 echoes
    id_echo = np.min([3, shape[-1]])
    # fft - [x, y, ch]
    k_process = k_data_xyz_ch_t[:, :, id_mid_slice, :, :id_echo]
    k_process, _ = helper_fns.data_fft_to_time(
        k_process, data_in_time_domain=False, axes=(0, 1)
    )
    # rSoS - possibly complex input, need to take abs
    k_sos = np.sqrt(
        np.sum(
            np.square(
                np.abs(k_process)
            ),
            axis=-2
        )
    )
    for k in range(id_echo):
        plot_k_data = np.zeros((2, *shape[:2]), dtype=float)
        # assign mag and phase data
        plot_k_data[0] = np.abs(k_sos[:, :, k])
        plot_k_data[1] = np.angle(k_sos[:, :, k])
        visualize_mag_phase_slice(plot_k_data, opts=opts, name=name+f"_echo{k}")

    # debugging single slice and single channel code
    # fft - [x, y, sl]
    id_channel = np.min([k_data_xyz_ch_t.shape[-2], 5])
    id_echo = np.min([k_data_xyz_ch_t.shape[-1], 2]) - 1
    k_process = k_data_xyz_ch_t[:, :, id_mid_slice-1:id_mid_slice+1, id_channel-1:id_channel+1, id_echo]
    k_process, _ = helper_fns.data_fft_to_time(data=k_process, data_in_time_domain=False, axes=(0, 1))

    for k in range(k_process.shape[2]):
        k_single_channel = np.zeros((2, *k_process.shape[:2]), dtype=float)
        k_single_channel[0] = np.abs(k_process[:, :, 0, k])
        k_single_channel[1] = np.angle(k_process[:, :, 0, k])
        k_single_slice = np.zeros((2, *k_process.shape[:2]), dtype=float)
        k_single_slice[0] = np.abs(k_process[:, :, k, 0])
        k_single_slice[1] = np.angle(k_process[:, :, k, 0])
        visualize_mag_phase_slice(
            k_single_slice, opts=opts,
            name=name+f"_slice-{id_mid_slice -1 + k}_single_channel-{id_channel+1}_echo-{id_echo+1}"
        )
        visualize_mag_phase_slice(
            k_single_channel, opts=opts,
            name=name+f"_slice-{id_mid_slice - 1}_single_channel-{id_channel - 1 + k}_echo-{id_echo+1}")


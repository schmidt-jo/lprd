import logging
import numpy as np
import tqdm
from twixtools import twixtools as tt
import pathlib as plib
import pandas as pd
import plotly.express as px
from lprd import helper_fns


def main():
    path = plib.Path(
        "/data/pt_np-jschmidt/data/01_in_vivo_scan_data/paper_protocol_mese/7T/"
        "2023-10-30/raw/meas_MID00044_FID27156_mod_semc_r0p7_fa140_z4000_pat3_36_sl35_200_esp9p5.dat"
    ).absolute()
    logging.info("__ Loading")
    logging.info(f"\tload file: {path.as_posix()}")
    twix = tt.read_twix(path.as_posix())[0]
    logging.info(f"\tassigning data")
    hdr = twix["hdr"]
    geom = twix["geometry"]
    mdbs = twix["mdb"]
    meas_dict = {}

    logging.info("\t\t-extract Meas info")
    for key, val in hdr["Meas"].items():
        if val:
            meas_dict.__setitem__(key, val)
    pd_meas = pd.Series(meas_dict)
    pd_meas_yaps_slice = pd.Series(hdr["MeasYaps"]["sSliceArray"])

    n_echoes = pd_meas.lContrasts
    n_fe = int(pd_meas.iNoOfFourierColumns)
    n_fe_base = pd_meas.lBaseResolution
    n_pe = pd_meas.lPhaseEncodingLines
    n_slices = pd_meas_yaps_slice.lSize
    n_coils = int(pd_meas.iMaxNoOfRxChannels)

    os_factor = int(n_fe / n_fe_base)
    fa = pd_meas.FlipAngle

    tr = pd_meas.TR
    tes = pd_meas.alTE[:n_echoes]
    ref_lines_pe = pd_meas.lRefLinesPE
    acc_factor = pd_meas.lAccelFactPE

    fov_read = pd_meas.ReadFoV
    fov_phase = pd_meas.PhaseFoV
    slice_thickness = pd_meas_yaps_slice.asSlice[0]["dThickness"]
    slice_pos = []
    for mdb in mdbs:
        z_pos = mdb.mdh.SliceData.SlicePos.Tra
        if slice_pos:
            if np.min(np.abs(np.array(slice_pos) - z_pos)) > 1e-6:
                slice_pos.append(z_pos)
        else:
            slice_pos.append(z_pos)
    slice_pos_ordered = np.sort(slice_pos)
    slice_index_mapping = np.searchsorted(slice_pos_ordered, slice_pos)
    vox_size = np.array([fov_read, fov_phase, slice_thickness]) / np.array([n_fe_base, n_pe, 1.0])
    clins = 1 + np.max([mdb.cLin for mdb in mdbs])
    n_part = 1 + np.max([mdb.cPar for mdb in mdbs])

    k_space_img = np.zeros((n_fe, n_pe, n_slices, n_coils, n_echoes), dtype=complex)
    k_space_ref = np.zeros((n_fe, n_pe, n_slices, n_coils, n_echoes), dtype=complex)

    for k in tqdm.trange(len(mdbs), desc="process mdbs"):
        mdb = mdbs[k]
        flags = mdb.get_active_flags()
        if "SWAPPED" in flags:
            data = np.flip(mdb.data.T, 0)
        else:
            data = mdb.data.T
        pe = mdb.cLin
        t = mdb.cEco
        sli = mdb.cSlc

        if mdb.is_image_scan():
            k_space_img[:, pe, slice_index_mapping[sli], :, t] = data
        else:
            k_space_ref[:, pe, sli, :, t] = data
    logging.info("plot k-space slice")
    fig_path = plib.Path("/data/pt_np-jschmidt/code/lprd/src/lprd/dev").absolute()
    fig = px.imshow(np.swapaxes(np.log(np.abs(k_space_img[:, :, 10, 0])), 0, 1),
                    facet_col=-1, facet_col_wrap=4)
    fig_file = fig_path.joinpath("k-space-sort-semc").with_suffix(".html")
    logging.info(f"\t\t - write image k-space peek file: {fig_path.as_posix()}")
    fig.write_html(fig_file.as_posix())
    fig = px.imshow(np.swapaxes(np.log(np.abs(k_space_ref[:, :, 10, 0])), 0, 1),
                    facet_col=-1, facet_col_wrap=4)
    fig_file = fig_path.joinpath("k-space-ref-semc").with_suffix(".html")
    logging.info(f"\t\t - write image k-space ref peek file: {fig_path.as_posix()}")
    fig.write_html(fig_file.as_posix())

    # remove os
    k_space_img_rm_os = helper_fns.remove_os(data=k_space_img, os_factor=os_factor)
    k_space_ref_rm_os = helper_fns.remove_os(data=k_space_ref, os_factor=os_factor)

    # fill data with reference scan (if there's no lines) for plotting
    k_space_all_data = k_space_img_rm_os.copy()
    k_space_all_data[
        (np.abs(k_space_img_rm_os) < 1e-9) & (np.abs(k_space_ref_rm_os) > 1e-9)
        ] = k_space_ref_rm_os[
        (np.abs(k_space_img_rm_os) < 1e-9) & (np.abs(k_space_ref_rm_os) > 1e-9)
    ]

    # naive recon
    logging.info(f"plot naive recon")
    img_naive_recon, _ = helper_fns.data_fft_to_time(data=k_space_all_data, axes=(0, 1))

    fig = px.imshow(np.abs(img_naive_recon[:, :, 9, 0, :]), facet_col=-1, facet_col_wrap=4)
    fig_file = fig_path.joinpath("img_naive_recon_semc_echoes_slice-10").with_suffix(".html")
    logging.info(f"\t\t - write image peek file: {fig_file.as_posix()}")
    fig.write_html(fig_file.as_posix())

    fig = px.imshow(np.abs(img_naive_recon[:, :, 10:18, 0, 0]), facet_col=-1, facet_col_wrap=4)
    fig_file = fig_path.joinpath("img_naive_recon_semc_echoe-0_slices").with_suffix(".html")
    logging.info(f"\t\t - write image peek file: {fig_file.as_posix()}")
    fig.write_html(fig_file.as_posix())


if __name__ == '__main__':
    # set up logging
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s -- %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    logging.info("_________________________________________________________")
    logging.info("____________ python - twix - raw data loader ____________")
    logging.info("_________________________________________________________")
    main()

import logging
from pypsi import Params
log_module = logging.getLogger(__name__)


def load_raw_data_and_interface(interface: Params, scale_to_max_val: float = 1000.0):
    """ returns k-space-image-data, k-space-sampling-mask,
        k-space-navigator-data (if applicable), k-space-navigator-mask (if applicable)
        affine-matrix, affine-nav-matrix (if-applicable)
    """
    # create shorthand
    img_params = interface.recon.multi_echo_img
    nav_params = interface.recon.navigator_img

    log_module.debug(f"allocate img arrays")
    etl = np.max(interface.sampling_k_traj.sampling_pattern["echo_num"].unique()) + 1
    if etl > img_params.etl:
        warn = f"recon interface set etl ({img_params.etl}) differs from num sampling pattern " \
               f"echo identifiers ({etl}), this might be okay if theres multiple sampling per readout pulse," \
               f" eg. megesse sequences"
        log_module.warning(warn)
    etl = np.max([etl, img_params.etl])

    # build image array
    # x is rl, y is pa
    if interface.pypulseq.acq_phase_dir == "RL":
        transpose_xy = True
    elif interface.pypulseq.acq_phase_dir == "PA":
        transpose_xy = False
    else:
        err = f"pypulseq phase encode direction {interface.pypulseq.acq_phase_dir} not recognized"
        log_module.exception(err)
        raise AttributeError(err)
    k_space = np.zeros(
        (img_params.n_read * img_params.os_factor, img_params.n_phase, img_params.n_slice, num_coils, etl),
        dtype=complex
    )
    k_sampling_mask = np.zeros(
        (img_params.n_read, img_params.n_phase, etl),
        dtype=bool
    )

    log_module.debug(f"allocate navigator arrays")
    nav = nav_params.n_read > 0
    if nav:
        nav_shape = (nav_params.n_read, nav_params.n_phase, nav_params.n_slice, num_coils)
        k_nav = np.zeros(nav_shape, dtype=complex)
        k_nav_mask = np.zeros(nav_shape[:2], dtype=bool)
    else:
        k_nav = None
        k_nav_mask = None

    # setup tkb
    # use tkbnufft interpolation to get k-space data from trajectory
    device = torch.device("cpu")
    # we want to regrid the data to the regular grid but have oversampling in read direction
    img_size = (img_params.os_factor * img_params.n_read,)
    grid_size = (img_params.os_factor * img_params.n_read,)
    tkbn_interp = tkbn.KbInterpAdjoint(
        im_size=img_size,
        grid_size=grid_size,
        device=device
    )

    log_module.debug("loop through acquisition types")
    for acq_type in interface.sampling_k_traj.k_trajectories["acquisition"].unique():
        log_module.info(f"processing {acq_type} acquisition data")
        # get all sampling pattern entries matching acq. type and no navigator
        sp_sub = interface.sampling_k_traj.sampling_pattern[
            (interface.sampling_k_traj.sampling_pattern["acq_type"] == acq_type)
        ]
        if nav:
            # split img data and navigators
            sp_img = sp_sub[~sp_sub["nav_acq"]]
            sp_nav = sp_sub[sp_sub["nav_acq"]]
        else:
            sp_img = sp_sub
            sp_nav = pd.DataFrame()

        # need trajectory
        k_traj_read_line = torch.from_numpy(
            interface.sampling_k_traj.k_trajectories[
                interface.sampling_k_traj.k_trajectories["acquisition"] == acq_type
                ]["k_traj_position"].to_numpy()
        )

        # get line numbers
        line_nums = sp_img.index.to_list()
        if line_nums:
            log_module.debug(f"\t\t___process img data")
            # get indices of entries
            ids_echo = torch.from_numpy(sp_img["echo_num"].to_numpy()).to(torch.int)
            ids_slice = torch.from_numpy(sp_img["slice_num"].to_numpy()).to(torch.int)
            ids_pe = torch.from_numpy(sp_img["pe_num"].to_numpy()).to(torch.int)

            log_module.info(f"\t\t - fill sampling mask")
            ids_pe_slice = torch.from_numpy(sp_img[sp_img["slice_num"] == 0]["pe_num"].to_numpy()).to(torch.int)
            ids_echo_slice = torch.from_numpy(sp_img[sp_img["slice_num"] == 0]["echo_num"].to_numpy()).to(torch.int)
            # k_sampling_mask[:, ids_pe_slice, ids_echo_slice] = 1

            log_module.info(f"\t\t - fill k-space data")
            data = torch.from_numpy(np.array([mdb_list[i].data.T for i in line_nums]))
            # want it in dimensions [N (batch), C (coils), k-length]
            data = torch.moveaxis(data, -1, 1)
            # small sanity check
            if data.shape[0] != len(sp_sub):
                err = "picked raw data size does not match number of sampling pattern entries"
                log_module.exception(err)
                raise AttributeError(err)

            if opts.grid_data:
                # from -pi to pi -- radians / voxel
                k_traj_read_line *= 2 * torch.pi
                traj_len = k_traj_read_line.shape[0]
                # calculate density compensation
                # log_module.info("\t\t - set density compensation")
                # dcomp = tkbn.calc_density_compensation_function(k_traj_read_line[None, :], im_size=img_size,
                #                                                 num_iterations=10)
                dcomp = 1.0

                # fill entries of trajectory    -> need dim [N (batch{, len(grid_size) == 1, k-length]
                k_traj_arr = torch.zeros((len(sp_sub), 1, traj_len))
                k_traj_arr[:, 0, :] = k_traj_read_line[None, :]
                # reshape to batch size
                if opts.visualize:
                    # peek at data
                    k_plot = torch.zeros(
                        (traj_len, img_params.n_phase, img_params.n_slice, num_coils, etl),
                        dtype=data.dtype
                    )
                    k_plot[:, ids_pe, ids_slice, :, ids_echo] = torch.moveaxis(data, -1, 1)
                    plotting.visualize_k_space(k_data_xyz_ch_t=k_plot.numpy(), opts=opts,
                                               name=f"test_data_{acq_type}_before_gridding")

                # set device
                k_traj_arr = k_traj_arr.to(device)
                data = data.to(device)

                # interpolate data
                log_module.info("\t\t - tkbn interpolation")
                data_interp = torch.fft.fftshift(tkbn_interp(dcomp * data, k_traj_arr).conj(), dim=-1)
                # we now have a k-space which should be having read direction interpolated from the trajectories.

                log_module.info(f"\t\t - assign to k-space")
                k_space[:, ids_pe, ids_slice, :, ids_echo] = torch.moveaxis(
                    data_interp.resolve_conj(), -1, 1
                ).numpy(force=True)

                # drop edges in pf region not acquired -> produces edge artifacts in interpolation
                if "pf" in acq_type:
                    # sanity check if 0 filled pf region starts at this side of the readout
                    # look at center of phase. supposed to be fully sampled
                    y = int(img_params.n_phase / 2)
                    # we extract nonzero elements of all respective entries (ids slice and echo)
                    # along the read direction and central phase encode
                    ids_nonzero_read = np.nonzero(np.abs(k_space[:, y, ids_slice, :, ids_echo]))[1]
                    # get unique entries
                    ids_nonzero_read_unique = np.unique(ids_nonzero_read)
                    # if pf not acquired region at lower side of k - space, the first two entries should have a bigger gap
                    # want to update sampling mask for pf readout, also extract 0 sampling reads
                    if np.diff(ids_nonzero_read_unique[:2]) > 3:
                        # 0 filled pf starts at x = 0, set outer line and first sampled line 0 to prevent edge artifacts
                        k_space[ids_nonzero_read_unique[0], ids_pe, ids_slice, :, ids_echo] = 0.0 + 0.0j
                        k_space[ids_nonzero_read_unique[1], ids_pe, ids_slice, :, ids_echo] = 0.0 + 0.0j
                        k_sampling_mask[:ids_nonzero_read_unique[1] + 1, ids_pe, ids_echo] = 0
                    elif np.diff(ids_nonzero_read_unique[-2:]) > 3:
                        # 0 filled part starts at x = img_read_n and stretches from this side towards central k space,
                        # set outer line and first sampled line 0 to prevent edge artifacts
                        k_space[ids_nonzero_read_unique[-1], ids_pe, ids_slice, :, ids_echo] = 0.0 + 0.0j
                        k_space[ids_nonzero_read_unique[-2], ids_pe, ids_slice, :, ids_echo] = 0.0 + 0.0j
                        k_sampling_mask[ids_nonzero_read_unique[-2:], ids_pe, ids_echo] = 0
                    else:
                        err = f"set partial fourier acquisition but couldn't find 0-filled outer k-space"
                        log_module.exception(err)
                        AttributeError(err)
            else:
                # no gridding
                # calculate the closest points on grid from trajectory. get positions
                k_traj_pos = (0.5 + k_traj_read_line) * img_params.n_read * img_params.os_factor
                # just round to the nearest point in 1d
                k_traj_pos = k_traj_pos.round().to(torch.int)
                # find indices of points that are actually within our k-space in case the trajectory maps too far out
                k_traj_pos_valid = torch.squeeze(
                    torch.nonzero((k_traj_pos >= 0) &
                                  (k_traj_pos < img_params.os_factor * img_params.n_read)))
                # find positions of those indices
                k_traj_pos_valid_use = k_traj_pos[k_traj_pos_valid]
                # # rm os
                # for idx in range(k_traj_pos_valid_use.shape[0]):
                #     traj_idx_rm_os = k_traj_pos_valid_use[idx]
                #     if traj_idx_rm_os % img_params.os_factor == 1:
                #         # would be rm by oversampling
                #         # check if theres a neighbor not removed, if not use the idx
                #         if not traj_idx_rm_os - 1 in k_traj_pos_valid_use and traj_idx_rm_os - 1 >= 0:
                #             k_traj_pos_valid_use[idx] = traj_idx_rm_os - 1
                # k_traj_pos_valid_rm_os = k_traj_pos_valid_use[k_traj_pos_valid_use % img_params.os_factor == 0]
                # split into data indices and k space indices without os
                indices_in_trajectory_data = torch.squeeze(
                    torch.nonzero(k_traj_pos_valid_use)
                )
                indices_in_k_space_img = k_traj_pos_valid_use.to(torch.int)
                indices_in_k_sampling_img = (
                        indices_in_k_space_img[::img_params.os_factor] / img_params.os_factor
                ).to(torch.int).numpy()
                # pick valid locations of data and remove os
                # k_traj_pos_valid_rm_os = k_traj_pos_valid_rm_os[k_traj_pos_valid_rm_os % img_params.os_factor == 0]
                k_no_grid_sub = data[:, :, indices_in_trajectory_data]
                # fill at respective positions
                # k_rm_os_pos = (k_traj_pos_valid_rm_os / img_params.os_factor).to(torch.int)
                k_space[
                    indices_in_k_space_img[None, :],
                    ids_pe[:, None], ids_slice[:, None],
                    :, ids_echo[:, None]
                ] = torch.moveaxis(k_no_grid_sub, -1, 1).numpy(force=True)
                k_sampling_mask[indices_in_k_sampling_img[:, None], ids_pe_slice[None, :], ids_echo_slice[None, :]] = 1

        # process navigators - no gridding
        # get line numbers
        line_nums = sp_nav.index.to_list()
        if line_nums:
            log_module.debug(f"\t\t___process nav data")
            # get indices of entries
            ids_slice = torch.from_numpy(sp_nav["slice_num"].to_numpy()).to(torch.int)
            ids_pe = torch.from_numpy(sp_nav["pe_num"].to_numpy()).to(torch.int)

            log_module.info(f"\t\t - fill sampling mask")
            ids_pe_slice = torch.from_numpy(sp_nav[sp_nav["slice_num"] == 0]["pe_num"].to_numpy()).to(torch.int)
            k_nav_mask[:, ids_pe_slice] = 1

            log_module.info(f"\t\t - fill k-space data")
            data = torch.from_numpy(np.array([mdb_list[i].data.T for i in line_nums]))
            # want it in dimensions [N (batch), C (coils), k-length]
            data = torch.moveaxis(data, -1, 1)
            # small sanity check
            if data.shape[0] != len(sp_sub):
                err = "picked raw data size does not match number of sampling pattern entries"
                log_module.exception(err)
                raise AttributeError(err)
            # no gridding
            # calculate the closest points on grid from trajectory. get positions
            k_traj_pos = (0.5 + k_traj_read_line) * nav_params.n_read * nav_params.os_factor
            # just round to the nearest point in 1d
            k_traj_pos = k_traj_pos.round().to(torch.int)
            # find indices of points that are actually within our k-space
            k_traj_pos_valid = torch.squeeze(
                torch.nonzero((k_traj_pos >= 0) &
                              (k_traj_pos < nav_params.os_factor * nav_params.n_read)))
            # find positions of those indices
            k_traj_pos_valid = k_traj_pos[k_traj_pos_valid]
            # pick valid locations of data and remove os
            # k_traj_pos_valid = k_traj_pos_valid_rm_os[k_traj_pos_valid_rm_os % nav_params.os_factor == 0]
            k_no_grid_sub = data[:, :, k_traj_pos_valid]
            # fill at respective positions
            k_nav[k_traj_pos_valid[None, :], ids_pe[:, None], ids_slice[:, None], :] = torch.moveaxis(
                k_no_grid_sub, -1, 1
            ).numpy(force=True)

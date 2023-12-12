import logging
from lprd import lprd_io, options


def main(opts: options.Config):
    # check files provided, check output path exist ifn make
    opts.check_paths()
    lprd_io.load_raw_data_and_interface(
        opts=opts
    )


if __name__ == '__main__':
    # create cli parser
    parser, prog_args = options.create_cli()
    if prog_args.config.debug_flag:
        level = logging.DEBUG
    else:
        level = logging.INFO
    # set up logging
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s -- %(message)s',
                        datefmt='%I:%M:%S', level=level)
    logging.info("_________________________________________________________")
    logging.info("____________ python - twix - raw data loader ____________")
    logging.info("_________________________________________________________")
    prog_opts = options.Config.from_cli_args(prog_args)
    prog_opts.display_settings()

    # run script
    try:
        main(prog_opts)
    except Exception as e:
        logging.exception(e)
        parser.print_usage()
        exit(-1)
    logging.info(f"Processing finished successful!")

import typing

import simple_parsing as sp
import dataclasses as dc
import pathlib as plib
import logging
import pandas as pd

pd.options.display.max_colwidth = 100
log_module = logging.getLogger(__name__)


@dc.dataclass
class Config(sp.Serializable):
    raw_data_file: str = sp.field(
        alias=["-i", "-rdf"],
        default="/data/pt_np-jschmidt/code/data_pipe_pulseq/example_data/pulseq_jstmc4b_fidnav.dat"
    )
    pyp_interface_file: str = sp.field(
        alias=["-p", "-pif"],
        default="/data/pt_np-jschmidt/code/data_pipe_pulseq/example_data/jstmc4b_sampling-pattern.pkl",
        help="necessary when dealing with pypulseq sequences. Input of recon data and tracing"
    )
    output_path: str = sp.field(
        alias="-o",
        default="")
    config_file: str = sp.field(
        alias="-c",
        default=""
    )
    visualize: bool = sp.field(
        alias="-v", default=True
    )
    debug_flag: bool = sp.field(
        alias="-d", default=False
    )
    grid_data: bool = sp.field(
        alias="-g", default=False
    )
    seq_type: str = sp.field(
        alias="-t", default="mese_fidnav"
    )

    def _set_output_path_to_input(self):
        in_path = plib.Path(self.raw_data_file).absolute().parent.joinpath("lprd")
        self.output_path = in_path.as_posix().__str__()

    def check_paths(self):
        o_p = plib.Path(self.output_path).absolute()
        if o_p.suffixes:
            o_p = o_p.parent
        o_p.mkdir(parents=True, exist_ok=True)
        self.output_path = o_p.as_posix()
        if self.visualize:
            o_p.joinpath("plots").mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_cli_args(cls, args: sp.ArgumentParser.parse_args):
        # to catch additional non default input via cli we compare to default
        # -> WILL NOT catch additional input args coinciding with defaults
        default_vals_dict = cls().to_dict()
        instance = cls()

        # if provided load config file
        if args.config.config_file:
            c_f = plib.Path(args.config.config_file).absolute()
            if c_f.is_file():
                log_module.info(f"loading config file: {c_f}")
                instance = cls.load(c_f)
        # check additional non default inputs
        for key, val in args.config.to_dict().items():
            if args.config.__getattribute__(key) != default_vals_dict.__getitem__(key):
                instance.__setattr__(key, val)

        if not instance.output_path:
            instance._set_output_path_to_input()
        return instance

    @classmethod
    def from_config_file(cls, config_file: typing.Union[str, plib.Path]):
        c_f = plib.Path(config_file).absolute()
        if not c_f.is_file():
            err = "config file path provided does not point to a file. exiting"
            log_module.exception(err)
            raise FileNotFoundError(c_f)
        instance = cls.load(config_file)
        if not instance.output_path:
            instance._set_output_path_to_input()
        return instance

    def display_settings(self):
        self_dict = self.to_dict()
        for key in ["raw_data_file", "pyp_interface_file", "output_path"]:
            plp = plib.Path(self_dict.get(key)).absolute()
            shortened = plp.name
            add_counter = 1
            while len(shortened) < 20:
                shortened = plp.parts[-1 - add_counter] + "/" + shortened
                add_counter += 1
            shortened = "... "+shortened
            self_dict.__setitem__(key, shortened)
        df = pd.concat((pd.Series({"": ""}), pd.Series(self_dict)), axis=0)
        log_module.info(df)


def create_cli():
    parser = sp.ArgumentParser(prog="raw_data_pipe")
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()
    return parser, args


if __name__ == '__main__':
    c = Config()
    c.save_json("/data/pt_np-jschmidt/code/data_pipe_pulseq/external_files/test_opts.json", indent=2)

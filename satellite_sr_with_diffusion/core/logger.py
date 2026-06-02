import logging
import os
import os.path as osp


def setup_paths(opt):
    """
    Create the experiment/result/log directory structure and finalise
    ``opt['path']`` for the current run (called by the Hydra config loader).
    """
    # Create experiment root and directory structure
    exp_name = opt["name"]
    if opt["phase"] == "train":
        exp_path = osp.join(os.getcwd(), "experiments", exp_name)
    else:
        exp_path = osp.join(os.getcwd(), "results", exp_name)

    if opt["phase"] == "train":
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        path_log = osp.join(exp_path, "logs")

        path_results = osp.join(exp_path, "results")
        path_checkpoint = osp.join(exp_path, "checkpoint")
        if not os.path.exists(path_log):
            os.makedirs(path_log)

        if not os.path.exists(path_results):
            os.makedirs(path_results)
        if not os.path.exists(path_checkpoint):
            os.makedirs(path_checkpoint)
        opt["path"]["experiments_root"] = exp_path
        opt["path"]["log"] = path_log

        opt["path"]["results"] = path_results
        opt["path"]["checkpoint"] = path_checkpoint
    else:  # validation
        # Prefer user-provided paths; fall back to defaults under exp_path
        path_log = opt["path"].get("log", osp.join(exp_path, "logs"))
        path_results = opt["path"].get("results", exp_path)

        # Only create the fallback root if we actually use it
        if (path_log.startswith(exp_path) or path_results == exp_path) and not os.path.exists(
            exp_path
        ):
            os.makedirs(exp_path)

        if not os.path.exists(path_log):
            os.makedirs(path_log)
        if not os.path.exists(path_results):
            os.makedirs(path_results)

        opt["path"]["log"] = path_log
        opt["path"]["results"] = path_results

    return opt


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    """Initializes and configures a logger."""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(message)s", datefmt="%y-%m-%d %H:%M:%S"
    )
    log_file = osp.join(root, f"{phase}.log")
    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(formatter)
    lg.setLevel(level)
    lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def dict_to_nonedict(opt):
    """Recursively converts a dictionary to a NoneDict."""
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(new_opt)
    if isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    return opt


class NoneDict(dict):
    """A dictionary that returns None for missing keys."""

    def __getitem__(self, key):
        return dict.get(self, key)


def dict2str(opt, indent_l=1):
    """Converts a dictionary to a formatted string for logging."""
    msg = ""
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += " " * (indent_l * 2) + k + ":[\n"
            msg += dict2str(v, indent_l + 1)
            msg += " " * (indent_l * 2) + "]\n"
        else:
            msg += " " * (indent_l * 2) + k + ": " + str(v) + "\n"
    return msg

import json
import os
import time

import torch


def get_hook_name(layerix: int, use_post=False):
    return f'blocks.{layerix}.mlp.hook_{"post" if use_post else "pre"}'


def time_function(function, *args, **kwargs):
    start_time = time.perf_counter()
    out = function(*args, **kwargs)
    return out, time.perf_counter() - start_time


def save_tensor(filename, tensor):
    with open(filename, 'wb') as f:
        torch.save(tensor, f)


def load_tensor(filename):
    with open(filename, 'rb') as f:
        return torch.load(f)


def save_json(filenname, data):
    with open(filenname, 'w') as f:
        json.dump(data, f)


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def get_experiment_dir(args):
    return os.path.join(
        os.environ.get('RESULTS_DIR', 'results'),
        args.experiment_type[0],
        args.model,
        args.feature_dataset,
        args.experiment_name if args.experiment_name else args.experiment_type
    )


def get_experiment_info_str(args):
    info_str = ""
    info_str += f'Running activations experiment "{args.experiment_name}" of type "{args.experiment_type}"\n'
    info_str += f'\tmodel: {args.model}\n'
    info_str += f'\tfeature dataset name: {args.feature_dataset}\n'
    info_str += f'\tn_threads: {args.n_threads}, device: {args.device}\n'
    info_str += f'\tbatch size: {args.batch_size}, output_precision: {args.output_precision}\n'
    return info_str


def get_experiment_metadata(args, total_time):
    arg_dict = vars(args)
    arg_dict['total_time'] = total_time
    arg_dict['current_time'] = time.time()
    return arg_dict


def expand_abs(path):
    return os.path.abspath(os.path.expanduser(path))

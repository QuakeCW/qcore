"""
Functions used throughout ucgmsim.
Mostly related to file system operations and other non-specific functionality.
"""

from shutil import rmtree
import os
import imp
import yaml
from collections import OrderedDict


class DotDictify(dict):
    MARKER = object()

    def __init__(self, value=None):
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError('expected dict')

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, DotDictify):
            value = DotDictify(value)
        super(DotDictify, self).__setitem__(key, value)

    def __getitem__(self, key):
        found = self.get(key, DotDictify.MARKER)
        if found is DotDictify.MARKER:
            found = DotDictify()
            super(DotDictify, self).__setitem__(key, found)
        return found

    __setattr__, __getattr__ = __setitem__, __getitem__



def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


def load_yaml(yaml_file, obj_type=OrderedDict):
    """obj_type: None for using a ordinary Dict; default using OrderedDict"""
    with open(yaml_file, 'r') as stream:
        try:
            if obj_type is None:
                return yaml.load(stream)
            else:
                return ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=obj_type)
        except yaml.YAMLError as exc:
            print(exc)


def ordered_dump(data, stream, Dumper=yaml.Dumper, representer=OrderedDict, **kwds):
    class OrderedDumper(Dumper):
        pass
    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())
    OrderedDumper.add_representer(representer, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def dump_yaml(input_dict, output_name, obj_type=OrderedDict):
    """obj_type: Dict for using a ordinary Dict; default using OrderedDict"""
    with open(output_name, 'w') as yaml_file:
        ordered_dump(input_dict, yaml_file, Dumper=yaml.SafeDumper, representer=obj_type, default_flow_style=False)
       # yaml.add_representer(OrderedDict, lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))
       # yaml.dump(input_dict, yaml_file, default_flow_style=False)


def load_params(*yaml_files):
    dot_dict = DotDictify({})
    for yaml_file in yaml_files:
        d = DotDictify(load_yaml(yaml_file))
        dot_dict.update(d)
    return dot_dict


def setup_dir(directory, empty=False):
    """
    Make sure a directory exists, optionally make sure it is empty.
    directory: path to directory
    empty: make sure directory is empty

    :param directory:
    :param empty:
    :return:
    """
    if os.path.exists(directory) and empty:
            rmtree(directory)
    if not os.path.exists(directory):
        # multi processing safety (not useful with empty set)
        try:
            os.makedirs(directory)
        except OSError:
            if not os.path.isdir(directory):
                raise


def load_py_cfg(f_path):
    """
    loads a python configuration file to a dictionary

        if you want to preserve the import params functionality, locals().update(cfg_dict) converts the returned dict to local variables.

    :param f_path: path to configuration file
    :return: dict of parameters
    """
    with open(f_path) as f:
        module = imp.load_module('params', f, f_path, ('.py', 'r', imp.PY_SOURCE))
        cfg_dict = module.__dict__

    return cfg_dict



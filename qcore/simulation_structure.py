"""
Gives access to the folder structure of the cybershake directory
"""
import os


def __get_fault_from_realisation(realisation):
    return realisation.split('_')[0]


def get_realisation_name(fault_name, rel_no):
    return "{}_REL{:0>2}".format(fault_name, rel_no)


# VM
def get_VM_dir(cybershake_root, realisation):
    fault = __get_fault_from_realisation(realisation)
    return os.path.join(cybershake_root, 'Data', 'VMs', fault)


# SRF
def get_srf_location(realisation):
    fault = __get_fault_from_realisation(realisation)
    return os.path.join(fault, 'Srf', realisation + '.srf')


def get_srf_path(cybershake_root, realisation):
    return os.path.join(cybershake_root, 'Data', 'Sources', get_srf_location(realisation))

# Source_params
def get_source_params_location(realisation):
    fault = __get_fault_from_realisation(realisation)
    return os.path.join(fault, 'Sim_params', realisation + '.yaml')


def get_source_params_path(cybershake_root, realisation):
    return os.path.join(cybershake_root, 'Data', 'Sources', get_source_params_location(realisation))

# Stoch
def get_stoch_location(realisation):
    fault = __get_fault_from_realisation(realisation)
    return os.path.join(fault, 'Stoch', realisation + '.stoch')

def get_sources_dir(cybershake_root):
    """Gets the cybershake sources directory"""
    return os.path.join(cybershake_root, 'Data', 'Sources')

def get_stoch_path(cybershake_root, realisation):
    return os.path.join(cybershake_root, 'Data', 'Sources', get_stoch_location(realisation))

# Runs
def get_runs_dir(cybershake_root):
    """Gets the path to the Runs directory of a cybershake run"""
    return os.path.join(cybershake_root, "Runs")

# Cybershake
def get_cybershake_config(cybershake_root):
    """Gets the path to the cybershake config json file"""
    return os.path.join(cybershake_root, "cybershake_config.json")

def get_cybershake_list(cybershake_root):
    """Gets the cybershake list, specifying the faults and number of realisation"""
    return os.path.join(cybershake_root, "list.txt")

# LF
def get_lf_dir(sim_root):
    return os.path.join(sim_root, 'LF')


def get_lf_outbin_dir(sim_root):
    return os.path.join(get_lf_dir(sim_root), 'OutBin')


# BB
def get_bb_dir(sim_root):
    return os.path.join(sim_root, 'BB')


def get_bb_acc_dir(sim_root):
    return os.path.join(get_bb_dir(sim_root), 'Acc')


def get_bb_bin_path(sim_root):
    return os.path.join(get_bb_acc_dir(sim_root), 'BB.bin')


# HF
def get_hf_dir(sim_root):
    return os.path.join(sim_root, 'HF')


def get_hf_acc_dir(sim_root):
    return os.path.join(get_hf_dir(sim_root), 'ACC')

def get_hf_bin_path(sim_root):
    return os.path.join(get_hf_acc_dir(sim_root), 'HF.bin')


# yaml
def get_fault_yaml_path(sim_root, fault_name=''):
    """
    manual: Albury_VM_home_melodypzhu_Albury_new_bench_Data_VMs_Albury-h0p4_EMODv3p0p4_190105/fault_params.yaml
    auto: Runs/Albury/fault_params.yaml
    :param sim_root: Albury_VM_home_melodypzhu_Albury_new_bench_Data_VMs_Albury-h0p4_EMODv3p0p4_190105 or Runs
    :param fault_name: '' or Albury
    :return: path to fault_params.yaml
    """
    return os.path.join(sim_root, fault_name, 'fault_params.yaml')


def get_root_yaml_path(sim_root):
    """
    manual: Albury_VM_home_melodypzhu_Albury_new_bench_Data_VMs_Albury-h0p4_EMODv3p0p4_190105/root_params.yaml
    auto: Runs/root_params.yaml
    :param sim_root: Albury_VM_home_melodypzhu_Albury_new_bench_Data_VMs_Albury-h0p4_EMODv3p0p4_190105 or Runs
    :return: path to root_params.yaml
    """
    return os.path.join(sim_root, 'root_params.yaml')










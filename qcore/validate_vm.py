#!/usr/bin/env python
"""
Checks EMOD3D VM folder for corectness. Compatible with Python 2.6+ and 3.0+.
Run script with VM folder location as first parameter. Returns 0 if successful.
or:
Import and use validate_vm directly.
Example of running in a bash script:
======================================
validate_vm.py /path/to/VMs/AlpineRegion
if [ $? -eq 0 ]; then
    echo success
else
    echo fail
fi
"""

import os
import argparse
from pathlib import Path

import matplotlib.path as mpltPath
from qcore.utils import load_yaml
from qcore.constants import VM_PARAMS_FILE_NAME, VMParams
from qcore.srf import get_bounds
from qcore.geo import ll_dist, compute_intermediate_latitudes

import numpy as np

SINGLE_FILE_SUB_PARSER = "file"
NZVM_SUB_PARSER = "files"
PARAMS_SUB_PARSER = "params"

SIZE_FLOAT = 4

MIN_LAT = -53
MAX_LAT = -28
MIN_LON = 160
MAX_LON = 185


def validate_vm_params(vm_params: str):
    """
    Validates the vm_params yaml file required for velocity model generation
    :param vm_params: The path to the vm_params yaml file
    :return: A bool and a string representing is the params file has validated successfully and the reason for failure if it did not
    """
    vm_params = Path(vm_params)

    # The vm directory and yaml file have to exist
    if not vm_params.parent.exists() or not vm_params.parent.is_dir():
        return False, f"VM dir is not a directory: {vm_params.parent}"

    if not vm_params.exists() or not vm_params.is_file():
        return False, f"VM params file is not a file that exists: {vm_params}"

    vm_params_dict = load_yaml(vm_params)

    # Check vm_params.yaml domain size consistency
    # Can probably check for the existence of key variable too if needed
    def get_expected_size(length):
        return int(round(length / vm_params_dict[VMParams.hh.value]))

    xyz_values = [
        (
            VMParams.nx.value,
            vm_params_dict[VMParams.nx.value],
            get_expected_size(vm_params_dict[VMParams.extent_x.value]),
        ),
        (
            VMParams.ny.value,
            vm_params_dict[VMParams.ny.value],
            get_expected_size(vm_params_dict[VMParams.extent_y.value]),
        ),
        (
            VMParams.nz.value,
            vm_params_dict[VMParams.nz.value],
            get_expected_size(
                vm_params_dict[VMParams.extent_zmax.value]
                - vm_params_dict[VMParams.extent_zmin.value]
            ),
        ),
    ]
    errors = []
    for name, actual, expected in xyz_values:
        if actual != expected:
            errors.append(
                f"Dimension {name} had a value mismatch. Grid size {actual} does not match up with calculated size: {expected}"
            )

    if errors:
        return False, "\n".join(errors)
    return True, ""


def validate_vm_files(vm_dir: str, srf: str = None):
    """
    Validates the files generated by the NZVM binary and supplementary scripts during VM generation.
    Attempts to test as much as possible before returning, in an attempt to reduce the number of retests required.
    :param vm_dir: The path to the VM directory
    :param srf: The path to the srf to be used with the VM. Used to check the srf is within the bounds of the VM
    :return: A bool and a string representing is the VM has validated successfully and the reason for failure if it did not
    """

    vm_dir = Path(vm_dir)

    errors = []
    vel_crns_file = vm_dir / "VeloModCorners.txt"
    vm_params_file_path = vm_dir / VM_PARAMS_FILE_NAME

    # Ensure the directory exists
    if not vm_dir.exists() or not vm_dir.is_dir():
        return False, "VM dir is not a directory: {}".format(vm_dir)

    # Ensure all required files exist
    vm_params_dict = load_yaml(vm_params_file_path)

    vm_files = [vm_dir / "vs3dfile.s", vm_dir / "vp3dfile.p", vm_dir / "rho3dfile.d"]
    all_files = [
        *vm_files,
        vel_crns_file,
        vm_params_file_path,
        vm_dir / f"gridout{vm_params_dict['sufx']}",
        vm_dir / f"gridfile{vm_params_dict['sufx']}",
        vm_dir / f"model_bounds{vm_params_dict['sufx']}",
        vm_dir / f"model_coords{vm_params_dict['sufx']}",
        vm_dir / f"model_params{vm_params_dict['sufx']}",
    ]
    for vm_file in all_files:
        if not vm_file.exists():
            errors.append(f"VM file not found: {vm_file}")

    # Check binary file sizes for files that exist
    vm_size = (
        vm_params_dict[VMParams.nx.value]
        * vm_params_dict[VMParams.ny.value]
        * vm_params_dict[VMParams.nz.value]
        * SIZE_FLOAT
    )
    for file_path in vm_files:
        # Test all files we can, so we get all the problems at once
        if file_path.exists():
            errors.extend(validate_vm_file(file_path, vm_size))

    if vel_crns_file.exists():
        polygon = []
        with open(vel_crns_file) as crns_fp:
            next(crns_fp)
            next(crns_fp)
            for line in crns_fp:
                lon, lat = map(float, line.split())
                lon = lon % 360
                polygon.append((lon, lat))
        if srf is not None:
            srf_bounds = get_bounds(srf)
        else:
            srf_bounds = None
        errors.extend(validate_vm_bounds(polygon, srf_bounds))

    if errors:
        return False, "\n".join(errors)
    return True, ""


def validate_vm_bounds(polygon, srf_bounds=None):
    """
    Validates the VM domain against the DEM and the srf bounds
    :param polygon: A list of (lon, lat) tuples giving the corners of the VM
    :param srf: A list of (lon, lat) tuples giving the corners of the VM. Not used if None
    :return: A list of error messages resulting from this validation
    """
    errors = []

    for lon, lat in polygon:
        if lon < MIN_LON or lon > MAX_LON or lat < MIN_LAT or lat > MAX_LAT:
            errors.append(f"VM extents not contained within NZVM DEM: {lon}, {lat}")
    # Check SRF is within bounds of the VM if it is given
    if srf_bounds is not None:
        edges = []
        for index, start_point in enumerate(polygon):
            end_point = polygon[(index + 1) % len(polygon)]
            lons = (
                    np.linspace(
                        start_point[0],
                        end_point[0],
                        int(ll_dist(*start_point, *end_point)),
                    )
                    % 360
            )
            lats = compute_intermediate_latitudes(start_point, end_point, lons)
            edges.extend(list(zip(lons, lats)))
        path = mpltPath.Path(edges)
        for bounds in srf_bounds:
            if not all(path.contains_points(bounds)):
                errors.append(
                    "Srf extents not contained within velocity model corners"
                )
    return errors


def validate_vm_file(file_name: Path, vm_size: int):
    """
    Validates that a velocity model file has the correct size, and no 0 values
    :param file_name: A Path object representing the file to test
    :param vm_size: The size of the VM
    :return: A possibly empty list of issues with the VM file
    """
    errors = []
    size = file_name.stat().st_size
    if size != vm_size * SIZE_FLOAT:
        errors.append(
            f"VM filesize for {file_name} expected: {vm_size * SIZE_FLOAT} found: {size}"
        )
    if (
        not np.min(
            np.fromfile(file_name, dtype="<f{}".format(SIZE_FLOAT), count=vm_size)
        )
        > 0
    ):
        errors.append(f"File {file_name} has minimum value of 0.0")
    return errors


def main():
    parser = argparse.ArgumentParser()

    sub_parser = parser.add_subparsers(dest="subparser_name")

    vm_params_parser = sub_parser.add_parser(
        PARAMS_SUB_PARSER,
        help="Validate a vm_params.yaml file. If the srf is given, will test that the srf is contained within the VM.",
    )
    vm_params_parser.add_argument(
        "vm_params", type=Path, help="Path to vm_params.yaml file"
    )
    vm_params_parser.add_argument(
        "srf", type=Path, help="Path to srf file", nargs="?", default=None
    )

    vm_file_parser = sub_parser.add_parser(
        NZVM_SUB_PARSER, help="Validate files generated by the NZVM"
    )
    vm_file_parser.add_argument("vm_dir", type=Path, help="path the VM folder")

    pert_parser = sub_parser.add_parser(
        SINGLE_FILE_SUB_PARSER,
        help="Validate a single VM file for size and 0s. Primarily intended for perturbation files.",
    )
    pert_parser.add_argument("vm_file", type=Path, help="Path the VM file to test")
    pert_parser.add_argument("vm_params", type=Path, help="Path to vm_params.yaml file")

    args = parser.parse_args()

    if args.subparser_name == PARAMS_SUB_PARSER:
        valid, error_message = validate_vm_params(args.vm_params)
    elif args.subparser_name == NZVM_SUB_PARSER:
        valid, error_message = validate_vm_files(args.vm_dir, args.srf)
    elif args.subparser_name == SINGLE_FILE_SUB_PARSER:
        vm_params_dict = load_yaml(args.vm_params)
        size = (
            vm_params_dict[VMParams.nx.value]
            * vm_params_dict[VMParams.ny.value]
            * vm_params_dict[VMParams.nz.value]
        )
        error_message = validate_vm_file(args.vm_file, size)
        valid = not error_message
    else:
        valid = False
        error_message = ""
        parser.error("Please choose a subcommand")

    if not valid:
        print(error_message)
        return 1
    else:
        return 0


if __name__ == "__main__":
    exit(main())

import h5py
import argparse
import numpy as np
import attr
import os
from os.path import isfile, join
import subprocess
from shutil import copyfile
from .pulse_aggregator import (
    aggregate_events_by_pulse,
    remove_data_not_used_by_mantid,
    patch_geometry,
)
import matplotlib.pylab as pl


@attr.s
class DatasetDetails(object):
    name = attr.ib()
    full_path = attr.ib()
    parent_path = attr.ib()
    text = attr.ib()


def find_variable_length_string_datasets(name, object):
    if isinstance(object, h5py.Dataset):
        if object.dtype == np.object:
            text = str(object[...])
            datasets_to_convert.append(
                DatasetDetails(
                    object.name.split("/")[-1],
                    object.name,
                    "/".join(object.name.split("/")[:-1]),
                    text,
                )
            )
    elif isinstance(object, h5py.Group):
        # Fix attributes
        for key, value in object.attrs.items():
            if isinstance(value, str):
                object.attrs[key] = np.string_(value)


def add_nx_class_to_group(group, nx_class_name):
    group.attrs.create(
        "NX_class", np.array(nx_class_name).astype(f"|S{len(nx_class_name)}")
    )


def add_nx_class_to_groups(group_names, nx_class_name, outfile):
    for group_name in group_names:
        add_nx_class_to_group(outfile[group_name], nx_class_name)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-i",
    "--input-directory",
    type=str,
    help="Directory with raw files to convert (all files ending .hdf assumed to be raw)",
    required=True,
)
parser.add_argument(
    "--format-convert",
    type=str,
    help="Path to h5format_convert and h5repack executables",
    required=True,
)
parser.add_argument(
    "--chopper-tdc-path",
    type=str,
    help="Path to the chopper TDC unix timestamps (ns) dataset in the file",
    default="/entry/instrument/chopper_1/top_dead_center/time",
)
parser.add_argument(
    "--tdc-pulse-time-difference",
    type=int,
    help="Time difference between TDC timestamps and pulse T0 in integer nanoseconds",
    default=0,
)
parser.add_argument(
    "--only-this-file", type=str, help="Only process file with this name"
)
args = parser.parse_args()

filenames = [
    join(args.input_directory, f)
    for f in os.listdir(args.input_directory)
    if isfile(join(args.input_directory, f))
]


def convert_to_fixed_length_strings():
    global datasets_to_convert
    datasets_to_convert = []
    output_file.visititems(find_variable_length_string_datasets)
    for dataset in datasets_to_convert:
        del output_file[dataset.full_path]
        output_file[dataset.parent_path].create_dataset(
            dataset.name,
            data=np.array(dataset.text).astype("|S" + str(len(dataset.text))),
        )


def add_attributes_to_node(node, attributes: dict):
    for key in attributes:
        if isinstance(attributes[key], str):
            # Since python 3 we have to treat strings like this
            node.attrs.create(
                key, np.array(attributes[key]).astype("|S" + str(len(attributes[key])))
            )
        else:
            node.attrs.create(key, np.array(attributes[key]))


def _link_log(outfile, log_group, source_path, target_name):
    log_group[target_name] = outfile[source_path]
    try:
        log_group[f"{target_name}/value"] = log_group[f"{target_name}/raw_value"]
    except:
        pass
    # Mantid doesn't assume relative unix epoch (should according to NeXus standard)
    if "start" not in log_group[f"{target_name}/time"].attrs:
        unix_epoch = "1970-01-01T00:00:00Z"
        log_group[f"{target_name}/time"].attrs.create(
            "start", np.array(unix_epoch).astype("|S" + str(len(unix_epoch)))
        )
    if "units" not in log_group[f"{target_name}/time"].attrs:
        nanosecs = "ns"
        log_group[f"{target_name}/time"].attrs.create(
            "units", np.array(nanosecs).astype("|S" + str(len(nanosecs)))
        )

    if log_group[f"{target_name}/time"].attrs.get("units") == b"ns":
        # Mantid doesn't know about nanoseconds, we'll have to reduce the precision to microseconds
        microsecs = "us"
        log_group[f"{target_name}/time"].attrs.modify(
            "units", np.array(microsecs).astype("|S" + str(len(microsecs)))
        )
        # Convert nanoseconds to microseconds and store as float not int
        times = log_group[f"{target_name}/time"][...].astype(float) * 0.001
        times_attrs = dict(log_group[f"{target_name}/time"].attrs)
        del log_group[f"{target_name}/time"]
        log_group[target_name].create_dataset("time", dtype=float, data=times)
        add_attributes_to_node(log_group[f"{target_name}/time"], times_attrs)


def link_logs():
    log_group = output_file["/entry"].create_group("logs")
    add_nx_class_to_group(log_group, "IXselog")
    _link_log(
        output_file,
        log_group,
        "/entry/instrument/linear_axis_1/target_value",
        "linear_axis_1_target_value",
    )
    _link_log(
        output_file,
        log_group,
        "/entry/instrument/linear_axis_1/value",
        "linear_axis_1_value",
    )
    _link_log(
        output_file,
        log_group,
        "/entry/instrument/linear_axis_2/target_value",
        "linear_axis_2_target_value",
    )
    _link_log(
        output_file,
        log_group,
        "/entry/instrument/linear_axis_2/value",
        "linear_axis_2_value",
    )
    _link_log(output_file, log_group, "/NTP_MRF_time_diff", "NTP_MRF_time_diff")
    for chopper_number in range(1, 9):
        _link_log(
            output_file,
            log_group,
            f"/entry/instrument/chopper_{chopper_number}/top_dead_center",
            f"chopper_{chopper_number}_TDC",
        )


for filename in filenames:
    name, extension = os.path.splitext(filename)
    if extension != ".hdf":
        continue

    onlypath, onlyname = os.path.split(filename)
    if args.only_this_file and onlyname != args.only_this_file:
        continue

    print(f"#############################################\nProcessing file: {filename}")

    # First run pulse aggregation
    output_filename = f"{name}.nxs"
    print("Copying input file")
    copyfile(filename, output_filename)
    with h5py.File(output_filename, "r+") as output_file:
        # DENEX detector
        print("Aggregating DENEX detector events")
        aggregate_events_by_pulse(
            output_file,
            args.chopper_tdc_path,
            "/entry/instrument/detector_1/raw_event_data",
            args.tdc_pulse_time_difference,
        )

        # Monitor
        print("Aggregating monitor events")
        aggregate_events_by_pulse(
            output_file,
            args.chopper_tdc_path,
            "/entry/monitor_1/events",
            args.tdc_pulse_time_difference,
            output_group_name="monitor_event_data",
            event_id_override=262144,
        )

        print("Adding missing NX_class attributes")
        add_nx_class_to_groups(
            [
                "/entry/instrument/linear_axis_1/speed",
                "/entry/instrument/linear_axis_1/status",
                "/entry/instrument/linear_axis_1/target_value",
                "/entry/instrument/linear_axis_1/value",
                "/entry/instrument/linear_axis_2/speed",
                "/entry/instrument/linear_axis_2/status",
                "/entry/instrument/linear_axis_2/target_value",
                "/entry/instrument/linear_axis_2/value",
                "/entry/sample/transformations/linear_stage_1_position",
                "/entry/sample/transformations/linear_stage_2_position",
                "/NTP_MRF_time_diff",
            ],
            "NXlog",
            output_file,
        )

        print("Removing groups without NX_class defined")
        remove_data_not_used_by_mantid(output_file, chatty=False)

        print("Patching geometry")
        patch_geometry(output_file)

        print("Converting to fixed length strings")
        convert_to_fixed_length_strings()

        print("Link logs to where Mantid can find them")
        link_logs()

    print("Running h5repack")
    name, extension = os.path.splitext(output_filename)
    repacked_filename = f"{name}_agg_with_monitor.nxs"
    subprocess.run(
        [
            os.path.join(args.format_convert, "h5repack"),
            output_filename,
            repacked_filename,
        ]
    )

    print("Deleting intermediate file")
    os.remove(output_filename)

    # Run h5format_convert on each file to improve compatibility with HDF5 1.8.x used by Mantid
    print("Running h5format_convert")
    subprocess.run(
        [os.path.join(args.format_convert, "h5format_convert"), repacked_filename]
    )

    pl.show()

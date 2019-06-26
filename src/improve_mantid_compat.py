import h5py
import argparse
import numpy as np
import attr
import os
from os.path import isfile, join
import subprocess
from shutil import copyfile
from pulse_aggregator import (
    aggregate_events_by_pulse,
    remove_data_not_used_by_mantid,
    patch_geometry,
)
import matplotlib.pylab as pl
from multiprocessing import Process


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


def convert_to_fixed_length_strings(output_file):
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


def link_logs(output_file):
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


def strip_to_essentials(fin="", fout="", pattern="waveform_data", chatty=False):
    ignores = pattern.replace(" ", "").split(',')
    input_file = h5py.File(fin, 'r')
    output_file = h5py.File(fout, 'w')

    console_output = ""
    if chatty:
        console_output += "Stripping out all entries that include:\n"
        for s in ignores:
            console_output += "{}\n".format(s)

    for a in input_file.attrs:
        output_file.attrs[a] = input_file.attrs[a]

    groups = []
    input_file.visit(groups.append)

    skipped = []
    for g in groups:
        to_be_copied = True
        for i in ignores:
            if g.count(i) > 0:
                to_be_copied = False
                break
        # Also remove groups without NX_class
        no_nx_class = False
        if isinstance(input_file[g], h5py.Group):
            if "NX_class" not in input_file[g].attrs.keys():
                no_nx_class = True
        if to_be_copied:
            if chatty:
                console_output += "Copying entry: {}\n".format(g)
            if type(input_file[g]) == h5py.highlevel.Dataset:
                indx = g.rfind("/")
                if indx < 0:
                    input_file.copy(g, output_file)
                else:
                    input_file.copy(g, output_file[g[:indx]])
            else:
                newGroup = output_file.create_group(g)
                for a in input_file[g].attrs:
                    newGroup.attrs[a] = input_file[g].attrs[a]
            if no_nx_class:
                add_nx_class_to_group(output_file[g], "NXlog")
        else:
            skipped.append(g)

    if chatty:
        console_output += "===================================\n"
        console_output += "The following entries were skipped:\n"
        for s in skipped:
            console_output += "{}\n".format(s)

    input_file.close()
    output_file.close()

    return console_output


def process_file(filelist, args, cpuid):

    for filename in filelist:

        console_output = "#################### CPU: {} ####################\n".format(cpuid)
        console_output += "Processing file: {}\n".format(filename)

        # First run pulse aggregation
        name, extension = os.path.splitext(filename)
        output_filename = f"{name}.nxs"
        console_output += "Copying input file\n"
        console_output += strip_to_essentials(fin=filename, fout=output_filename,
                            pattern="waveforms")
        with h5py.File(output_filename, "r+") as output_file:
            # DENEX detector
            console_output += "Aggregating DENEX detector events\n"
            console_output += aggregate_events_by_pulse(
                output_file,
                args.chopper_tdc_path,
                "/entry/instrument/detector_1/raw_event_data",
                args.tdc_pulse_time_difference,
            )

            # Monitor
            console_output += "Aggregating monitor events\n"
            console_output += aggregate_events_by_pulse(
                output_file,
                args.chopper_tdc_path,
                "/entry/monitor_1/events",
                args.tdc_pulse_time_difference,
                output_group_name="monitor_event_data",
                event_id_override=262144,
            )

            console_output += "Adding missing NX_class attributes\n"
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

            console_output += "Patching geometry\n"
            patch_geometry(output_file)

            console_output += "Converting to fixed length strings\n"
            convert_to_fixed_length_strings(output_file)

            console_output += "Link logs to where Mantid can find them\n"
            link_logs(output_file)

        # Run h5format_convert on each file to improve compatibility with HDF5 1.8.x used by Mantid
        console_output += "Running h5format_convert\n"
        subprocess.run(
            [os.path.join(args.format_convert, "h5format_convert"), output_filename]
        )

        print(console_output)
        # pl.show()

    return


#==============================================================================


if __name__ == "__main__":

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
    parser.add_argument(
        "-n",
        "--ncpus",
        type=int,
        help="Number of cpus to use for parallel processing of multiple files",
        default=1,
    )
    args = parser.parse_args()

    # Get all files in current directory
    all_files = [
        join(args.input_directory, f)
        for f in os.listdir(args.input_directory)
        if isfile(join(args.input_directory, f))
    ]

    # Clean file list
    filenames = []
    for f in all_files:
        name, extension = os.path.splitext(f)
        if extension != ".hdf":
            continue
        onlypath, onlyname = os.path.split(f)
        if args.only_this_file and onlyname != args.only_this_file:
            continue
        filenames.append(f)

    # Now process each file from the clean list =========================

    # Prepare file list for each process
    # Try to split the workload according to the file sizes
    filesizes = []
    for f in filenames:
        filesizes.append(os.path.getsize(f))

    # Containers for file names and total workload volume
    file_list_per_cpu = []
    for i in range(args.ncpus):
        file_list_per_cpu.append([])
    bytes_per_cpu = np.zeros([args.ncpus], dtype=np.int64)

    # Go through each file in the list
    for i, f in enumerate(filenames):
        # Find the cpu that currently holds the least bytes
        icpu = np.argmin(bytes_per_cpu)
        # Append the current file to its list and update its workload
        file_list_per_cpu[icpu].append(f)
        bytes_per_cpu[icpu] += filesizes[i]

    # Prepare array to hold processes
    processes = []
    for ip in range(args.ncpus):
        processes.append(Process(target=process_file, args=(file_list_per_cpu[ip], args, ip)))
    # Start and join processes
    for p in processes:
        p.start()
    for p in processes:
        p.join()

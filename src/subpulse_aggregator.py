import h5py
import numpy as np
import argparse
from shutil import copyfile
from tqdm import tqdm
from matplotlib import pyplot as pl
from utils import delete_path_from_nexus

# Uncomment for nicer styling on plots
# import seaborn as sns
# sns.set()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-i", "--input-filename", type=str, help="Input file to convert.", required=True
)
parser.add_argument("-o", "--output-filename", type=str, help="Output filename.")
parser.add_argument(
    "--in-place",
    action="store_true",
    help="Writes output data into the input file instead of creating a new output file",
)
parser.add_argument(
    "--tdc-pulse-time-difference",
    type=int,
    help="Time difference between TDC timestamps and pulse T0 in integer nanoseconds",
    default=0,
)
parser.add_argument(
    "--raw-event-path",
    type=str,
    help="Path to the raw event NXevent_data group in the file",
    default="/entry/instrument/detector_1/raw_event_data",
)
parser.add_argument(
    "--output-event-path",
    type=str,
    help="Path to the group where the output should be written",
    default="/entry/event_data",
)
parser.add_argument(
    "--chopper-tdc-path",
    type=str,
    help="Path to the chopper TDC unix timestamps (ns) dataset in the file",
    default="/entry/instrument/chopper_1/top_dead_center/time",
)
parser.add_argument(
    "--wfm-chopper-tdc-path",
    type=str,
    help="Path to the chopper TDC unix timestamps (ns) dataset in the file",
    default="/entry/instrument/chopper_3/top_dead_center/time",
)
parser.add_argument(
    "--wfm-2-chopper-tdc-path",
    type=str,
    help="Path to the chopper TDC unix timestamps (ns) dataset in the file",
    default="/entry/instrument/chopper_4/top_dead_center/time",
)
args = parser.parse_args()


def position_to_index(pos, count):
    uint_max = 2 ** 16 - 1
    # What if count (Nx or Ny) does not divide uint_max?
    return np.floor_divide(pos, (uint_max // count))


def convert_id(detector_id, id_offset=0):
    Nx = 512

    x = np.bitwise_and(detector_id[:], 0xFFFF)
    y = np.right_shift(detector_id[:], 16)

    # Hist, XEdge, YEdge = np.histogram2d(x, y, bins=(100, 100))
    # fig = pl.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(Hist)
    # pl.show()

    # Mantid requires 32 bit unsigned, so this should be correct dtype already.
    # Need offset here unless the banks event ids start at zero (Mantid
    # will simply discard events that do not correspond to IDF).
    # return id_offset + position_to_index(x, Nx) + Nx * position_to_index(y, Ny)
    return id_offset + x + (Nx * y)


def write_event_data(
    output_data_group,
    event_ids,
    event_index_output,
    event_offset_output,
    event_time_zero_output,
):
    output_data_group.create_dataset(
        "event_id", data=event_ids, compression="gzip", compression_opts=1
    )
    event_time_zero_ds = output_data_group.create_dataset(
        "event_time_zero",
        data=event_time_zero_output,
        compression="gzip",
        compression_opts=1,
    )
    event_time_zero_ds.attrs.create("units", np.array("ns").astype("|S2"))
    event_time_zero_ds.attrs.create(
        "offset", np.array("1970-01-01T00:00:00").astype("|S19")
    )
    output_data_group.create_dataset(
        "event_index",
        data=event_index_output.astype(np.uint64),
        compression="gzip",
        compression_opts=1,
    )
    event_offset_ds = output_data_group.create_dataset(
        "event_time_offset",
        data=event_offset_output,
        compression="gzip",
        compression_opts=1,
        dtype=np.uint32,
    )
    event_offset_ds.attrs.create("units", np.array("ns").astype("|S2"))


def truncate_to_chopper_time_range(
    chopper_times, event_id_array, event_times, wfm_chopper_times, wfm_2_chopper_times
):
    # Chopper timestamps are our pulse timestamps, we can only aggregate events per pulse
    # for time periods in which we actually have chopper timestamps
    # truncate any other events
    start = np.searchsorted(event_times, chopper_times[0], "left")
    end = np.searchsorted(event_times, chopper_times[-2], "right")
    event_id_array = event_id_array[start:end]
    event_times = event_times[start:end]

    # We need wfm chopper times up to the following chopper_time
    wfm_chopper_times = wfm_chopper_times[
        np.searchsorted(wfm_chopper_times, chopper_times[0], "left") : np.searchsorted(
            wfm_chopper_times, chopper_times[-1], "right"
        )
    ]
    wfm_2_chopper_times = wfm_2_chopper_times[
        np.searchsorted(
            wfm_2_chopper_times, chopper_times[0], "left"
        ) : np.searchsorted(wfm_2_chopper_times, chopper_times[-1], "right")
    ]

    chopper_times = chopper_times[:-2]

    return (
        chopper_times,
        event_id_array,
        event_times,
        wfm_chopper_times,
        wfm_2_chopper_times,
    )


def _wfm_psc_1():
    """"
    Definition V20 for wfm pulse shaping chopper 1 (closest to source)
    :return: Returns the sorted angles of all edges in degrees. First entry is start angle of the first cut-out
    second entry is end angle of first cut-out. Cut-outs are in order from the position that the top-dead-centre (TDC)
    timestamp is recorded. The values in the array are from the closing edge of the largest window, TDC position is 15
    degrees after this.
    """
    return (
        np.array(
            [
                83.71,
                94.7,
                140.49,
                155.79,
                193.26,
                212.56,
                242.32,
                265.33,
                287.91,
                314.37,
                330.3,
                360.0,
            ]
        )
        + 15.0
    )


def _wfm_psc_2():
    """
    Definition V20 for wfm pulse shaping chopper 2 (closest to sample)
    :return: Returns the sorted angles of all edges in degrees. First entry is start angle of the first cut-out
    second entry is end angle of first cut-out. Cut-outs are in order from the position that the top-dead-centre (TDC)
    timestamp is recorded.
    """
    return (
        np.array(
            [
                65.04,
                76.03,
                126.1,
                141.4,
                182.88,
                202.18,
                235.67,
                254.97,
                284.73,
                307.74,
                330.00,
                360.0,
            ]
        )
        + 15.0
    )


def _tof_shifts(pscdata, psc_frequency=0.0):
    """
    This is the time shift from the WFM chopper top-dead-centre timestamp to the t0 of each sub-pulse
    """
    cut_out_centre = np.reshape(pscdata, (len(pscdata) // 2, 2)).mean(1)
    tof_shifts = cut_out_centre / (360.0 * psc_frequency)
    # TODO What about the 17.1 degree phase shift from the chopper signal,
    #  which Peter mentioned, do we need to apply that here?
    return tof_shifts


def _which_subpulse(time_after_source_tdc, threshold):
    return np.searchsorted(threshold, time_after_source_tdc, "left")


def plot_histograms(offset_from_source_chopper, offset_from_wfm_windows, threshold):
    fig, (ax1, ax2) = pl.subplots(2, 1)
    ax1.hist(offset_from_source_chopper, bins=2 * 144, range=(0, 72000000))
    for value in threshold:
        ax1.axvline(x=value, color="r", linestyle="dashed", linewidth=1)
    ax1.set_title(
        "Time offset from source chopper TDC timestamp, vertical lines indicate thresholds for subpulses"
    )
    ax1.set_xlabel("Time (nanoseconds)")
    ax1.set_ylabel("Counts")
    ax2.hist(offset_from_wfm_windows, bins=2 * 144, range=(0, 72000000))
    ax2.set_title(
        "Time offset from WFM chopper TDC timestamp, adjusted for the window timing for each subpulse"
    )
    ax2.set_xlabel("Time (nanoseconds)")
    ax2.set_ylabel("Counts")
    # Leave a little more whitespace between the plots, so that the labels don't collide
    pl.subplots_adjust(hspace=0.3)


def aggregate_events_by_subpulse(
    out_file,
    optargs,
    input_group_path,
    output_group_name="event_data",
    event_id_override=None,
):
    # Nasty hardcoded thresholds for subpulses
    # TODO calculate these from beamline geometry
    component_name = input_group_path.split("/")[-2]
    if component_name == "monitor_1":
        threshold = np.array(
            [23500000, 32800000, 40500000, 48000000, 55000000], dtype=int
        )
    else:
        threshold = np.array(
            [28000000, 39000000, 48000000, 57000000, 65600000], dtype=int
        )
    relative_shifts = (
        _tof_shifts(_wfm_psc_1(), psc_frequency=70.0)
        + _tof_shifts(_wfm_psc_2(), psc_frequency=70.0)
    ) * 5.0e08  # factor of 0.5 * 1.0e9 (taking mean and converting to nanoseconds)
    relative_shifts = relative_shifts.astype(np.uint64)
    # Create output event group
    output_data_group = out_file["/entry"].create_group(output_group_name)
    output_data_group.attrs.create("NX_class", "NXevent_data", None, dtype="<S12")
    # Shift the TDC times
    source_tdc_times = out_file[optargs.chopper_tdc_path][...]
    source_tdc_times += optargs.tdc_pulse_time_difference
    wfm_tdc_times = out_file[optargs.wfm_chopper_tdc_path][...]
    wfm_2_tdc_times = out_file[optargs.wfm_2_chopper_tdc_path][...]
    event_ids = out_file[input_group_path + "/event_id"][...]
    event_ids = convert_id(event_ids)
    event_time_zero_input = out_file[input_group_path + "/event_time_zero"][...]
    source_tdc_times, event_ids, event_time_zero_input, wfm_tdc_times, wfm_2_tdc_times = truncate_to_chopper_time_range(
        source_tdc_times,
        event_ids,
        event_time_zero_input,
        wfm_tdc_times,
        wfm_2_tdc_times,
    )
    source_tdc_index = 0
    wfm_tdc_index = 0
    wfm_2_tdc_index = 0  # for the second WFM chopper
    event_index = 0
    event_offset_output = np.zeros_like(event_time_zero_input, dtype=np.uint32)
    event_id_output = np.zeros_like(event_time_zero_input, dtype=np.uint32)
    offset_from_source_chopper_tdc = np.zeros_like(event_time_zero_input)
    event_index_output = np.zeros((wfm_tdc_times.size * 6 + 1,), dtype=np.uint64)
    event_time_zero_output = np.zeros((wfm_tdc_times.size * 6,), dtype=np.uint64)
    subpulse_uuid = (0, 0)
    subpulse_count = 0
    print("Aggregating events by subpulse...", flush=True)
    for event_input_number, (event_wallclock_time, event_id) in enumerate(
        tqdm(zip(event_time_zero_input, event_ids), total=len(event_time_zero_input))
    ):
        # Find relevant source chopper timestamp
        tdc_index = (
            np.searchsorted(
                source_tdc_times[source_tdc_index:], event_wallclock_time, "right"
            )
            - 1
            + source_tdc_index
        )
        source_tdc = source_tdc_times[tdc_index]

        # Find relevant WFM chopper timestamps
        wfm_tdc_index = (
            np.searchsorted(wfm_tdc_times[wfm_tdc_index:], source_tdc, "right")
            + wfm_tdc_index
        )
        wfm_tdc = wfm_tdc_times[wfm_tdc_index]
        wfm_2_tdc_index = (
            np.searchsorted(wfm_2_tdc_times[wfm_2_tdc_index:], source_tdc, "right")
            + wfm_2_tdc_index
        )
        wfm_2_tdc = wfm_2_tdc_times[wfm_2_tdc_index]
        wfm_tdc_mean = np.uint64((wfm_tdc + wfm_2_tdc) // 2)

        # Determine which subpulse the event is in
        offset_from_source_chopper_tdc[event_input_number] = (
            event_wallclock_time - source_tdc
        )
        subpulse_index = _which_subpulse(
            offset_from_source_chopper_tdc[event_input_number], threshold
        )
        t0 = wfm_tdc_mean + relative_shifts[subpulse_index]

        if t0 > event_wallclock_time:
            # Throw this event away
            # TODO does this ever happen? (this check costs ~10000 events per second in processing time)
            continue

        event_offset_output[event_index] = event_wallclock_time - t0
        event_id_output[event_index] = event_id

        next_subpulse_uuid = (wfm_tdc_index, subpulse_index)
        if next_subpulse_uuid == subpulse_uuid:
            # This event is from the same subpulse as the previous one
            event_index_output[subpulse_count] = event_index
        else:
            # Append a new subpulse
            # + 1 to event_index as it indicates the start of the next pulse, not end of current one
            event_index_output[subpulse_count + 1] = event_index + 1
            event_time_zero_output[subpulse_count] = np.uint64(t0)
            subpulse_count += 1

        subpulse_uuid = next_subpulse_uuid
        event_index += 1
    # Truncate space from arrays which wasn't needed due to bad events
    event_offset_output = event_offset_output[:event_index]
    event_id_output = event_id_output[:event_index]
    offset_from_source_chopper_tdc = offset_from_source_chopper_tdc[:event_index]

    # Truncate subpulse arrays which may have been preallocated too large
    event_index_output = event_index_output[:subpulse_count]
    event_time_zero_output = event_time_zero_output[:subpulse_count]

    plot_histograms(offset_from_source_chopper_tdc, event_offset_output, threshold)

    if event_id_override is not None:
        event_id_output = event_id_override * np.ones_like(event_id_output)
    else:
        event_id_output[event_id_output > 262143] = 262143
    write_event_data(
        output_data_group,
        event_id_output,
        event_index_output,
        event_offset_output,
        event_time_zero_output,
    )


def remove_data_not_used_by_mantid(out_file):
    global groups_to_remove
    # Delete waveform groups (not read by Mantid)
    for channel in range(3):
        group_name = f"/entry/instrument/detector_1/waveforms_channel_{channel}"
        delete_path_from_nexus(out_file, group_name)
    groups_to_remove = []

    def remove_groups_without_nxclass(name, object):
        if isinstance(object, h5py.Group):
            if "NX_class" not in object.attrs.keys():
                groups_to_remove.append(name)

    out_file.visititems(remove_groups_without_nxclass)
    for group in reversed(groups_to_remove):
        print(group)
        delete_path_from_nexus(out_file, group)


def patch_geometry(out_file):
    pixels_per_axis = 512
    pixel_ids = np.arange(0, pixels_per_axis ** 2, 1, dtype=int)
    pixel_ids = np.reshape(pixel_ids, (pixels_per_axis, pixels_per_axis))
    delete_path_from_nexus(out_file, "entry/instrument/detector_1/detector_number")
    out_file["entry/instrument/detector_1/"].create_dataset(
        "detector_number", pixel_ids.shape, dtype=np.int64, data=pixel_ids
    )
    neutron_sensitive_width = 0.28  # metres, from DENEX data sheet
    # This pixel size is approximate, in practice the EFU configuration/calibration affects both the division
    # into 512 pixels and the actual active width we see of the detector
    # I suspect the actually detector area we collect data from is smaller than 0.28x0.28
    pixel_size = neutron_sensitive_width / pixels_per_axis
    single_axis_offsets = (
        (pixel_size * np.arange(0, pixels_per_axis, 1, dtype=np.float))
        - (neutron_sensitive_width / 2.0)
        + (pixel_size / 2.0)
    )
    x_offsets, y_offsets = np.meshgrid(single_axis_offsets, single_axis_offsets)
    delete_path_from_nexus(out_file, "entry/instrument/detector_1/x_pixel_offset")
    delete_path_from_nexus(out_file, "entry/instrument/detector_1/y_pixel_offset")
    out_file["entry/instrument/detector_1/"].create_dataset(
        "x_pixel_offset", x_offsets.shape, dtype=np.float64, data=x_offsets
    )
    out_file["entry/instrument/detector_1/"].create_dataset(
        "y_pixel_offset", y_offsets.shape, dtype=np.float64, data=y_offsets
    )
    delete_path_from_nexus(out_file, "entry/monitor_1/waveforms")
    delete_path_from_nexus(out_file, "entry/instrument/detector_1/waveforms_channel_3")
    delete_path_from_nexus(out_file, "entry/instrument/linear_axis_1")
    delete_path_from_nexus(out_file, "entry/instrument/linear_axis_2")
    delete_path_from_nexus(out_file, "entry/sample/transformations/offset_stage_1_to_default_sample")
    delete_path_from_nexus(out_file, "entry/sample/transformations/offset_stage_2_to_sample")
    delete_path_from_nexus(out_file, "entry/sample/transformations/offset_stage_2_to_stage_1")
    # Correct the source position, to be location of WFM choppers
    out_file["entry/instrument/source/transformations/location"][...] = 20.55
    # Correct detector_1 position and orientation
    delete_path_from_nexus(out_file, "entry/instrument/detector_1/depends_on")
    depend_on_path = "/entry/instrument/detector_1/transformations/x_offset"
    out_file["entry/instrument/detector_1"].create_dataset(
        "depends_on",
        data=np.array(depend_on_path).astype("|S" + str(len(depend_on_path))),
    )
    delete_path_from_nexus(out_file, "entry/instrument/detector_1/transformations/orientation")
    location_path = "entry/instrument/detector_1/transformations/location"
    out_file[location_path][...] = 3.5
    out_file[location_path].attrs["vector"] = [0.0, 0.0, 1.0]
    out_file[location_path].attrs["depends_on"] = "."
    delete_path_from_nexus(out_file, "entry/instrument/detector_1/transformations/beam_direction_offset")
    x_offset_dataset = out_file[
        "entry/instrument/detector_1/transformations"
    ].create_dataset("x_offset", (1,), dtype=np.float64, data=0.065)
    x_offset_dataset.attrs.create("units", np.array("m").astype("|S1"))
    translation_label = "translation"
    x_offset_dataset.attrs.create(
        "transformation_type",
        np.array(translation_label).astype("|S" + str(len(translation_label))),
    )
    x_offset_dataset.attrs.create(
        "depends_on",
        np.array("/" + location_path).astype("|S" + str(len(location_path) + 1)),
    )
    x_offset_dataset.attrs.create("vector", [-1.0, 0.0, 0.0])
    # Correct monitor position and id
    out_file["/entry/monitor_1/transformations/transformation"][...] = -1.8
    out_file["/entry/monitor_1/detector_id"][...] = 262144
    # Link monitor in the instrument group so that Mantid finds it
    out_file["/entry/instrument/monitor_1"] = out_file["/entry/monitor_1"]
    # Link monitor event datasets to monitor in instrument group (for Mantid)
    out_file["/entry/instrument/monitor_1/event_id"] = out_file[
        "/entry/monitor_event_data/event_id"
    ]
    out_file["/entry/instrument/monitor_1/event_index"] = out_file[
        "/entry/monitor_event_data/event_index"
    ]
    out_file["/entry/instrument/monitor_1/event_time_offset"] = out_file[
        "/entry/monitor_event_data/event_time_offset"
    ]
    out_file["/entry/instrument/monitor_1/event_time_zero"] = out_file[
        "/entry/monitor_event_data/event_time_zero"
    ]
    out_file["/entry/instrument/monitor_1/monitor_number"] = out_file[
        "/entry/instrument/monitor_1/detector_id"
    ]


if __name__ == "__main__":
    if not args.in_place:
        print("Copying input file contents to output file...")
        copyfile(args.input_filename, args.output_filename)
        print("...done copying.")
        output_file = args.output_filename
    else:
        output_file = args.input_filename

    with h5py.File(output_file, "r+") as raw_file:
        # DENEX detector
        aggregate_events_by_subpulse(raw_file, args, args.raw_event_path)

        # Monitor
        aggregate_events_by_subpulse(
            raw_file,
            args,
            "/entry/monitor_1/events",
            "monitor_event_data",
            event_id_override=262144,
        )

        remove_data_not_used_by_mantid(raw_file)
        patch_geometry(raw_file)

        pl.show()

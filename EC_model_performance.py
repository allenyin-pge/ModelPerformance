import pandas as pd
from typing import Tuple, Optional, List, Dict, NamedTuple, Union
import logging
import numpy as np
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from EC_corrosion_stats import (
    filter_identifications,
    filter_manufacturing_anomalies,
    get_external_anomaly_row_mask,
)
from dataclasses import dataclass
from scipy import stats

"""
Shared functions to measuring risk model performance from spatial-joined
ILI data and EC_LOF_Risk table outputs

Variable namings specific to keeping track of the steps
"""

# Field names that we really care about:
FAILURE_PRESSURE_FIELDS = ['Estimated Failure Pressure (Pf)', 'Pf/MAOP', 'Pf* (with tool tolerances)', 'Pf*/MAOP']
# S_EC ranges from 0 to 10 -- linear scale
# af_EC ranges from 0.01 to 100 -- ln(af_EC) is linear scale
CALCULATED_RISK_FIELDS =  ['EC_LOF_Leak', 'EC_LOF_Rupture', 'S_EC', 'af_EC']

# Fields that are added through augment_dataframe_inplace
# risk fields have a single value per pipe segment
# metric fields are from ILI and need to be aggregated over a pipe segment
AUGMENTED_RISK_FIELDS = ["ln_EC_LOF_Leak", "ln_EC_LOF_Rupture", "ln_af_EC"]
AUGMENTED_METRIC_FIELDS = ["-Depth (%)"]

# Also get the stationing number, so we are not simply comparing individual anomaly to pipe values
# Note that `beginstationseriesid` is unique, meaning that one route can have multiple different `beginstationseriesid`.
LOCATION_FIELDS = ["route", "beginstationseriesid", "beginstationnum", "endstationseriesid", "endstationnum"]
STATIONING_VARS = ["route", "beginstationseriesid", "beginstationnum", "endstationnum"]

# We als want to later compare the risk values to volumetric loss, so keep track of those fields as well
VOLUMETRIC_LOSS_FIELDS = ["Identification", "Internal", "Depth (%)", "WT (in)", "OD (in)", "Width (in)", "Length (in)"]


# =============== Data loading and Data cleaning ==============

def load_source_files(
    dir_for_year: str,
    cleaned_ILI_csv_fname: str,
    ILI_ECLOF_Pipesegment_joined_fname: str,
    EC_Risk_LOF_csv_fname: Optional[str] = None,
) -> Union[
    Tuple[List[str], pd.DataFrame],
    Tuple[List[str], pd.DataFrame, pd.DataFrame]
]:
    """
    Read in the spatially-joined csv file and get the
    proper ILI headers.

    Args:
        dir_for_year: Directory where the data for this year can be found.
            If we are measuring model performance for year 2022, the directory
            should contain the joined data with 2022's risk results.
        cleaned_ILI_csv_fname: Cleaned ILI csv file used to form the spatial join,
            <dir_for_year>/<cleaned_ILI_csv_fname> specifies the file location.
        ILI_ECLOF_Pipesegment_joined_fname: This csv file produced from spatial-joining
            EC_LOF tables (spatialized to pipesegment) to ILI data (spatialized to pipesegment)
        EC_Risk_LOF_csv_name: csv for corresponding year's EC_LOF_Risk table.

    Returns:
        ILI_headers: The column headers for cleaned ILI data. The same columns are present
            in `ILI_ECLOF_Pipesegment_joined` but with the str values cut off.
        master_dataset: This is the panda dataframe loaded from:
            <dir_for_year>/<ILI_ECLOF_Pipesegment_joined>
        EC_Risk_LOF_table: This is the panda dataframe loaded from:
            <dir_for_year>/<EC_LOF_Risk_csv_fname>, if given
    """
    ILI_file_name = fr'{dir_for_year}\{cleaned_ILI_csv_fname}'
    cleaned_ILI_data = pd.read_csv(ILI_file_name, low_memory=False, encoding='unicode_escape')

    print(f"ILI data has {len(cleaned_ILI_data)} rows")
    ILI_headers = list(cleaned_ILI_data.columns)
    ILI_headers = [s.strip() for s in ILI_headers]
    print(f"ILI_headers:\n{ILI_headers}")

    ILI_ECLOF_Pipesegment_joined_fname = fr'{dir_for_year}\{ILI_ECLOF_Pipesegment_joined_fname}'
    master_dataset = pd.read_csv(ILI_ECLOF_Pipesegment_joined_fname, low_memory=False)
    print(f"Spatial-joined master dataset columns: {master_dataset.columns.values}")

    # In master_dataset, columns starting at the occurence of ["Vendor", "Source", "Route"...]
    # are ILI column contents of the spatial join. Find where this is happening and
    # update the column header strings with ILI_headers.
    ILI_start = np.flatnonzero(master_dataset.columns.values == "Vendor")[0]
    assert master_dataset.columns.values[ILI_start + 1] == "Source", \
        "Expected 'Source' column after 'Vendor' column!"
    master_dataset.columns.values[ILI_start: ILI_start + len(ILI_headers)] = ILI_headers
    print(f"Cleaned Spatial-joined master dataset columns: {master_dataset.columns.values}")

    if EC_Risk_LOF_csv_fname is not None:
        EC_Risk_LOF_csv_fname = fr'{dir_for_year}\{EC_Risk_LOF_csv_fname}'
        EC_Risk_LOF_table = pd.read_csv(EC_Risk_LOF_csv_fname, low_memory=False, encoding='unicode_escape')
        return ILI_headers, master_dataset, EC_Risk_LOF_table
    else:
        return ILI_headers, master_dataset



def standardize_column_names_inplace(master_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Update column names in place -- mostly column names from the EC LOF table side.
    They get distorted after spatial join
    """
    master_dataset.rename(
        columns={
            "EC_LOF_Rup": "EC_LOF_Rupture",
            "EC_LOF_Lea": "EC_LOF_Leak",
        },
        inplace=True
    )

    master_dataset.rename(
        columns={
            "Route": "route",
            "beginstati": "beginstationseriesid",
            "beginsta_1": "beginstationnum",
            "endstation": "endstationseriesid",
            "endstati_1": "endstationnum",
        },
        inplace=True
    )
    return master_dataset

def convert_numeric_cols(master_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Sometimes columns are not numeric, convert the ones
    we care about
    """
    # Clean out complete white spaces..
    master_dataset = master_dataset.replace(r'^\s*$', np.nan, regex=True)
    numeric_fields = (
        FAILURE_PRESSURE_FIELDS
        + CALCULATED_RISK_FIELDS
        + VOLUMETRIC_LOSS_FIELDS[2:]
    )
    for nf in numeric_fields:
        master_dataset[nf] = pd.to_numeric(master_dataset[nf], errors="raise")
    return master_dataset

def sanity_check_ILI_mileage_from_stationing(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, float]:
    """
    Function to sanity check ILI totaly mileage.
    1. Satvinder's dashboard (https://shorturl.at/ajqvG) records down the number
      of miles of ILI done per year. This is calculated from mile-point (MP) markers.
    2. From the spatial-joined ILI and EC_LOF table, we can calculate the total
      number of ILI miles from adding all the unique segments defined by the
      combination of [beginstationseriesid, beginstationnum, endstationnum]
    3. This second quantity should be less or equal to the ILI mileage calculated
        from MP markers.
    """
    unique_segments = df[[
        "beginstationseriesid",
        "beginstationnum",
        "endstationnum",
        "endstationseriesid"
    ]].drop_duplicates().dropna()
    assert (
        unique_segments["beginstationseriesid"] == unique_segments["endstationseriesid"]
    ).all(), "begin and end station series id should be same"
    ILI_mileage = (unique_segments["endstationnum"] - unique_segments["beginstationnum"]).sum() / 5280
    print(f"Found {ILI_mileage:.3f} miles of ILI data from stationing")
    return unique_segments, ILI_mileage


def check_null_stationing(
    master_dataset: pd.DataFrame
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, float]]:
    """
    `sanity_check_ILI_mileage_from_stationing` assertion might fail sometimes
    due to rows with no stationing variables.

    This is when ILI and EC_LOF table don't overlap.
    Usually this are rows where the longitude/latitude in ILI data is invalid
    (i.e. all 0's)

    This function checks this, and returns a new df without the null stationing
    rows.

    Returns:
        (Optionally)
        - df_no_null_stationing
        - unique_segments: dataframe of unique segments defined by stationing
        - ILI_mileage: calculated from no null stationing data
    """
    routes_with_null_stationing = master_dataset[master_dataset["beginstationseriesid"].isna()]["route"].unique()
    print(f"Found {len(routes_with_null_stationing)} routes with null stationing:\n{routes_with_null_stationing}")

    null_stationing = master_dataset[LOCATION_FIELDS].isnull()
    if null_stationing.any().any():
        print(f"Rows with location fields with null values:\n----\n{null_stationing.sum()}")
        print(f"Droppiong these rows")
        master_dataset_no_null_stationing = master_dataset.dropna(subset=LOCATION_FIELDS)
        # Now run ILI mileage check again from stationing
        unique_segments, ILI_mileage = sanity_check_ILI_mileage_from_stationing(master_dataset_no_null_stationing)
        return master_dataset_no_null_stationing, unique_segments, ILI_mileage
    
    # otherwise return nothing
    return None, None, None


def calculate_ILI_mileage_from_dataset_MP(df: pd.DataFrame) -> float:
    """
    Calculate total mileage from looking at max MP2 and min MP1 grouped by route.
    This is still <= actual ILI mileage from Satvinder's dashboard. This is because
    there can be dynamic pipe segments where no ILI anomaly is found.
    """
    unique_MP_pairs = df[["route", "MP1", "MP2"]].drop_duplicates()
    unique_MP_pairs["diff_MP"] = unique_MP_pairs["MP2"] - unique_MP_pairs["MP1"]
    MP_mileage = unique_MP_pairs['diff_MP'].sum()
    print(f"ILI mileage calculated from MP markers = {MP_mileage:.3f} miles")
    
    # print(f"MP and identification by Routes:") -- run to investigate
    # # Group by route and look at the data here...mileage mismatch
    # group_by_route = df_no_null_stationing.groupby(["route"])
    # for name, group in group_by_route:
    #     display(group[LOCATION_FIELDS + ["MP1", "MP2", "Identification"]])

    return MP_mileage


def sanitize_relevant_fields_inplace(
    master_dataset_no_null_stationing: pd.DataFrame,
    auto_drop: bool = True,
) -> pd.DataFrame:
    """
    Do additioinal data cleaning of the spatial joined dataset.
    Return the final cleaned dataset ready for plotting and computation

    This step will change the input dataset in place as well.

    Args:
        auto_drop: If True, will automatically drop rows with 0 failure pressure.
            If the printed output don't look right, set this to False and
            run the function again, and the filtered performance-related 
            dataframe will be returned, without dropping 0 failure pressure rows.
    """

    # Get the data frame with only the relevant fields
    relevant_fields = (
        FAILURE_PRESSURE_FIELDS
        + CALCULATED_RISK_FIELDS
        + LOCATION_FIELDS
        + VOLUMETRIC_LOSS_FIELDS
    )

    # Failure pressures can't be 0. If they are 0, that's probably because
    # when ArcMap imported csv files null values were set to 0.
    # Here we set them back.
    num_rows_with_0FP = master_dataset_no_null_stationing[
        FAILURE_PRESSURE_FIELDS
    ].isin([0]).any(axis=1).sum()
    total_rows = master_dataset_no_null_stationing.shape[0]
    perc_0FP = num_rows_with_0FP / total_rows * 100
    print(
        f"{num_rows_with_0FP}/{total_rows} rows with correct stationing"
        f" have 0 failure pressures ({perc_0FP:.3f}%). Fixing them"
    )
    master_dataset_no_null_stationing[
        FAILURE_PRESSURE_FIELDS
    ] = master_dataset_no_null_stationing[FAILURE_PRESSURE_FIELDS].replace(0, np.nan)
    performance_df = master_dataset_no_null_stationing[relevant_fields]

    # Rows for which failure pressure fields are NaN should have Identification
    # as "Girth Weld", "Flange", i.e. not really anomalies
    print(f"After fixing 0 failure pressure fields:\n {performance_df.head()}")

    # Now we filter out non-EC related rows
    # Messages will be printed out to break down number of 
    # manufacturing, non-manufacturing, and EC-related rows.
    _, filtered_performance_df = filter_identifications(performance_df)
    filtered_performance_df = filter_manufacturing_anomalies(filtered_performance_df)
    has_EC_row_mask = get_external_anomaly_row_mask(filtered_performance_df)
    num_EC_rows = has_EC_row_mask.sum()
    num_non_manufacturing_rows = has_EC_row_mask.shape[0]
    perc_EC_rows = num_EC_rows / num_non_manufacturing_rows * 100
    print(
        f"{num_EC_rows} / {num_non_manufacturing_rows} non-manufacturing rows"
        f" are EC-related ({perc_EC_rows:.3f}%)."
    )
    filtered_performance_df = filtered_performance_df.loc[has_EC_row_mask]

    # Now we check if the filtered rows still have null in failure pressure fields
    num_rows_with_null_FP = filtered_performance_df[
        FAILURE_PRESSURE_FIELDS
    ].isnull().any(axis=1).sum()
    if num_rows_with_null_FP:
        rows_with_null_FP = filtered_performance_df[filtered_performance_df["Pf/MAOP"].isnull()]
        print(
            "After filtering, rows with NA in failure pressure fields still exist:\n"
            f"{rows_with_null_FP.head()}"
        )
        if auto_drop:
            print(f"Dropping these rows")
            for field in FAILURE_PRESSURE_FIELDS:
                filtered_performance_df = filtered_performance_df[
                    filtered_performance_df[field].notna()
                ]
                
            for field in FAILURE_PRESSURE_FIELDS:
                print(filtered_performance_df[filtered_performance_df[field].isnull()])
    
    return filtered_performance_df


# ================== Computation ==========

def sanity_check_stationing(
    filtered_performance_df: pd.DataFrame
) -> Tuple[pd.DataFrame, int, int]:
    """
    Sanity check to make sure the spatial join process has properly assigned
    the ILI anomalies to the correct dynamic segment used in the risk model outputs.
    A dynamic segment is defined by the [beginstationseriesid, beginstationnum, endstationnum]
    combinations

    Returns:
        - unique_segments: DataFrame with the stationing variable values for the
            unique pipeline dynamic segments.
        - num_dynamic_segments_with_multiple_LOF_leak: int
        - num_dynamic_segments_with_multiple_LOF_rupture: int
    """
    # How many different `beginstationseriesid` are there?
    num_unique_beginstationseriesid = len(
        np.unique(filtered_performance_df["beginstationseriesid"])
    )
    num_unique_endstationseriesid = len(
        np.unique(filtered_performance_df["endstationseriesid"])
    )
    assert num_unique_beginstationseriesid == num_unique_endstationseriesid,\
        "We expect number of unique beginstationseriesid equal to that of endstationseriesid"

    # Are begin and end stationseries id the same?
    same_begin_and_end_stationseriesid = np.all(
        filtered_performance_df["beginstationseriesid"]
        == filtered_performance_df["endstationseriesid"]
    )
    assert same_begin_and_end_stationseriesid,\
        "We expect begin and end stationseriesid for each row to be the same"

    # Now get unique segments and segment statistics
    unique_segments = filtered_performance_df[STATIONING_VARS].drop_duplicates()
    print(f"ILI data is mapped to {len(unique_segments)} unique risk-model dynamic pipe segments")

    num_dynamic_segments_with_multiple_LOF_leak = np.sum(
        filtered_performance_df.groupby(STATIONING_VARS)["EC_LOF_Leak"].nunique() != 1
    )
    num_dynamic_segments_with_multiple_LOF_rupture = np.sum(
        filtered_performance_df.groupby(STATIONING_VARS)["EC_LOF_Rupture"].nunique() != 1
    )
    print(
        f"Number of dynamic segments with multiple LOF leak={num_dynamic_segments_with_multiple_LOF_leak}, "
        f"with multiple LOF rupture={num_dynamic_segments_with_multiple_LOF_rupture}"
    )
    assert (
        (num_dynamic_segments_with_multiple_LOF_leak == 0)
        and (num_dynamic_segments_with_multiple_LOF_rupture == 0)
    ), "Risk values for anomalies mapped to the same dynamic segment should be the same!"
    return (
        unique_segments,
        num_dynamic_segments_with_multiple_LOF_leak,
        num_dynamic_segments_with_multiple_LOF_rupture
    )

def augment_dataframe_inplace(
    filtered_performance_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add additional columns to make plotting easier:
    1. ln_EC_LOF_Leak -- nan if EC_LOF_Leak is 0
    2. ln_EC_LOF_Rupture -- nan if EC_LOF_Rupture is 0
    3. ln_af_EC -- nan if af_EC is 0
    4. -Depth (%) -- inverted, so now ranges from -1 to 0. Bigger value = smaller risk
    """
    filtered_performance_df["ln_EC_LOF_Leak"] = np.nan
    has_nonzero_LOF_Leak = (filtered_performance_df["EC_LOF_Leak"] > 0)
    filtered_performance_df["ln_EC_LOF_Leak"][has_nonzero_LOF_Leak] = np.log(
        filtered_performance_df["EC_LOF_Leak"][has_nonzero_LOF_Leak]
    )

    filtered_performance_df["ln_EC_LOF_Rupture"] = np.nan
    has_nonzero_LOF_Rupture = (filtered_performance_df["EC_LOF_Rupture"] > 0)
    filtered_performance_df["ln_EC_LOF_Rupture"][has_nonzero_LOF_Rupture] = np.log(
        filtered_performance_df["EC_LOF_Rupture"][has_nonzero_LOF_Rupture]
    )

    filtered_performance_df["ln_af_EC"] = np.nan
    has_nonzero_af_EC = (filtered_performance_df["af_EC"] > 0)
    filtered_performance_df["ln_af_EC"][has_nonzero_af_EC] = np.log(
        filtered_performance_df["af_EC"][has_nonzero_af_EC]
    )

    if "-Depth (%)" not in filtered_performance_df.columns:
        filtered_performance_df["-Depth (%)"] = -filtered_performance_df["Depth (%)"]

    return filtered_performance_df


def do_segment_level_aggregation(
    filtered_performance_df: pd.DataFrame,
    aggregation_functions: List[str] = ["mean", "min", "max"],
) -> Dict[str, pd.DataFrame]:
    """
    We want to compare each pipe segment's risk value with a single metric
    representing that pipe segment's ground-truth health value. This single
    metric is derived from the EC-related ILI columns for anomaly rows that
    fall within each pipe segment.

    To arrive at a single metric from multiple anomaly rows, different
    aggregation functions can be used. This function will return a dictionary with:
    - keys=name of the aggregation function,
    - values=aggregated metric data frame, one row per pipe segment.
    """
    dict_filtered_performance_df_aggregated = {}
    metric_fields = FAILURE_PRESSURE_FIELDS + AUGMENTED_METRIC_FIELDS
    risk_fields = CALCULATED_RISK_FIELDS + AUGMENTED_RISK_FIELDS
    for func_str in aggregation_functions:
        aggregation = {field: func_str for field in metric_fields}
        aggregation.update({field: "first" for field in risk_fields})
        dict_filtered_performance_df_aggregated[func_str] = filtered_performance_df.groupby(
            STATIONING_VARS
        ).agg(aggregation).reset_index()
    return dict_filtered_performance_df_aggregated


# =================== Select route types ===================
# These functions go through data frame containing valid "route" column values
# and return pd.Series that act as boolean mask to select rows of different route types.

BACK_BONE_IDENTIFIER = [
    "300A", "300B", "400", "401", "002", "057A", "057B", "057C", "107", "303"
]
BACK_BONE_REGEX_PATTERN = '|'.join(r"\b{}\b".format(x) for x in BACK_BONE_IDENTIFIER)

@dataclass
class RouteMasks:
    DFM: pd.Series
    DREG: pd.Series
    SP: pd.Series
    Xtie: pd.Series
    backbone: pd.Series
    numbered_line: pd.Series


def get_route_types(
    df: pd.DataFrame,
    combine_SP_backbone: bool=False,
) -> RouteMasks:
    is_DFM = df["route"].str.contains("DFM")
    is_DREG = df["route"].str.contains("DREG")
    is_SP = df["route"].str.contains("SP")
    is_Xtie = df["route"].str.contains("X")
    is_backbone = df["route"].str.contains(BACK_BONE_REGEX_PATTERN)
    is_numbered_line = (~is_DFM) & (~is_DREG) & (~is_SP) & (~is_Xtie) & (~is_backbone)

    print(f"DFM lines found: {df[is_DFM]['route'].unique()}")
    print(f"DREG lines found: {df[is_DREG]['route'].unique()}")
    print(f"SP lines found: {df[is_SP]['route'].unique()}")
    print(f"Xtie lines found: {df[is_Xtie]['route'].unique()}")
    print(f"Backbone lines found: {df[is_backbone]['route'].unique()}")
    print(f"Numbered lines found: {df[is_numbered_line]['route'].unique()}")
    if not combine_SP_backbone:
        return RouteMasks(
            DFM=is_DFM,
            DREG=is_DREG,
            SP=is_SP,  # Gordon says to merge SP and backbone lines together
            Xtie=is_Xtie,
            backbone=is_backbone,
            numbered_line=is_numbered_line,
        )
    else:
        return RouteMasks(
            DFM=is_DFM,
            DREG=is_DREG,
            SP=pd.Series(), # empty series
            Xtie=is_Xtie,
            backbone=is_backbone & is_SP,
            numbered_line=is_numbered_line,
        )


# =================== Interpolate segments functions ==========

"""
Not all pipe segments covered by ILI have EC-related anomalies.
If we only compare the risk model outputs vs. ILI-based metrics for segments
with EC-related anomalies, the measurement results can bias toward more
unhealthy segments.

To mitigate this, we do the following interpolation procedures to assign failure pressure or %-corrosion
values to dynamic segments without ILI anomaly entries:
- For each route available in the ILI data, find the minimum and maximum
  (beginstationseriesid, beginstationnum, endstationnum) range. Note that each segment can be
  uniquely identified by these combinations.
- Out of these segments, find how many segments does not have any ILI anomalies found.
- Assign "safe" values to these segments without any found ILI anomaly:
    - For negative-perc-depth lost, use top 95% value from the max-aggregation results. For this
        we assume "-Depth (%)" was aggregated after calculation rather than vice versa.
    - For Pf/MAOP and other failure pressure fields, use top 95% value from max-aggregation results.
"""

def make_interpolate_segments(
    EC_Risk_LOF_table: pd.DataFrame,
    unique_ILI_segments: pd.DataFrame,
    dict_df_aggregated: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Create interpolated segment dataframe from EC_Risk_LOF table
    and aggregated segment metrics from spatial-joined ILI
    and EC_Risk_LOF table.

    Args:
        EC_Risk_LOF_table: The original EC_Risk_LOF_table. Pipe segments
            are uniquely identified by (beginstationseriesid, beginstationnum, endstationnum)
            Contains a superset of all segments avaiable in each of `dict_df_aggregated`'s
            values.
        unique_ILI_segments: DataFrame containing information for unqiue pipe segments present
            in ILI data. See output of `sanity_check_stationing()`.
        dict_df_aggregated: Dictionary of aggregation type (e.g. "mean", "min", "max") to
            dataframe of aggregated segment values. See for example how
            `do_segment_level_aggregation()` produce its output.

    Returns:
        df_interpolated_segments: Dataframe with same columns as the values of
            `dict_df_aggregated`. Contains information for pipe segments within the
            ILI data range but had no EC anomaly found.
            The `CALCULATED_RISK_FIELDS` and `AUGMENTED_RISK_FIELDS` are derived from
            original EC_Risk_LOF_table data for these interpolated segments.
            The `FAILURE_PRESSURE_FIELDS` and `AUGMENTED_METRIC_FIELS` are derived
            from `dict_df_aggregated`'s values for interpolation.
    """
    EC_Risk_LOF_table_relevant = EC_Risk_LOF_table[
        LOCATION_FIELDS + CALCULATED_RISK_FIELDS
    ]
    grouped_by_segs = EC_Risk_LOF_table_relevant.groupby(
        ["route", "beginstationseriesid", "beginstationnum", "endstationnum", "endstationseriesid"]
    )
    assert np.all(grouped_by_segs.size().values == 1), "Expect all pipe segments to be unique in risk table"

    # Now we compile the segments that need to be interpolated and the segment information
    df_interpolated_segments = None
    for route in unique_ILI_segments["route"].unique():
    
        ILI_segments_for_route = unique_ILI_segments[unique_ILI_segments["route"] == route]
        # find all the beginstationseriesid for these routes
        unique_beginstationseriesid = ILI_segments_for_route["beginstationseriesid"].unique()    
        for bssid in unique_beginstationseriesid:
            # for each of these beginstationseriesid, find smallest and largest beginstationnum
            cur_ILI_segments = ILI_segments_for_route[ILI_segments_for_route["beginstationseriesid"] == bssid]
            min_beginstationnum = cur_ILI_segments["beginstationnum"].min()
            max_beginstationnum = cur_ILI_segments["beginstationnum"].max()
            # look for segments that fall within this range in the EC_LOF table's segments:
            # beginstationseriesid uniquely identifies route as well!
            cur_LOF_segments = EC_Risk_LOF_table_relevant[EC_Risk_LOF_table_relevant["beginstationseriesid"] == bssid]
            cur_LOF_segments_in_range = cur_LOF_segments[
                (cur_LOF_segments["beginstationnum"] > min_beginstationnum) & (cur_LOF_segments["beginstationnum"] < max_beginstationnum)
            ]
            # get number of segments not already in ILI
            # convert the unique segments to python set
            cur_ILI_set = set(map(tuple, cur_ILI_segments[["beginstationseriesid", "beginstationnum", "endstationnum"]].values))
            cur_LOF_set = set(map(tuple, cur_LOF_segments_in_range[["beginstationseriesid", "beginstationnum", "endstationnum"]].values))
            set_diff = cur_LOF_set.difference(cur_ILI_set)

            if len(set_diff) > 0:
                diff_segments = cur_LOF_segments[
                    cur_LOF_segments[["beginstationseriesid", "beginstationnum", "endstationnum"]].apply(tuple, axis=1).isin(set_diff)
                ]
                diff_segments["route"] = route
                if df_interpolated_segments is None:
                    df_interpolated_segments = diff_segments
                else:
                    df_interpolated_segments = pd.concat([df_interpolated_segments, diff_segments], ignore_index=True)

    # Now need to interpolate this frame with all the other values -- use aggregate over all route types
    fields_to_interpolate = ["-Depth (%)"] + FAILURE_PRESSURE_FIELDS
    df_interpolated_segments[fields_to_interpolate] = np.nan
    df_interpolated_segments["-Depth (%)"] = dict_df_aggregated["max"]["-Depth (%)"].quantile(0.95)
    for f in FAILURE_PRESSURE_FIELDS:
        df_interpolated_segments[f] = dict_df_aggregated["max"][f].quantile(0.95)
    df_interpolated_segments = augment_dataframe_inplace(df_interpolated_segments)
    return df_interpolated_segments

# =================== Plotting functions ===================

"""
Performance measurement for leak:
- Select rows whose EC_LOF_Rupture=0
- Plot mean (Depth %) vs. {S_EC, ln(af_EC)} in linear scale
- For each of the route types, and overall

Performance measurement for rupture:
- Select rows whose EC_LOF_Leak=0
- Plot mean (Pf*/MAOP) vs. {S_EC, ln(af_EC)} in linear scale
- For each of the route types, and overall
"""

def make_linear_scale_scatter_with_reg_line(
    df: pd.DataFrame,
    x_field: str,
    x_label: str,
    y_field: str,
    y_label: str,
    ax: matplotlib.axes._axes.Axes,
    robust: bool = False,
) -> None:
    """
    `df` contains the data to be plotted, which can be selected by
    `x_field` and `y_field`.

    The (x,y) scatter plot will be drawn as is.
    A regression line will be fitted with the form `Y ~ X` and super 
    imposed onto the scatter plot.
    In the end, the new field will be removed to prevent future errors.

    `ax` will contain the Axes object where the plot will be drawn.
    """       
    sns.regplot(
        x=x_field, y=y_field, data=df,
        line_kws={'color': 'red'},
        logx=False, 
        ax=ax,
        robust=robust,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def make_performance_plot(
    dict_df_aggregated: Dict[str, pd.DataFrame],
    suptitle_postfix: str,
    risk_output_type: str = "leak", # or "rupture"
    ILI_aggregation_type: str = "mean", # or anything else in dict_df_aggregated.keys()
    route_types_to_ignore: Optional[List[str]] = None,
    df_interpolate_segments: Optional[pd.DataFrame] = None,
) -> Tuple[
    matplotlib.figure.Figure,
    List[matplotlib.axes.Axes],
]:
    """
    Wrapper for making big plots
    For each subplot:
    - x_axis = risk model output. Bigger = more risk
    - y_axis = ILI pipe health proxy. Bigger = better health = less risk
    - Ideally we want to see inverted relationship, or regression line going
      from top left to bottom right.

    Args:
        dict_df_aggregated: Dictionary of aggregation type (e.g. "mean", "min", "max") to
            dataframe of aggregated segment values. See for example how
            `do_segment_level_aggregation()` produce its output.
        suptitle_postfix: Title of the overall plot is always in the format:
            "Model performance metric for EC <risk_output_type>", the suptitle_postfix
            appends to it and specifies additional information (e.g. year of the results)
        risk_output_type: Either "leak" or "rupture". The risk model assigns a segment with
            either leak or rupture risk, and we use different ILI measurement as the
            ground-truth comparison.
        route_types_to_ignore: List of route types to ignore, if given
        df_interpolate_segments: Dataframe containing for each route, how many segments
            to interpolate ground-truth values for, with what value, for what risk type.
            See also `make_interpolate_segments()`.
    
    Returns:
        fig: The drawn matplotlib figure
        axes: The different subplot axes of the figure
    """
    df_to_plot = dict_df_aggregated[ILI_aggregation_type]
    if df_interpolate_segments is not None:
        df_to_plot = pd.concat([df_to_plot, df_interpolate_segments], ignore_index=True)
    
    # Note each row's y_label will be prefixed with route type
    if risk_output_type == "leak":
        y_field = "-Depth (%)"
        y_label = "Negative Depth (%) lost"
        x_fields = [
            # "ln_EC_LOF_Leak",
            "ln_af_EC",
            "S_EC",
        ]
        x_labels = [
            # "ln(EC_LOF_Leak)",
            "ln(af_EC)",
            "S_EC",
        ]
        # Look at leak scores when rupture LOF is 0
        risk_score_mask = df_to_plot["EC_LOF_Rupture"] == 0
    elif risk_output_type == "rupture":
        y_field = "Pf*/MAOP"
        y_label = "Pf*/MAOP"
        x_fields = [
            # "ln_EC_LOF_Rupture",
            "ln_af_EC",
            "S_EC",
        ]
        x_labels = [
            # "ln(EC_LOF_Rupture)",
            "ln(af_EC)",
            "S_EC",
        ]
        # Look at rupture scores when leak LOF is 0
        risk_score_mask = df_to_plot["EC_LOF_Leak"] == 0

    df_to_plot = df_to_plot[risk_score_mask]
    route_masks = get_route_types(df_to_plot)
    dict_route_masks = {
        key: value
        for (key, value) in route_masks.__dict__.items()
        if sum(value) > 0
    }
    if route_types_to_ignore is not None:
        for route_type in route_types_to_ignore:
            if route_type in dict_route_masks:
                del dict_route_masks[route_type]

    route_type_list = list(dict_route_masks.keys())
    num_route_types = len(dict_route_masks)

    # We want to plot a separate row for each route type, and then overall
    n_rows = num_route_types + 1
    # For each route type, make one plot for vs. {ln(EC_LOF_xx), S_EC, ln(af_EC)}
    n_cols = len(x_fields)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        sharey="row",
        sharex="col",
        figsize=(
            n_cols * 4.0,
            n_rows * 3.5,
        )
    )

    # Now plot each one
    for i_route in range(n_rows):
        for j_risk in range(n_cols):
            # Top row is always overall
            if i_route == 0:
                y_label_post_fix = "\nAll lines"
                cur_df_to_plot = df_to_plot
            else:
                route_type = route_type_list[i_route - 1]
                y_label_post_fix = f"\n{route_type}"
                cur_df_to_plot = df_to_plot[dict_route_masks[route_type]]

            make_linear_scale_scatter_with_reg_line(
                df=cur_df_to_plot,
                x_field=x_fields[j_risk],
                x_label=x_labels[j_risk],
                y_field=y_field,
                y_label=f"{y_label}{y_label_post_fix}",
                ax=axes[i_route][j_risk]
            )
            stats_result = stats.spearmanr(
                cur_df_to_plot[x_fields[j_risk]],
                cur_df_to_plot[y_field],
            )
            sig_str = "*" if stats_result.pvalue < 0.05 else ""
            axes[i_route][j_risk].set_title(f"rank correlation={stats_result.statistic:.3f}{sig_str}")
    
    plt.suptitle(f"Model performance metric for EC {risk_output_type}{suptitle_postfix}")
    return fig, axes


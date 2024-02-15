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

"""
Utility scripts to perform common filtering and visualization
for calculating external corrosion volumetric loss and other
statistics based on ILI data
"""


# ==== Filtering ILI tally ====

def filter_identifications(
    cleaned_ILI_df: pd.DataFrame,
) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
    """
    Look at `"Identification"` column and filter by:
    1. Drop null columns
    2. Drop empty str
    3. Convert to lower case and aggregate unique ones

    Return:
    - unique_identifications: Series representing the unique identifications in
        the ILI data
    - filtered_ILI_df: Filtered version of the original ILI data frame with
        only eligibile identifications.

    """
    unique_identifications = cleaned_ILI_df["Identification"].dropna().astype(str).apply(lambda x: x.strip())
    unique_identifications = unique_identifications.apply(lambda x: x.lower())

    # remove empty str
    empty_str_identification = unique_identifications == ''
    unique_identifications = unique_identifications[~empty_str_identification]

    # filter out all the rows whose identification is not in unique_identification
    no_NA_identification_ILI = cleaned_ILI_df.dropna(subset=["Identification"])
    all_identification = no_NA_identification_ILI["Identification"].astype(str).apply(lambda x: x.strip()).apply(lambda x: x.lower())
    is_ok_identification = all_identification.isin(unique_identifications)

    filtered_ILI_df = no_NA_identification_ILI[is_ok_identification]
    filtered_ILI_df["Identification"] = filtered_ILI_df["Identification"].astype(str).apply(lambda x: x.strip()).apply(lambda x: x.lower())

    # Print out some statistics
    unique_identification_freq = filtered_ILI_df["Identification"].value_counts()
    print(
        f"{np.sum(unique_identification_freq == 1) / len(unique_identification_freq)*100:.3f}% "
        "of all identifications has only 1 occurence"
    )
    return unique_identifications, filtered_ILI_df


def filter_manufacturing_anomalies(cleaned_identification_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out rows with whose identifications contain anything `"manufacturing"`-related
    """
    unique_identification_freq = cleaned_identification_df["Identification"].value_counts()
    manufacturing_mask = ["manufacturing" in category for category in unique_identification_freq.index]
    print(f"Total {unique_identification_freq[manufacturing_mask].sum()} manufacturing Identifications")
    print(f"Total {unique_identification_freq.sum()} Identifications")
    print(f"{unique_identification_freq[manufacturing_mask].sum() / unique_identification_freq.sum() * 100: .3f}% of identifications are manufacturing related")

    mask = cleaned_identification_df["Identification"].isin(unique_identification_freq[manufacturing_mask].keys())
    filtered_df = cleaned_identification_df[~mask]
    print(f"Total {len(filtered_df)} non-manufacturing identifications")
    return filtered_df


def _get_external_anomaly_row_mask(segment: pd.DataFrame) -> pd.Series:
    # Return row mask indidating presence of external corrosion

    # keep rows with non-null "Depth (%)" field
    mask_has_depth = ~segment["Depth (%)"].isna()
    # keep rows with is "External"
    mask_is_external = segment["Internal"].str.lower() == "external"
    # keep rows with "WT (in)" and "OD (in)" present
    mask_has_WT = ~segment["WT (in)"].isna()
    mask_has_OD = ~segment["OD (in)"].isna()
    return (
        mask_has_depth & mask_is_external & mask_has_WT & mask_has_OD
    )


def get_external_anomaly_row_mask(segment: pd.DataFrame) -> pd.Series:
    # Return row mask indicating presence of external corrosion
    # with width or length

    is_external_mask = _get_external_anomaly_row_mask(segment)
    mask_has_width = ~segment["Width (in)"].isna()
    mask_has_length = ~segment["Length (in)"].isna()
    return (
        is_external_mask & (mask_has_length | mask_has_width)
    )


def get_volumetric_row_mask(segment: pd.DataFrame) -> pd.Series:
    # Return row mask indicating presence of external corrosion
    # with all fields present to calculate volumetric loss
    is_external_mask = _get_external_anomaly_row_mask(segment)
    mask_has_width = ~segment["Width (in)"].isna()
    mask_has_length = ~segment["Length (in)"].isna()
    return (
        is_external_mask & mask_has_length & mask_has_width
    )


@dataclass
class GirthWeldStats:
    list_average_girth_weld_dist: List[float]
    num_route_no_girth_welds: int
    total_route_dist_no_girth_welds: float
    total_route_dist_no_girth_welds: float
    num_route_no_survey_dist: int
    total_routes: int


def get_girth_weld_stats(
    cleaned_identification_df: pd.DataFrame
) -> Tuple[
    Dict[int, pd.DataFrame],
    Dict[int, Dict[str, pd.DataFrame]],
    List[Tuple[int, str]],
    GirthWeldStats,
]:
    """
    Go through ILI dataframe with cleaned identificiations and group them by 
    year and route.

    For each route, group the anomalies between successive girth welds and get
    statistics.

    Args:
        cleaned_identification_df: ILI dataframe that has invalid identification and
            anomalies identified with manufacturing-related issues removed.

    Returns:
        grouped_by_year_dict: Dict[key, dataframe], keys=year, value=rows associated
            with that year.
        grouped_by_year_by_route_dict: Dict[key, Dict[key, str]].
            First-level key=year
            Second-level key=route name, whose values=rows associated with that (year, route)
        list_dict_keys_with_girth_welds: List[[int, str]]. Each item contains the
            (year, route) that indexes into `grouped_by_year_by_route_dict` for routes
            that contains girth welds
        girth_weld_stats: GirthWeldStats object containing relevant stats about the different
            routes.
    """

    # for each year
    grouped_by_year_dict = {key: group for (key, group) in cleaned_identification_df.groupby("Year")}
    # for each route
    grouped_by_year_by_route_dict = {
        year: {
            route: route_group for (route, route_group) in
            year_group.groupby("Route")
        }
        for (year, year_group) in grouped_by_year_dict.items()
    }

    # find all girth weld
    # and check their index difference and also survey distance diff

    # This includes (year, route) such that `grouped_by_year_by_route_dict[year][route]` has
    # girth welds
    list_dict_keys_with_girth_welds =[]

    list_average_girth_weld_dist = []
    num_route_no_girth_welds = 0
    total_route_dist_no_girth_welds = 0
    num_route_no_survey_dist = 0
    total_routes = 0

    for (year, year_group) in grouped_by_year_by_route_dict.items():    
        for (route, route_group) in year_group.items():
            total_routes += 1
            is_girth_weld = route_group["Identification"] == "girth weld"
            if route_group["ILI Survey Distance (ft)"].isna().all():
                print(f"Year {year}, Route {route}: No survey distance available!")
                num_route_no_survey_dist += 1
                continue
            if not any(is_girth_weld):
                max_dist = route_group["ILI Survey Distance (ft)"].max() - route_group["ILI Survey Distance (ft)"].min()
                print(f"Year {year}, Route {route}: No girth weld available, total survedy distance available={max_dist}")
                num_route_no_girth_welds += 1
                total_route_dist_no_girth_welds += max_dist
                continue
            if np.sum(is_girth_weld) == 1:
                print(f"Year {year}, Route {route}: Only one girth weld found, moving to no girth weld category")
                if len(route_group) > 1:
                    max_dist = route_group["ILI Survey Distance (ft)"].max() - route_group["ILI Survey Distance (ft)"].min()
                    print(f"  Year {year}, Route {route}: No girth weld available, total survedy distance available={max_dist}")
                    total_route_dist_no_girth_welds += max_dist
                num_route_no_girth_welds += 1
                continue
                
            survey_dist_between_welds = route_group[is_girth_weld]["ILI Survey Distance (ft)"].diff()[1:]        
            average_girth_weld_dist = survey_dist_between_welds.mean()
            print(f"Year {year}, Route {route}: Average distance between girth welds is {average_girth_weld_dist}")
            list_average_girth_weld_dist.append(average_girth_weld_dist)

            # Add routes with girth-welds do dict
            list_dict_keys_with_girth_welds.append((year, route))

    print(
        "\n",
        f"Total {total_routes} routes available,\n"    
        f"Total {num_route_no_girth_welds} routes have no girth welds found, for combined dist of {total_route_dist_no_girth_welds}ft \n",
        f"Total {num_route_no_survey_dist} routes have no survey distance available \n",
        f"Average distance between girth welds for routes = {np.mean(list_average_girth_weld_dist)}"
    )

    girth_weld_stats =  GirthWeldStats(
        list_average_girth_weld_dist=list_average_girth_weld_dist,
        num_route_no_girth_welds=num_route_no_girth_welds,
        total_route_dist_no_girth_welds=total_route_dist_no_girth_welds,
        num_route_no_survey_dist=num_route_no_survey_dist,
        total_routes=total_routes,
    )
    assert len(list_dict_keys_with_girth_welds) == (
        total_routes - num_route_no_girth_welds - num_route_no_survey_dist
    )

    return (
        grouped_by_year_dict,
        grouped_by_year_by_route_dict,
        list_dict_keys_with_girth_welds,
        girth_weld_stats,
    )

# ==== Segment ILI results into pipe sections ====

class SegmentDescription(NamedTuple):
    year: int
    route: str
    group: int  # either girth weld group or dist-group


def segment_route_by_girth_welds(
    grouped_by_year_by_route_dict: Dict[int, Dict[str, pd.DataFrame]],
    list_dict_keys_with_girth_welds: List[Tuple[int, str]],
) -> Dict[SegmentDescription, pd.DataFrame]:
    """
    Return a dictionary that segments ILI tally items into girth-weld sections
    for eligibile routes.

    Args:
        grouped_by_year_by_route_dict: Dict[key, Dict[key, str]].
            First-level key=year
            Second-level key=route name, whose values=rows associated with that (year, route)
        list_dict_keys_with_girth_welds: List[[int, str]]. Each item contains the
            (year, route) that indexes into `grouped_by_year_by_route_dict` for routes
            that contains girth welds

    Returns:
        year_route_segment_dict: {SegmentDescription, Dataframe},
            The dictionary contains all segments' description and the associated dataframe.
            The dataframe value will have at least one row, corresponding to each girth-weld
            that's found on each route. Rows in addition to that are other ILI tally items
            corresponding to that girth weld section.
    """
    year_route_segment_dict = {}
    sections_with_anomalies = 0
    for (year, route) in list_dict_keys_with_girth_welds:
        df = grouped_by_year_by_route_dict[year][route]
        is_girth_weld = (df["Identification"] == "girth weld")
        group = is_girth_weld.cumsum()
        for g in range(group.iloc[-1]):
            year_route_segment_dict[SegmentDescription(year, route, g)] = df[group == g]
    return year_route_segment_dict


def get_girth_weld_segment_length(
    year_route_segment_dict: Dict[SegmentDescription, pd.DataFrame],
    segment_key: SegmentDescription,
) -> Optional[float]:
    """
    Assuming `year_route_segment_dict` is generated from `segment_route_by_girth_welds()`,
    calculate the length of a girth weld segment, characterized by increasing `SegmentDescription.group`
    attribute, by indexing the survey distance of the successive girth weld survey distance values.

    If a segment is the last segment, use the previous segment's value.
    If it's the only segment available of a (year, route) combination, we ignore this section
    and return None
    """
    year, route, group = segment_key
    cur_segment_survey_distance = year_route_segment_dict[
        segment_key
    ]["ILI Survey Distance (ft)"].to_numpy()[0]
    if (year, route, group + 1) in year_route_segment_dict:
        return year_route_segment_dict[
            (year, route, group + 1)
        ]["ILI Survey Distance (ft)"].to_numpy()[0] - cur_segment_survey_distance
    elif (year, route, group - 1) in year_route_segment_dict:
        return year_route_segment_dict[
            (year, route, group - 1)
        ]["ILI Survey Distance (ft)"].to_numpy()[0] - cur_segment_survey_distance
    else:
        return None 


def segment_route_by_survey_dist(
    grouped_by_year_by_route_dict: Dict[int, Dict[str, pd.DataFrame]],
    segment_dist: float=40.0,
):
    """
    Return a dictionary that segments ILI tally items into survey-distance sections
    for eligibile routes.

    Args:
        grouped_by_year_by_route_dict: Dict[key, Dict[key, str]].
            First-level key=year
            Second-level key=route name, whose values=rows associated with that (year, route)
        segment_dist: float, length of pipe distance to segment

    Returns:
        year_route_segment_dict: {SegmentDescription, Dataframe},
            The dictionary contains all segments' description and the associated dataframe.
            If the dataframe doesn't include anything, it means no ILI entries available for
            that section of the pipe.
        discarded_no_survey_dist: int
    """
    year_route_segment_dict = {}
    discarded_no_survey_dist = 0
    for year, year_routes in grouped_by_year_by_route_dict.items():
        for route, df in year_routes.items():
            survey_dist = df["ILI Survey Distance (ft)"]
            bucket = (survey_dist // segment_dist)
            bucket_vals = bucket.dropna().unique().astype(int)
            for b_val in bucket_vals:
                year_route_segment_dict[SegmentDescription(year, route, b_val)] = df[bucket == b_val]
    return year_route_segment_dict


# ===== Volume calculations ====

def pipe_segment_volume(
    h: Union[float, np.ndarray],
    WT: Union[float, np.ndarray],
    OD: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Given a pipe segment of length `h`, with outer diameter `OD`, and
    wall thickness `WT`, the volume of this pipe is equal to

        pi * h * WT * (OD - WT)

    This is derived and simplified using the relationship:

    pipe_inner_diameter = OD - 2 * WT

    Note input and output types can be arrays since function is vectorized.
    Will be faster than giving it pandas data frame.

    The units of h, WT, and OD are assumed to be the same.
    """
    return np.pi * h * (OD - WT)

def anomaly_volume(
    length: Union[float, np.ndarray],
    width: Union[float, np.ndarray],
    depth: Union[float, np.ndarray],
    OD: Union[float, np.ndarray],
    WT: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Given an anomaly on the outer surface of the pipe (EC), with
    - `length` measured along the axial direction,
    - `width` measured circumferentially,
    - `depth` measured in the radial direction
    - `OD` = outer dimater of the pipe
    - `WT` = thickness of the pipe,
    - pipe inner diameter = OD - 2 * WT

    The volume of this anomaly is then an annulus sector that can be calculated by:

        volume = length * cross_section_area,

    `cross_section_area` is the annulus area of the anomaly, where the arc angle `theta`
    is proportional to the width of the corrosion anomaly.

        cross_section_area = (theta / 2) * (R^2 - r^2)
        theta = width / (pi * OD)

    here `R` and `r` are the outer and inner radius of the pipe:

        R = OD / 2
        r = R - depth
    
    Note input and output types can be arrays since function is vectorized.
    Will be faster than giving it pandas data frame.
    """
    theta = width / (np.pi * OD)
    R = OD / 2
    r = R - depth
    cross_section_area = theta / 2 * (R**2 - r**2)
    volume = length * cross_section_area
    return volume


def calculate_segment_frac_volumetric_loss(
    segment: pd.DataFrame,
    segment_length_ft: float,
    volumetric_mask_name: str = "volumetric_row_mask",
) -> Tuple[float, float, float]:
    """
    Calculate volumetric loss from the input segment.

    Args:
        segment: dataframe whose rows all belong to the same pipe segment. There should
            be a field in it with bool value, indicating whether volumetric loss can be
            calculated from that row.
        segment_length: float, length of the pipe segment
        volumetric_mask_name: str, indicates the column name that contains the boolean
            indicator value. If not given or doesn't exist, will throw error.

    Returns:
        frac_volumetric_loss: float, volumetric loss on the entire segment after
            aggregating all eligible anomaly entries, normalized by total pipeline segment
            volume
        WT_CV: float, std(WT)/mean(WT). Normally 0, but may not be if the measured
            WT on anomaly entries aross segment is not the same.
        OD_CV: float, std(OD)/mean(OD). Normally 0, but may not be if the measured
            OD on anomaly entries across segment is not the same.
    """
    assert volumetric_mask_name in segment.columns, "Require volumetric_mask_name"

    segment_anomalies = segment[segment[volumetric_mask_name]]
    if segment_anomalies.empty:
        return (0, 0, 0)

    # calculate anomaly volume
    depth_in = (segment_anomalies["Depth (%)"] * segment_anomalies["WT (in)"]).to_numpy() / 100.
    length = segment_anomalies["Length (in)"].to_numpy()
    width = segment_anomalies["Width (in)"].to_numpy()
    OD = segment_anomalies["OD (in)"].to_numpy()
    WT = segment_anomalies["WT (in)"].to_numpy()
    volumetric_loss = anomaly_volume(length, width, depth_in, OD, WT)
    assert np.all(volumetric_loss >= 0)

    # calculate total pipe volume
    segment_WT = WT[0]
    WT_CV = 0.
    if not np.all(WT == WT[0]):
        WT_CV = np.std(WT) / np.mean(WT)
        segment_WT = np.mean(WT)
        logging.info(
            f"Segment WT not uniform, taking average."
            f" WT_CV = {WT_CV}")
    segment_OD = OD[0]
    OD_CV = 0.
    if not np.all(OD == OD[0]):
        OD_CV = np.std(OD) / np.mean(OD)
        segment_OD = np.mean(OD)
        logging.info(
            f"Segment OD not uniform, taking average."
            f" OD_CV = {OD_CV}")

    # All units should be in "inches"
    pipe_volume = pipe_segment_volume(
        segment_length_ft * 12.0, segment_WT, segment_OD
    )
    assert np.all(pipe_volume >= 0)
    return (
        np.sum(volumetric_loss) / pipe_volume,
        WT_CV,
        OD_CV,
    )


def compile_volumetric_loss_for_all_segments(
    year_route_segment_dict: Dict[SegmentDescription, pd.DataFrame],
    segment_type: str,
    volumetric_mask_name: str = "volumetric_row_mask",
    fixed_segment_length_ft: Optional[float] = None,
) -> Dict[int, np.ndarray]:
    """
    Calculate frac volumetric loss for all available pipe segments.

    Args:
        year_route_segment_dict: Dict[SegmentDescription, pd.DataFrame] generated
            by `segment_route_by_girth_welds()` or ``segment_route_by_survey_dist()`.
            The data frames corresponding to each pipe segment may or may not have
            eligible anomalies.
        segment_type: str, "girth weld" or "fixed distance". Indicates what kind of 
            segment aggregation.
        volumetric_mask_name: str, column in the data frames indicating whether a row
            can be used to calculate volumetric loss
        fixed_segment_length_ft: float, If segment_type is `fixed distance`, this is the
            length with which anomalies are aggregated by. Unit is ft.

    Returns:
        dict_result: Dict[int, np.ndarray]. Keys are
            the year, values are arrays with 3 columns: (frac_vol_loss, WT_CV, OD_CV)
            If the last two columns are not 0, that means the pipe segment does not have
            uniform WT or OD.
    """
    segment_length_ft = fixed_segment_length_ft
    if segment_type == "fixed distance":
        assert segment_length_ft is not None, "Need to specify segment length for fixed distance segments"

    dict_result = defaultdict(list)
    for (year, route, group), segment in year_route_segment_dict.items():
        if segment.empty:
            continue
        elif segment_type == "girth weld" and len(segment) <= 1:
            continue

        if segment_type == "girth weld":
            segment_length_ft = get_girth_weld_segment_length(
                year_route_segment_dict, (year, route, group)
            )
            if segment_length_ft is None:
                continue
        
        segment_result = calculate_segment_frac_volumetric_loss(
            segment, segment_length_ft, volumetric_mask_name,
        )
        dict_result[year].append(segment_result)
    return {
        key: np.array(np.vstack(value))
        for key, value in dict_result.items()
    }

# ===== Visualization ====

"""
- Cut-off for WD and OD uniformity
- remove 0's -- only look at segments with observed corrosion
"""

def plot_distribution_and_sd(
    data: np.ndarray,
    log_transform_data: bool = True,
    ylog: bool = False,
    kde: bool = True,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
) -> Tuple[float, float, float]:
    """
    Returns:
        (mean, +1*SD, +2*SD) values in the original scale of the data
    """
    if log_transform_data:
        data = np.log(data)

    sns.histplot(data, kde=kde)
    mean = data.mean()
    std = data.std()
    one_sd, two_sd = std, 2 * std
    plt.axvline(
        mean, 
        color="green", 
        linestyle="--", 
        label=f"Mean={mean: .2E}"
    )
    plt.axvline(
        (mean + one_sd), 
        color="red", 
        linestyle="--", 
        label=f"1 SD={(mean + one_sd): .2E}"
    )
    plt.axvline(
        (mean + two_sd), 
        color="orange", 
        linestyle="--", 
        label=f"2 SD={(mean + two_sd): .2E}"
    )
    plt.legend()
    if ylog:
        plt.yscale("log")
    if xlabel:
        if log_transform_data:
            xlabel = f"{xlabel}, (log transformed)"
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    return (mean, mean + one_sd, mean + two_sd)

# annual statistics
def plot_annual_frac_volumetric_loss(
    annual_result: Dict[int, np.ndarray],
    WT_CV_thres: float = 0.15,
    OD_CV_thres: float = 0.15,
    segment_type: Optional[str] = None,
    fixed_segment_length_ft: Optional[float] = None,
) -> Tuple[
    List[matplotlib.figure.Figure],
    List[Tuple[int, int, int]],
]:
    """
    Args:
        segment_type: "girth weld" or "fixed distance", useful for adding plot title.
        OD_CV_thres, WT_CV_thres: If given, remove all segments whose OD_CV or WT_CV
            values are greater than this.
        fixed_segment_length_ft: If `segment_type` is "fixed_distance", this should be
            that used to do pipe segmentation. Will add this to plot title

    Returns:
        fig_list: Figure per year, if only that year has good eligible anomalies though.
        stats_list: (mean, mean+1SD, mean+2SD) of segment level 
            volumetric loss stats.
    """
    fig_list = []
    stats_list = []
    for year, results in annual_result.items():
        if np.all(results[:, 0] == 0):
            # Year without any volumetric loss results
            continue
        # plot only non-zero results
        loss_mask = results[:, 0] > 0
        # filter results with OD and WT uniformity, these are mostly outliers
        WT_mask = results[:, 1] <= WT_CV_thres
        OD_mask = results[:, 2] <= OD_CV_thres
        data = results[loss_mask & WT_mask & OD_mask, 0]
        fig = plt.figure()
        title_str = (
            f"{year}: normalized volumetric loss, segment type={segment_type}\n"
            f"WT_CV thresh={WT_CV_thres}, OD_CV thresh={OD_CV_thres}"
        )
        stats = plot_distribution_and_sd(
            data=data,
            log_transform_data=True,
            ylog=False,
            kde=False,
            xlabel="normalized segment volumetric loss",
            ylabel="Count",
            title=title_str,
        )
        fig_list.append(fig)
        stats_list.append(stats)
    return fig_list, stats_list


def plot_all_frac_volumetric_loss(
    annual_result: Dict[int, np.ndarray],
    WT_CV_thres: float = 0.15,
    OD_CV_thres: float = 0.15,
    segment_type: Optional[str] = None,
    fixed_segment_length_ft: Optional[float] = None,
) -> Tuple[matplotlib.figure.Figure, Tuple[int, int, int]]:
    """
    Same as `plot_annual_frac_volumetric_loss` but compile all available
    volumetric losses.
    """
    all_values = np.concatenate([value for _, value in annual_result.items()])
    loss_mask = all_values[:, 0] > 0
    WT_mask = all_values[:, 1] <= WT_CV_thres
    OD_mask = all_values[:, 2] <= OD_CV_thres
    data = all_values[loss_mask & WT_mask & OD_mask, 0]
    fig = plt.figure()
    
    title_str = f"All year: normalized volumetric loss, segment type={segment_type}\n"
    if segment_type == "fixed distance":
        title_str += f"segment length = {fixed_segment_length_ft}ft\n"
    title_str += f"WT_CV thresh={WT_CV_thres}, OD_CV thresh={OD_CV_thres}"

    stats = plot_distribution_and_sd(
        data=data,
        log_transform_data=True,
        ylog=False,
        kde=False,
        xlabel="normalized segment volumetric loss",
        ylabel="Count",
        title=title_str,
    )
    return fig, stats


# ===== Utility functions ====
def calculate_threshold_volumetric_loss(
    h: Union[float, np.array],
    WT: Union[float, np.array],
    OD: Union[float, np.array],
    threshold_normalized_volumetric_loss: float,
) -> Tuple[Union[float, np.array], Union[float, np.array]]:
    """
    Given pipe statistics all in inches, and the normalized volumetric
    loss threshold, calculate the equivalent in absolute volumetric
    loss.

    Args:
        h: length of pipe segment
        WT: pipe wall thickness
        OD: pipe outer diameter

    Output:
        absolute_loss: float or array corresponding to the absolute
            volumetric loss corresponding to the normalized threshold value.
        segment_volume: float or array corresponding to the total segment
            volume for each pipe dimensions
    """
    segment_volume = pipe_segment_volume(h, WT, OD)
    absolute_loss = segment_volume * threshold_normalized_volumetric_loss
    return absolute_loss, segment_volume
    
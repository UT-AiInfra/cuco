import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple, List
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from cuco.utils import get_path_to_best_node
import matplotlib.transforms as transforms


def plot_improvement(
    df: pd.DataFrame,
    title: str = "Best Combined Score Over Time",
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    xlabel: str = "Number of Evaluated LLM Program Proposals",
    ylabel: str = "Evolved Performance Score",
    ylim: Optional[Tuple[float, float]] = None,
    plot_path_to_best_node: bool = True,
):
    """
    Plots the improvement of a program over generations.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))

    # Plot best score line
    # Calculate cumulative maximum and back-fill leading NaNs
    # to ensure the line is continuous from the start.
    df = df.sort_values(by="generation")
    df_filtered = df[df["correct"]].copy()

    line1 = ax.plot(
        df_filtered["generation"],
        df_filtered["combined_score"].cummax(),
        linewidth=3,
        color="red",
        label="Best Score",
    )

    # Plot individual evaluations as scatter points
    scatter1 = ax.scatter(
        df_filtered["generation"],
        df_filtered["combined_score"],
        alpha=1.0,
        s=40,
        color="black",
        label="Individual Evals",
    )

    if ylim is not None:
        ax.set_ylim(*ylim)

    # Get the path to the best node
    if plot_path_to_best_node:
        best_path_df = get_path_to_best_node(df_filtered, score_column="combined_score")
    else:
        best_path_df = pd.DataFrame()
    line_best_path_plot = []  # Initialize to empty list

    if not best_path_df.empty:
        # Plot the path to the best node
        line_best_path_plot = ax.plot(
            best_path_df["generation"],  # Use generation for x-axis
            best_path_df["combined_score"],
            linestyle="-.",
            marker="o",
            color="blue",
            label="Path to Best Node",
            markersize=5,
            linewidth=2,
        )
        # Add annotations if 'patch_name' column exists
        if "patch_name" in best_path_df.columns:
            _place_non_overlapping_annotations(
                ax, best_path_df, "generation", "combined_score", "patch_name"
            )

    # Create a second y-axis for cumulative API cost
    ax2 = ax.twinx()
    handles = line1 + [scatter1]
    if line_best_path_plot:  # If the best path was plotted
        handles.extend(line_best_path_plot)

    labels = [h.get_label() for h in handles]

    if "api_costs" in df_filtered.columns:
        cumulative_api_cost = df["api_costs"].cumsum().bfill()
        line2 = ax2.plot(
            df["generation"],
            cumulative_api_cost,
            linewidth=2,
            color="orange",
            linestyle="--",
            label="Cumulative Cost",
        )
        ax2.set_ylabel(
            "Cumulative API Cost ($)",
            fontsize=22,
            weight="bold",
            color="orange",
            labelpad=15,
        )
        ax2.tick_params(axis="y", which="major", labelsize=25)
        handles.extend(line2)
        labels = [h.get_label() for h in handles]  # Recreate labels

    ax.legend(handles, labels, fontsize=25, loc="lower right")

    # Customize plot
    ax.set_xlabel(xlabel, fontsize=30, weight="bold")
    ax.set_ylabel(ylabel, fontsize=30, weight="bold", labelpad=25)
    ax.set_title(title, fontsize=40, weight="bold")
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.grid(True, alpha=0.3)

    # Remove top and right spines for the primary axis
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(
        False
    )  # Keep right spine if ax2 is present, or manage ax2 spines

    if "api_cost" in df_filtered.columns and ax2:
        # Ensure ax2 spine is visible if it exists
        ax2.spines["top"].set_visible(False)  # Match primary axis top spine
        ax2.tick_params(axis="y", which="major", labelsize=30)

    fig.tight_layout()  # Adjust layout to prevent overlapping labels

    return fig, ax


def _place_non_overlapping_annotations(
    ax: Axes, df: pd.DataFrame, x_col: str, y_col: str, text_col: str
):
    """
    Places annotations with minimal overlap using a systematic approach.
    """
    # Define multiple offset positions to try (in order of preference)
    offset_positions = [
        (40, -30),  # bottom-right
        (40, 30),  # top-right
        (-40, 30),  # top-left
        (-40, -30),  # bottom-left
        (60, 0),  # right
        (-60, 0),  # left
        (0, 40),  # top
        (0, -40),  # bottom
        (70, -50),  # far bottom-right
        (-70, 50),  # far top-left
    ]

    placed_boxes = []  # Store bounding boxes of placed annotations

    for _, row in df.iterrows():
        patch_name_val = str(row.get(text_col, ""))
        if pd.notna(patch_name_val) and patch_name_val != "":
            if patch_name_val == "nan" or patch_name_val == "none":
                patch_name_val = "Base"

            # Wrap long patch names
            patch_name_to_plot = _wrap_text(patch_name_val, max_length=15)

            x_pos = float(row[x_col])
            y_pos = float(row[y_col])

            # Find the best position with minimal overlap
            best_offset, best_ha, best_va = _find_best_position(
                ax, x_pos, y_pos, patch_name_to_plot, offset_positions, placed_boxes
            )

            # Place the annotation
            annotation = ax.annotate(
                patch_name_to_plot,
                (x_pos, y_pos),
                textcoords="offset points",
                xytext=best_offset,
                ha=best_ha,
                va=best_va,
                fontsize=11,
                fontweight="bold",
                color="darkgreen",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="lightyellow",
                    ec="black",
                    alpha=0.7,
                ),
                arrowprops=dict(
                    arrowstyle="-",
                    shrinkA=5,
                    shrinkB=5,
                    connectionstyle="arc3,rad=0.2",
                    color="black",
                ),
                zorder=10,
            )

            # Store the bounding box for future collision detection
            try:
                # Get the bounding box in data coordinates
                bbox = annotation.get_window_extent()
                inv_transform = ax.transData.inverted()
                bbox_data = inv_transform.transform_bbox(bbox)
                placed_boxes.append(bbox_data)
            except Exception:
                # Fallback: approximate bounding box
                approx_width = len(patch_name_to_plot) * 0.01  # rough estimate
                approx_height = patch_name_to_plot.count("\n") * 0.02 + 0.02
                placed_boxes.append(
                    transforms.Bbox.from_bounds(
                        x_pos - approx_width / 2,
                        y_pos - approx_height / 2,
                        approx_width,
                        approx_height,
                    )
                )


def _wrap_text(text: str, max_length: int = 15) -> str:
    """
    Wraps text at word boundaries for better readability.
    """
    if len(text) <= max_length:
        return text

    # Try to find a good breaking point
    mid_point = len(text) // 2

    # Look for a space near the middle
    for offset in range(min(5, mid_point)):
        # Check before midpoint
        if mid_point - offset > 0 and text[mid_point - offset] == " ":
            break_point = mid_point - offset
            part1 = text[:break_point].strip()
            part2 = text[break_point + 1 :].strip()
            return f"{part1}\n{part2}"

        # Check after midpoint
        if mid_point + offset < len(text) and text[mid_point + offset] == " ":
            break_point = mid_point + offset
            part1 = text[:break_point].strip()
            part2 = text[break_point + 1 :].strip()
            return f"{part1}\n{part2}"

    # No good space found, break at midpoint
    return f"{text[:mid_point]}\n{text[mid_point:]}"


def _find_best_position(
    ax: Axes,
    x_pos: float,
    y_pos: float,
    text: str,
    offset_positions: List[Tuple[int, int]],
    placed_boxes: List[transforms.Bbox],
) -> Tuple[Tuple[int, int], str, str]:
    """
    Finds the best annotation position with minimal overlap.
    """
    best_offset = offset_positions[0]
    best_overlap_count = float("inf")

    for offset in offset_positions:
        # Determine alignment based on offset
        ha = "left" if offset[0] >= 0 else "right"
        va = "bottom" if offset[1] >= 0 else "top"

        # Estimate the bounding box for this position
        estimated_bbox = _estimate_annotation_bbox(
            ax, x_pos, y_pos, text, offset, ha, va
        )

        # Count overlaps with existing annotations
        overlap_count = sum(1 for bbox in placed_boxes if estimated_bbox.overlaps(bbox))

        # If no overlaps, use this position
        if overlap_count == 0:
            return offset, ha, va

        # Track the position with minimum overlaps
        if overlap_count < best_overlap_count:
            best_overlap_count = overlap_count
            best_offset = offset

    # Return the alignment for the best offset
    ha = "left" if best_offset[0] >= 0 else "right"
    va = "bottom" if best_offset[1] >= 0 else "top"

    return best_offset, ha, va


def _estimate_annotation_bbox(
    ax: Axes,
    x_pos: float,
    y_pos: float,
    text: str,
    offset: Tuple[int, int],
    ha: str,
    va: str,
) -> transforms.Bbox:
    """
    Estimates the bounding box of an annotation in data coordinates.
    """
    # Rough estimation based on text length and number of lines
    lines = text.split("\n")
    max_line_length = max(len(line) for line in lines)
    num_lines = len(lines)

    # Approximate dimensions (these are rough estimates)
    char_width_data = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 100
    line_height_data = (ax.get_ylim()[1] - ax.get_ylim()[0]) / 50

    width = max_line_length * char_width_data
    height = num_lines * line_height_data

    # Convert offset from points to data coordinates (approximate)
    x_offset_data = offset[0] * char_width_data / 8  # rough conversion
    y_offset_data = offset[1] * line_height_data / 12  # rough conversion

    # Calculate annotation position based on alignment
    if ha == "left":
        left = x_pos + x_offset_data
        right = left + width
    else:  # ha == "right"
        right = x_pos + x_offset_data
        left = right - width

    if va == "bottom":
        bottom = y_pos + y_offset_data
        top = bottom + height
    else:  # va == "top"
        top = y_pos + y_offset_data
        bottom = top - height

    return transforms.Bbox.from_bounds(left, bottom, width, height)

"""
spatial_selection.py
====================
Coverage-Aware Spatially Uniform Tie-Point Selection for Lunar Imagery.

Primary Innovation #2:
Rather than greedily taking the top-N highest-confidence matches (which cluster
densely on crater rims and leave vast mare basalt regions empty), this module
enforces uniform spatial distribution using adaptive grid bucketing and
non-maximal suppression across an 8x8 spatial partition.

Computes:
- Spatial Coverage (%) = (Occupied Cells / Total Active Overlap Cells) * 100
- Spatial Shannon Entropy (distribution dispersion metric)
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np


class SpatialTiePointSelector:
    """
    Partitions the overlap field into an adaptive grid and selects
    spatially distributed, high-confidence tie points.
    """

    def __init__(self,
                 grid_rows: int = 8,
                 grid_cols: int = 8,
                 max_points_per_cell: int = 3,
                 target_total_points: int = 60):
        """
        Parameters:
        - grid_rows: number of horizontal partitions (default 8)
        - grid_cols: number of vertical partitions (default 8)
        - max_points_per_cell: cap on points selected per cell to avoid local clustering
        - target_total_points: desired total number of distributed tie points
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.max_points_per_cell = max_points_per_cell
        self.target_total_points = target_total_points

    def select_uniform_tie_points(self,
                                 pts_src: np.ndarray,
                                 pts_ref: np.ndarray,
                                 reproj_errors: np.ndarray,
                                 confidences: Optional[np.ndarray],
                                 img_shape_ref: Tuple[int, int]) -> Dict[str, Any]:
        """
        Executes coverage-aware selection on inlier correspondences.

        Parameters:
        - pts_src: ndarray of shape (N, 2)
        - pts_ref: ndarray of shape (N, 2)
        - reproj_errors: ndarray of shape (N,)
        - confidences: ndarray of shape (N,) or None
        - img_shape_ref: (height, width) of reference image

        Returns:
        - Dict with:
          * 'selected_pts_src': (M, 2)
          * 'selected_pts_ref': (M, 2)
          * 'selected_indices': indices of chosen points from input array
          * 'grid_info': cell counts and occupancy matrix
          * 'spatial_coverage_pct': percentage of occupied grid cells
          * 'spatial_entropy': Shannon entropy of point distribution
          * 'normalized_entropy': entropy scaled to [0, 1]
          * 'comparison_conventional': spatial coverage if top-N global points were used
        """
        n_pts = len(pts_ref)
        h, w = img_shape_ref[:2]

        if n_pts == 0:
            return {
                "selected_pts_src": np.empty((0, 2), dtype=np.float32),
                "selected_pts_ref": np.empty((0, 2), dtype=np.float32),
                "selected_indices": np.array([], dtype=int),
                "spatial_coverage_pct": 0.0,
                "spatial_entropy": 0.0,
                "normalized_entropy": 0.0,
                "grid_cell_counts": np.zeros((self.grid_rows, self.grid_cols), dtype=int).tolist(),
                "occupied_cells": 0,
                "total_cells": self.grid_rows * self.grid_cols,
                "selected_count": 0
            }

        if confidences is None or len(confidences) != n_pts:
            confidences = np.ones((n_pts,), dtype=np.float32)

        # Composite quality score: high confidence, low reprojection error
        # Normalized between 0 and 1
        err_penalty = 1.0 / (1.0 + reproj_errors)
        quality_score = confidences * 0.5 + err_penalty * 0.5

        # Determine cell bounds
        cell_h = h / float(self.grid_rows)
        cell_w = w / float(self.grid_cols)

        # Assign each reference point to a grid cell (r, c)
        grid_bins: Dict[Tuple[int, int], List[int]] = {}
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                grid_bins[(r, c)] = []

        for idx in range(n_pts):
            x, y = pts_ref[idx]
            c = int(np.clip(x // cell_w, 0, self.grid_cols - 1))
            r = int(np.clip(y // cell_h, 0, self.grid_rows - 1))
            grid_bins[(r, c)].append(idx)

        # Sort indices inside each cell by quality score descending
        for cell_coord in grid_bins:
            grid_bins[cell_coord].sort(key=lambda idx: quality_score[idx], reverse=True)

        # Round 1: Select up to max_points_per_cell from every non-empty cell
        selected_indices_set = set()
        for cell_coord, idx_list in grid_bins.items():
            take_n = min(len(idx_list), self.max_points_per_cell)
            for i in range(take_n):
                selected_indices_set.add(idx_list[i])

        # Round 2: If target_total_points not yet reached, adaptively take next best points
        if len(selected_indices_set) < self.target_total_points and len(selected_indices_set) < n_pts:
            remaining_needed = self.target_total_points - len(selected_indices_set)
            remaining_indices = [i for i in range(n_pts) if i not in selected_indices_set]
            remaining_indices.sort(key=lambda idx: quality_score[idx], reverse=True)
            for idx in remaining_indices[:remaining_needed]:
                selected_indices_set.add(idx)

        selected_indices = np.array(sorted(list(selected_indices_set)), dtype=int)
        sel_src = pts_src[selected_indices]
        sel_ref = pts_ref[selected_indices]

        # Calculate grid occupancy metrics on selected tie points
        grid_counts = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
        for idx in selected_indices:
            x, y = pts_ref[idx]
            c = int(np.clip(x // cell_w, 0, self.grid_cols - 1))
            r = int(np.clip(y // cell_h, 0, self.grid_rows - 1))
            grid_counts[r, c] += 1

        total_cells = self.grid_rows * self.grid_cols
        occupied_cells = int(np.sum(grid_counts > 0))
        coverage_pct = float(occupied_cells / total_cells * 100.0)

        # Spatial Shannon Entropy
        pts_in_cells = grid_counts.ravel()[grid_counts.ravel() > 0]
        p = pts_in_cells / float(np.sum(pts_in_cells))
        entropy = float(-np.sum(p * np.log2(p))) if len(p) > 0 else 0.0
        max_possible_entropy = np.log2(total_cells)
        norm_entropy = float(entropy / max_possible_entropy) if max_possible_entropy > 0 else 0.0

        # Comparative metric: what would coverage be with greedy top-N global selection?
        top_n_global_indices = np.argsort(quality_score)[-len(selected_indices):]
        conv_grid_counts = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
        for idx in top_n_global_indices:
            x, y = pts_ref[idx]
            c = int(np.clip(x // cell_w, 0, self.grid_cols - 1))
            r = int(np.clip(y // cell_h, 0, self.grid_rows - 1))
            conv_grid_counts[r, c] += 1
        conv_occupied = int(np.sum(conv_grid_counts > 0))
        conv_coverage_pct = float(conv_occupied / total_cells * 100.0)

        return {
            "selected_pts_src": sel_src,
            "selected_pts_ref": sel_ref,
            "selected_indices": selected_indices,
            "spatial_coverage_pct": round(coverage_pct, 2),
            "spatial_entropy": round(entropy, 3),
            "normalized_entropy": round(norm_entropy, 3),
            "grid_cell_counts": grid_counts.tolist(),
            "occupied_cells": occupied_cells,
            "total_cells": total_cells,
            "selected_count": len(selected_indices),
            "conventional_comparison": {
                "conventional_coverage_pct": round(conv_coverage_pct, 2),
                "conventional_occupied_cells": conv_occupied,
                "coverage_improvement_pct": round(coverage_pct - conv_coverage_pct, 2)
            }
        }

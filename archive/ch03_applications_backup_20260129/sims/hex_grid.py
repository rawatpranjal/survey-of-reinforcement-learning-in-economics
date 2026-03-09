"""
Hexagonal Grid Utilities for DiDi Dispatch Simulation
Chapter 3: Applications of RL
Implements hex grid with axial coordinates for spatial ride-hailing simulation.
"""

import numpy as np
from typing import List, Tuple, Dict, Set


class HexGrid:
    """
    Hexagonal grid using axial coordinates (q, r).

    Axial coordinates map naturally to hex grids:
    - q: column offset
    - r: row

    Hex distance = (|q1-q2| + |q1-q2+r1-r2| + |r1-r2|) / 2
    """

    def __init__(self, radius: int = 2):
        """
        Generate hex grid with given radius from center.

        Radius 0: 1 zone (center only)
        Radius 1: 7 zones
        Radius 2: 19 zones
        Radius 3: 37 zones

        Parameters
        ----------
        radius : int
            Grid radius (number of rings around center)
        """
        self.radius = radius
        self.zones = self._generate_hex_grid(radius)
        self.num_zones = len(self.zones)
        self.zone_to_idx = {zone: i for i, zone in enumerate(self.zones)}
        self.idx_to_zone = {i: zone for i, zone in enumerate(self.zones)}
        self.distances = self._precompute_distances()
        self.neighbors = self._precompute_neighbors()

    def _generate_hex_grid(self, radius: int) -> List[Tuple[int, int]]:
        """
        Generate all hex coordinates within radius of origin.

        Uses axial coordinates (q, r). A hex at (q, r) is within radius R
        if max(|q|, |r|, |q+r|) <= R.
        """
        zones = []
        for q in range(-radius, radius + 1):
            r_min = max(-radius, -q - radius)
            r_max = min(radius, -q + radius)
            for r in range(r_min, r_max + 1):
                zones.append((q, r))
        return sorted(zones)

    def _hex_distance(self, zone_a: Tuple[int, int], zone_b: Tuple[int, int]) -> int:
        """
        Compute hex distance between two axial coordinates.

        Distance formula: (|dq| + |dq + dr| + |dr|) / 2
        """
        dq = zone_a[0] - zone_b[0]
        dr = zone_a[1] - zone_b[1]
        return (abs(dq) + abs(dq + dr) + abs(dr)) // 2

    def _precompute_distances(self) -> np.ndarray:
        """Precompute pairwise distances between all zones."""
        n = self.num_zones
        distances = np.zeros((n, n), dtype=np.float32)
        for i, zone_a in enumerate(self.zones):
            for j, zone_b in enumerate(self.zones):
                distances[i, j] = self._hex_distance(zone_a, zone_b)
        return distances

    def _precompute_neighbors(self) -> Dict[int, List[int]]:
        """
        Precompute immediate neighbors (distance 1) for each zone.

        In axial coordinates, the 6 neighbors of (q, r) are:
        (q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)
        """
        neighbor_offsets = [
            (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)
        ]
        neighbors = {}
        for idx, zone in enumerate(self.zones):
            zone_neighbors = []
            for dq, dr in neighbor_offsets:
                neighbor = (zone[0] + dq, zone[1] + dr)
                if neighbor in self.zone_to_idx:
                    zone_neighbors.append(self.zone_to_idx[neighbor])
            neighbors[idx] = zone_neighbors
        return neighbors

    def distance(self, zone_a_idx: int, zone_b_idx: int) -> float:
        """
        Get precomputed distance between two zones by index.

        Parameters
        ----------
        zone_a_idx : int
            Index of first zone
        zone_b_idx : int
            Index of second zone

        Returns
        -------
        float
            Hex distance between zones
        """
        return self.distances[zone_a_idx, zone_b_idx]

    def get_neighbors(self, zone_idx: int) -> List[int]:
        """
        Get indices of neighboring zones (distance 1).

        Parameters
        ----------
        zone_idx : int
            Index of zone

        Returns
        -------
        List[int]
            Indices of neighboring zones
        """
        return self.neighbors[zone_idx]

    def zones_within_distance(self, zone_idx: int, max_dist: int) -> List[int]:
        """
        Get all zones within given distance of a zone.

        Parameters
        ----------
        zone_idx : int
            Index of center zone
        max_dist : int
            Maximum distance (inclusive)

        Returns
        -------
        List[int]
            Indices of zones within distance
        """
        return [i for i in range(self.num_zones)
                if self.distances[zone_idx, i] <= max_dist]

    def to_cartesian(self, zone_idx: int) -> Tuple[float, float]:
        """
        Convert zone index to Cartesian coordinates for visualization.

        Hex spacing: horizontal = 1.5 * size, vertical = sqrt(3) * size
        Using unit size.

        Parameters
        ----------
        zone_idx : int
            Index of zone

        Returns
        -------
        Tuple[float, float]
            (x, y) Cartesian coordinates
        """
        q, r = self.idx_to_zone[zone_idx]
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2)
        return (x, y)

    def get_zone_centers(self) -> np.ndarray:
        """
        Get Cartesian coordinates for all zone centers.

        Returns
        -------
        np.ndarray
            Shape (num_zones, 2) array of (x, y) coordinates
        """
        centers = np.zeros((self.num_zones, 2), dtype=np.float32)
        for i in range(self.num_zones):
            centers[i] = self.to_cartesian(i)
        return centers


def test_hex_grid():
    """Unit tests for HexGrid class."""
    # Test zone counts
    assert len(HexGrid(0).zones) == 1
    assert len(HexGrid(1).zones) == 7
    assert len(HexGrid(2).zones) == 19
    assert len(HexGrid(3).zones) == 37

    grid = HexGrid(2)

    # Test self-distance is zero
    for i in range(grid.num_zones):
        assert grid.distance(i, i) == 0

    # Test symmetry
    for i in range(grid.num_zones):
        for j in range(grid.num_zones):
            assert grid.distance(i, j) == grid.distance(j, i)

    # Test center has 6 neighbors
    center_idx = grid.zone_to_idx[(0, 0)]
    assert len(grid.get_neighbors(center_idx)) == 6

    # Test neighbor distances are all 1
    for neighbor_idx in grid.get_neighbors(center_idx):
        assert grid.distance(center_idx, neighbor_idx) == 1

    # Test maximum distance in radius-2 grid is 4
    max_dist = grid.distances.max()
    assert max_dist == 4

    # Test zones_within_distance
    center_idx = grid.zone_to_idx[(0, 0)]
    zones_dist_1 = grid.zones_within_distance(center_idx, 1)
    assert len(zones_dist_1) == 7  # center + 6 neighbors

    print("All hex grid tests passed.")
    print(f"Radius 2 grid: {grid.num_zones} zones")
    print(f"Distance matrix shape: {grid.distances.shape}")
    print(f"Max distance: {max_dist}")


if __name__ == "__main__":
    test_hex_grid()

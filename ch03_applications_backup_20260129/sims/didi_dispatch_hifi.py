"""
High-Fidelity DiDi Dispatch Environment
Chapter 3: Applications of RL
Implements ride-hailing dispatch as Gymnasium environment following Qin et al. (2020).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from scipy.optimize import linear_sum_assignment

from hex_grid import HexGrid


@dataclass
class Driver:
    """Driver state representation."""
    zone: int
    status: int  # 0=idle, 1=en_route_to_pickup, 2=serving
    time_until_free: int  # Timesteps until idle again
    assigned_order: Optional[int] = None


@dataclass
class Order:
    """Order (ride request) representation."""
    origin: int
    destination: int
    fare: float
    wait_time: int  # Timesteps waiting for pickup
    max_wait: int  # Maximum wait before cancellation


class DiDiDispatchEnv(gym.Env):
    """
    High-fidelity ride-hailing dispatch environment.

    Follows the semi-MDP formulation from Qin et al. (2020) "Ride-Hailing
    Order Dispatching at DiDi via Reinforcement Learning."

    State: Driver locations/status + pending orders + time
    Action: Selection among K matching variants (discrete)
    Reward: Fare revenue - pickup costs - idle penalties - cancellation penalties
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_drivers: int = 20,
        grid_radius: int = 2,
        max_batch_size: int = 30,
        episode_length: int = 288,  # 24h with 5-min timesteps
        gamma: float = 0.99,
        fare_per_km: float = 2.0,
        base_fare: float = 3.0,
        pickup_cost_per_km: float = 0.5,
        idle_penalty: float = 0.1,
        cancel_penalty: float = 5.0,
        base_demand_rate: float = 0.3,
        max_wait_timesteps: int = 3,
        speed_km_per_timestep: float = 2.0,  # ~24 km/h with 5-min timesteps
        km_per_hex: float = 1.0,
        num_matching_variants: int = 5,
        seed: Optional[int] = None,
    ):
        """
        Initialize the dispatch environment.

        Parameters
        ----------
        num_drivers : int
            Number of drivers in the fleet
        grid_radius : int
            Hex grid radius (radius 2 = 19 zones)
        max_batch_size : int
            Maximum orders per batch
        episode_length : int
            Timesteps per episode
        gamma : float
            Discount factor
        fare_per_km : float
            Revenue per km of trip
        base_fare : float
            Base fare per trip
        pickup_cost_per_km : float
            Cost per km for pickup
        idle_penalty : float
            Penalty per idle driver per timestep
        cancel_penalty : float
            Penalty per cancelled order
        base_demand_rate : float
            Base Poisson rate per zone per timestep
        max_wait_timesteps : int
            Maximum wait before order cancels
        speed_km_per_timestep : float
            Driver speed in km per timestep
        km_per_hex : float
            Distance in km per hex unit
        num_matching_variants : int
            Number of discrete matching actions
        seed : int, optional
            Random seed
        """
        super().__init__()

        self.num_drivers = num_drivers
        self.grid = HexGrid(grid_radius)
        self.num_zones = self.grid.num_zones
        self.max_batch_size = max_batch_size
        self.episode_length = episode_length
        self.gamma = gamma

        # Economic parameters
        self.fare_per_km = fare_per_km
        self.base_fare = base_fare
        self.pickup_cost_per_km = pickup_cost_per_km
        self.idle_penalty = idle_penalty
        self.cancel_penalty = cancel_penalty
        self.km_per_hex = km_per_hex
        self.speed_km_per_timestep = speed_km_per_timestep

        # Demand parameters
        self.base_demand_rate = base_demand_rate
        self.max_wait_timesteps = max_wait_timesteps

        # Action space: select among K matching variants
        self.num_matching_variants = num_matching_variants
        self.action_space = spaces.Discrete(num_matching_variants)

        # Observation space
        # Driver features: zone, time_bucket, status, time_until_free (N x 4)
        # Order features: origin, dest, wait_time (M x 3, padded)
        # Global: time_bucket, num_idle, num_pending
        driver_dim = num_drivers * 4
        order_dim = max_batch_size * 3
        global_dim = 3
        self.obs_dim = driver_dim + order_dim + global_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # State variables
        self.drivers: List[Driver] = []
        self.orders: List[Order] = []
        self.timestep = 0
        self.total_revenue = 0.0
        self.total_cancellations = 0

        # Demand pattern: higher in center, peaks at rush hours
        self._init_demand_pattern()

        # Random state
        self._np_random = None
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

    def _init_demand_pattern(self):
        """Initialize spatiotemporal demand pattern."""
        # Spatial: higher demand near center
        center_idx = self.grid.zone_to_idx[(0, 0)]
        self.zone_demand_weights = np.zeros(self.num_zones)
        for i in range(self.num_zones):
            dist_to_center = self.grid.distance(i, center_idx)
            self.zone_demand_weights[i] = 1.0 / (1.0 + 0.3 * dist_to_center)
        self.zone_demand_weights /= self.zone_demand_weights.sum()

        # Temporal: rush hours at 8am (96) and 6pm (216) for 5-min buckets
        self.time_demand_multiplier = np.ones(self.episode_length)
        for t in range(self.episode_length):
            # Morning rush: peak at t=96 (8am)
            morning = 1.5 * np.exp(-((t - 96) ** 2) / (2 * 20 ** 2))
            # Evening rush: peak at t=216 (6pm)
            evening = 1.8 * np.exp(-((t - 216) ** 2) / (2 * 25 ** 2))
            self.time_demand_multiplier[t] = 1.0 + morning + evening

    def _get_rng(self) -> np.random.Generator:
        """Get random number generator."""
        if self._np_random is None:
            self._np_random = np.random.default_rng()
        return self._np_random

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Parameters
        ----------
        seed : int, optional
            Random seed
        options : dict, optional
            Additional options

        Returns
        -------
        observation : np.ndarray
            Initial observation
        info : dict
            Additional information
        """
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        rng = self._get_rng()

        # Initialize drivers uniformly across zones
        self.drivers = []
        for _ in range(self.num_drivers):
            zone = rng.integers(0, self.num_zones)
            self.drivers.append(Driver(zone=zone, status=0, time_until_free=0))

        # Clear orders
        self.orders = []

        # Reset counters
        self.timestep = 0
        self.total_revenue = 0.0
        self.total_cancellations = 0

        # Generate initial batch of orders
        self._generate_orders()

        obs = self._get_observation()
        info = {"timestep": 0, "num_orders": len(self.orders)}

        return obs, info

    def _generate_orders(self):
        """Generate new orders for current timestep using Poisson process."""
        rng = self._get_rng()

        # Demand rate varies by time
        time_mult = self.time_demand_multiplier[min(self.timestep, self.episode_length - 1)]

        for zone in range(self.num_zones):
            # Zone-specific rate
            rate = (self.base_demand_rate *
                    self.zone_demand_weights[zone] *
                    self.num_zones *
                    time_mult)

            # Sample number of orders from Poisson
            num_orders = rng.poisson(rate)

            for _ in range(num_orders):
                if len(self.orders) >= self.max_batch_size:
                    break

                # Destination: biased toward center
                dest_probs = self.zone_demand_weights.copy()
                dest_probs[zone] *= 0.5  # Less likely same zone
                dest_probs /= dest_probs.sum()
                dest = rng.choice(self.num_zones, p=dest_probs)

                # Fare based on trip distance
                trip_dist = self.grid.distance(zone, dest) * self.km_per_hex
                fare = self.base_fare + self.fare_per_km * trip_dist

                order = Order(
                    origin=zone,
                    destination=dest,
                    fare=fare,
                    wait_time=0,
                    max_wait=self.max_wait_timesteps
                )
                self.orders.append(order)

    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector.

        Layout:
        - Driver features: [zone/num_zones, timestep/episode_len, status/2, time_until_free/10]
        - Order features: [origin/num_zones, dest/num_zones, wait_time/max_wait] (padded)
        - Global: [timestep/episode_len, num_idle/num_drivers, num_pending/max_batch]
        """
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        idx = 0

        # Driver features (normalized)
        for driver in self.drivers:
            obs[idx] = driver.zone / max(1, self.num_zones - 1)
            obs[idx + 1] = self.timestep / max(1, self.episode_length - 1)
            obs[idx + 2] = driver.status / 2.0
            obs[idx + 3] = min(driver.time_until_free, 10) / 10.0
            idx += 4

        # Order features (normalized, padded)
        for i in range(self.max_batch_size):
            if i < len(self.orders):
                order = self.orders[i]
                obs[idx] = order.origin / max(1, self.num_zones - 1)
                obs[idx + 1] = order.destination / max(1, self.num_zones - 1)
                obs[idx + 2] = order.wait_time / max(1, self.max_wait_timesteps)
            else:
                obs[idx] = 0
                obs[idx + 1] = 0
                obs[idx + 2] = 0
            idx += 3

        # Global features
        num_idle = sum(1 for d in self.drivers if d.status == 0)
        obs[idx] = self.timestep / max(1, self.episode_length - 1)
        obs[idx + 1] = num_idle / self.num_drivers
        obs[idx + 2] = len(self.orders) / self.max_batch_size

        return obs

    def _compute_matching_variants(self) -> List[List[Tuple[int, int]]]:
        """
        Compute K different matching variants for action selection.

        Variants:
        0: Optimal Hungarian matching (minimize pickup - fare)
        1: Greedy nearest driver
        2: Greedy highest fare
        3: Optimal with noise (exploration)
        4: Random valid matching
        """
        idle_drivers = [(i, d) for i, d in enumerate(self.drivers) if d.status == 0]

        if not idle_drivers or not self.orders:
            return [[] for _ in range(self.num_matching_variants)]

        rng = self._get_rng()
        variants = []

        # Build cost matrix
        n_drivers = len(idle_drivers)
        n_orders = len(self.orders)
        cost = np.zeros((n_drivers, n_orders))

        for di, (driver_idx, driver) in enumerate(idle_drivers):
            for oi, order in enumerate(self.orders):
                pickup_dist = self.grid.distance(driver.zone, order.origin) * self.km_per_hex
                pickup_cost = pickup_dist * self.pickup_cost_per_km
                cost[di, oi] = pickup_cost - order.fare  # Minimize cost = maximize value

        # Variant 0: Optimal Hungarian
        if n_drivers > 0 and n_orders > 0:
            row_ind, col_ind = linear_sum_assignment(cost)
            matching_0 = [(idle_drivers[r][0], col_ind[i])
                          for i, r in enumerate(row_ind)
                          if col_ind[i] < n_orders]
            variants.append(matching_0)
        else:
            variants.append([])

        # Variant 1: Greedy nearest driver
        matching_1 = self._greedy_nearest_matching(idle_drivers, self.orders)
        variants.append(matching_1)

        # Variant 2: Greedy highest fare
        matching_2 = self._greedy_fare_matching(idle_drivers, self.orders)
        variants.append(matching_2)

        # Variant 3: Optimal with noise
        if n_drivers > 0 and n_orders > 0:
            noisy_cost = cost + rng.normal(0, 0.5, cost.shape)
            row_ind, col_ind = linear_sum_assignment(noisy_cost)
            matching_3 = [(idle_drivers[r][0], col_ind[i])
                          for i, r in enumerate(row_ind)
                          if col_ind[i] < n_orders]
            variants.append(matching_3)
        else:
            variants.append([])

        # Variant 4: Random valid matching
        matching_4 = self._random_matching(idle_drivers, self.orders, rng)
        variants.append(matching_4)

        return variants

    def _greedy_nearest_matching(
        self,
        idle_drivers: List[Tuple[int, Driver]],
        orders: List[Order]
    ) -> List[Tuple[int, int]]:
        """Match orders to nearest available driver greedily."""
        matching = []
        assigned_drivers = set()
        assigned_orders = set()

        # Sort orders by wait time (prioritize older orders)
        order_indices = sorted(range(len(orders)), key=lambda i: -orders[i].wait_time)

        for oi in order_indices:
            if oi in assigned_orders:
                continue
            order = orders[oi]

            best_driver = None
            best_dist = float('inf')

            for di, (driver_idx, driver) in enumerate(idle_drivers):
                if driver_idx in assigned_drivers:
                    continue
                dist = self.grid.distance(driver.zone, order.origin)
                if dist < best_dist:
                    best_dist = dist
                    best_driver = driver_idx

            if best_driver is not None:
                matching.append((best_driver, oi))
                assigned_drivers.add(best_driver)
                assigned_orders.add(oi)

        return matching

    def _greedy_fare_matching(
        self,
        idle_drivers: List[Tuple[int, Driver]],
        orders: List[Order]
    ) -> List[Tuple[int, int]]:
        """Match highest fare orders first to nearest driver."""
        matching = []
        assigned_drivers = set()
        assigned_orders = set()

        # Sort orders by fare descending
        order_indices = sorted(range(len(orders)), key=lambda i: -orders[i].fare)

        for oi in order_indices:
            if oi in assigned_orders:
                continue
            order = orders[oi]

            best_driver = None
            best_dist = float('inf')

            for di, (driver_idx, driver) in enumerate(idle_drivers):
                if driver_idx in assigned_drivers:
                    continue
                dist = self.grid.distance(driver.zone, order.origin)
                if dist < best_dist:
                    best_dist = dist
                    best_driver = driver_idx

            if best_driver is not None:
                matching.append((best_driver, oi))
                assigned_drivers.add(best_driver)
                assigned_orders.add(oi)

        return matching

    def _random_matching(
        self,
        idle_drivers: List[Tuple[int, Driver]],
        orders: List[Order],
        rng: np.random.Generator
    ) -> List[Tuple[int, int]]:
        """Random valid matching."""
        matching = []
        driver_indices = [d[0] for d in idle_drivers]
        order_indices = list(range(len(orders)))

        rng.shuffle(driver_indices)
        rng.shuffle(order_indices)

        num_matches = min(len(driver_indices), len(order_indices))
        for i in range(num_matches):
            matching.append((driver_indices[i], order_indices[i]))

        return matching

    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep.

        Parameters
        ----------
        action : int
            Index of matching variant to use

        Returns
        -------
        observation : np.ndarray
            New observation
        reward : float
            Reward for this step
        terminated : bool
            Whether episode ended
        truncated : bool
            Whether episode was truncated
        info : dict
            Additional information
        """
        reward = 0.0
        info = {}

        # Get matching variants and select one
        variants = self._compute_matching_variants()
        action = min(action, len(variants) - 1)
        matching = variants[action]

        # Execute matching
        matched_orders = set()
        for driver_idx, order_idx in matching:
            if order_idx >= len(self.orders):
                continue

            driver = self.drivers[driver_idx]
            order = self.orders[order_idx]

            # Compute trip parameters
            pickup_dist = self.grid.distance(driver.zone, order.origin) * self.km_per_hex
            trip_dist = self.grid.distance(order.origin, order.destination) * self.km_per_hex

            # Time calculations
            pickup_time = max(1, int(np.ceil(pickup_dist / self.speed_km_per_timestep)))
            trip_time = max(1, int(np.ceil(trip_dist / self.speed_km_per_timestep)))

            # Update driver
            driver.status = 1  # En route to pickup
            driver.time_until_free = pickup_time + trip_time
            driver.assigned_order = order_idx
            driver.zone = order.destination  # Will end at destination

            # Compute reward
            fare_revenue = order.fare
            pickup_cost = pickup_dist * self.pickup_cost_per_km
            reward += fare_revenue - pickup_cost
            self.total_revenue += fare_revenue

            matched_orders.add(order_idx)

        # Update unmatched orders (increment wait time, check cancellation)
        new_orders = []
        cancellations = 0
        for i, order in enumerate(self.orders):
            if i in matched_orders:
                continue
            order.wait_time += 1
            if order.wait_time >= order.max_wait:
                cancellations += 1
                reward -= self.cancel_penalty
            else:
                new_orders.append(order)

        self.orders = new_orders
        self.total_cancellations += cancellations

        # Update driver states
        for driver in self.drivers:
            if driver.time_until_free > 0:
                driver.time_until_free -= 1
                if driver.time_until_free == 0:
                    driver.status = 0
                    driver.assigned_order = None
                elif driver.status == 1 and driver.time_until_free <= 2:
                    # Transition from pickup to serving
                    driver.status = 2

        # Idle penalty
        num_idle = sum(1 for d in self.drivers if d.status == 0)
        reward -= self.idle_penalty * num_idle

        # Advance time and generate new orders
        self.timestep += 1
        self._generate_orders()

        # Check termination
        terminated = self.timestep >= self.episode_length
        truncated = False

        # Observation
        obs = self._get_observation()

        info.update({
            "timestep": self.timestep,
            "num_orders": len(self.orders),
            "num_idle": num_idle,
            "num_matches": len(matching),
            "cancellations": cancellations,
            "total_revenue": self.total_revenue,
            "total_cancellations": self.total_cancellations,
        })

        return obs, reward, terminated, truncated, info


# Baseline heuristics as standalone policies


def nearest_driver_policy(env: DiDiDispatchEnv) -> int:
    """Always select nearest-driver matching (action 1)."""
    return 1


def greedy_fare_policy(env: DiDiDispatchEnv) -> int:
    """Always select highest-fare matching (action 2)."""
    return 2


def random_policy(env: DiDiDispatchEnv) -> int:
    """Random action selection."""
    return env._get_rng().integers(0, env.num_matching_variants)


def optimal_hungarian_policy(env: DiDiDispatchEnv) -> int:
    """Always select optimal Hungarian matching (action 0)."""
    return 0


def evaluate_policy(
    env: DiDiDispatchEnv,
    policy_fn,
    num_episodes: int = 10,
    seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate a policy over multiple episodes.

    Parameters
    ----------
    env : DiDiDispatchEnv
        Environment instance
    policy_fn : callable
        Policy function: env -> action
    num_episodes : int
        Number of evaluation episodes
    seed : int
        Base random seed

    Returns
    -------
    dict
        Evaluation metrics
    """
    total_rewards = []
    total_revenues = []
    total_cancels = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        episode_reward = 0.0

        while True:
            action = policy_fn(env)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)
        total_revenues.append(info["total_revenue"])
        total_cancels.append(info["total_cancellations"])

    return {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "mean_revenue": np.mean(total_revenues),
        "mean_cancellations": np.mean(total_cancels),
    }


def test_environment():
    """Smoke test for DiDiDispatchEnv."""
    print("Testing DiDiDispatchEnv...")

    env = DiDiDispatchEnv(
        num_drivers=10,
        grid_radius=2,
        max_batch_size=20,
        episode_length=50,
        seed=42
    )

    # Test reset
    obs, info = env.reset(seed=42)
    assert obs.shape == (env.obs_dim,), f"Obs shape: {obs.shape}"
    print(f"Observation dimension: {env.obs_dim}")

    # Test step
    total_reward = 0.0
    for t in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"Episode completed: {info['timestep']} steps")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Total revenue: {info['total_revenue']:.2f}")
    print(f"Total cancellations: {info['total_cancellations']}")

    # Test baseline policies
    print("\nEvaluating baseline policies (5 episodes each)...")

    env = DiDiDispatchEnv(num_drivers=20, episode_length=100, seed=42)

    policies = {
        "Hungarian": optimal_hungarian_policy,
        "Nearest": nearest_driver_policy,
        "Greedy Fare": greedy_fare_policy,
        "Random": random_policy,
    }

    for name, policy in policies.items():
        results = evaluate_policy(env, policy, num_episodes=5, seed=42)
        print(f"{name:12s}: reward={results['mean_reward']:8.1f} +/- {results['std_reward']:6.1f}, "
              f"revenue={results['mean_revenue']:7.1f}, cancels={results['mean_cancellations']:.1f}")

    print("\nAll environment tests passed.")


if __name__ == "__main__":
    test_environment()

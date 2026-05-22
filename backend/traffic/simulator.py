"""
Programmable traffic simulation engine
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrafficIncident:
    """Represents a traffic incident"""

    edge_id: Tuple[int, int]
    start_time: float
    end_time: float
    congestion_factor: float
    severity: float = 1.0


class TrafficSimulator:
    """
    Programmable traffic simulation engine

    Simulates various traffic scenarios:
    - Time-varying congestion (peak hours)
    - Incidents (accidents, road works)
    - Weather conditions
    - Regional congestion patterns
    """

    def __init__(self, base_distance_matrix: np.ndarray):
        """
        Initialize traffic simulator

        Args:
            base_distance_matrix: Base distance matrix without traffic
        """
        self.base_distances = base_distance_matrix.copy()
        self.dimension = len(base_distance_matrix)
        self.scenarios = {}
        self.current_scenario = None
        self.current_time = 0.0
        self.history = []

    def create_scenario(self, name: str, config: Dict) -> "TrafficScenario":
        """
        Create a new traffic scenario

        Args:
            name: Scenario name
            config: Scenario configuration

        Returns:
            TrafficScenario instance
        """
        scenario_type = config.get("type", "custom")

        if scenario_type == "time_varying":
            scenario = TimeVaryingScenario(name, config)
        elif scenario_type == "incident":
            scenario = IncidentScenario(name, config)
        elif scenario_type == "weather":
            scenario = WeatherScenario(name, config)
        else:
            scenario = TrafficScenario(name, config)

        self.scenarios[name] = scenario
        return scenario

    def activate_scenario(self, scenario_name: str):
        """Activate a traffic scenario"""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")

        self.current_scenario = self.scenarios[scenario_name]
        logger.info(f"Activated traffic scenario: {scenario_name}")

    def get_edge_weights(self, time: float = None) -> np.ndarray:
        """
        Get edge weight matrix at current time with traffic applied

        Args:
            time: Simulation time (uses current_time if None)

        Returns:
            Modified distance matrix with traffic
        """
        if time is not None:
            self.current_time = time

        weights = self.base_distances.copy()

        if self.current_scenario:
            weights = self.current_scenario.apply(
                weights, self.current_time, self.base_distances
            )

        return weights

    def update_time(self, time: float):
        """Update simulation time"""
        self.current_time = time

    def add_incident(self, incident: TrafficIncident):
        """Add incident to current scenario"""
        if isinstance(self.current_scenario, IncidentScenario):
            self.current_scenario.add_incident(incident)

    def remove_incident(self, edge_id: Tuple[int, int]):
        """Remove incident from current scenario"""
        if isinstance(self.current_scenario, IncidentScenario):
            self.current_scenario.remove_incident(edge_id)

    def log_event(self, event: Dict):
        """Log a traffic event for history"""
        self.history.append({**event, "timestamp": self.current_time})


class TrafficScenario:
    """Base traffic scenario class"""

    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config

    def apply(
        self, weights: np.ndarray, time: float, base_weights: np.ndarray
    ) -> np.ndarray:
        """Apply traffic effects to weights"""
        return weights.copy()


class TimeVaryingScenario(TrafficScenario):
    """Time-varying congestion scenario (peak hours, etc.)"""

    def __init__(self, name: str, config: Dict):
        super().__init__(name, config)
        self.peak_hours = config.get("peak_hours", [(8, 10), (17, 19)])
        self.congestion_multiplier = config.get("congestion_multiplier", 1.8)
        self.base_multiplier = config.get("base_multiplier", 1.0)

    def apply(
        self, weights: np.ndarray, time: float, base_weights: np.ndarray
    ) -> np.ndarray:
        """Apply time-varying congestion"""
        weights = weights.copy()

        # Get hour from time (assuming time in seconds, simulation period = 86400 seconds)
        hour = (time % 86400) / 3600

        # Check if in peak hours
        in_peak = False
        for start, end in self.peak_hours:
            if start <= hour < end:
                in_peak = True
                break

        multiplier = self.congestion_multiplier if in_peak else self.base_multiplier

        # Apply multiplier to all edges
        weights = base_weights * multiplier

        return weights


class IncidentScenario(TrafficScenario):
    """Incident-based congestion (accidents, road works)"""

    def __init__(self, name: str, config: Dict):
        super().__init__(name, config)
        self.incidents: Dict[Tuple[int, int], TrafficIncident] = {}

        # Add initial incidents from config
        for incident_config in config.get("incidents", []):
            incident = TrafficIncident(
                edge_id=tuple(incident_config["edge_id"]),
                start_time=incident_config.get("start_time", 0),
                end_time=incident_config.get("end_time", float("inf")),
                congestion_factor=incident_config.get("congestion_factor", 2.0),
                severity=incident_config.get("severity", 1.0),
            )
            self.incidents[incident.edge_id] = incident

    def add_incident(self, incident: TrafficIncident):
        """Add a traffic incident"""
        self.incidents[incident.edge_id] = incident

    def remove_incident(self, edge_id: Tuple[int, int]):
        """Remove a traffic incident"""
        if edge_id in self.incidents:
            del self.incidents[edge_id]

    def apply(
        self, weights: np.ndarray, time: float, base_weights: np.ndarray
    ) -> np.ndarray:
        """Apply incident-based congestion"""
        weights = base_weights.copy()

        for edge_id, incident in self.incidents.items():
            # Check if incident is active
            if incident.start_time <= time <= incident.end_time:
                i, j = edge_id
                # Apply congestion to both directions
                weights[i][j] *= incident.congestion_factor
                weights[j][i] *= incident.congestion_factor

        return weights


class WeatherScenario(TrafficScenario):
    """Weather-based congestion"""

    def __init__(self, name: str, config: Dict):
        super().__init__(name, config)
        self.regions = config.get("regions", [])
        self.global_multiplier = config.get("global_multiplier", 1.0)

    def apply(
        self, weights: np.ndarray, time: float, base_weights: np.ndarray
    ) -> np.ndarray:
        """Apply weather-based congestion"""
        weights = base_weights.copy() * self.global_multiplier

        # Apply regional effects
        for region in self.regions:
            bounds = region["bounds"]
            multiplier = region.get("multiplier", 1.2)

            # Apply multiplier to edges within region
            # (simplified: assumes edge list available from weights)
            # In real implementation, would need node coordinates

        return weights


class AdaptiveTrafficSimulator:
    """
    Adaptive traffic simulator that learns and adjusts patterns
    """

    def __init__(self, base_simulator: TrafficSimulator):
        self.simulator = base_simulator
        self.congestion_history = []
        self.max_history = 1000

    def record_congestion(self, edge_id: Tuple[int, int], congestion_level: float):
        """Record observed congestion level"""
        self.congestion_history.append(
            {
                "edge": edge_id,
                "congestion": congestion_level,
                "time": self.simulator.current_time,
            }
        )

        if len(self.congestion_history) > self.max_history:
            self.congestion_history.pop(0)

    def get_predicted_weights(self, future_time: float) -> np.ndarray:
        """Get predicted weights at future time"""
        # Simple prediction: linear interpolation from history
        # In production: use ML models (LSTM, Prophet, etc.)
        return self.simulator.get_edge_weights(future_time)

    def update_scenario_from_data(self, observed_weights: np.ndarray):
        """Update scenario parameters based on observed data"""
        # Compare observed vs base weights to infer congestion factors
        if self.simulator.current_scenario:
            multipliers = observed_weights / self.simulator.base_distances
            avg_multiplier = np.nanmean(multipliers[~np.isinf(multipliers)])

            # Update scenario parameters
            if hasattr(self.simulator.current_scenario, "congestion_multiplier"):
                self.simulator.current_scenario.congestion_multiplier = avg_multiplier

"""
Unit tests for quantum optimizer
"""
import pytest
import numpy as np
from quantum.optimizer import QuantumRouteOptimizer


@pytest.fixture
def sample_graph_data():
    """Create sample graph data for testing"""
    num_nodes = 6
    distance_matrix = np.array(
        [
            [0, 10, 20, 30, 40, 50],
            [10, 0, 15, 25, 35, 45],
            [20, 15, 0, 10, 20, 30],
            [30, 25, 10, 0, 15, 25],
            [40, 35, 20, 15, 0, 10],
            [50, 45, 30, 25, 10, 0],
        ]
    )

    laplacian = np.array(
        [
            [5, -1, -1, -1, -1, -1],
            [-1, 5, -1, -1, -1, -1],
            [-1, -1, 5, -1, -1, -1],
            [-1, -1, -1, 5, -1, -1],
            [-1, -1, -1, -1, 5, -1],
            [-1, -1, -1, -1, -1, 5],
        ]
    )

    return {
        "name": "test_data",
        "dimension": num_nodes,
        "num_customers": num_nodes - 1,
        "distance_matrix": distance_matrix,
        "laplacian": laplacian,
        "demands": np.array([10, 20, 15, 25, 30]),
        "capacity": 100,
    }


def test_optimizer_initialization():
    """Test optimizer initialization"""
    optimizer = QuantumRouteOptimizer()
    assert optimizer.quantum_time == 5.0
    assert optimizer.num_iterations == 10
    assert optimizer.num_qubits == 20


def test_quantum_walk(sample_graph_data):
    """Test quantum walk"""
    optimizer = QuantumRouteOptimizer()
    laplacian = sample_graph_data["laplacian"]
    distance_matrix = sample_graph_data["distance_matrix"]

    amplitudes = optimizer._quantum_walk(laplacian, distance_matrix)

    assert len(amplitudes) == laplacian.shape[0]
    assert np.all(amplitudes >= 0)
    assert np.isclose(np.sum(amplitudes), 1.0, atol=0.1)  # Roughly normalized


def test_decode_routes(sample_graph_data):
    """Test route decoding"""
    optimizer = QuantumRouteOptimizer()
    amplitudes = np.array([0.3, 0.2, 0.2, 0.15, 0.1, 0.05])

    routes = optimizer._decode_routes(
        amplitudes,
        sample_graph_data["dimension"],
        vehicle_count=2,
        demands=sample_graph_data["demands"],
        capacity=sample_graph_data["capacity"],
    )

    # All routes should start and end with depot (0)
    for route in routes:
        assert route[0] == 0
        assert route[-1] == 0

    # Routes should not be empty
    assert len(routes) > 0
    assert len(routes) <= 2


def test_optimize(sample_graph_data):
    """Test full optimization"""
    optimizer = QuantumRouteOptimizer()
    parameters = {"quantum_time": 2.0, "iterations": 2}

    routes, metrics = optimizer.optimize(sample_graph_data, vehicle_count=2, parameters=parameters)

    assert isinstance(routes, list)
    assert isinstance(metrics, dict)
    assert "total_distance" in metrics
    assert "vehicle_loads" in metrics
    assert metrics["total_distance"] > 0
    assert len(routes) <= 2


def test_route_distance_calculation(sample_graph_data):
    """Test distance calculation"""
    optimizer = QuantumRouteOptimizer()
    routes = [[0, 1, 2, 0], [0, 3, 4, 0]]
    distance_matrix = sample_graph_data["distance_matrix"]

    total_dist = optimizer._calculate_route_distance(routes, distance_matrix)

    expected = 10 + 15 + 20 + 30 + 15 + 40  # Route distances
    assert total_dist == expected


def test_two_opt_improvement(sample_graph_data):
    """Test 2-opt local search"""
    optimizer = QuantumRouteOptimizer()
    original_route = [0, 1, 4, 2, 3, 0]
    distance_matrix = sample_graph_data["distance_matrix"]

    improved_route = optimizer._two_opt(original_route, distance_matrix)

    # Route should still start and end with depot
    assert improved_route[0] == 0
    assert improved_route[-1] == 0

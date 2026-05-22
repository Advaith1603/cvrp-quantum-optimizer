"""
API routes for CVRP optimization and traffic management
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
from datetime import datetime

from app.services import OptimizationService, TrafficService
from app.schemas import (
    OptimizeRequest,
    OptimizeResponse,
    TrafficScenarioRequest,
    TrafficScenarioResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["optimization"])

optimization_service = OptimizationService()
traffic_service = TrafficService()


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
    }


@router.post("/optimize", response_model=OptimizeResponse)
async def optimize_route(request: OptimizeRequest, background_tasks: BackgroundTasks):
    """
    Optimize routes using CTQW algorithm with traffic adaptation

    Args:
        request: Optimization request with dataset, parameters, and traffic scenario

    Returns:
        Optimized routes with metrics
    """
    try:
        logger.info(f"Optimization request: {request.dataset}")

        # Load dataset
        dataset = optimization_service.load_dataset(request.dataset)

        # Get traffic weights
        traffic_weights = None
        if request.traffic_scenario:
            traffic_weights = traffic_service.get_scenario_weights(
                request.traffic_scenario, request.adaptation
            )

        # Run optimization
        result = optimization_service.optimize(
            dataset=dataset,
            vehicle_count=request.vehicle_count,
            traffic_weights=traffic_weights,
            parameters=request.parameters,
        )

        # Log result
        background_tasks.add_task(
            optimization_service.log_optimization,
            dataset=request.dataset,
            result=result,
        )

        return OptimizeResponse(**result)

    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/datasets")
async def list_datasets():
    """List available CVRP datasets"""
    try:
        datasets = optimization_service.get_available_datasets()
        return {"datasets": datasets}
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/datasets/{dataset_name}")
async def get_dataset_info(dataset_name: str):
    """Get information about a specific dataset"""
    try:
        info = optimization_service.get_dataset_info(dataset_name)
        return info
    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/traffic/scenarios", response_model=TrafficScenarioResponse)
async def create_traffic_scenario(request: TrafficScenarioRequest):
    """Create a custom traffic scenario"""
    try:
        logger.info(f"Creating traffic scenario: {request.name}")
        scenario = traffic_service.create_scenario(request)
        return TrafficScenarioResponse(**scenario)
    except Exception as e:
        logger.error(f"Error creating scenario: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/traffic/scenarios")
async def list_traffic_scenarios():
    """List available traffic scenarios"""
    try:
        scenarios = traffic_service.get_available_scenarios()
        return {"scenarios": scenarios}
    except Exception as e:
        logger.error(f"Error listing scenarios: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/traffic/scenarios/{scenario_name}")
async def get_scenario(scenario_name: str):
    """Get traffic scenario details"""
    try:
        scenario = traffic_service.get_scenario(scenario_name)
        return scenario
    except Exception as e:
        logger.error(f"Error getting scenario: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/traffic/incidents")
async def add_incident(incident_data: Dict):
    """Add a traffic incident"""
    try:
        logger.info(f"Adding traffic incident")
        result = traffic_service.add_incident(incident_data)
        return result
    except Exception as e:
        logger.error(f"Error adding incident: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/traffic/incidents/{edge_id}")
async def remove_incident(edge_id: str):
    """Remove a traffic incident"""
    try:
        result = traffic_service.remove_incident(edge_id)
        return result
    except Exception as e:
        logger.error(f"Error removing incident: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/optimize/adaptive")
async def optimize_adaptive(
    request: OptimizeRequest, background_tasks: BackgroundTasks
):
    """
    Optimize with traffic adaptation - re-optimize as traffic changes

    Args:
        request: Optimization request
        background_tasks: Background task handler

    Returns:
        Initial optimized routes, then adapts as traffic changes
    """
    try:
        logger.info("Starting adaptive optimization")

        # Initial optimization
        result = await optimize_route(request, background_tasks)

        # Set up adaptation monitoring
        background_tasks.add_task(
            optimization_service.monitor_traffic_adaptation,
            dataset=request.dataset,
            scenario=request.traffic_scenario,
        )

        return result

    except Exception as e:
        logger.error(f"Adaptive optimization error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/results")
async def get_results(
    limit: int = 10, offset: int = 0, dataset: Optional[str] = None
):
    """Get optimization results history"""
    try:
        results = optimization_service.get_results_history(
            limit=limit, offset=offset, dataset=dataset
        )
        return results
    except Exception as e:
        logger.error(f"Error getting results: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/results/{result_id}")
async def get_result_detail(result_id: str):
    """Get detailed result information"""
    try:
        result = optimization_service.get_result_detail(result_id)
        return result
    except Exception as e:
        logger.error(f"Error getting result: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/compare")
async def compare_scenarios(scenarios: List[OptimizeRequest]):
    """Compare optimization results across multiple scenarios"""
    try:
        logger.info(f"Comparing {len(scenarios)} scenarios")
        results = optimization_service.compare_scenarios(scenarios)
        return results
    except Exception as e:
        logger.error(f"Error comparing scenarios: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

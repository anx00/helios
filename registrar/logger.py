"""
Helios Weather Lab - Logger (Physics Engine)
Logs physics predictions to database and console.
"""

from datetime import datetime
from typing import Optional

from database import insert_prediction
from config import STATIONS
from synthesizer import PhysicsPrediction, format_physics_log


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return round((celsius * 9/5) + 32, 1)


def log_prediction(
    station_id: str,
    timestamp: datetime,
    temp_actual_c: float,
    temp_hrrr_hourly_c: float,
    soil_moisture: Optional[float],
    radiation: Optional[float],
    sky_condition: str,
    wind_dir: Optional[int],
    prediction: PhysicsPrediction,
    target_date: Optional[str] = None,
) -> int:
    """
    Log a physics prediction to the database and console.
    
    Returns:
        The record ID
    """
    # Convert to Fahrenheit for storage
    temp_actual_f = celsius_to_fahrenheit(temp_actual_c)
    temp_hrrr_hourly_f = celsius_to_fahrenheit(temp_hrrr_hourly_c)
    
    # Prepare data for database
    data = {
        "timestamp": timestamp.isoformat(),
        "station_id": station_id,
        "temp_actual_f": temp_actual_f,
        "temp_hrrr_hourly_f": temp_hrrr_hourly_f,
        "soil_moisture": soil_moisture,
        "radiation": radiation,
        "sky_condition": sky_condition,
        "wind_dir": wind_dir,
        "hrrr_max_raw_f": prediction.hrrr_max_raw_f,
        "current_deviation_f": prediction.current_deviation_f,
        "delta_weight": prediction.delta_weight,
        "physics_adjustment_f": prediction.physics_adjustment_f,
        "physics_reason": prediction.physics_reason,
        "final_prediction_f": prediction.final_prediction_f,
        "target_date": target_date,
        "velocity_ratio": getattr(prediction, 'velocity_ratio', None),
        "aod_550nm": getattr(prediction, 'aod_550nm', None),
        "sst_delta_c": getattr(prediction, 'sst_delta_c', None),
        "advection_adj_c": getattr(prediction, 'advection_adj_c', None),
        "verified_floor_f": getattr(prediction, 'verified_floor_f', None),
    }
    
    record_id = insert_prediction(data)
    
    # Console output - one line format
    log_line = format_physics_log(station_id, timestamp, prediction)
    print(log_line)
    
    return record_id

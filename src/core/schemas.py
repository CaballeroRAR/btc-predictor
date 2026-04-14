from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Any, Literal
from datetime import datetime

class PredictionSchema(BaseModel):
    """Schema for individual price predictions."""
    sim_run_date: str = Field(..., description="YYYYMMDD identifier for the simulation run")
    forecast_date: str = Field(..., description="YYYY-MM-DD target date for the prediction")
    predicted_price: float = Field(..., description="USD price predicted by the LSTM")
    actual_price: Optional[float] = Field(None, description="Actual price matched later")
    schema_version: int = Field(1, description="Schema version for backward compatibility")

class CalibrationSchema(BaseModel):
    """Schema for sentiment/market drift calibration state."""
    model_config = ConfigDict(protected_namespaces=())
    
    last_calibration_date: str = Field(..., description="Timestamp of the last calibration")
    drift_value: float = Field(..., description="Calculated psychological drift offset")
    reference_price: float = Field(..., description="The BTC price at the moment of calibration")
    model_path: str = Field(..., description="Path to the model artifact used")
    schema_version: int = Field(1, description="Schema version")

class SnapshotSchema(BaseModel):
    """Schema for complete system forecasting snapshots (UI Cache)."""
    dates: List[str] = Field(..., description="List of forecast target dates")
    prices: List[float] = Field(..., description="List of predicted prices")
    std: List[float] = Field(..., description="Confidence intervals (Standard Deviation)")
    backtest_values: List[float] = Field(..., description="Historical performance prices")
    backtest_dates: List[str] = Field(..., description="Dates for the backtest period")
    avg_drift: float = Field(..., description="Mean drift applied to the simulation")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="Generation timestamp")
    schema_version: int = Field(1, description="Schema version")

class InvestmentSchema(BaseModel):
    """Schema for persisted investment journal entries."""
    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)

    id: Optional[str] = Field(None, description="Unique identifier (timestamp-based)")
    amount: float = Field(..., gt=0, description="Investment amount in USD")
    entry_price: float = Field(..., gt=0, description="BTC price at entry")
    target_pct: float = Field(..., description="Target profit percentage")
    target_price: float = Field(..., gt=0, description="Calculated exit price")
    
    simulation_status: Literal["SUCCESS", "TARGET_NOT_REACHED"] = Field(
        ..., description="Whether the plan hit the target in the forecast window"
    )
    confidence_score: float = Field(
        ..., ge=0, le=1, description="Model confidence based on standard deviation"
    )
    
    projected_withdrawal_date: Optional[str] = Field(None, description="YYYY-MM-DD exit date")
    forecast_prices: List[float] = Field(..., description="Mean price vector")
    forecast_dates: List[str] = Field(..., description="Date vector")
    std: List[float] = Field(..., description="Standard deviation vector")
    
    created_at: Optional[datetime] = Field(default_factory=datetime.now, description="Record creation time")
    schema_version: int = Field(1, description="Schema version")

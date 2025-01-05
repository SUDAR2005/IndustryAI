from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime
import pandas as pd
from pydantic import BaseModel, Field, validator
import numpy as np

class ProjectBase(BaseModel):
    # Basic Project Information
    project_id: str = Field(..., description="Unique identifier for the project")
    project_name: str = Field(..., description="Name of the project")
    project_description: str
    project_type: str = Field(..., description="Type of green project (solar, wind, etc.)")
    project_status: str = Field(..., description="Current status of the project")
    start_date: datetime
    expected_completion_date: datetime
    
    # Location Information
    country: str
    region: str
    coordinates: Dict[str, float] = Field(..., description="Latitude and longitude")
    
    class Config:
        validate_assignment = True

class FinancialMetrics(BaseModel):
    # Investment Details
    total_investment_required: float
    current_funding: float
    funding_gap: float
    expected_roi: float
    payback_period: float
    
    # Costs
    capital_expenditure: float
    operational_expenditure: float
    maintenance_costs: float
    
    # Revenue Projections
    projected_revenue_5yr: List[float]
    projected_cashflow_5yr: List[float]
    
    # Risk Metrics
    financial_risk_score: float = Field(..., ge=0, le=100)
    market_volatility_index: float
    currency_risk_exposure: float

class EnvironmentalMetrics(BaseModel):
    # Carbon Metrics
    carbon_reduction_tons: float
    carbon_intensity: float
    emission_savings_forecast: List[float]
    
    # Resource Usage
    water_usage_reduction: Optional[float]
    energy_efficiency_score: float = Field(..., ge=0, le=100)
    renewable_energy_generation: Optional[float]
    
    # Environmental Impact
    biodiversity_impact_score: float = Field(..., ge=0, le=100)
    land_use_change: Optional[float]
    waste_reduction_tons: Optional[float]
    
    # Climate Risk
    climate_risk_score: float = Field(..., ge=0, le=100)
    natural_disaster_risk: float
    climate_adaptation_score: float

class SocialMetrics(BaseModel):
    # Community Impact
    jobs_created: int
    community_benefit_score: float = Field(..., ge=0, le=100)
    local_business_impact: float
    
    # Social Development
    healthcare_impact_score: Optional[float]
    education_impact_score: Optional[float]
    poverty_reduction_impact: Optional[float]
    
    # Stakeholder Engagement
    community_engagement_level: float = Field(..., ge=0, le=100)
    indigenous_peoples_impact: Optional[float]
    stakeholder_satisfaction_score: float

class GovernanceMetrics(BaseModel):
    # Project Governance
    compliance_score: float = Field(..., ge=0, le=100)
    transparency_index: float
    corruption_risk_score: float
    
    # Management Quality
    management_experience_score: float
    track_record_score: float
    regulatory_compliance_score: float
    
    # Reporting & Monitoring
    reporting_quality_score: float
    monitoring_framework_score: float
    audit_frequency: int


def validate_project_data(df: pd.DataFrame) -> bool:
    """Validate if all required columns are present with correct data types"""
    required_columns = {
        'project_id': str,
        'project_name': str,
        'total_investment_required': float,
        'carbon_reduction_tons': float,
        'jobs_created': int,
        'compliance_score': float,
        # Add more required columns as needed
    }
    
    # Check if all required columns exist
    missing_columns = set(required_columns.keys()) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Validate data types
    for col, dtype in required_columns.items():
        if not all(isinstance(x, dtype) for x in df[col].dropna()):
            raise ValueError(f"Invalid data type in column {col}")
    
    return True

def calculate_project_scores(project_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate aggregate scores for different aspects of the project"""
    
    def normalize_score(score):
        return (score - score.min()) / (score.max() - score.min()) * 100
    
    # Calculate Environmental Score
    project_data['environmental_score'] = normalize_score(
        project_data['carbon_reduction_tons'] * 0.4 +
        project_data['energy_efficiency_score'] * 0.3 +
        project_data['climate_risk_score'] * 0.3
    )
    
    # Calculate Social Score
    project_data['social_score'] = normalize_score(
        project_data['jobs_created'] * 0.3 +
        project_data['community_benefit_score'] * 0.4 +
        project_data['stakeholder_satisfaction_score'] * 0.3
    )
    
    # Calculate Governance Score
    project_data['governance_score'] = normalize_score(
        project_data['compliance_score'] * 0.4 +
        project_data['transparency_index'] * 0.3 +
        project_data['management_experience_score'] * 0.3
    )
    
    # Calculate Overall ESG Score
    project_data['overall_esg_score'] = (
        project_data['environmental_score'] * 0.4 +
        project_data['social_score'] * 0.3 +
        project_data['governance_score'] * 0.3
    )
    
    return project_data
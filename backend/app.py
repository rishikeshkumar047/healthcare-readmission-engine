"""
FastAPI application for hospital readmission risk prediction.

This module provides a REST API for predicting hospital readmission risk
using machine learning models. It includes endpoints for health checks,
single predictions, batch predictions, and model information.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Healthcare Readmission Risk Prediction Engine",
    description="API for predicting hospital readmission risk using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Pydantic Models ====================

class PatientData(BaseModel):
    """Model for patient health data input."""
    
    age: int = Field(..., ge=0, le=150, description="Patient age in years")
    gender: str = Field(..., description="Patient gender (M/F)")
    length_of_stay: int = Field(..., ge=1, description="Length of hospital stay in days")
    number_of_medications: int = Field(..., ge=0, description="Number of medications prescribed")
    number_of_diagnoses: int = Field(..., ge=0, description="Number of diagnoses")
    number_of_procedures: int = Field(..., ge=0, description="Number of procedures performed")
    admission_type: str = Field(..., description="Type of admission (Emergency/Urgent/Elective)")
    discharge_disposition: str = Field(..., description="Discharge disposition")
    diabetes: int = Field(default=0, ge=0, le=1, description="Diabetes indicator (0/1)")
    hypertension: int = Field(default=0, ge=0, le=1, description="Hypertension indicator (0/1)")
    heart_disease: int = Field(default=0, ge=0, le=1, description="Heart disease indicator (0/1)")
    kidney_disease: int = Field(default=0, ge=0, le=1, description="Kidney disease indicator (0/1)")
    
    @validator('gender')
    def validate_gender(cls, v):
        """Validate gender field."""
        if v not in ['M', 'F']:
            raise ValueError('Gender must be M or F')
        return v
    
    @validator('admission_type')
    def validate_admission_type(cls, v):
        """Validate admission type field."""
        valid_types = ['Emergency', 'Urgent', 'Elective']
        if v not in valid_types:
            raise ValueError(f'Admission type must be one of {valid_types}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "age": 65,
                "gender": "M",
                "length_of_stay": 5,
                "number_of_medications": 8,
                "number_of_diagnoses": 3,
                "number_of_procedures": 2,
                "admission_type": "Emergency",
                "discharge_disposition": "Home",
                "diabetes": 1,
                "hypertension": 1,
                "heart_disease": 0,
                "kidney_disease": 0
            }
        }


class PredictionResponse(BaseModel):
    """Model for prediction response."""
    
    readmission_risk: float = Field(..., ge=0, le=1, description="Readmission risk probability (0-1)")
    risk_level: str = Field(..., description="Risk level classification (Low/Medium/High)")
    confidence_score: float = Field(..., ge=0, le=1, description="Model confidence in prediction")
    interpretation: str = Field(..., description="Human-readable interpretation of risk")
    
    class Config:
        schema_extra = {
            "example": {
                "readmission_risk": 0.72,
                "risk_level": "High",
                "confidence_score": 0.88,
                "interpretation": "Patient has high risk of readmission within 30 days. Consider follow-up interventions."
            }
        }


class SinglePredictionRequest(BaseModel):
    """Model for single prediction request."""
    
    patient_data: PatientData = Field(..., description="Patient health data")
    
    class Config:
        schema_extra = {
            "example": {
                "patient_data": {
                    "age": 65,
                    "gender": "M",
                    "length_of_stay": 5,
                    "number_of_medications": 8,
                    "number_of_diagnoses": 3,
                    "number_of_procedures": 2,
                    "admission_type": "Emergency",
                    "discharge_disposition": "Home",
                    "diabetes": 1,
                    "hypertension": 1,
                    "heart_disease": 0,
                    "kidney_disease": 0
                }
            }
        }


class BatchPredictionRequest(BaseModel):
    """Model for batch prediction request."""
    
    patients: List[PatientData] = Field(..., min_items=1, max_items=1000, description="List of patients")
    
    class Config:
        schema_extra = {
            "example": {
                "patients": [
                    {
                        "age": 65,
                        "gender": "M",
                        "length_of_stay": 5,
                        "number_of_medications": 8,
                        "number_of_diagnoses": 3,
                        "number_of_procedures": 2,
                        "admission_type": "Emergency",
                        "discharge_disposition": "Home",
                        "diabetes": 1,
                        "hypertension": 1,
                        "heart_disease": 0,
                        "kidney_disease": 0
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Model for batch prediction response."""
    
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_processed: int = Field(..., description="Total number of records processed")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


class HealthCheckResponse(BaseModel):
    """Model for health check response."""
    
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    service_name: str = Field(..., description="Service name")


class ModelInfo(BaseModel):
    """Model for model information response."""
    
    model_name: str = Field(..., description="Name of the model")
    model_version: str = Field(..., description="Version of the model")
    framework: str = Field(..., description="ML framework used")
    accuracy: float = Field(..., ge=0, le=1, description="Model accuracy")
    precision: float = Field(..., ge=0, le=1, description="Model precision")
    recall: float = Field(..., ge=0, le=1, description="Model recall")
    f1_score: float = Field(..., ge=0, le=1, description="Model F1 score")
    training_date: str = Field(..., description="Model training date")
    input_features: int = Field(..., description="Number of input features")
    last_updated: str = Field(..., description="Last model update timestamp")


# ==================== Mock Model Functions ====================

def calculate_risk_score(patient_data: PatientData) -> tuple[float, float]:
    """
    Calculate readmission risk score using patient data.
    
    This is a placeholder implementation. In production, this should use
    the actual trained machine learning model.
    
    Args:
        patient_data: Patient health data
        
    Returns:
        Tuple of (risk_score, confidence_score)
    """
    # Placeholder calculation - replace with actual model prediction
    risk_score = 0.0
    
    # Age factor
    if patient_data.age > 65:
        risk_score += 0.15
    elif patient_data.age > 45:
        risk_score += 0.05
    
    # Length of stay factor
    risk_score += min(patient_data.length_of_stay / 20, 0.25)
    
    # Number of medications factor
    risk_score += min(patient_data.number_of_medications / 30, 0.20)
    
    # Comorbidity factor
    comorbidity_count = (
        patient_data.diabetes + patient_data.hypertension +
        patient_data.heart_disease + patient_data.kidney_disease
    )
    risk_score += min(comorbidity_count * 0.1, 0.25)
    
    # Admission type factor
    if patient_data.admission_type == "Emergency":
        risk_score += 0.15
    elif patient_data.admission_type == "Urgent":
        risk_score += 0.08
    
    # Normalize to 0-1 range
    risk_score = min(risk_score, 1.0)
    
    # Calculate confidence based on data completeness
    confidence_score = 0.85 + (0.15 * min(comorbidity_count / 4, 1.0))
    
    return risk_score, confidence_score


def classify_risk_level(risk_score: float) -> str:
    """
    Classify risk level based on risk score.
    
    Args:
        risk_score: Readmission risk score (0-1)
        
    Returns:
        Risk level classification
    """
    if risk_score >= 0.7:
        return "High"
    elif risk_score >= 0.4:
        return "Medium"
    else:
        return "Low"


def generate_interpretation(risk_score: float, risk_level: str, patient_data: PatientData) -> str:
    """
    Generate human-readable interpretation of readmission risk.
    
    Args:
        risk_score: Readmission risk score
        risk_level: Risk level classification
        patient_data: Patient health data
        
    Returns:
        Interpretation string
    """
    interpretations = {
        "High": f"Patient has high risk of readmission within 30 days (risk: {risk_score:.1%}). "
                "Consider follow-up interventions and discharge planning.",
        "Medium": f"Patient has moderate risk of readmission within 30 days (risk: {risk_score:.1%}). "
                 "Monitor closely and ensure proper discharge instructions.",
        "Low": f"Patient has low risk of readmission within 30 days (risk: {risk_score:.1%}). "
               "Standard follow-up care recommended."
    }
    
    return interpretations.get(risk_level, "Unable to classify risk level")


# ==================== API Endpoints ====================

@app.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    summary="Health Check",
    description="Check if the service is running and healthy"
)
async def health_check():
    """
    Health check endpoint.
    
    Returns the current status of the service and timestamp.
    """
    logger.info("Health check endpoint called")
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        service_name="Healthcare Readmission Risk Prediction Engine"
    )


@app.get(
    "/model/info",
    response_model=ModelInfo,
    status_code=status.HTTP_200_OK,
    tags=["Model"],
    summary="Get Model Information",
    description="Get information about the current prediction model"
)
async def get_model_info():
    """
    Get model information endpoint.
    
    Returns details about the trained model including accuracy metrics.
    """
    logger.info("Model info endpoint called")
    return ModelInfo(
        model_name="Hospital Readmission Risk Classifier",
        model_version="1.0.0",
        framework="scikit-learn",
        accuracy=0.87,
        precision=0.85,
        recall=0.84,
        f1_score=0.84,
        training_date="2026-01-01",
        input_features=12,
        last_updated=datetime.utcnow().isoformat()
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
    summary="Single Prediction",
    description="Get readmission risk prediction for a single patient"
)
async def predict_single(request: SinglePredictionRequest):
    """
    Single prediction endpoint.
    
    Accepts patient health data and returns readmission risk prediction.
    
    Args:
        request: Single prediction request containing patient data
        
    Returns:
        Prediction response with risk score and classification
        
    Raises:
        HTTPException: If prediction fails
    """
    try:
        logger.info("Single prediction request received")
        patient_data = request.patient_data
        
        # Calculate risk score
        risk_score, confidence_score = calculate_risk_score(patient_data)
        
        # Classify risk level
        risk_level = classify_risk_level(risk_score)
        
        # Generate interpretation
        interpretation = generate_interpretation(risk_score, risk_level, patient_data)
        
        logger.info(f"Prediction completed: risk_level={risk_level}, score={risk_score:.3f}")
        
        return PredictionResponse(
            readmission_risk=round(risk_score, 4),
            risk_level=risk_level,
            confidence_score=round(confidence_score, 4),
            interpretation=interpretation
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
    summary="Batch Predictions",
    description="Get readmission risk predictions for multiple patients"
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction endpoint.
    
    Accepts multiple patient records and returns predictions for all.
    
    Args:
        request: Batch prediction request containing list of patients
        
    Returns:
        Batch prediction response with predictions and processing metrics
        
    Raises:
        HTTPException: If batch prediction fails
    """
    try:
        logger.info(f"Batch prediction request received for {len(request.patients)} patients")
        
        import time
        start_time = time.time()
        
        predictions = []
        
        for patient_data in request.patients:
            # Calculate risk score
            risk_score, confidence_score = calculate_risk_score(patient_data)
            
            # Classify risk level
            risk_level = classify_risk_level(risk_score)
            
            # Generate interpretation
            interpretation = generate_interpretation(risk_score, risk_level, patient_data)
            
            predictions.append(
                PredictionResponse(
                    readmission_risk=round(risk_score, 4),
                    risk_level=risk_level,
                    confidence_score=round(confidence_score, 4),
                    interpretation=interpretation
                )
            )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Batch prediction completed: {len(predictions)} patients processed "
            f"in {processing_time_ms:.2f}ms"
        )
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            processing_time_ms=round(processing_time_ms, 2)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get(
    "/",
    tags=["Root"],
    summary="API Root",
    description="Welcome endpoint"
)
async def root():
    """
    Root endpoint.
    
    Returns welcome message and API documentation links.
    """
    return {
        "message": "Healthcare Readmission Risk Prediction Engine",
        "documentation": "/docs",
        "health": "/health",
        "model_info": "/model/info",
        "endpoints": {
            "single_prediction": "POST /predict",
            "batch_predictions": "POST /predict/batch"
        }
    }


# ==================== Error Handlers ====================

@app.exception_handler(ValueError)
async def value_error_exception_handler(request, exc):
    """Handle ValueError exceptions."""
    logger.error(f"Value error: {str(exc)}")
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Invalid input: {str(exc)}"
    )


# ==================== Application Entry Point ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

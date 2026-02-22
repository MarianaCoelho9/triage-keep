"""API request and response schemas for TriageKeep."""
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Optional, Any
from voice_agent.core.types import ChatHistory


# ============= Request Schemas =============

class TriageRequest(BaseModel):
    """Request schema for triage interaction endpoints.
    
    Used by /analyze, /extract, and /report endpoints to accept
    user input and conversation history for triage processing.
    """
    user_input: str = Field(
        ..., 
        description="Current user input text",
        min_length=1
    )
    chat_history: ChatHistory = Field(
        default_factory=list,
        description="Conversation history with role and content"
    )


class TriageReportRequest(BaseModel):
    """Request schema for /report endpoint.

    user_input is optional to avoid appending placeholder text when
    generating a final report from history.
    """
    user_input: Optional[str] = Field(
        default=None,
        description="Optional current user input text"
    )
    chat_history: ChatHistory = Field(
        default_factory=list,
        description="Conversation history with role and content"
    )


class FHIRExportRequest(BaseModel):
    """Request schema for /report/fhir endpoint."""

    report: Dict[str, Any] = Field(
        ...,
        description="Structured triage report payload to map into FHIR",
    )
    include_validation: bool = Field(
        default=True,
        description="Whether to run lightweight FHIR structure validation",
    )
    model_config = ConfigDict(extra="forbid")


# ============= Response Schemas =============

class TriageAnalysisResponse(BaseModel):
    """Response from /analyze endpoint.
    
    Contains the agent's response to user input based on triage interaction.
    """
    analysis: Dict[str, str] = Field(
        ...,
        description="Analysis results containing the agent response"
    )


class TriageExtractionResponse(BaseModel):
    """Envelope response from /extract endpoint."""

    success: bool = Field(..., description="Whether extraction succeeded")
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured extraction payload",
    )
    error: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Error metadata when success is false",
    )
    model_config = ConfigDict(extra="forbid")


class TriageReportResponse(BaseModel):
    """Envelope response from /report endpoint."""

    success: bool = Field(..., description="Whether report generation succeeded")
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured report payload",
    )
    error: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Error metadata when success is false",
    )
    model_config = ConfigDict(extra="forbid")


class FHIRExportResponse(BaseModel):
    """Envelope response from /report/fhir endpoint."""

    success: bool = Field(..., description="Whether FHIR export succeeded")
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="FHIR export payload with bundle and warnings",
    )
    error: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Error metadata when success is false",
    )
    model_config = ConfigDict(extra="forbid")


class TranscriptionResponse(BaseModel):
    """Response from /transcribe endpoint."""
    transcript: str = Field(
        ..., 
        description="Transcribed text from audio"
    )


class StatusResponse(BaseModel):
    """Generic status response for health check endpoints."""
    status: str = Field(..., description="Service status")
    system: Optional[str] = Field(None, description="System identifier")

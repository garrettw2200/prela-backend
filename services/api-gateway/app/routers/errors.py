"""
Error Analysis API Router

Provides endpoints for analyzing trace errors and generating
actionable recommendations for debugging.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..auth import require_tier
from ..middleware.ai_feature_limiter import check_ai_feature_limit

from shared import get_clickhouse_client
from shared.error_analyzer import ErrorAnalysis, ErrorAnalyzer, ErrorCategory, ErrorSeverity
from shared.error_explainer import ErrorExplainer, ErrorExplanation
from shared.hallucination_detector import (
    HallucinationDetector,
    HallucinationAnalysis,
    ClaimGrounding,
)

router = APIRouter()

# Initialize error explainer (reused across requests)
_error_explainer: ErrorExplainer | None = None
_hallucination_detector: HallucinationDetector | None = None


def get_error_explainer() -> ErrorExplainer:
    """Get or create error explainer instance."""
    global _error_explainer
    if _error_explainer is None:
        _error_explainer = ErrorExplainer()
    return _error_explainer


def get_hallucination_detector() -> HallucinationDetector:
    """Get or create hallucination detector instance."""
    global _hallucination_detector
    if _hallucination_detector is None:
        _hallucination_detector = HallucinationDetector(
            similarity_threshold=0.7,
            model_name="all-MiniLM-L6-v2",
        )
    return _hallucination_detector


# Response Models


class ErrorRecommendationResponse(BaseModel):
    """Response model for a single error recommendation."""

    title: str = Field(..., description="Short title of the recommendation")
    description: str = Field(..., description="Detailed explanation of the fix")
    action_type: str = Field(..., description="Type of action: replay, code_change, or config_change")
    replay_params: dict[str, Any] | None = Field(
        None, description="Parameters for one-click replay"
    )
    code_snippet: str | None = Field(None, description="Example code to implement the fix")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    estimated_cost_impact: str | None = Field(
        None, description="Estimated cost impact of applying this fix"
    )


class ErrorAnalysisResponse(BaseModel):
    """Response model for a single span error analysis."""

    category: ErrorCategory = Field(..., description="Error category")
    severity: ErrorSeverity = Field(..., description="Error severity level")
    error_type: str | None = Field(None, description="Exception class name")
    error_message: str = Field(..., description="Full error message")
    error_code: int | None = Field(None, description="HTTP status code if applicable")
    recommendations: list[ErrorRecommendationResponse] = Field(
        ..., description="List of actionable recommendations"
    )
    context: dict[str, Any] = Field(..., description="Additional error context")


class SpanErrorResponse(BaseModel):
    """Response model for a single failed span."""

    span_id: str = Field(..., description="Unique span identifier")
    span_name: str = Field(..., description="Name of the span (e.g., 'openai.chat.completions.create')")
    analysis: ErrorAnalysisResponse = Field(..., description="Error analysis and recommendations")


class TraceErrorAnalysisResponse(BaseModel):
    """Response model for complete trace error analysis."""

    trace_id: str = Field(..., description="Trace identifier")
    error_count: int = Field(..., description="Number of errors found in this trace")
    errors: list[SpanErrorResponse] = Field(..., description="Detailed analysis of each failed span")
    has_critical_errors: bool = Field(
        ..., description="Whether any errors are marked as CRITICAL severity"
    )


class ErrorExplanationResponse(BaseModel):
    """Response model for AI-generated error explanation."""

    why_it_happened: str = Field(..., description="Natural language explanation of why the error occurred")
    what_to_do: str = Field(..., description="Step-by-step guidance for fixing the error")
    related_patterns: list[str] = Field(..., description="Similar error patterns seen in production")
    estimated_fix_time: str = Field(..., description="Estimated time to fix (e.g., '< 1 minute')")


class ErrorExplanationFullResponse(BaseModel):
    """Complete response with error analysis and AI explanation."""

    trace_id: str = Field(..., description="Trace identifier")
    span_id: str = Field(..., description="Span identifier")
    analysis: ErrorAnalysisResponse = Field(..., description="Structured error analysis")
    explanation: ErrorExplanationResponse = Field(..., description="AI-generated explanation")


# API Endpoints


@router.get(
    "/traces/{trace_id}/error-analysis",
    response_model=TraceErrorAnalysisResponse,
    summary="Analyze trace errors",
    description="Analyzes all errors in a trace and provides actionable recommendations for fixes.",
    responses={
        200: {
            "description": "Successful analysis",
            "content": {
                "application/json": {
                    "example": {
                        "trace_id": "trace-abc-123",
                        "error_count": 1,
                        "errors": [
                            {
                                "span_id": "span-def-456",
                                "span_name": "openai.chat.completions.create",
                                "analysis": {
                                    "category": "rate_limit",
                                    "severity": "HIGH",
                                    "error_type": "RateLimitError",
                                    "error_message": "Rate limit exceeded. Retry after 30 seconds.",
                                    "error_code": 429,
                                    "recommendations": [
                                        {
                                            "title": "Wait and retry with exponential backoff",
                                            "description": "Rate limits reset after a short period...",
                                            "action_type": "replay",
                                            "replay_params": {},
                                            "code_snippet": None,
                                            "confidence": 0.95,
                                            "estimated_cost_impact": None,
                                        }
                                    ],
                                    "context": {"model": "gpt-4", "max_tokens": 2048},
                                },
                            }
                        ],
                        "has_critical_errors": False,
                    }
                }
            },
        },
        404: {"description": "No errors found in trace or trace not found"},
    },
)
async def get_trace_error_analysis(
    trace_id: str,
    project_id: str = Query(..., description="Project identifier for filtering"),
) -> TraceErrorAnalysisResponse:
    """
    Analyzes all errors in a trace and provides actionable recommendations.

    This endpoint:
    1. Queries ClickHouse for all failed spans in the trace
    2. Analyzes each error using pattern matching and heuristics
    3. Generates 1-3 actionable recommendations per error
    4. Returns structured analysis with replay parameters and code snippets

    Args:
        trace_id: The trace ID to analyze
        project_id: Project ID for filtering (from query param)

    Returns:
        TraceErrorAnalysisResponse with detailed error analysis

    Raises:
        HTTPException: 404 if no errors found or trace doesn't exist
    """
    client = get_clickhouse_client()

    # Query all failed spans from ClickHouse
    query = """
    SELECT
        span_id,
        name,
        span_type,
        status,
        attributes,
        events,
        started_at,
        duration_ms
    FROM spans
    WHERE trace_id = %(trace_id)s
      AND service_name LIKE %(project_pattern)s
      AND status = 'error'
    ORDER BY started_at ASC
    """

    try:
        result = client.execute(
            query, {"trace_id": trace_id, "project_pattern": f"%{project_id}%"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No errors found in trace '{trace_id}' for project '{project_id}'",
        )

    # Analyze each error
    error_analyses: list[SpanErrorResponse] = []

    for row in result:
        # Parse span data from ClickHouse row
        span_data = {
            "span_id": row[0],
            "name": row[1],
            "span_type": row[2],
            "status": row[3],
            "attributes": json.loads(row[4]) if row[4] else {},
            "events": json.loads(row[5]) if row[5] else [],
            "started_at": row[6],
            "duration_ms": row[7],
        }

        # Run error analysis
        analysis: ErrorAnalysis = ErrorAnalyzer.analyze_span_error(span_data)

        # Convert to response models
        error_analyses.append(
            SpanErrorResponse(
                span_id=span_data["span_id"],
                span_name=span_data["name"],
                analysis=ErrorAnalysisResponse(
                    category=analysis.category,
                    severity=analysis.severity,
                    error_type=analysis.error_type,
                    error_message=analysis.error_message,
                    error_code=analysis.error_code,
                    recommendations=[
                        ErrorRecommendationResponse(
                            title=rec.title,
                            description=rec.description,
                            action_type=rec.action_type,
                            replay_params=rec.replay_params,
                            code_snippet=rec.code_snippet,
                            confidence=rec.confidence,
                            estimated_cost_impact=rec.estimated_cost_impact,
                        )
                        for rec in analysis.recommendations
                    ],
                    context=analysis.context,
                ),
            )
        )

    # Check for critical errors
    has_critical = any(e.analysis.severity == ErrorSeverity.CRITICAL for e in error_analyses)

    return TraceErrorAnalysisResponse(
        trace_id=trace_id,
        error_count=len(error_analyses),
        errors=error_analyses,
        has_critical_errors=has_critical,
    )


@router.get(
    "/traces/{trace_id}/error-explanation",
    response_model=ErrorExplanationFullResponse,
    summary="Get AI-powered error explanation",
    description="Generates natural language explanation for a specific error using GPT-4o-mini.",
    responses={
        200: {
            "description": "Successful explanation generation",
            "content": {
                "application/json": {
                    "example": {
                        "trace_id": "trace-abc-123",
                        "span_id": "span-def-456",
                        "analysis": {
                            "category": "rate_limit",
                            "severity": "HIGH",
                            "error_type": "RateLimitError",
                            "error_message": "Rate limit exceeded (429)",
                            "error_code": 429,
                            "recommendations": [],
                            "context": {"model": "gpt-4"},
                        },
                        "explanation": {
                            "why_it_happened": "Your application hit OpenAI's rate limit because you're sending too many requests in a short time period. This commonly happens during testing or when scaling up.",
                            "what_to_do": "1. Wait 30 seconds for the rate limit to reset\n2. Switch to gpt-4o-mini (83% cheaper with higher limits)\n3. Implement exponential backoff in your retry logic",
                            "related_patterns": [
                                "Common during peak usage hours",
                                "Often resolved by switching to cheaper models",
                            ],
                            "estimated_fix_time": "< 1 minute",
                        },
                    }
                }
            },
        },
        404: {"description": "Span not found or not an error"},
    },
)
async def get_error_explanation(
    trace_id: str,
    span_id: str = Query(..., description="Span ID to explain"),
    project_id: str = Query(..., description="Project identifier for filtering"),
) -> ErrorExplanationFullResponse:
    """
    Generates AI-powered natural language explanation for a specific error.

    This endpoint:
    1. Queries ClickHouse for the failed span
    2. Analyzes the error using ErrorAnalyzer
    3. Generates natural language explanation using GPT-4o-mini
    4. Returns both structured analysis and friendly explanation

    Args:
        trace_id: The trace ID containing the error
        span_id: The specific span ID with error
        project_id: Project ID for filtering (from query param)

    Returns:
        ErrorExplanationFullResponse with analysis and AI explanation

    Raises:
        HTTPException: 404 if span not found or not an error
        HTTPException: 500 if explanation generation fails
    """
    client = get_clickhouse_client()

    # Query specific span from ClickHouse
    query = """
    SELECT
        span_id,
        name,
        span_type,
        status,
        attributes,
        events,
        started_at,
        duration_ms
    FROM spans
    WHERE trace_id = %(trace_id)s
      AND span_id = %(span_id)s
      AND service_name LIKE %(project_pattern)s
      AND status = 'error'
    LIMIT 1
    """

    try:
        result = client.execute(
            query,
            {
                "trace_id": trace_id,
                "span_id": span_id,
                "project_pattern": f"%{project_id}%",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Error span '{span_id}' not found in trace '{trace_id}' for project '{project_id}'",
        )

    # Parse span data
    row = result[0]
    span_data = {
        "span_id": row[0],
        "name": row[1],
        "span_type": row[2],
        "status": row[3],
        "attributes": json.loads(row[4]) if row[4] else {},
        "events": json.loads(row[5]) if row[5] else [],
        "started_at": row[6],
        "duration_ms": row[7],
    }

    # Run error analysis
    analysis: ErrorAnalysis = ErrorAnalyzer.analyze_span_error(span_data)

    # Generate AI explanation
    try:
        explainer = get_error_explainer()
        explanation: ErrorExplanation = explainer.explain_error(
            category=analysis.category.value,
            severity=analysis.severity.value,
            error_message=analysis.error_message,
            context=analysis.context,
        )
    except Exception as e:
        # Log error but continue with fallback
        print(f"Error generating explanation: {e}")
        explanation = ErrorExplanation(
            why_it_happened="AI explanation temporarily unavailable. Please check error details above.",
            what_to_do="Review the error message and try the suggested fixes.",
            related_patterns=[],
            estimated_fix_time="Unknown",
        )

    # Convert to response models
    return ErrorExplanationFullResponse(
        trace_id=trace_id,
        span_id=span_id,
        analysis=ErrorAnalysisResponse(
            category=analysis.category,
            severity=analysis.severity,
            error_type=analysis.error_type,
            error_message=analysis.error_message,
            error_code=analysis.error_code,
            recommendations=[
                ErrorRecommendationResponse(
                    title=rec.title,
                    description=rec.description,
                    action_type=rec.action_type,
                    replay_params=rec.replay_params,
                    code_snippet=rec.code_snippet,
                    confidence=rec.confidence,
                    estimated_cost_impact=rec.estimated_cost_impact,
                )
                for rec in analysis.recommendations
            ],
            context=analysis.context,
        ),
        explanation=ErrorExplanationResponse(
            why_it_happened=explanation.why_it_happened,
            what_to_do=explanation.what_to_do,
            related_patterns=explanation.related_patterns,
            estimated_fix_time=explanation.estimated_fix_time,
        ),
    )


# Hallucination Detection Models


class ClaimResponse(BaseModel):
    """Response model for a single claim."""

    text: str = Field(..., description="The claim text")
    sentence_index: int = Field(..., description="Position in original output (0-indexed)")
    start_char: int = Field(..., description="Character offset in original text")
    end_char: int = Field(..., description="End character offset")


class ClaimGroundingResponse(BaseModel):
    """Response model for claim grounding assessment."""

    claim: ClaimResponse = Field(..., description="The claim being assessed")
    is_grounded: bool = Field(..., description="True if supported by context")
    confidence: float = Field(..., description="Confidence in grounding assessment (0.0-1.0)")
    similarity_score: float = Field(..., description="Semantic similarity to context (0.0-1.0)")
    supporting_context: str | None = Field(None, description="Best matching context chunk")
    context_index: int | None = Field(None, description="Index of supporting context")


class HallucinationAnalysisResponse(BaseModel):
    """Response model for hallucination analysis."""

    trace_id: str = Field(..., description="Trace identifier")
    span_id: str = Field(..., description="Span identifier")
    output_text: str = Field(..., description="Original LLM output")
    context_chunks: list[str] = Field(..., description="Retrieved context used for comparison")
    claims: list[ClaimGroundingResponse] = Field(
        ..., description="All extracted claims with grounding assessment"
    )
    hallucination_detected: bool = Field(..., description="True if any ungrounded claims found")
    overall_confidence: float = Field(..., description="Average confidence across claims (0.0-1.0)")
    ungrounded_claim_count: int = Field(..., description="Number of claims without support")
    grounded_claim_count: int = Field(..., description="Number of claims with support")
    similarity_threshold: float = Field(..., description="Threshold used for grounding (0.0-1.0)")
    encoder_available: bool = Field(..., description="Whether sentence-transformers is available")


@router.get(
    "/traces/{trace_id}/hallucination-analysis",
    response_model=list[HallucinationAnalysisResponse],
    summary="Analyze trace for hallucinations",
    description="Detect hallucinations in LLM outputs by comparing against retrieved context",
)
async def analyze_hallucinations(
    trace_id: str,
    similarity_threshold: float = Query(
        0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score to consider a claim grounded (0.0-1.0)",
    ),
    user: dict = Depends(require_tier("pro")),
) -> list[HallucinationAnalysisResponse]:
    """
    Analyze LLM spans in a trace for hallucinations.

    Compares LLM outputs against retrieved context to detect unsupported claims.
    Uses semantic similarity (sentence-transformers) when available, falls back
    to text matching otherwise.

    Args:
        trace_id: The trace to analyze
        similarity_threshold: Minimum similarity to consider grounded (default: 0.7)

    Returns:
        List of hallucination analyses for each LLM span with retrieval context

    Raises:
        HTTPException: 404 if trace not found, 500 on processing errors
    """
    # Check and increment AI usage limit
    await check_ai_feature_limit(user["user_id"], user["tier"], "hallucination")

    client = get_clickhouse_client()

    try:
        # Query all LLM spans in the trace
        query = """
            SELECT
                span_id,
                name,
                attributes,
                span_type
            FROM spans
            WHERE trace_id = %(trace_id)s
              AND span_type = 'llm'
            ORDER BY started_at ASC
        """

        result = client.execute(query, {"trace_id": trace_id})

        if not result:
            raise HTTPException(
                status_code=404, detail=f"No LLM spans found in trace {trace_id}"
            )

        # Get hallucination detector
        detector = get_hallucination_detector()
        # Update threshold if provided
        detector.similarity_threshold = similarity_threshold

        analyses = []

        for row in result:
            span_id, name, attributes_json, span_type = row

            # Parse attributes
            try:
                attributes = json.loads(attributes_json) if isinstance(attributes_json, str) else attributes_json
            except json.JSONDecodeError:
                attributes = {}

            # Extract LLM output
            output_text = None

            # Try common output fields
            for field in ["llm.response", "response", "output", "content"]:
                if field in attributes:
                    output_text = str(attributes[field])
                    break

            # Try from events
            if not output_text:
                # Events are stored in span table, need to query separately or include in main query
                # For now, skip if no output in attributes
                continue

            if not output_text:
                continue

            # Extract context chunks from span
            span_data = {"attributes": attributes}
            context_chunks = HallucinationDetector.extract_context_from_span(span_data)

            # Skip if no context to compare against
            if not context_chunks:
                continue

            # Analyze for hallucinations
            analysis = detector.analyze(output_text, context_chunks)

            # Convert to response model
            analyses.append(
                HallucinationAnalysisResponse(
                    trace_id=trace_id,
                    span_id=span_id,
                    output_text=analysis.output_text,
                    context_chunks=analysis.context_chunks,
                    claims=[
                        ClaimGroundingResponse(
                            claim=ClaimResponse(
                                text=cg.claim.text,
                                sentence_index=cg.claim.sentence_index,
                                start_char=cg.claim.start_char,
                                end_char=cg.claim.end_char,
                            ),
                            is_grounded=cg.is_grounded,
                            confidence=cg.confidence,
                            similarity_score=cg.similarity_score,
                            supporting_context=cg.supporting_context,
                            context_index=cg.context_index,
                        )
                        for cg in analysis.claims
                    ],
                    hallucination_detected=analysis.hallucination_detected,
                    overall_confidence=analysis.overall_confidence,
                    ungrounded_claim_count=analysis.ungrounded_claim_count,
                    grounded_claim_count=analysis.grounded_claim_count,
                    similarity_threshold=detector.similarity_threshold,
                    encoder_available=detector._encoder is not None,
                )
            )

        if not analyses:
            raise HTTPException(
                status_code=404,
                detail=f"No LLM spans with retrieval context found in trace {trace_id}",
            )

        return analyses

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze hallucinations: {str(e)}",
        ) from e

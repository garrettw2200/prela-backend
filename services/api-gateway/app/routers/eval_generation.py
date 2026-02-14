"""
Eval Generation API Router

Provides endpoints to trigger eval suite generation from production traces,
poll generation status, download completed YAML files, and list past generations.

Generation runs asynchronously via FastAPI BackgroundTasks because analyzing
hundreds of traces with LLM calls takes 30-120 seconds.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field

from shared import get_clickhouse_client, settings
from ..auth import require_tier
from shared.eval_generator import EvalGenerator, EvalGenerationConfig

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class EvalGenerationRequest(BaseModel):
    suite_name: str | None = None
    time_window_hours: int = Field(default=168, ge=1, le=720)
    max_traces: int = Field(default=500, ge=10, le=5000)
    max_cases: int = Field(default=50, ge=5, le=200)
    include_failure_modes: bool = True
    include_edge_cases: bool = True
    include_positive_examples: bool = True
    agent_name_filter: str | None = None
    service_name_filter: str | None = None


class EvalGenerationResponse(BaseModel):
    generation_id: str
    status: str
    started_at: str


class PatternSummaryItem(BaseModel):
    category: str
    subcategory: str
    count: int
    description: str


class EvalGenerationStatus(BaseModel):
    generation_id: str
    status: str
    suite_name: str
    cases_generated: int
    traces_analyzed: int
    patterns_found: int
    pattern_summary: list[PatternSummaryItem]
    error: str | None = None
    started_at: str
    completed_at: str | None = None


class EvalGenerationListItem(BaseModel):
    generation_id: str
    suite_name: str
    status: str
    cases_generated: int
    traces_analyzed: int
    started_at: str
    completed_at: str | None = None


class EvalGenerationHistoryResponse(BaseModel):
    generations: list[EvalGenerationListItem]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------------------
# Background Task
# ---------------------------------------------------------------------------


def _run_generation(generation_id: str, project_id: str, config: EvalGenerationConfig) -> None:
    """Execute eval generation in background and store result in ClickHouse.

    This runs synchronously inside a BackgroundTasks thread.
    """
    try:
        client = get_clickhouse_client()

        generator = EvalGenerator()
        result = generator.generate(project_id, config, clickhouse_client=client)

        # Store completed result
        _store_result(client, generation_id, project_id, result)

    except Exception as e:
        logger.error(f"Background eval generation failed: {e}", exc_info=True)
        try:
            client = get_clickhouse_client()
            _store_error(client, generation_id, project_id, str(e))
        except Exception:
            logger.error(f"Failed to store error for generation {generation_id}")


def _store_result(client, generation_id: str, project_id: str, result) -> None:
    """Store generation result in analysis_results table."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
    result_json = json.dumps({
        "suite_name": result.suite_name,
        "suite_yaml": result.suite_yaml,
        "cases_generated": result.cases_generated,
        "traces_analyzed": result.traces_analyzed,
        "patterns_found": result.patterns_found,
        "pattern_summary": result.pattern_summary,
        "status": result.status,
        "error": result.error,
        "started_at": result.started_at,
        "completed_at": result.completed_at,
    })

    client.insert(
        "analysis_results",
        [[
            generation_id,
            "",  # trace_id — project-level, not trace-level
            project_id,
            "eval_generation",
            result_json,
            result.cases_generated / max(result.traces_analyzed, 1),
            now,
        ]],
        column_names=["result_id", "trace_id", "project_id", "analysis_type", "result", "score", "created_at"],
    )


def _store_error(client, generation_id: str, project_id: str, error: str) -> None:
    """Store a failed generation result."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
    result_json = json.dumps({
        "suite_name": "",
        "suite_yaml": None,
        "cases_generated": 0,
        "traces_analyzed": 0,
        "patterns_found": 0,
        "pattern_summary": [],
        "status": "failed",
        "error": error,
        "started_at": now,
        "completed_at": now,
    })

    client.insert(
        "analysis_results",
        [[generation_id, "", project_id, "eval_generation", result_json, 0.0, now]],
        column_names=["result_id", "trace_id", "project_id", "analysis_type", "result", "score", "created_at"],
    )


def _store_initial(client, generation_id: str, project_id: str, suite_name: str) -> None:
    """Store initial 'running' record so status polling works immediately."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
    result_json = json.dumps({
        "suite_name": suite_name,
        "suite_yaml": None,
        "cases_generated": 0,
        "traces_analyzed": 0,
        "patterns_found": 0,
        "pattern_summary": [],
        "status": "running",
        "error": None,
        "started_at": now,
        "completed_at": None,
    })

    client.insert(
        "analysis_results",
        [[generation_id, "", project_id, "eval_generation", result_json, 0.0, now]],
        column_names=["result_id", "trace_id", "project_id", "analysis_type", "result", "score", "created_at"],
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/generate", response_model=EvalGenerationResponse)
async def trigger_eval_generation(
    request: EvalGenerationRequest,
    background_tasks: BackgroundTasks,
    project_id: str = Query(..., description="Project ID"),
    user: dict = Depends(require_tier("pro")),
) -> EvalGenerationResponse:
    """Trigger eval suite generation from production traces.

    Starts generation in the background and returns immediately with a
    generation_id that can be used to poll status and download results.
    """
    generation_id = str(uuid.uuid4())
    suite_name = request.suite_name or f"Generated Suite - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"

    config = EvalGenerationConfig(
        project_id=project_id,
        suite_name=suite_name,
        time_window_hours=request.time_window_hours,
        max_traces=request.max_traces,
        max_cases=request.max_cases,
        include_failure_modes=request.include_failure_modes,
        include_edge_cases=request.include_edge_cases,
        include_positive_examples=request.include_positive_examples,
        agent_name_filter=request.agent_name_filter,
        service_name_filter=request.service_name_filter,
        model=settings.eval_generation_model,
        max_tokens=settings.eval_generation_max_tokens,
    )

    # Store initial "running" record
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        client = get_clickhouse_client()
        _store_initial(client, generation_id, project_id, suite_name)
    except Exception as e:
        logger.error(f"Failed to store initial generation record: {e}")
        raise HTTPException(status_code=500, detail="Failed to start generation")

    # Launch background task
    background_tasks.add_task(_run_generation, generation_id, project_id, config)

    return EvalGenerationResponse(
        generation_id=generation_id,
        status="running",
        started_at=started_at,
    )


@router.get("/{generation_id}/status", response_model=EvalGenerationStatus)
async def get_generation_status(
    generation_id: str,
    project_id: str = Query(..., description="Project ID"),
    user: dict = Depends(require_tier("pro")),
) -> EvalGenerationStatus:
    """Get the status of an eval generation run."""
    try:
        client = get_clickhouse_client()
        result = client.query(
            """
            SELECT result, created_at
            FROM analysis_results
            WHERE result_id = %(generation_id)s
              AND project_id = %(project_id)s
              AND analysis_type = 'eval_generation'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            parameters={"generation_id": generation_id, "project_id": project_id},
        )

        if not result.result_rows:
            raise HTTPException(status_code=404, detail=f"Generation {generation_id} not found")

        row = result.result_rows[0]
        data = json.loads(row[0])

        return EvalGenerationStatus(
            generation_id=generation_id,
            status=data.get("status", "unknown"),
            suite_name=data.get("suite_name", ""),
            cases_generated=data.get("cases_generated", 0),
            traces_analyzed=data.get("traces_analyzed", 0),
            patterns_found=data.get("patterns_found", 0),
            pattern_summary=[
                PatternSummaryItem(**p) for p in data.get("pattern_summary", [])
            ],
            error=data.get("error"),
            started_at=data.get("started_at", str(row[1])),
            completed_at=data.get("completed_at"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get generation status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get generation status")


@router.get("/{generation_id}/download")
async def download_eval_suite(
    generation_id: str,
    project_id: str = Query(..., description="Project ID"),
    user: dict = Depends(require_tier("pro")),
) -> Response:
    """Download a completed eval suite as a YAML file."""
    try:
        client = get_clickhouse_client()
        result = client.query(
            """
            SELECT result
            FROM analysis_results
            WHERE result_id = %(generation_id)s
              AND project_id = %(project_id)s
              AND analysis_type = 'eval_generation'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            parameters={"generation_id": generation_id, "project_id": project_id},
        )

        if not result.result_rows:
            raise HTTPException(status_code=404, detail=f"Generation {generation_id} not found")

        data = json.loads(result.result_rows[0][0])

        if data.get("status") != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Generation is not completed (status: {data.get('status')})",
            )

        suite_yaml = data.get("suite_yaml")
        if not suite_yaml:
            raise HTTPException(status_code=404, detail="No YAML content available")

        suite_name = data.get("suite_name", "eval_suite").replace(" ", "_").lower()
        filename = f"{suite_name}.yaml"

        return Response(
            content=suite_yaml,
            media_type="application/x-yaml",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download eval suite: {e}")
        raise HTTPException(status_code=500, detail="Failed to download eval suite")


@router.get("/history", response_model=EvalGenerationHistoryResponse)
async def list_eval_generations(
    project_id: str = Query(..., description="Project ID"),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    user: dict = Depends(require_tier("pro")),
) -> EvalGenerationHistoryResponse:
    """List past eval generation runs for a project."""
    try:
        client = get_clickhouse_client()

        # Get total count
        count_result = client.query(
            """
            SELECT count()
            FROM analysis_results
            WHERE project_id = %(project_id)s
              AND analysis_type = 'eval_generation'
            """,
            parameters={"project_id": project_id},
        )
        total = count_result.result_rows[0][0] if count_result.result_rows else 0

        # Get page
        offset = (page - 1) * page_size
        result = client.query(
            """
            SELECT result_id, result, created_at
            FROM analysis_results
            WHERE project_id = %(project_id)s
              AND analysis_type = 'eval_generation'
            ORDER BY created_at DESC
            LIMIT %(limit)s OFFSET %(offset)s
            """,
            parameters={"project_id": project_id, "limit": page_size, "offset": offset},
        )

        generations = []
        for row in result.result_rows:
            data = json.loads(row[1])
            generations.append(EvalGenerationListItem(
                generation_id=row[0],
                suite_name=data.get("suite_name", ""),
                status=data.get("status", "unknown"),
                cases_generated=data.get("cases_generated", 0),
                traces_analyzed=data.get("traces_analyzed", 0),
                started_at=data.get("started_at", str(row[2])),
                completed_at=data.get("completed_at"),
            ))

        return EvalGenerationHistoryResponse(
            generations=generations,
            total=total,
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        logger.error(f"Failed to list eval generations: {e}")
        raise HTTPException(status_code=500, detail="Failed to list eval generations")

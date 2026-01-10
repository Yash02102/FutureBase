from typing import List, Optional

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    step: str = Field(..., description="A single, concrete task step.")


class Plan(BaseModel):
    goal: str
    steps: List[PlanStep]


class ToolResult(BaseModel):
    tool: str
    output: str
    metadata: Optional[dict] = None
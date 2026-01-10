from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class MCPToolSpec:
    name: str
    description: str
    input_schema: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    url: str
    transport: str = "http"
    headers: Optional[Dict[str, str]] = None

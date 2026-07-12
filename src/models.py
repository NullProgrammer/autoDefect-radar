from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src import config


@dataclass(frozen=True)
class VehicleQuery:
    make: str = config.DEFAULT_MAKE
    model: str = config.DEFAULT_MODEL
    model_year: int = config.DEFAULT_MODEL_YEAR

    @property
    def slug(self) -> str:
        parts = [str(self.model_year), self.make, self.model]
        return "_".join(part.lower().replace(" ", "_") for part in parts)

    @property
    def label(self) -> str:
        return f"{self.model_year} {self.make.upper()} {self.model.upper()}"

    def as_api_params(self) -> dict[str, str | int]:
        return {
            "make": self.make,
            "model": self.model,
            "modelYear": self.model_year,
        }


class VehicleRecords(BaseModel):
    query: VehicleQuery
    recalls: list[dict[str, Any]] = Field(default_factory=list)
    complaints: list[dict[str, Any]] = Field(default_factory=list)


class NormalizedDocument(BaseModel):
    source_type: str
    evidence_id: str
    title: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def filename(self) -> str:
        safe_id = "".join(char if char.isalnum() else "_" for char in self.evidence_id)
        return f"{self.source_type}_{safe_id}.md"

    def path_under(self, root: Path) -> Path:
        return root / self.source_type / self.filename

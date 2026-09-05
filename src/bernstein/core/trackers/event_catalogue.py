"""Schema-validated canonical event-type catalogue for tracker ingest.

Source webhook event names map through this catalogue only. Parsers must
not compare raw event-name string literals; unknown source events resolve
to the explicit ``unmapped`` canonical type and are counted.

The catalogue file lives beside this module
(``event_catalogue.yaml``) and ships in the wheel. It is loaded and
validated once at process start.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

logger = logging.getLogger(__name__)

__all__ = [
    "UNMAPPED_ACTION",
    "UNMAPPED_TYPE",
    "CanonicalAction",
    "EventCatalogue",
    "EventCatalogueError",
    "ExtractShape",
    "ResolvedSourceEvent",
    "catalogue_content_hash",
    "get_event_catalogue",
    "load_event_catalogue",
    "reset_event_catalogue_for_tests",
]

#: Explicit canonical type / action for source events with no catalogue entry.
UNMAPPED_TYPE: Final[str] = "unmapped"
UNMAPPED_ACTION: Final[str] = "unmapped"

CanonicalAction = Literal["created", "updated", "transitioned", "commented", "unmapped"]
ExtractShape = Literal["issue_or_pr", "object_attributes", "nested_issue"]

_CATALOGUE_FILENAME: Final[str] = "event_catalogue.yaml"
_CATALOGUE_PATH: Final[Path] = Path(__file__).resolve().parent / _CATALOGUE_FILENAME

_REQUIRED_CANONICAL_TYPES: Final[frozenset[str]] = frozenset({"issue", "issue_comment", UNMAPPED_TYPE})
_REQUIRED_ACTIONS: Final[frozenset[str]] = frozenset(
    {"created", "updated", "transitioned", "commented", UNMAPPED_ACTION}
)


class EventCatalogueError(ValueError):
    """Raised when the catalogue file is missing or fails schema validation."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CanonicalTypeEntry(_StrictModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class SourceEventMapping(_StrictModel):
    canonical_type: str = Field(min_length=1)
    extract: ExtractShape | None = None


class EventCatalogueDocument(_StrictModel):
    """On-disk shape of ``event_catalogue.yaml``."""

    schema_version: int = Field(ge=1)
    canonical_types: list[CanonicalTypeEntry] = Field(min_length=1)
    canonical_actions: list[str] = Field(min_length=1)
    mappings: dict[str, dict[str, SourceEventMapping]] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_closed_sets(self) -> EventCatalogueDocument:
        type_ids = {entry.id for entry in self.canonical_types}
        missing_types = _REQUIRED_CANONICAL_TYPES - type_ids
        if missing_types:
            msg = f"canonical_types missing required ids: {sorted(missing_types)}"
            raise ValueError(msg)
        missing_actions = _REQUIRED_ACTIONS - set(self.canonical_actions)
        if missing_actions:
            msg = f"canonical_actions missing required values: {sorted(missing_actions)}"
            raise ValueError(msg)
        for adapter, events in self.mappings.items():
            if not adapter or not isinstance(events, dict) or not events:
                msg = f"mappings[{adapter!r}] must be a non-empty mapping"
                raise ValueError(msg)
            for source_name, mapping in events.items():
                if not source_name:
                    msg = f"mappings[{adapter!r}] has an empty source event name"
                    raise ValueError(msg)
                if mapping.canonical_type not in type_ids:
                    msg = (
                        f"mappings[{adapter!r}][{source_name!r}].canonical_type "
                        f"{mapping.canonical_type!r} is not in canonical_types"
                    )
                    raise ValueError(msg)
                if mapping.canonical_type == UNMAPPED_TYPE:
                    msg = (
                        f"mappings[{adapter!r}][{source_name!r}] must not target "
                        f"the explicit {UNMAPPED_TYPE!r} bucket; omit the entry instead"
                    )
                    raise ValueError(msg)
        return self


@dataclass(frozen=True, slots=True)
class ResolvedSourceEvent:
    """Result of mapping one source event name through the catalogue."""

    adapter: str
    source_event: str
    canonical_type: str
    extract: ExtractShape | None
    catalogue_content_hash: str

    @property
    def is_unmapped(self) -> bool:
        return self.canonical_type == UNMAPPED_TYPE


@dataclass
class EventCatalogue:
    """Loaded, validated catalogue with an unmapped counter."""

    document: EventCatalogueDocument
    content_hash: str
    _raw: dict[str, Any]
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _unmapped_counts: dict[tuple[str, str], int] = field(default_factory=dict, repr=False)

    def resolve(self, adapter: str, source_event: str) -> ResolvedSourceEvent:
        """Map a source event name to a canonical type.

        Unknown ``(adapter, source_event)`` pairs resolve to
        :data:`UNMAPPED_TYPE` and increment the unmapped counter.
        """

        adapter_key = str(adapter or "")
        source_key = str(source_event or "")
        mapping = self.document.mappings.get(adapter_key, {}).get(source_key)
        if mapping is None:
            with self._lock:
                key = (adapter_key, source_key)
                self._unmapped_counts[key] = self._unmapped_counts.get(key, 0) + 1
            return ResolvedSourceEvent(
                adapter=adapter_key,
                source_event=source_key,
                canonical_type=UNMAPPED_TYPE,
                extract=None,
                catalogue_content_hash=self.content_hash,
            )
        return ResolvedSourceEvent(
            adapter=adapter_key,
            source_event=source_key,
            canonical_type=mapping.canonical_type,
            extract=mapping.extract,
            catalogue_content_hash=self.content_hash,
        )

    def unmapped_count(self) -> int:
        """Return the total number of unmapped source events observed."""

        with self._lock:
            return sum(self._unmapped_counts.values())

    def unmapped_counts(self) -> dict[tuple[str, str], int]:
        """Return a copy of per-(adapter, source_event) unmapped counts."""

        with self._lock:
            return dict(self._unmapped_counts)

    def known_source_events(self, adapter: str) -> frozenset[str]:
        """Return the closed set of source event names for ``adapter``."""

        return frozenset(self.document.mappings.get(adapter, {}))

    def is_canonical_action(self, action: str) -> bool:
        """Return whether ``action`` is in the catalogue's action vocabulary."""

        return action in self.document.canonical_actions


def catalogue_content_hash(raw: dict[str, Any]) -> str:
    """Return the SHA-256 hex digest of the catalogue's canonical JSON form.

    Mirrors :func:`bernstein.core.persistence.work_ledger.compute_entry_hash`'s
    sort-keys / minimal-separators approach so the digest is byte-stable
    across hosts.
    """

    document = {
        "canonical_actions": raw.get("canonical_actions"),
        "canonical_types": raw.get("canonical_types"),
        "mappings": raw.get("mappings"),
        "schema_version": raw.get("schema_version"),
    }
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_event_catalogue(path: Path | None = None) -> EventCatalogue:
    """Load and validate the event catalogue from ``path`` (or the package default).

    Raises:
        EventCatalogueError: Missing file or schema validation failure.
    """

    catalogue_path = path if path is not None else _CATALOGUE_PATH
    if not catalogue_path.exists():
        raise EventCatalogueError(f"Event catalogue not found: {catalogue_path}")
    try:
        with catalogue_path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise EventCatalogueError(f"Malformed event catalogue: {exc}") from exc
    if not isinstance(raw, dict):
        raise EventCatalogueError("Event catalogue must be a YAML mapping at the top level")
    try:
        document = EventCatalogueDocument.model_validate(raw)
    except ValidationError as exc:
        raise EventCatalogueError(f"Event catalogue failed schema validation: {exc}") from exc
    digest = catalogue_content_hash(raw)
    return EventCatalogue(document=document, content_hash=digest, _raw=dict(raw))


_catalogue: EventCatalogue | None = None
_catalogue_lock = threading.Lock()


def get_event_catalogue() -> EventCatalogue:
    """Return the process-wide catalogue, loading it on first call."""

    global _catalogue
    if _catalogue is not None:
        return _catalogue
    with _catalogue_lock:
        if _catalogue is None:
            _catalogue = load_event_catalogue()
        return _catalogue


def reset_event_catalogue_for_tests(catalogue: EventCatalogue | None = None) -> None:
    """Replace or clear the process-wide catalogue (tests only)."""

    global _catalogue
    with _catalogue_lock:
        _catalogue = catalogue


try:
    get_event_catalogue()
except EventCatalogueError:
    logger.exception("Event catalogue failed to load at import time")

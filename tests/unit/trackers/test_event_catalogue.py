"""Tests for the canonical tracker event-type catalogue (#5132)."""

from __future__ import annotations

import ast
import hashlib
import hmac
import json
from pathlib import Path

import pytest

from bernstein.core.trackers.event_catalogue import (
    UNMAPPED_ACTION,
    UNMAPPED_TYPE,
    EventCatalogueError,
    get_event_catalogue,
    load_event_catalogue,
    reset_event_catalogue_for_tests,
)
from bernstein.core.trackers.webhook_receiver import (
    ReplayLedger,
    WebhookConfig,
    WebhookReceiver,
    _github_parse,
    _gitlab_parse,
    register_builtin_handlers,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WEBHOOK_RECEIVER = _REPO_ROOT / "src" / "bernstein" / "core" / "trackers" / "webhook_receiver.py"
_CATALOGUE_YAML = _REPO_ROOT / "src" / "bernstein" / "core" / "trackers" / "event_catalogue.yaml"

#: Source event names that must not appear as string-literal comparisons on
#: the GitHub / GitLab ingest parsers. The catalogue is the only place these
#: names may be declared.
_FORBIDDEN_SOURCE_EVENT_LITERALS = frozenset(
    {
        "issues",
        "issue_comment",
        "pull_request",
        "issue",
        "note",
    }
)


@pytest.fixture(autouse=True)
def _reset_catalogue() -> None:
    """Reload the packaged catalogue around each test so counters stay fresh."""

    reset_event_catalogue_for_tests(None)
    get_event_catalogue()
    yield
    reset_event_catalogue_for_tests(None)


def test_catalogue_loads_and_validates_at_start() -> None:
    catalogue = load_event_catalogue(_CATALOGUE_YAML)
    assert catalogue.document.schema_version >= 1
    assert UNMAPPED_TYPE in {entry.id for entry in catalogue.document.canonical_types}
    assert UNMAPPED_ACTION in catalogue.document.canonical_actions
    assert catalogue.content_hash
    assert len(catalogue.content_hash) == 64
    # Process-wide singleton is the same validated document.
    live = get_event_catalogue()
    assert live.content_hash == catalogue.content_hash
    assert "issues" in live.known_source_events("github")
    assert "note" in live.known_source_events("gitlab")


def test_catalogue_rejects_invalid_document(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 1\ncanonical_types: []\n", encoding="utf-8")
    with pytest.raises(EventCatalogueError):
        load_event_catalogue(bad)


def test_unknown_source_event_maps_to_unmapped_and_is_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalogue = get_event_catalogue()
    before = catalogue.unmapped_count()
    resolved = catalogue.resolve("github", "workflow_run")
    assert resolved.is_unmapped
    assert resolved.canonical_type == UNMAPPED_TYPE
    assert catalogue.unmapped_count() == before + 1

    register_builtin_handlers()
    event = _github_parse(
        {"x-github-event": "workflow_run", "x-github-delivery": "u-1"},
        {"action": "completed"},
    )
    assert event is not None
    assert event.action == UNMAPPED_ACTION
    assert event.canonical_type == UNMAPPED_TYPE
    assert event.catalogue_content_hash == catalogue.content_hash
    assert catalogue.unmapped_count() == before + 2

    monkeypatch.setenv("TEST_CAT_SECRET", "shh")
    receiver = WebhookReceiver()
    receiver.configure("github", WebhookConfig(enabled=True, secret_env="TEST_CAT_SECRET"))
    body = b'{"action":"completed"}'
    sig = "sha256=" + hmac.new(b"shh", body, hashlib.sha256).hexdigest()
    result = receiver.receive(
        "github",
        {
            "x-hub-signature-256": sig,
            "x-github-event": "workflow_run",
            "x-github-delivery": "u-recv-1",
        },
        body,
    )
    assert result.status == "unmapped"
    assert result.event is not None
    assert result.event.action == UNMAPPED_ACTION


def test_github_parser_has_no_literal_event_name_comparisons() -> None:
    tree = ast.parse(_WEBHOOK_RECEIVER.read_text(encoding="utf-8"), filename=str(_WEBHOOK_RECEIVER))
    targets = {
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_github_parse", "_gitlab_parse"}
    }
    assert {fn.name for fn in targets} == {"_github_parse", "_gitlab_parse"}

    findings: list[str] = []
    for fn in targets:
        for node in ast.walk(fn):
            if not isinstance(node, ast.Compare):
                continue
            literals = _string_literals_in_compare(node)
            hit = literals & _FORBIDDEN_SOURCE_EVENT_LITERALS
            if hit:
                findings.append(f"{fn.name}:{node.lineno}: {sorted(hit)}")
    assert not findings, (
        f"ingest parsers must map through the event catalogue; found literal source-event comparisons: {findings}"
    )


def _string_literals_in_compare(node: ast.Compare) -> set[str]:
    values: set[str] = set()
    for candidate in (node.left, *node.comparators):
        if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
            values.add(candidate.value)
        elif isinstance(candidate, ast.Set | ast.List | ast.Tuple):
            for elt in candidate.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    values.add(elt.value)
    return values


def test_journal_entry_records_catalogue_content_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalogue = get_event_catalogue()
    expected_hash = catalogue.content_hash
    assert len(expected_hash) == 64

    ledger_path = tmp_path / "ledger.jsonl"
    receiver = WebhookReceiver(ledger=ReplayLedger(ledger_path))
    receiver.configure("github", WebhookConfig(enabled=True, secret_env="TEST_CAT_SECRET"))
    monkeypatch.setenv("TEST_CAT_SECRET", "shh")

    body = json.dumps(
        {
            "action": "opened",
            "issue": {
                "id": 1,
                "number": 7,
                "html_url": "https://github.com/acme/repo/issues/7",
                "title": "t",
                "body": "b",
                "state": "open",
                "labels": [],
            },
            "repository": {"full_name": "acme/repo"},
        }
    ).encode("utf-8")
    headers = {
        "x-hub-signature-256": "sha256=" + hmac.new(b"shh", body, hashlib.sha256).hexdigest(),
        "x-github-event": "issues",
        "x-github-delivery": "journal-1",
    }
    result = receiver.receive("github", headers, body)
    assert result.status == "accepted"
    assert result.event is not None
    assert result.event.catalogue_content_hash == expected_hash

    lines = ledger_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["catalogue_content_hash"] == expected_hash
    assert entry["delivery_id"] == "github:journal-1"


def test_gitlab_note_extracts_via_catalogue_shape() -> None:
    event = _gitlab_parse(
        {"x-gitlab-event-uuid": "gl-note-1"},
        {
            "object_kind": "note",
            "object_attributes": {"action": "create"},
            "issue": {
                "iid": 9,
                "title": "Comment target",
                "description": "",
                "state": "opened",
                "url": "https://gitlab.example.com/acme/repo/-/issues/9",
            },
            "project": {"path_with_namespace": "acme/repo"},
            "labels": [],
        },
    )
    assert event is not None
    assert event.adapter == "gitlab"
    assert event.ticket.id == "acme/repo#9"
    assert event.canonical_type == "issue_comment"
    assert event.action == "created"
    assert event.catalogue_content_hash == get_event_catalogue().content_hash

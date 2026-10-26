"""
Integrity and provenance validation tests for the Q26-Q50 additional
evaluation set (golden_qa_25_additional.json).

Mirrors the contracts of test_golden_dataset.py, adapted for the second
25-question set: exact size, Q26-Q50 IDs, negative-control and
conversational quotas, page ranges, current-corpus chunk existence, and
non-duplication against the Q01-Q25 set.
"""

import json
from pathlib import Path

import pytest

from due_diligence.evaluation.schemas import GoldenDataset


@pytest.fixture
def additional_dataset() -> GoldenDataset:
    dataset_path = Path("tests/eval/data/golden_qa_25_additional.json")
    assert dataset_path.exists(), f"Additional dataset file not found at {dataset_path}"
    return GoldenDataset.from_file(dataset_path)


def test_additional_dataset_size_and_metadata(additional_dataset: GoldenDataset) -> None:
    assert len(additional_dataset.questions) == 25, "Additional dataset must contain exactly 25 questions"
    assert additional_dataset.company == "HDFC Bank Limited"
    assert additional_dataset.document_name == "hdfc-bank__annual-report__fy2025-26.pdf"
    assert additional_dataset.version == "2.0.0"


def test_additional_unique_question_ids(additional_dataset: GoldenDataset) -> None:
    ids = [q.id for q in additional_dataset.questions]
    assert len(ids) == len(set(ids)), f"Duplicate question IDs found: {ids}"
    expected_ids = [f"Q{i:02d}" for i in range(26, 51)]
    assert ids == expected_ids, f"Question IDs must be sequential Q26 to Q50, got {ids}"


def test_no_duplication_with_set1(additional_dataset: GoldenDataset) -> None:
    set1_path = Path("tests/eval/data/golden_qa_25.json")
    if not set1_path.exists():
        pytest.skip("Q01-Q25 dataset not present for duplication check")
    set1_ids = {q.id for q in GoldenDataset.from_file(set1_path).questions}
    set1_texts = {q.question.strip().lower() for q in GoldenDataset.from_file(set1_path).questions}
    for q in additional_dataset.questions:
        assert q.id not in set1_ids, f"Question ID {q.id} collides with Q01-Q25 set"
        assert q.question.strip().lower() not in set1_texts, f"Question {q.id} duplicates a Q01-Q25 question"


def test_additional_negative_controls_contract(additional_dataset: GoldenDataset) -> None:
    negatives = [q for q in additional_dataset.questions if not q.evidence_should_exist]
    assert len(negatives) >= 3, "Must contain at least 3 negative/unanswerable questions"

    for n in negatives:
        assert n.is_uncertain is True, f"Negative question {n.id} must have is_uncertain=True"
        assert n.expected_pages == [], f"Negative question {n.id} must have empty expected_pages"
        assert len(n.expected_facts) > 0, f"Negative question {n.id} must specify expected refusal phrases"


def test_additional_conversational_contract(additional_dataset: GoldenDataset) -> None:
    followups = [q for q in additional_dataset.questions if q.failure_mode == "followup_query_rewriting"]
    assert len(followups) >= 3, "Must contain at least 3 follow-up queries requiring rewriting"

    for f in followups:
        assert f.conversation_history is not None, f"Follow-up question {f.id} must define conversation_history"
        assert len(f.conversation_history) >= 2, "Conversation history must contain at least user and assistant turns"


def test_additional_answerable_evidence_provenance(additional_dataset: GoldenDataset) -> None:
    answerables = [q for q in additional_dataset.questions if q.evidence_should_exist]
    assert len(answerables) == 22

    for q in answerables:
        assert len(q.expected_facts) > 0, f"Question {q.id} must define non-empty expected facts"
        assert len(q.expected_pages) > 0, f"Question {q.id} must define non-empty expected pages"
        for page in q.expected_pages:
            assert 1 <= page <= 678, f"Question {q.id} has out-of-range page {page}"


def test_additional_coverage_quotas(additional_dataset: GoldenDataset) -> None:
    by_mode = {q.failure_mode for q in additional_dataset.questions}
    assert "table_lookup" in by_mode
    assert "list_completeness" in by_mode
    assert "calculation_from_explicit_values" in by_mode
    assert "ambiguous_terminology" in by_mode

    tables = [q for q in additional_dataset.questions if q.failure_mode == "table_lookup"]
    assert len(tables) >= 5, "Must contain at least 5 table/structured-data questions"
    lists = [q for q in additional_dataset.questions if q.failure_mode == "list_completeness"]
    assert len(lists) >= 2, "Must contain at least 2 list-completeness questions"
    entity_qs = [q for q in additional_dataset.questions if q.category == "subsidiary"]
    assert len(entity_qs) >= 4, "Must contain at least 4 entity/subsidiary-scope questions"


def test_additional_target_chunk_ids_exist_in_canonical_corpus(
    additional_dataset: GoldenDataset,
) -> None:
    canonical_chunks_file = Path("data/processed/hdfc_canonical_chunks.jsonl")
    if not canonical_chunks_file.exists():
        pytest.skip("Canonical chunks file not present")

    all_chunk_ids: set[str] = set()
    with open(canonical_chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            chunk_data = json.loads(line)
            all_chunk_ids.add(chunk_data["chunk_id"])

    missing_chunks: list[tuple[str, str]] = []
    for q in additional_dataset.questions:
        for cid in q.target_chunk_ids:
            if cid not in all_chunk_ids:
                missing_chunks.append((q.id, cid))

    assert not missing_chunks, f"Target chunks not found in canonical corpus: {missing_chunks}"

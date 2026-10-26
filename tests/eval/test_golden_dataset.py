"""
Integrity and provenance validation tests for the 25-question Golden Benchmark Dataset.
"""

import json
from pathlib import Path

import pytest

from due_diligence.evaluation.schemas import GoldenDataset


@pytest.fixture
def golden_dataset() -> GoldenDataset:
    dataset_path = Path("tests/eval/data/golden_qa_25.json")
    assert dataset_path.exists(), f"Golden dataset file not found at {dataset_path}"
    return GoldenDataset.from_file(dataset_path)


def test_dataset_size_and_metadata(golden_dataset: GoldenDataset) -> None:
    assert len(golden_dataset.questions) == 25, "Golden dataset must contain exactly 25 questions"
    assert golden_dataset.company == "HDFC Bank Limited"
    assert golden_dataset.document_name == "hdfc-bank__annual-report__fy2025-26.pdf"
    assert golden_dataset.version == "1.0.1"


def test_unique_question_ids(golden_dataset: GoldenDataset) -> None:
    ids = [q.id for q in golden_dataset.questions]
    assert len(ids) == len(set(ids)), f"Duplicate question IDs found: {ids}"
    expected_ids = [f"Q{i:02d}" for i in range(1, 26)]
    assert ids == expected_ids, f"Question IDs must be sequential Q01 to Q25, got {ids}"


def test_all_16_failure_modes_covered(golden_dataset: GoldenDataset) -> None:
    expected_failure_modes = {
        "direct_factual_lookup",
        "financial_metric_lookup",
        "table_lookup",
        "ratio_capital",
        "year_over_year_comparison",
        "risk_question",
        "cybersecurity_question",
        "business_segment_question",
        "management_business_narrative",
        "multi_evidence_synthesis",
        "cross_page_evidence",
        "calculation_from_explicit_values",
        "not_disclosed_question",
        "negative_no_evidence_question",
        "ambiguous_terminology",
        "followup_query_rewriting",
    }
    present_failure_modes = {q.failure_mode for q in golden_dataset.questions}
    missing = expected_failure_modes - present_failure_modes
    assert not missing, f"Missing required failure modes in golden dataset: {missing}"


def test_negative_controls_contract(golden_dataset: GoldenDataset) -> None:
    negatives = [q for q in golden_dataset.questions if not q.evidence_should_exist]
    assert len(negatives) >= 2, "Must contain at least 2 negative/unanswerable questions"

    for n in negatives:
        assert n.is_uncertain is True, f"Negative question {n.id} must have is_uncertain=True"
        assert n.expected_pages == [], f"Negative question {n.id} must have empty expected_pages"
        assert len(n.expected_facts) > 0, f"Negative question {n.id} must specify expected refusal phrases"


def test_conversational_followup_contract(golden_dataset: GoldenDataset) -> None:
    followups = [q for q in golden_dataset.questions if q.failure_mode == "followup_query_rewriting"]
    assert len(followups) >= 1, "Must contain at least one follow-up query requiring rewriting"

    for f in followups:
        assert f.conversation_history is not None, f"Follow-up question {f.id} must define conversation_history"
        assert len(f.conversation_history) >= 2, "Conversation history must contain at least user and assistant turns"


def test_answerable_questions_evidence_provenance(golden_dataset: GoldenDataset) -> None:
    answerables = [q for q in golden_dataset.questions if q.evidence_should_exist]
    assert len(answerables) == 23

    for q in answerables:
        assert len(q.expected_facts) > 0, f"Question {q.id} must define non-empty expected facts"
        assert len(q.expected_pages) > 0, f"Question {q.id} must define non-empty expected pages"
        for page in q.expected_pages:
            assert 1 <= page <= 678, f"Question {q.id} has out-of-range page {page}"


def test_target_chunk_ids_exist_in_canonical_corpus(golden_dataset: GoldenDataset) -> None:
    canonical_chunks_file = Path("data/processed/hdfc_canonical_chunks.jsonl")
    if not canonical_chunks_file.exists():
        pytest.skip("Canonical chunks file not present")

    all_chunk_ids: set[str] = set()
    with open(canonical_chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            chunk_data = json.loads(line)
            all_chunk_ids.add(chunk_data["chunk_id"])

    missing_chunks: list[tuple[str, str]] = []
    for q in golden_dataset.questions:
        for cid in q.target_chunk_ids:
            if cid not in all_chunk_ids:
                missing_chunks.append((q.id, cid))

    assert not missing_chunks, f"Target chunks not found in canonical corpus: {missing_chunks}"

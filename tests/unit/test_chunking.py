"""
Unit tests for document-aware chunking (chunks.py, chunker.py).
Tests all 40 requirements across Narrative, Headings, Lists, Tables, Provenance, Determinism, and Configuration.
"""

from due_diligence.ingestion.chunker import (
    ChunkingConfig,
    DocumentChunker,
    count_tokens,
    verify_chunking_provenance,
)
from due_diligence.ingestion.elements import (
    Heading,
    ListItem,
    NarrativeText,
    Table,
)


class TestNarrativeChunking:
    """Tests 1-7: Narrative accumulation, sizing, splitting, and overlap."""

    def test_multiple_narrative_paragraphs_accumulate(self) -> None:
        """1. Multiple adjacent narrative paragraphs accumulate into a single chunk."""
        elems = [
            NarrativeText(
                element_id="doc_p1_n1",
                page_number=1,
                bbox=(40, 50, 500, 80),
                document_name="test.pdf",
                text="The Bank recorded strong asset quality throughout the fiscal year.",
            ),
            NarrativeText(
                element_id="doc_p1_n2",
                page_number=1,
                bbox=(40, 90, 500, 120),
                document_name="test.pdf",
                text="Net interest margin expanded by 15 basis points to 3.45%.",
            ),
        ]
        chunker = DocumentChunker()
        chunks = chunker.chunk_document_elements(elems)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "narrative"
        assert "asset quality" in chunks[0].text
        assert "Net interest margin" in chunks[0].text
        assert chunks[0].source_element_ids == ["doc_p1_n1", "doc_p1_n2"]

    def test_narrative_target_size_respected(self) -> None:
        """2. Narrative chunk flushes when target_max_tokens is reached."""
        # 10 paragraphs each ~30 tokens => ~300 tokens total. With target=100, splits into multiple chunks.
        elems = [
            NarrativeText(
                element_id=f"doc_p1_n{i}",
                page_number=1,
                bbox=(40, 50 + i * 20, 500, 70 + i * 20),
                document_name="test.pdf",
                text=f"Paragraph {i}: The quick brown fox jumps over the lazy dog repeatedly to generate token volume.",
            )
            for i in range(10)
        ]
        cfg = ChunkingConfig(target_max_tokens=50, hard_max_tokens=100, narrative_overlap_tokens=10)
        chunker = DocumentChunker(cfg)
        chunks = chunker.chunk_document_elements(elems)
        assert len(chunks) > 1
        for c in chunks:
            assert count_tokens(c.text) <= 120

    def test_narrative_overlap_is_created_when_split(self) -> None:
        """3 & 4. Overlap is generated across split narrative sequences and present in next chunk."""
        elems = [
            NarrativeText(
                element_id="doc_p1_n1",
                page_number=1,
                bbox=(40, 50, 500, 80),
                document_name="test.pdf",
                text="First segment of the narrative detailing risk management policies and regulatory compliance.",
            ),
            NarrativeText(
                element_id="doc_p1_n2",
                page_number=1,
                bbox=(40, 90, 500, 120),
                document_name="test.pdf",
                text="Second segment detailing capital adequacy, liquidity coverage ratio, and leverage metrics.",
            ),
        ]
        cfg = ChunkingConfig(target_max_tokens=15, hard_max_tokens=30, narrative_overlap_tokens=8)
        chunker = DocumentChunker(cfg)
        chunks = chunker.chunk_document_elements(elems)

        assert len(chunks) == 2
        assert chunks[0].overlap_tokens == 0
        assert chunks[1].overlap_tokens > 0
        # Overlap text from chunk 0 must appear at the beginning of chunk 1
        tail_words = chunks[0].text.split()[-3:]
        assert any(w in chunks[1].text for w in tail_words)

    def test_narrative_content_is_not_lost(self) -> None:
        """5. Full narrative content across elements is preserved without omission."""
        elems = [
            NarrativeText(
                element_id=f"doc_p1_n{i}",
                page_number=1,
                bbox=(40, 50 + i * 30, 500, 75 + i * 30),
                document_name="test.pdf",
                text=f"UniqueKeyword{i} detailing specialized financial disclosures for segment {i}.",
            )
            for i in range(5)
        ]
        cfg = ChunkingConfig(target_max_tokens=25, hard_max_tokens=50)
        chunker = DocumentChunker(cfg)
        chunks = chunker.chunk_document_elements(elems)
        all_text = " ".join(c.text for c in chunks)
        for i in range(5):
            assert f"UniqueKeyword{i}" in all_text

    def test_oversized_single_narrative_split(self) -> None:
        """6 & 7. Single NarrativeText exceeding hard limit is split internally, preserving element ID."""
        long_text = " ".join([f"Sentence {i} explains risk factor {i} in exhaustive detail." for i in range(40)])
        elem = NarrativeText(
            element_id="doc_p1_huge_narrative",
            page_number=1,
            bbox=(40, 50, 500, 400),
            document_name="test.pdf",
            text=long_text,
        )
        cfg = ChunkingConfig(target_max_tokens=50, hard_max_tokens=70, narrative_overlap_tokens=15)
        chunker = DocumentChunker(cfg)
        chunks = chunker.chunk_document_elements([elem])

        assert len(chunks) > 1
        for c in chunks:
            assert "doc_p1_huge_narrative" in c.source_element_ids
        assert chunks[0].overlap_tokens == 0
        assert chunks[1].overlap_tokens > 0


class TestHeadingRules:
    """Tests 8-11: Heading context, propagation, hierarchy, and multi-page persistence."""

    def test_heading_updates_section_heading_and_path(self) -> None:
        """8 & 9. Heading updates section_heading and heading_path."""
        elems = [
            Heading(
                element_id="h1",
                page_number=1,
                bbox=(40, 20, 400, 40),
                document_name="test.pdf",
                text="FINANCIAL HIGHLIGHTS",
                level=1,
                metadata={"heading_path": ["FINANCIAL HIGHLIGHTS"]},
            ),
            Heading(
                element_id="h2",
                page_number=1,
                bbox=(40, 50, 400, 70),
                document_name="test.pdf",
                text="Capital Adequacy",
                level=2,
                metadata={"heading_path": ["FINANCIAL HIGHLIGHTS", "Capital Adequacy"]},
            ),
            NarrativeText(
                element_id="n1",
                page_number=1,
                bbox=(40, 80, 500, 110),
                document_name="test.pdf",
                text="The Bank CET1 ratio stood at 16.8%.",
            ),
        ]
        chunker = DocumentChunker()
        chunks = chunker.chunk_document_elements(elems)

        assert len(chunks) == 1
        assert chunks[0].section_heading == "Capital Adequacy"
        assert chunks[0].heading_path == ["FINANCIAL HIGHLIGHTS", "Capital Adequacy"]
        assert "h1" in chunks[0].source_element_ids
        assert "h2" in chunks[0].source_element_ids
        assert "n1" in chunks[0].source_element_ids

    def test_heading_does_not_become_orphan_chunk(self) -> None:
        """10. Heading preceding narrative does not create an isolated tiny chunk."""
        elems = [
            Heading(
                element_id="h1",
                page_number=1,
                bbox=(40, 20, 400, 40),
                document_name="test.pdf",
                text="Management Discussion and Analysis",
                level=1,
            ),
            NarrativeText(
                element_id="n1",
                page_number=1,
                bbox=(40, 50, 500, 100),
                document_name="test.pdf",
                text="Macroeconomic indicators remained stable during the period under review.",
            ),
        ]
        chunker = DocumentChunker()
        chunks = chunker.chunk_document_elements(elems)
        assert len(chunks) == 1
        assert "Management Discussion and Analysis" in chunks[0].text
        assert "Macroeconomic indicators" in chunks[0].text

    def test_heading_context_survives_page_boundaries(self) -> None:
        """11. Heading on Page 100 propagates to Narrative on Page 101."""
        elems = [
            Heading(
                element_id="h_p100",
                page_number=100,
                bbox=(40, 20, 400, 40),
                document_name="test.pdf",
                text="Note 18 - Capital Adequacy",
                level=2,
                metadata={"heading_path": ["Notes", "Note 18 - Capital Adequacy"]},
            ),
            NarrativeText(
                element_id="n_p101",
                page_number=101,
                bbox=(40, 50, 500, 100),
                document_name="test.pdf",
                text="The following disclosures represent regulatory capital computation under Basel III.",
            ),
        ]
        chunker = DocumentChunker()
        chunks = chunker.chunk_document_elements(elems)
        assert len(chunks) == 1
        assert chunks[0].section_heading == "Note 18 - Capital Adequacy"
        assert chunks[0].heading_path == ["Notes", "Note 18 - Capital Adequacy"]
        assert chunks[0].page_numbers == [100, 101]


class TestListChunking:
    """Tests 12-14: Consecutive list grouping, splitting without bullet fragmentation, zero overlap."""

    def test_consecutive_list_items_grouped(self) -> None:
        """12. Consecutive ListItems remain grouped together."""
        elems = [
            NarrativeText(
                element_id="n1",
                page_number=1,
                bbox=(40, 30, 500, 50),
                document_name="test.pdf",
                text="Key corporate objectives include:",
            ),
            ListItem(
                element_id="l1",
                page_number=1,
                bbox=(40, 60, 500, 80),
                document_name="test.pdf",
                text="Sustained credit growth across retail segments.",
                marker="•",
            ),
            ListItem(
                element_id="l2",
                page_number=1,
                bbox=(40, 90, 500, 110),
                document_name="test.pdf",
                text="Prudent provisioning and asset quality governance.",
                marker="•",
            ),
        ]
        chunker = DocumentChunker()
        chunks = chunker.chunk_document_elements(elems)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "mixed"
        assert "Key corporate objectives include:" in chunks[0].text
        assert "• Sustained credit growth" in chunks[0].text
        assert "• Prudent provisioning" in chunks[0].text
        assert chunks[0].source_element_ids == ["n1", "l1", "l2"]

    def test_large_lists_split_only_between_complete_items(self) -> None:
        """13 & 14. Large lists split between complete items, no bullet splitting, overlap_tokens == 0."""
        elems = [
            ListItem(
                element_id=f"li_{i}",
                page_number=1,
                bbox=(40, 30 + i * 20, 500, 48 + i * 20),
                document_name="test.pdf",
                text=f"Corporate policy guideline number {i} regarding operational governance.",
                marker=f"{i}.",
            )
            for i in range(1, 15)
        ]
        cfg = ChunkingConfig(target_max_tokens=40, hard_max_tokens=60)
        chunker = DocumentChunker(cfg)
        chunks = chunker.chunk_document_elements(elems)

        assert len(chunks) > 1
        for c in chunks:
            assert c.chunk_type == "list"
            assert c.overlap_tokens == 0
            # Every line in chunk should start with a valid bullet marker
            lines = [ln.strip() for ln in c.text.split("\n\n") if ln.strip()]
            for ln in lines:
                assert any(ln.startswith(f"{k}.") for k in range(1, 15))


class TestTableChunking:
    """Tests 15-31: Table atomicity, header repetition, row splitting, metadata, zero overlap."""

    def test_normal_table_becomes_one_chunk(self) -> None:
        """15, 18, 19, 20. Small table produces one chunk, preserves markdown, caption, footnotes."""
        headers = ["Category", "Amount (INR Cr)"]
        rows = [["Deposits", "2,500,000"], ["Advances", "2,100,000"]]
        md = Table.render_markdown(headers, rows)
        table = Table(
            element_id="t1",
            page_number=5,
            bbox=(40, 100, 500, 250),
            document_name="test.pdf",
            headers=headers,
            rows=rows,
            markdown=md,
            caption="Key Balance Sheet Figures",
            footnotes=["Unaudited numbers."],
            confidence_score=0.99,
        )
        chunker = DocumentChunker()
        chunks = chunker.chunk_document_elements([table])

        assert len(chunks) == 1
        assert chunks[0].chunk_type == "table"
        assert "[Table: Key Balance Sheet Figures]" in chunks[0].text
        assert "| Category" in chunks[0].text
        assert "| Deposits" in chunks[0].text
        assert "Footnotes:\n- Unaudited numbers." in chunks[0].text
        assert chunks[0].overlap_tokens == 0
        assert chunks[0].table_group_id == "t1"
        assert chunks[0].table_slice_index == 0
        assert chunks[0].table_slice_count == 1

    def test_narrative_before_and_after_table_flushed(self) -> None:
        """16 & 17. Narrative before table is flushed, narrative after starts new chunk."""
        elems = [
            NarrativeText(
                element_id="n_before",
                page_number=1,
                bbox=(40, 20, 500, 50),
                document_name="test.pdf",
                text="Text before table.",
            ),
            Table(
                element_id="t_middle",
                page_number=1,
                bbox=(40, 60, 500, 200),
                document_name="test.pdf",
                headers=["H1", "H2"],
                rows=[["V1", "V2"]],
                markdown="| H1 | H2 |\n|---|---|\n| V1 | V2 |",
            ),
            NarrativeText(
                element_id="n_after",
                page_number=1,
                bbox=(40, 220, 500, 250),
                document_name="test.pdf",
                text="Text after table.",
            ),
        ]
        chunker = DocumentChunker()
        chunks = chunker.chunk_document_elements(elems)

        assert len(chunks) == 3
        assert chunks[0].chunk_type == "narrative"
        assert chunks[0].text == "Text before table."
        assert chunks[1].chunk_type == "table"
        assert "| H1" in chunks[1].text
        assert chunks[2].chunk_type == "narrative"
        assert chunks[2].text == "Text after table."

    def test_oversized_table_split_rules(self) -> None:
        """21-26, 28-31. Oversized table row splitting, header repetition, contiguity, invariants."""
        headers = ["Item", "FY26", "FY25", "Variance"]
        # 60 rows will easily exceed table_max_tokens=150
        rows = [[f"Line Item {i:02d}", f"{1000 + i}", f"{900 + i}", f"+{100}"] for i in range(60)]
        table = Table(
            element_id="t_huge",
            page_number=10,
            bbox=(40, 50, 500, 750),
            document_name="test.pdf",
            headers=headers,
            rows=rows,
            markdown=Table.render_markdown(headers, rows),
            caption="Comprehensive Income Statement",
            footnotes=["Note: In INR Crores."],
        )
        cfg = ChunkingConfig(table_max_tokens=150)
        chunker = DocumentChunker(cfg)
        chunks = chunker.chunk_document_elements([table])

        assert len(chunks) > 1
        slice_count = len(chunks)

        all_rendered_items: list[str] = []
        for idx, c in enumerate(chunks):
            assert c.chunk_type == "table"
            assert c.overlap_tokens == 0
            assert c.table_group_id == "t_huge"
            assert c.table_slice_index == idx
            assert c.table_slice_count == slice_count

            # 22. Complete header repeated in every table slice
            assert "| Item" in c.text
            assert "| Variance" in c.text

            # Check continuation caption on slices > 0
            if idx > 0:
                assert "Continuation" in c.text

            # Collect line items in order
            for i in range(60):
                tag = f"Line Item {i:02d}"
                if tag in c.text:
                    all_rendered_items.append(tag)

        # 24, 25, 26. Every row appears exactly once, none lost, none duplicated, order preserved
        assert len(all_rendered_items) == 60
        assert len(set(all_rendered_items)) == 60
        expected = [f"Line Item {i:02d}" for i in range(60)]
        assert all_rendered_items == expected

    def test_oversized_individual_row_intact(self) -> None:
        """27. Single huge table row exceeding table_max_tokens remains intact in its own slice."""
        headers = ["Clause", "Detailed Terms"]
        huge_text = " ".join([f"Subclause {i} stipulations." for i in range(80)])
        rows = [
            ["Row 1", "Normal short row"],
            ["Row 2", huge_text],
            ["Row 3", "Another short row"],
        ]
        table = Table(
            element_id="t_huge_row",
            page_number=12,
            bbox=(40, 50, 500, 600),
            document_name="test.pdf",
            headers=headers,
            rows=rows,
            markdown=Table.render_markdown(headers, rows),
        )
        cfg = ChunkingConfig(table_max_tokens=50)
        chunker = DocumentChunker(cfg)
        chunks = chunker.chunk_document_elements([table])

        assert len(chunks) == 3
        assert "Normal short row" in chunks[0].text
        assert huge_text in chunks[1].text
        assert "Another short row" in chunks[2].text


class TestProvenanceAndDeterminism:
    """Tests 32-38: Provenance verification, header/footer filtering, and determinism."""

    def test_page_numbers_unique_and_sorted(self) -> None:
        """32. page_numbers is sorted and unique."""
        elems = [
            NarrativeText(
                element_id="n1",
                page_number=2,
                bbox=(40, 50, 500, 80),
                document_name="test.pdf",
                text="Text on page 2.",
            ),
            NarrativeText(
                element_id="n2",
                page_number=1,
                bbox=(40, 50, 500, 80),
                document_name="test.pdf",
                text="Text on page 1.",
            ),
        ]
        chunker = DocumentChunker()
        chunks = chunker.chunk_document_elements(elems)
        assert chunks[0].page_numbers == [1, 2]

    def test_header_footer_filtering(self) -> None:
        """34. Elements with metadata is_header_footer == True are filtered out."""
        elems = [
            NarrativeText(
                element_id="hdr",
                page_number=1,
                bbox=(40, 10, 500, 30),
                document_name="test.pdf",
                text="HDFC BANK LIMITED | ANNUAL REPORT 2024-25",
                metadata={"is_header_footer": True},
            ),
            NarrativeText(
                element_id="body",
                page_number=1,
                bbox=(40, 50, 500, 100),
                document_name="test.pdf",
                text="Main substantive body content.",
            ),
        ]
        chunker = DocumentChunker()
        chunks = chunker.chunk_document_elements(elems)
        assert len(chunks) == 1
        assert "HDFC BANK LIMITED" not in chunks[0].text
        assert "Main substantive body content." in chunks[0].text
        assert chunks[0].source_element_ids == ["body"]

    def test_provenance_validation_invariant(self) -> None:
        """33 & 35. verify_chunking_provenance proves zero lost elements and full table row preservation."""
        elems = [
            Heading(
                element_id="h1",
                page_number=1,
                bbox=(40, 20, 400, 40),
                document_name="test.pdf",
                text="Income Statement",
            ),
            Table(
                element_id="t1",
                page_number=1,
                bbox=(40, 50, 500, 250),
                document_name="test.pdf",
                headers=["Account", "Amount"],
                rows=[[f"Account {i}", f"{i * 100}"] for i in range(25)],
                markdown=Table.render_markdown(["Account", "Amount"], [[f"Account {i}", f"{i * 100}"] for i in range(25)]),
            ),
            NarrativeText(
                element_id="n1",
                page_number=1,
                bbox=(40, 260, 500, 300),
                document_name="test.pdf",
                text="Substantive notes on financial results.",
            ),
        ]
        cfg = ChunkingConfig(table_max_tokens=60)
        chunker = DocumentChunker(cfg)
        chunks = chunker.chunk_document_elements(elems)

        is_valid, errors = verify_chunking_provenance(elems, chunks)
        assert is_valid, f"Provenance errors: {errors}"

    def test_determinism_repeated_runs(self) -> None:
        """36, 37, 38. Running twice produces identical chunks, chunk IDs, and table slices."""
        elems = [
            Heading(
                element_id="h1",
                page_number=1,
                bbox=(40, 20, 400, 40),
                document_name="test.pdf",
                text="Executive Summary",
            ),
            NarrativeText(
                element_id="n1",
                page_number=1,
                bbox=(40, 50, 500, 100),
                document_name="test.pdf",
                text="The year witnessed resilient performance across all metrics.",
            ),
            Table(
                element_id="t1",
                page_number=1,
                bbox=(40, 110, 500, 300),
                document_name="test.pdf",
                headers=["Metric", "Value"],
                rows=[[f"M{i}", f"V{i}"] for i in range(30)],
                markdown=Table.render_markdown(["Metric", "Value"], [[f"M{i}", f"V{i}"] for i in range(30)]),
            ),
        ]
        chunker = DocumentChunker(ChunkingConfig(table_max_tokens=80))
        run1 = chunker.chunk_document_elements(elems)
        run2 = chunker.chunk_document_elements(elems)

        assert len(run1) == len(run2)
        for c1, c2 in zip(run1, run2, strict=True):
            assert c1.chunk_id == c2.chunk_id
            assert c1.text == c2.text
            assert c1.chunk_type == c2.chunk_type
            assert c1.page_numbers == c2.page_numbers
            assert c1.source_element_ids == c2.source_element_ids
            assert c1.overlap_tokens == c2.overlap_tokens
            assert c1.table_metadata == c2.table_metadata


class TestConfigurationChanges:
    """Tests 39 & 40: Predictable behavior under configuration threshold changes."""

    def test_target_max_tokens_configuration(self) -> None:
        """39. Changing target_max_tokens changes narrative chunk count predictably."""
        elems = [
            NarrativeText(
                element_id=f"n{i}",
                page_number=1,
                bbox=(40, 50 + i * 20, 500, 68 + i * 20),
                document_name="test.pdf",
                text=f"Paragraph {i} contains detailed prose for token accumulation testing.",
            )
            for i in range(20)
        ]
        # Larger limit => fewer chunks
        c_large = DocumentChunker(ChunkingConfig(target_max_tokens=400, hard_max_tokens=500)).chunk_document_elements(elems)
        # Smaller limit => more chunks
        c_small = DocumentChunker(ChunkingConfig(target_max_tokens=50, hard_max_tokens=80)).chunk_document_elements(elems)

        assert len(c_small) > len(c_large)

    def test_table_max_tokens_configuration(self) -> None:
        """40. Changing table_max_tokens changes table slicing predictably."""
        headers = ["Col1", "Col2"]
        rows = [[f"Val1_{i}", f"Val2_{i}"] for i in range(40)]
        table = Table(
            element_id="t_cfg",
            page_number=1,
            bbox=(40, 50, 500, 500),
            document_name="test.pdf",
            headers=headers,
            rows=rows,
            markdown=Table.render_markdown(headers, rows),
        )
        c_large = DocumentChunker(ChunkingConfig(table_max_tokens=1500)).chunk_document_elements([table])
        c_small = DocumentChunker(ChunkingConfig(table_max_tokens=100)).chunk_document_elements([table])

        assert len(c_large) == 1
        assert len(c_small) > 1


class TestScheduleBoundariesAndTotalRows:
    """Regression tests for the Q06 failure: schedule isolation and Total-row placement."""

    def _schedule_table(
        self,
        element_id: str,
        heading_id: str,
        schedule_title: str,
        total_value: str,
        caption: str | None = None,
    ) -> list:
        heads = ["Schedule", "FY26", "FY25"]
        rows = [
            ["I Item one", "100.00", "90.00"],
            ["II Item two", "200.00", "180.00"],
            ["Total", total_value, "270.00"],
        ]
        return [
            Heading(
                element_id=heading_id,
                page_number=508,
                bbox=(40, 20, 400, 40),
                document_name="test.pdf",
                text=schedule_title,
                level=1,
            ),
            Table(
                element_id=element_id,
                page_number=508,
                bbox=(40, 60, 500, 300),
                document_name="test.pdf",
                headers=heads,
                rows=rows,
                markdown=Table.render_markdown(heads, rows),
                caption=caption,
            ),
        ]

    def test_consecutive_schedule_tables_stay_in_separate_chunks(self) -> None:
        """A new SCHEDULE heading forces a chunk boundary; totals never cross schedules."""
        elems = self._schedule_table("tab11", "h11", "SCHEDULE 11  -  OTHER ASSETS", "248,623.68")
        elems += self._schedule_table(
            "tab12", "h12", "SCHEDULE  12 - CONTINGENT LIABILITIES", "3,479,487.31",
            caption="Schedule 12",
        )
        chunks = DocumentChunker().chunk_document_elements(elems)

        assert len(chunks) == 2
        assert chunks[0].section_heading == "SCHEDULE 11  -  OTHER ASSETS"
        assert chunks[1].section_heading == "SCHEDULE  12 - CONTINGENT LIABILITIES"
        assert "248,623.68" in chunks[0].text
        assert "248,623.68" not in chunks[1].text
        assert "3,479,487.31" in chunks[1].text
        assert "3,479,487.31" not in chunks[0].text

    def test_atomic_table_keeps_total_row_with_line_items(self) -> None:
        """A normal-sized table chunk contains both line items and its Total row."""
        elems = self._schedule_table("tab1", "h1", "SCHEDULE  12 - CONTINGENT LIABILITIES", "34,179,458,972")
        chunks = DocumentChunker().chunk_document_elements(elems)

        assert len(chunks) == 1
        assert "| I Item one" in chunks[0].text
        assert "| Total " in chunks[0].text
        assert "34,179,458,972" in chunks[0].text

    def test_sliced_table_total_row_in_final_slice_with_header(self) -> None:
        """An oversized table's Total row lands in the final slice with headers repeated."""
        headers = ["Item", "FY26"]
        rows = [[f"Line Item {i:02d}", f"{1000 + i}"] for i in range(40)]
        rows.append(["Total", "999,999"])
        table = Table(
            element_id="t_big_total",
            page_number=389,
            bbox=(40, 50, 500, 750),
            document_name="test.pdf",
            headers=headers,
            rows=rows,
            markdown=Table.render_markdown(headers, rows),
        )
        cfg = ChunkingConfig(table_max_tokens=150)
        chunks = DocumentChunker(cfg).chunk_document_elements([table])

        assert len(chunks) > 1
        for c in chunks[:-1]:
            assert "| Total " not in c.text
        final = chunks[-1]
        assert "| Total " in final.text
        assert "999,999" in final.text
        assert "| Item " in final.text  # headers repeated in final slice
        # Every row exactly once across slices
        assert sum(c.text.count("Line Item") for c in chunks) == 40
        assert sum(c.text.count("| Total ") for c in chunks) == 1

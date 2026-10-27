import pytest
from dataclasses import replace

from agentic_rag.evaluation import eval_retrieval
from agentic_rag.retrieval import build_pipeline, rrf_fuse
from agentic_rag.retrieval.chunking import chunk_document
from agentic_rag.retrieval.stores import IndexMismatchError
from agentic_rag.text import tokenize

DOC = """---
year: 2025
doc_type: quy chế
---
# Quy chế
## Chương 3
### Điều 12. Bảo lưu
1. Sinh viên được bảo lưu tối đa 2 học kỳ.
2. Nộp đơn trước ít nhất 2 tuần.
### Điều 13. Thôi học
1. Nộp đơn thôi học tại phòng Đào tạo.
"""


def test_stopword_does_not_eat_bao_luu():
    assert "bao" not in tokenize("trước bao lâu", stop=True)      # "bao lâu" là stopword
    assert tokenize("bảo lưu", stop=True) == ["bao", "luu"]       # nhưng "bảo lưu" thì giữ


def test_chunking_parent_child_breadcrumb_and_metadata():
    parents, chunks = chunk_document(DOC, "qc", "qc.md", child_max_chars=60)
    assert [p.title for p in parents] == ["Quy chế > Chương 3 > Điều 12. Bảo lưu", "Quy chế > Chương 3 > Điều 13. Thôi học"]
    assert all(c.text.startswith("[Quy chế > Chương 3 > ") for c in chunks)
    assert all(c.metadata["year"] == 2025 and c.metadata["source"] == "qc.md" for c in chunks)
    assert len({c.parent_id for c in chunks}) == 2


def test_rrf_rewards_agreement():
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]])
    assert [x for x, _ in fused[:2]] == ["a", "b"] or [x for x, _ in fused[:2]] == ["b", "a"]
    assert fused[-1][0] in ("c", "d")


def test_ingest_is_idempotent(pipeline, settings):
    n = pipeline.store.count()
    pipeline.ingest_dir(settings.docs_dir)
    assert pipeline.store.count() == n


def test_retrieve_finds_right_article(pipeline):
    r = pipeline.retrieve("Nộp đơn bảo lưu trước bao lâu?")
    assert r.passages and "Điều 12" in r.passages[0].label


def test_out_of_scope_returns_nothing(pipeline):
    assert pipeline.retrieve("Cách nấu phở bò ngon").passages == []


def test_metadata_filter(pipeline):
    r = pipeline.retrieve("giờ giấc", {"doc_type": "nội quy"})
    assert r.passages and all(p.metadata["doc_type"] == "nội quy" for p in r.passages)
    assert pipeline.retrieve("bảo lưu", {"doc_type": "nội quy"}).passages == []


def test_index_persists_and_fingerprint_mismatch_is_caught(pipeline, settings):
    again = build_pipeline(settings)                       # nạp lại từ đĩa
    assert again.store.count() == pipeline.store.count()
    assert again.retrieve("giờ đóng cổng ký túc xá").passages
    with pytest.raises(IndexMismatchError):
        build_pipeline(replace(settings, embedding_dim=512))  # đổi số chiều => phải ingest lại


def test_retrieval_quality_on_golden_set(pipeline):
    r = eval_retrieval(pipeline, "data/eval/retrieval.jsonl")
    assert r["recall_at_k"] >= 0.9 and r["no_answer_accuracy"] == 1.0, r

import threading

from aqg.ledger import Ledger


def test_ledger_round_trip(tmp_path):
    ledger = Ledger(tmp_path / "nested" / "ledger.sqlite")
    ledger.start_run("r1", {"limit": 3, "languages": ["python3"]})
    ledger.record_stage("r1", "triage", "ok", "3 candidates", content_item_id="c1")
    ledger.finish_run("r1", "completed", {"accepted": 1})
    run = ledger.get_run("r1")
    assert run["params"] == {"limit": 3, "languages": ["python3"]}
    assert run["summary"] == {"accepted": 1} and run["status"] == "completed"
    assert run["started_at"].endswith("+00:00") and run["finished_at"]
    assert ledger.get_run("missing") is None

    ledger.mark_source("c1", 1, "h1", "rejected", "r1", problem_title="Encode a CompactSize Integer")
    assert ledger.concept_used("c1", "h1")
    assert not ledger.title_used("Encode a CompactSize Integer")
    ledger.mark_source("c1", 1, "h1", "accepted", "r1", problem_title="Encode a CompactSize Integer")  # upsert
    assert ledger.title_used("encode a compactsize integer!")
    ledger.mark_source("c2", 1, "", "skipped:no_text", "r1")
    ledger.mark_source("c3", 1, "", "rejected", "r1")
    assert ledger.used_content_item_ids() == {"c1", "c2"}
    assert ledger.used_content_item_ids(("rejected",)) == {"c3"}
    assert not ledger.concept_used("c2", "")
    rows = ledger._read("SELECT COUNT(*) AS n FROM used_sources WHERE content_item_id = 'c1'")
    assert rows[0]["n"] == 1


def test_llm_calls(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite")
    threads = [threading.Thread(target=ledger.record_llm_call,
                                kwargs=dict(run_id="r1", stage="extract", provider="ollama", model="m",
                                            input_tokens=10, output_tokens=5, latency_ms=100, ok=True,
                                            unknown_key="ignored"))
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ledger.record_llm_call(run_id="r2", ok=False, error="boom")
    assert ledger.llm_totals("r1") == {"calls": 8, "ok_calls": 8, "input_tokens": 80, "output_tokens": 40, "latency_ms": 800}
    assert ledger.llm_totals("r2")["calls"] == 1
    assert ledger.llm_totals("none") == {"calls": 0, "ok_calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_ms": 0}
    reopened = Ledger(tmp_path / "ledger.sqlite")
    assert reopened.llm_totals("r1")["calls"] == 8

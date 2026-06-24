"""Unit tests for the dataset validation endpoint.

Fast and dependency-light — no model downloads, no torch. Exercises every
supported file format plus the error paths.
"""

import json

from llmtune.server.routers.dataset import ValidateRequest, validate_dataset


def _validate(path):
    return validate_dataset(ValidateRequest(path=str(path)))


def test_jsonl(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"instruction":"hi","response":"yo"}\n{"instruction":"a","response":"b"}\n')
    out = _validate(p)
    assert out["status"] == "ok"
    assert out["row_count"] == 2
    assert out["format"] == "instruction + response"
    assert out["columns"] == ["instruction", "response"]


def test_json_array(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps([{"text": "a"}, {"text": "b"}, {"text": "c"}]))
    out = _validate(p)
    assert out["status"] == "ok"
    assert out["row_count"] == 3
    assert out["format"] == "text"


def test_csv(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("instruction,response\nq1,a1\nq2,a2\n")
    out = _validate(p)
    assert out["status"] == "ok"
    assert out["row_count"] == 2
    assert out["columns"] == ["instruction", "response"]


def test_txt_skips_blank_lines(tmp_path):
    p = tmp_path / "d.txt"
    p.write_text("one\n\ntwo\nthree\n")
    out = _validate(p)
    assert out["status"] == "ok"
    assert out["row_count"] == 3
    assert out["columns"] == ["text"]


def test_missing_file():
    out = _validate("/nope/does-not-exist.jsonl")
    assert out["status"] == "not_found"


def test_directory(tmp_path):
    out = _validate(tmp_path)
    assert out["status"] == "not_file"


def test_empty_path():
    out = _validate("")
    assert out["status"] == "empty"


def test_bad_json(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text("not json at all\n")
    out = _validate(p)
    assert out["status"] == "parse_error"

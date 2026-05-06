from __future__ import annotations

import os

import ds2api_oracle


def test_go_executable_uses_existing_candidate_when_go_is_not_on_path(monkeypatch, tmp_path) -> None:
    go_name = "go.exe" if os.name == "nt" else "go"
    go_exe = tmp_path / go_name
    go_exe.write_text("", encoding="utf-8")
    monkeypatch.setattr(ds2api_oracle.shutil, "which", lambda name: None)

    assert ds2api_oracle._go_executable(extra_candidates=[go_exe]) == str(go_exe)

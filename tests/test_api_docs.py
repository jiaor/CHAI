"""Every Python block in ``docs/api.md`` runs.

The page is the forward-model API reference; a reference whose examples have rotted is
worse than none.  Each block runs on its own, on the CPU, so each must be
self-contained -- a reader pastes one block, not the page.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

DOC = Path(__file__).resolve().parents[1] / "docs" / "api.md"
BLOCKS = re.findall(r"```python\n(.*?)```", DOC.read_text(), flags=re.DOTALL)


def test_the_page_has_examples():
    assert len(BLOCKS) >= 4


@pytest.mark.parametrize("k", range(len(BLOCKS)))
def test_api_block_runs(k, monkeypatch):
    # Hide any GPU: the page picks "cuda" when it can, and the suite must not.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    exec(compile(BLOCKS[k], f"{DOC.name}[{k}]", "exec"), {"__name__": "api_doc"})  # noqa: S102

from __future__ import annotations

import pytest

from examples.jobpacer.diagnostics import gpu_interference_profile as diagnostic


def test_interference_diagnostic_refuses_without_two_visible_gpus(tmp_path, monkeypatch):
    output = tmp_path / "interference"
    monkeypatch.setattr(diagnostic.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(diagnostic.torch.cuda, "device_count", lambda: 1)

    with pytest.raises(RuntimeError, match="exactly two visible CUDA devices"):
        diagnostic.run_diagnostic(output)

    assert not output.exists()

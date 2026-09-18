"""Test interface contracts independently of the invoking experiment shell."""
import os
import pytest

INTERFACE_ENV = ("VIA_CONTROL_INTERFACE", "VIA_EXTRA_GUIDE", "VIA_DG_FEEDBACK")
for key in INTERFACE_ENV:
    os.environ.pop(key, None)

@pytest.fixture(autouse=True)
def neutral_interface_environment(monkeypatch):
    for key in INTERFACE_ENV:
        monkeypatch.delenv(key, raising=False)

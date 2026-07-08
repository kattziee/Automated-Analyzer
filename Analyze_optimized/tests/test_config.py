import sys

from config import python_runtime_support


def test_python_runtime_support_reports_version_details():
    info = python_runtime_support()

    assert "version" in info
    assert "supported" in info
    assert info["version"] == sys.version.split()[0]
    assert isinstance(info["supported"], bool)

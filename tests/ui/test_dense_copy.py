"""Behaviour: instructional chrome is a tooltip, not a caption."""

from __future__ import annotations

from pathlib import Path

HINT_PHRASES = (
    "Drag the title bar to reorder",
    "Internal quantities stay SI",
    "Start from wheat datasets",
    "Select datasets in the tree",
    "Simulate enabled configurations at this location",
    "Defaults is readonly",
    "Uses Inputs plus each enabled configuration",
    "Select a process, then edit inputs",
    "Run the process to plot output variables",
)


def test_instructional_hints_are_not_shown_as_labels():
    root = Path(__file__).resolve().parents[2] / "orion" / "ui"
    missing = list(HINT_PHRASES)
    labeled = []
    for path in sorted(root.glob("*.py")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for phrase in HINT_PHRASES:
                if phrase not in line:
                    continue
                if phrase in missing:
                    missing.remove(phrase)
                if "ui.label(" in line and "tooltip(" not in line:
                    labeled.append(f"{path.name}:{line_no}")
    assert not missing, f"hint phrases missing from UI: {missing}"
    assert not labeled, f"instructional copy used as a caption: {labeled}"

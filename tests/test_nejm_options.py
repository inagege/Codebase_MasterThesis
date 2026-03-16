from pathlib import Path
import sys

# ensure repo root is on sys.path so imports work under pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.benchmark_data_loading import (
    _extract_nejm_option_labels,
    _extract_nejm_options_for_row,
)


def test_extract_nejm_option_labels_from_raw_options_column():
    raw_options = (
        "['A: Erythema infectiosum' 'B: Hand, foot, and mouth disease'\n"
        " 'C: Kawasaki disease' 'D: Measles' 'E: Pityriasis rosea']"
    )
    assert _extract_nejm_option_labels(raw_options) == [
        "Erythema infectiosum",
        "Hand, foot, and mouth disease",
        "Kawasaki disease",
        "Measles",
        "Pityriasis rosea",
    ]


def test_extract_nejm_option_labels_fallback_to_conversations_column():
    row = {
        "options": "",
        "conversations": (
            "[{'from': 'human', 'value': '<image>\\nQuestion\\nA: Foo\\nB: Bar\\nC: Baz\\nD: Qux\\nE: Quux'}\n"
            " {'from': 'gpt', 'value': 'B: Bar'}]"
        ),
    }
    assert _extract_nejm_options_for_row(row) == ["Foo", "Bar", "Baz", "Qux", "Quux"]

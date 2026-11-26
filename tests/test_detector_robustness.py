import itertools
import re
import time

import pandas as pd
import pytest

from app.schemas.findings import DetectionConfig
from app.services import detectors

# The pattern as it was before the lookbehind was added.
OLD_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}", re.ASCII)

TIME_LIMIT_SECONDS = 5.0
HOSTILE_VALUES = {
    "long-local-part-run": "a" * 300_000,
    "run-then-at-sign": "a" * 300_000 + "@",
    "dots": "a." * 150_000,
    "domain-like-labels": "x@" + "a." * 150_000,
    "many-at-signs": "a@" * 150_000,
    "long-digit-run": "1" * 300_000,
    "phone-like-separators": "(111) 111-" * 30_000,
}


def test_the_email_pattern_matches_exactly_what_it_matched_before() -> None:
    checked = 0
    for length in range(8):
        for characters in itertools.product("a1.@- ", repeat=length):
            text = "".join(characters)
            assert bool(OLD_EMAIL.search(text)) == bool(detectors._EMAIL_PATTERN.search(text)), repr(text)
            checked += 1

    assert checked > 300_000  # every string of up to seven characters over this alphabet


@pytest.mark.parametrize(
    "text",
    ["jane.doe@example.com", "contact: bob@mail.example.org today", "x+tag@sub.example.io", "A_B%c@d-e.co"],
)
def test_real_emails_are_still_found(text: str) -> None:
    assert detectors._EMAIL_PATTERN.search(text)


@pytest.mark.parametrize("text", ["user@host", "a@b.c", "no at sign", "@example.com", "plain text"])
def test_non_emails_are_still_ignored(text: str) -> None:
    assert not detectors._EMAIL_PATTERN.search(text)


@pytest.mark.parametrize("text", list(HOSTILE_VALUES.values()), ids=list(HOSTILE_VALUES))
def test_hostile_values_are_scanned_in_bounded_time(text: str) -> None:
    start = time.perf_counter()
    detectors._EMAIL_PATTERN.search(text)
    detectors._PHONE_PATTERN.search(text)

    assert time.perf_counter() - start < TIME_LIMIT_SECONDS


def test_a_huge_hostile_cell_does_not_stall_the_detectors() -> None:
    frame = pd.DataFrame({"notes": ["a" * 300_000 + "@", "ok", None]})

    start = time.perf_counter()
    findings = detectors.run_detectors(frame, DetectionConfig())

    assert time.perf_counter() - start < TIME_LIMIT_SECONDS
    assert not [f for f in findings if f.category.value in {"potential_email", "potential_phone"}]
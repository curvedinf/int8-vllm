# SPDX-License-Identifier: Apache-2.0
"""Readable code/table formatting must not hide actual generation failures."""

import pytest

from scripts.ua_live_soak import check_reply


@pytest.mark.parametrize(
    "text",
    [
        'print(is_palindrome("race a car"))                      # Output: False',
        "| Generation | Description |\n|------------|-------------|\n| Young | New |",
        "| Left | Right |\n|:--------------|--------------:|\n| one | two |",
        "An explanation.\n----------------------\nFurther details.",
    ],
)
def test_readable_layout_is_not_character_salad(text):
    assert check_reply(text) is None


@pytest.mark.parametrize("text", ["a" * 13, "界" * 13])
def test_repeated_visible_glyphs_are_rejected(text):
    assert check_reply(text).startswith("character salad")


def test_repeated_prose_tail_is_still_rejected():
    assert check_reply("This is the same long repeated sentence. " * 7).startswith(
        "repeat tail"
    )


@pytest.mark.parametrize("text", ["", " \n\t", "----------------------"])
def test_missing_final_answer_is_still_rejected(text):
    assert check_reply(text) == "empty reply"

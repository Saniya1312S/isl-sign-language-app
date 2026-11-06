"""Spelling buffer: smoothing, hold time, re-arming, editing, events."""
from isl.spelling import SpellingBuffer
from isl.vocab import suggest


def feed(buf, labels):
    typed = []
    for lab in labels:
        t = buf.update(lab)
        if t:
            typed.append(t)
    return typed


def test_letter_is_typed_only_after_hold_time():
    b = SpellingBuffer(hold_frames=5)
    assert feed(b, ["H"] * 4) == []
    assert feed(b, ["H"] * 6) == ["H"]
    assert b.text == "H"


def test_flicker_does_not_type_wrong_letters():
    b = SpellingBuffer(hold_frames=6)
    noisy = ["A", "A", "B", "A", "A", "C", "A", "A", "A", "B", "A", "A", "A", "A"]
    typed = feed(b, noisy)
    assert typed == ["A"] and b.text == "A"


def test_same_letter_needs_release_before_repeating():
    b = SpellingBuffer(hold_frames=4, release_frames=3)
    feed(b, ["L"] * 10)
    assert b.text == "L"                               # holding does not repeat the letter
    feed(b, [None] * 8 + ["L"] * 8)                     # hand lowered, then signed again
    assert b.text == "LL"


def test_different_letters_chain_without_release():
    b = SpellingBuffer(hold_frames=4)
    feed(b, ["H"] * 8 + ["I"] * 8)
    assert b.text == "HI"


def test_low_confidence_none_frames_never_type():
    b = SpellingBuffer(hold_frames=3)
    assert feed(b, [None] * 50) == [] and b.text == ""


def test_space_backspace_clear_and_events():
    b = SpellingBuffer(hold_frames=3)
    feed(b, ["H"] * 8 + [None] * 8 + ["I"] * 10)      # the vote window adds ~5 frames of delay
    assert b.text == "HI"
    b.add_space()
    assert b.text == "HI " and b.pop_events() == ["word:HI"]
    b.add_space()
    assert b.text == "HI "                              # no double spaces
    b.backspace(); b.backspace()
    assert b.text == "H"
    b.clear()
    assert b.text == ""


def test_replace_current_word_for_autocomplete():
    b = SpellingBuffer()
    b.set_text("I NEED DOC")
    b.replace_current_word("DOCTOR")
    assert b.text == "I NEED DOCTOR "


def test_word_mode_adds_space_after_each_word():
    b = SpellingBuffer(hold_frames=3, word_mode=True)
    feed(b, ["doctor"] * 8 + [None] * 8 + ["small"] * 10)
    assert b.text == "doctor small "


def test_progress_grows_towards_one():
    b = SpellingBuffer(hold_frames=10)
    feed(b, ["A"] * 4)
    early = b.snapshot().progress
    feed(b, ["A"] * 3)
    assert 0 < early < b.snapshot().progress < 1


def test_suggestions():
    assert "doctor" in suggest("doc")
    assert suggest("d") == []                           # too short
    assert suggest("zzzz") == []


def feed_timed(buf, labels, fps):
    """Feed labels as if the camera ran at `fps` frames per second."""
    typed = []
    for i, lab in enumerate(labels):
        t = buf.update(lab, now=i / fps)
        if t:
            typed.append(t)
    return typed


def test_hold_time_is_in_seconds_not_frames():
    """The same 1-second hold must work for a fast and for a slow camera."""
    for fps in (30, 5):
        b = SpellingBuffer(hold_seconds=1.0)
        typed = feed_timed(b, ["K"] * int(fps * 2.5), fps)
        assert typed == ["K"], f"fps={fps}"
        b2 = SpellingBuffer(hold_seconds=1.0)
        assert feed_timed(b2, ["K"] * int(fps * 0.6), fps) == [], f"too early at fps={fps}"


def test_time_based_progress():
    b = SpellingBuffer(hold_seconds=2.0)
    feed_timed(b, ["A"] * 15, 10)                 # 1.5 seconds of steady sign
    assert 0.5 < b.snapshot().progress < 1.0

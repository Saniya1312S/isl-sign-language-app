"""Translation (offline part) and voice helpers."""
from isl.translate import translate
from isl.vocab import GLOSSARY, LETTER_NAMES
from isl.voice import browser_speech_html, spell_out


def test_glossary_hit_needs_no_network():
    t = translate("Doctor", "hi", allow_online=False)
    assert t.text == "डॉक्टर" and t.source == "glossary"
    assert translate("water", "mr", allow_online=False).text == "पाणी"


def test_single_letter_uses_devanagari_letter_name():
    assert translate("B", "hi", allow_online=False).text == LETTER_NAMES["B"]


def test_multi_word_glossary_phrase():
    t = translate("help doctor", "hi", allow_online=False)
    assert t.text == f"{GLOSSARY['help'][0]} {GLOSSARY['doctor'][0]}"


def test_unknown_word_reports_unavailable_instead_of_fake_text():
    t = translate("quasar", "hi", allow_online=False)
    assert t.text is None and t.source == "unavailable"          # old app returned "[Translation Error]"


def test_english_and_devanagari_pass_through():
    assert translate("hello", "en").source == "same"
    assert translate("नमस्ते", "mr").source == "same"


def test_every_glossary_entry_is_complete():
    for word, pair in GLOSSARY.items():
        assert len(pair) == 2 and all(pair), word


def test_spell_out():
    assert spell_out("cat") == "C. A. T."
    assert spell_out("ab", "hi") == f"{LETTER_NAMES['A']}. {LETTER_NAMES['B']}."


def test_browser_speech_html_escapes_script_end_tag():
    html = browser_speech_html("hi </script><b>x", "hi", nonce=3)
    assert "</script><b>" not in html
    assert "hi-IN" in html

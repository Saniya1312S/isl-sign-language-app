"""SignAssistant logic, tested with a fake Gemini client (no network, no key)."""
from types import SimpleNamespace

from isl.assistant import SYSTEM_PROMPT, SignAssistant, friendly_error, resolve_api_key


class FakeModels:
    def __init__(self, text="Go to the nearest hospital.", error=None):
        self.text, self.error, self.calls = text, error, []

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.text)

    def list(self):
        return [SimpleNamespace(name="models/gemini-2.5-flash", supported_actions=["generateContent"]),
                SimpleNamespace(name="models/gemini-2.5-flash-image", supported_actions=["generateContent"]),
                SimpleNamespace(name="models/text-embedding-004", supported_actions=["embedContent"])]


def make(**kw):
    models = FakeModels(**kw)
    return SignAssistant("k", model="test-model", client=SimpleNamespace(models=models)), models


def test_successful_answer_and_request_shape():
    bot, models = make()
    r = bot.ask("where is hospital", [("user", "hi"), ("assistant", "hello")], "Hindi")
    assert r.ok and r.text == "Go to the nearest hospital."
    model, contents, config = models.calls[0]
    assert model == "test-model" and len(contents) == 3
    assert "Hindi" in contents[-1].parts[0].text
    assert config.system_instruction == SYSTEM_PROMPT


def test_empty_question_makes_no_request():
    bot, models = make()
    assert not bot.ask("   ").ok and models.calls == []


def test_missing_key_gives_friendly_message():
    r = SignAssistant("").ask("hello")
    assert not r.ok and "API key" in r.error


def test_errors_are_translated_to_plain_language():
    for exc, fragment in [(Exception("403 PERMISSION_DENIED api key"), "key"), (Exception("429 RESOURCE_EXHAUSTED"), "limit"),
                          (Exception("404 model not found"), "model"), (Exception("Connection timeout"), "internet")]:
        bot, _ = make(error=exc)
        assert fragment in bot.ask("hi").error.lower() or fragment in friendly_error(exc).lower()


def test_history_is_trimmed():
    bot, models = make()
    bot.ask("q", [("user", str(i)) for i in range(20)])
    assert len(models.calls[0][1]) == 7                 # last 6 history turns + the new question


def test_list_models_filters_non_text_models():
    bot, _ = make()
    assert bot.list_models() == ["gemini-2.5-flash"]


def test_api_key_priority(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert resolve_api_key("typed", {"GEMINI_API_KEY": "secret"}) == "typed"
    assert resolve_api_key("", {"GEMINI_API_KEY": "secret"}) == "secret"
    assert resolve_api_key("", {}) == "from-env"
    monkeypatch.delenv("GEMINI_API_KEY")
    assert resolve_api_key("", {}) == ""

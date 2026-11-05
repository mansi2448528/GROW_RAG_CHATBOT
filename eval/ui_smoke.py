"""End-to-end UI smoke test using Streamlit's AppTest (no browser needed).

Run from the project root after the index has been built:

    python eval/ui_smoke.py

The app is driven exactly as a user would drive it: sidebar, suggested
questions, chat input, scheme selector, history, and the trace panel. The run
writes real query artifacts and ``qa_log.csv`` rows, which is intentional for a
smoke test.
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app" / "ui.py"
BRAND_HEADLINE = (
    '<h1 class="welcome-title">HDFC Mutual Fund '
    '<span class="welcome-accent">AI Assistant</span></h1>'
)
WELCOME_TITLE = "How can I help you today?"
PLACEHOLDER = "Ask anything about HDFC Mutual Funds..."
SUGGESTIONS = [
    "Expense ratio of HDFC Flexi Cap Direct?",
    "ELSS lock-in period?",
    "Minimum SIP for HDFC Mid Cap?",
    "Exit load for HDFC Large Cap?",
]
TRACE_TABS = ["Gate", "Retrieval", "Prompt", "Raw output", "Validators"]


def _app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=180).run()


def _markdown(at: AppTest) -> str:
    return "\n".join(str(getattr(node, "value", "")) for node in at.markdown)


def _card(at: AppTest, kind: str) -> str:
    """Return the most recent card of ``kind`` from the visible chat history."""
    marker = f'class="answer-card {kind}"'
    found = ""
    for node in at.markdown:
        value = str(getattr(node, "value", ""))
        if marker in value:
            found = value
    return found


def _button(at: AppTest, label: str):
    matches = [button for button in at.button if button.label == label]
    if not matches:
        raise AssertionError(
            f"button not found: {label}; have {[b.label for b in at.button]}"
        )
    return matches[-1]


def _labels(at: AppTest) -> list[str]:
    return [button.label for button in at.button]


def _check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"[PASS] {name}")
        return
    print(f"[FAIL] {name}" + (f" — {detail}" if detail else ""))
    raise AssertionError(name)


def main() -> int:
    print(f"UI smoke test: {APP}\n")

    # 1. Startup / static shell
    at = _app()
    _check("app starts without exceptions", not at.exception,
           "; ".join(str(e.value) for e in at.exception))
    shell = _markdown(at)
    _check("sidebar brand is rendered", 'class="sidebar-brand"' in shell)
    _check("chat history label is rendered", "Chat history" in shell)
    _check("guest profile is rendered", "Guest user" in shell)
    _check("main brand heading is rendered", BRAND_HEADLINE in shell)
    _check("welcome tagline is exact",
           f'<p class="welcome-tagline">{WELCOME_TITLE}</p>' in shell)
    _check("persistent disclaimer is rendered",
           "Facts-only. No investment advice." in shell)
    labels = _labels(at)
    _check("four suggested questions are rendered",
           all(suggestion in labels for suggestion in SUGGESTIONS), str(labels))
    _check("new chat control is rendered", "New Chat" in labels, str(labels))
    _check("scheme selector is rendered",
           [s.label for s in at.selectbox] == ["Optional filter hint"])
    _check("answer trace setting is rendered",
           [c.label for c in at.checkbox] == ["Show answer trace"])
    _check("chat search is rendered",
           [t.label for t in at.text_input] == ["Search chats"])
    _check("sidebar settings expander is rendered",
           "Settings" in [e.label for e in at.expander])
    _check("chat input placeholder is exact",
           [c.placeholder for c in at.chat_input] == [PLACEHOLDER],
           str([c.placeholder for c in at.chat_input]))

    # 2. Answer path (first suggested question)
    _button(at, SUGGESTIONS[0]).click().run()
    _check("suggested question answers without exceptions", not at.exception,
           "; ".join(str(e.value) for e in at.exception))
    answer = _card(at, "ok")
    _check("answer card is shown", bool(answer))
    _check("answer uses the verified-answer label", "Verified answer" in answer)
    _check("answer contains the indexed expense ratio",
           "0.75% per annum" in answer, answer[:180])
    _check("answer cites one official Groww URL",
           "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth" in answer)
    _check("answer shows the freshness date",
           "Last updated from sources" in answer)
    _check("user message appears as a chat bubble",
           'class="user-bubble"' in _markdown(at))
    _check("copy control is rendered", len(at.get("iframe")) >= 1)
    _check("regenerate control is rendered", "Regenerate" in _labels(at))
    _check("question becomes the chat title", "Expense ratio" in _markdown(at))
    _check("trace expander is available",
           any(e.label == "How this answer was produced" for e in at.expander))
    _check("trace has the five documented tabs",
           [t.label for t in at.tabs] == TRACE_TABS,
           str([t.label for t in at.tabs]))

    # 3. Regenerate reruns the same pipeline question
    _button(at, "Regenerate").click().run()
    _check("regenerate runs without exceptions", not at.exception,
           "; ".join(str(e.value) for e in at.exception))
    _check("regenerate keeps the verified answer",
           "0.75% per annum" in _card(at, "ok"))

    # 4. Refusal path
    at.chat_input[0].set_value("Should I buy HDFC Mid Cap?").run()
    _check("refusal runs without exceptions", not at.exception,
           "; ".join(str(e.value) for e in at.exception))
    refusal = _card(at, "warn")
    _check("refusal card is shown", bool(refusal))
    _check("refusal uses the declined label", "Request declined" in refusal)
    _check("refusal has no scheme citation", "groww.in" not in refusal,
           refusal[:180])
    _check("prior answer remains in chat history",
           "0.75% per annum" in _markdown(at))

    # 5. No-context path
    at.chat_input[0].set_value("Exit load after 5 years?").run()
    _check("no-context query runs without exceptions", not at.exception,
           "; ".join(str(e.value) for e in at.exception))
    no_context = _card(at, "neutral")
    _check("no-context card is shown", bool(no_context))
    _check("no-context uses the not-covered label", "Not covered" in no_context)
    _check("no-context answer says it could not find the fact",
           "official pages I cover" in no_context, no_context[:240])

    # 6. Scheme selector / filter hint
    at.selectbox[0].select("HDFC ELSS Tax Saver Fund").run()
    at.chat_input[0].set_value("What is the exit load?").run()
    _check("scheme-hint query runs without exceptions", not at.exception,
           "; ".join(str(e.value) for e in at.exception))
    hinted = _card(at, "ok")
    _check("scheme hint produces the ELSS answer",
           "HDFC ELSS Tax Saver Fund" in hinted, hinted[:180])
    _check("scheme-hint answer uses the ELSS source",
           "hdfc-elss-tax-saver-fund-direct-plan-growth" in hinted)
    _check("scheme hint is recorded in the UI",
           any("Scheme filter applied" in str(c.value) for c in at.caption))

    # 7. Chat history, new chat and search
    _button(at, "New Chat").click().run()
    _check("new chat shows the welcome screen",
           f'<p class="welcome-tagline">{WELCOME_TITLE}</p>' in _markdown(at))
    _check("new chat hides the previous conversation",
           "0.75% per annum" not in _markdown(at))
    history = [
        label for label in _labels(at)
        if "Expense ratio" in label and "·" in label
    ]
    _check("previous chat is listed in history", bool(history), str(_labels(at)))
    _button(at, history[0]).click().run()
    _check("history restores the previous conversation",
           "0.75% per annum" in _markdown(at))

    at.text_input[0].set_value("zzzz-no-such-chat").run()
    _check("chat search filters history",
           any("No chats match your search." in str(c.value) for c in at.caption))
    at.text_input[0].set_value("").run()

    # 8. Trace visibility setting
    at.checkbox[0].uncheck().run()
    _check("trace can be hidden from settings",
           not any(e.label == "How this answer was produced" for e in at.expander))
    at.checkbox[0].check().run()
    _check("trace can be shown again from settings",
           any(e.label == "How this answer was produced" for e in at.expander))

    # 9. PII path (fresh session)
    pii_at = _app()
    pii_at.chat_input[0].set_value(
        "My PAN is ABCDE1234F, check my folio").run()
    _check("PII query runs without exceptions", not pii_at.exception,
           "; ".join(str(e.value) for e in pii_at.exception))
    pii_card = _card(pii_at, "warn")
    _check("PII query is refused", bool(pii_card))
    _check("PII value is never echoed", "ABCDE1234F" not in _markdown(pii_at))
    _check("PII question is withheld in the chat",
           "[personal data withheld]" in _markdown(pii_at))
    _check("PII refusal explains the missing trace",
           any("No trace was written" in str(i.value) for i in pii_at.info))
    _check("PII chat is titled without personal data",
           any("Personal data question" in label for label in _labels(pii_at)))
    _check("PII answer cannot be regenerated",
           all(button.disabled for button in pii_at.button
               if button.label == "Regenerate"))

    print("\nAll UI smoke checks passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError:
        sys.exit(1)

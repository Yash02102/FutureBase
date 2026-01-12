from src.guardrails import Guardrails, MaxLengthGuardrail, RegexBlocklistGuardrail


def test_blocklist_guardrail_blocks_output():
    guardrails = Guardrails([RegexBlocklistGuardrail(["secret"])])
    verdict = guardrails.check_output("task", "contains secret token")
    assert verdict.passed is False


def test_max_length_guardrail_blocks_large_input():
    guardrails = Guardrails([MaxLengthGuardrail(max_input_chars=5, max_output_chars=100)])
    verdict = guardrails.check_input("hello world", "")
    assert verdict.passed is False

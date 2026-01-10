from src.deepagents_harness import DeepAgentsHarness, StepResult


def test_harness_runs_steps():
    def step(task: str) -> StepResult:
        return StepResult(name="echo", output=task)

    harness = DeepAgentsHarness([step])
    results = harness.run("hello")
    assert results[0].output == "hello"
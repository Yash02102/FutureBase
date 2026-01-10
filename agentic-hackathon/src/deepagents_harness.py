from typing import Callable, List, Optional


class StepResult:
    def __init__(self, name: str, output: str):
        self.name = name
        self.output = output


class DeepAgentsHarness:
    def __init__(self, steps: Optional[List[Callable[[str], StepResult]]] = None):
        self.steps = steps or []

    def add_step(self, step: Callable[[str], StepResult]) -> None:
        self.steps.append(step)

    def run(self, task: str) -> List[StepResult]:
        results: List[StepResult] = []
        for step in self.steps:
            results.append(step(task))
        return results

    def run_subagent(self, name: str, task: str, fn: Callable[[str], str]) -> StepResult:
        output = fn(task)
        return StepResult(name=name, output=output)
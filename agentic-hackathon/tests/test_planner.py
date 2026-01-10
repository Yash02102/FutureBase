from src.utils.schemas import Plan, PlanStep


def test_plan_schema():
    plan = Plan(goal="test", steps=[PlanStep(step="one")])
    assert plan.goal == "test"
    assert plan.steps[0].step == "one"
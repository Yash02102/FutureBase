from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .utils.schemas import Plan


class Planner:
    def __init__(self, model: str):
        self.llm = ChatOpenAI(model=model)

    def generate_plan(self, goal: str) -> Plan:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a precise planner. Return a concise plan with 3-7 steps.",
                ),
                ("human", "Goal: {goal}"),
            ]
        )
        chain = prompt | self.llm.with_structured_output(Plan)
        return chain.invoke({"goal": goal})
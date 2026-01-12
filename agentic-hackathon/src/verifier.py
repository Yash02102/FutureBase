from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


class Verification(BaseModel):
    is_valid: bool = Field(..., description="Whether the result satisfies the task.")
    notes: str = Field(..., description="Short reason or corrections.")


class Verifier:
    def __init__(self, llm):
        self.llm = llm

    def verify(self, task: str, result: str) -> Verification:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Check if the result satisfies the task. Return a verdict and notes.",
                ),
                ("human", "Task: {task}\nResult: {result}"),
            ]
        )
        chain = prompt | self.llm.with_structured_output(Verification)
        return chain.invoke({"task": task, "result": result})

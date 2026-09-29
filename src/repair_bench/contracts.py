"""Runtime contracts for the first controlled, completed-utterance slice."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Plan(Contract):
    day: (
        Literal[
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        ]
        | None
    )
    time: str | None = Field(pattern=r"^([01][0-9]|2[0-3]):[0-5][0-9]$")
    party_size: int | None = Field(ge=1, le=20)
    status: Literal["proposed", "cancelled", "needs_clarification"]


class StartRun(Contract):
    scenario_id: Literal["dinner-date-dev"] = "dinner-date-dev"


NUMBERS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
).split()


def verbalize(plan: Plan) -> str:
    if plan.status == "cancelled":
        return "The dinner plan is cancelled. No reservation has been made."
    if plan.status == "needs_clarification" or any(
        x is None for x in (plan.day, plan.time, plan.party_size)
    ):
        return "Please clarify the day, time, or number of people for the dinner plan."
    hour, minute = map(int, plan.time.split(":"))
    spoken = NUMBERS[hour % 12 or 12]
    if minute:
        tens = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty"}
        spoken += " " + (
            ("oh " + NUMBERS[minute])
            if minute < 10
            else NUMBERS[minute]
            if minute < 20
            else tens[minute // 10]
            + (" " + NUMBERS[minute % 10] if minute % 10 else "")
        )
    spoken += " A M" if hour < 12 else " P M"
    return f"The proposed dinner is {plan.day} at {spoken} for {NUMBERS[plan.party_size]} people. No reservation has been made."

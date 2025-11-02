
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Customer:
    id: int
    alexa_rank: float
    employee_range: str
    industry: str
    closedate: datetime | None
    mrr: float
    

@dataclass(frozen=True)
class NonCustomer:
    id: int
    alexa_rank: float
    employee_range: str
    industry: str


@dataclass(frozen=True)
class UsageAction:
    id: int
    when_timestamp: datetime
    actions_crm_contacts: int
    actions_crm_companies: int
    actions_crm_deals: int
    actions_email: int
    users_crm_contacts: int
    users_crm_companies: int
    users_crm_deals: int
    users_email: int

__all__ = [
    "Customer",
    "NonCustomer",
    "UsageAction",
]
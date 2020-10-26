from pydantic import BaseModel

## DATA MODEL
## This describes the format of the json that our API expects. All fields are required
class Payload(BaseModel):
    Store: int
    DayOfWeek: int
    Date: str
    Customers: int
    Open: int
    Promo: int
    StateHoliday: str
    SchoolHoliday: int
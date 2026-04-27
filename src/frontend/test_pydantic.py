from pydantic import BaseModel


class Test(BaseModel):
    name: str


print(Test(name="hello", foo="bar").model_dump())

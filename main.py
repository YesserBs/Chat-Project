from fastapi import FastAPI
from pydantic import BaseModel

from greeting_detector import get_short_circuit_response

app = FastAPI()


class AskRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "API is running"}


@app.post("/ask")
def ask(payload: AskRequest):
    # First: check if it's only a greeting
    short_response = get_short_circuit_response(payload.text)

    if short_response is not None:
        return {
            "status": "success",
            "type": "greeting",
            "answer": short_response["response"]
        }

    # Future logic can go here
    return {
        "status": "success",
        "type": "generic",
        "answer": "processing request.."
    }
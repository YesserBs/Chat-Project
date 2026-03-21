from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Missing GROQ_API_KEY in environment variables")

client = Groq(api_key=api_key)

app = FastAPI(title="NexBot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str  # "user" ou "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class ChatResponse(BaseModel):
    reply: str


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    success: bool
    username: str


@app.post("/auth/login", response_model=LoginResponse)
def login(body: LoginRequest):
    """Mock login — accepte n'importe quel email/mot de passe non vide."""
    if not body.email or not body.password:
        raise HTTPException(status_code=400, detail="Email et mot de passe requis")
    return LoginResponse(success=True, username=body.email.split("@")[0])


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    """Envoie l'historique à l'API Groq et retourne la réponse."""
    if not body.messages:
        raise HTTPException(status_code=400, detail="Messages requis")

    for msg in body.messages:
        if msg.role not in ("user", "assistant"):
            raise HTTPException(status_code=400, detail=f"Rôle invalide : {msg.role!r}")

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1000,
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un assistant IA utile et concis. Réponds en français sauf si l'utilisateur écrit dans une autre langue."
                },
                *[{"role": m.role, "content": m.content} for m in body.messages],
            ],
        )
        reply = response.choices[0].message.content
        return ChatResponse(reply=reply)

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erreur API LLM : {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok"}
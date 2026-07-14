from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="Deployment Risk API",
    version="1.0"
)

app.include_router(router)

@app.get("/")
def home():
    return {
        "message": "Deployment Risk API running successfully"
    }
from fastapi import FastAPI
from api.routers import production

app = FastAPI(title="Predap API")

app.include_router(production.router)

@app.get("/")
async def root():
    return {"message": "Welcome, the Predap API is running!"}
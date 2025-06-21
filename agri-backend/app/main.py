from fastapi import FastAPI
from app.routes.route import router
from app.utils.logger import setup_logger
from app.config import settings

# Initialize logger
setup_logger()

app = FastAPI(title="Smart Agriculture Backend")

# Register routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Smart Agriculture Backend Running"}
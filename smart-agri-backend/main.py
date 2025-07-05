from fastapi import FastAPI
from routes.route import router
from utils.logger import setup_logger

# Initialize logger
setup_logger()

app = FastAPI(title="Smart Agriculture Backend")

# Register routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Smart Agriculture Backend Running"}
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import AIDataOrchestrator

app = FastAPI(title="AI Tool Orchestrator", version="1.0.0")

class RequestPayload(BaseModel):
    request: str

class ResponsePayload(BaseModel):
    result: str

orchestrator = AIDataOrchestrator()

@app.post("/orchestrate", response_model=ResponsePayload)
async def orchestrate_task(payload: RequestPayload):
    """Process a task using multiple tools"""
    try:
        result = orchestrator.orchestrate(payload.request)
        return ResponsePayload(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
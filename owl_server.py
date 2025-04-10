from fastapi import FastAPI, Body
from pydantic import BaseModel
import uvicorn

from owl import construct_society, run_society

app = FastAPI()

class TaskRequest(BaseModel):
    title: str
    description: str

@app.post("/api/tasks")
async def create_task(task_data: TaskRequest):
    try:
        # Sử dụng Owl để xử lý task
        society = construct_society(task_data.description)
        answer, chat_history, token_count = run_society(society)
        
        return {
            "success": True,
            "data": {
                "owlTaskId": str(hash(task_data.description))[:10],  # ID giả
                "result": answer,
                "tokenCount": token_count
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
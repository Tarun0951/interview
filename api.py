import asyncio
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks,File ,Form,UploadFile
from fastapi.middleware.cors import CORSMiddleware
import json
import uuid
import websockets
import os
from typing import Dict
from app import AdvancedAIInterviewer  

# Initialize FastAPI app
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)

# Global object for the interviewer (we'll create a new one for each session)
interviewer = None


@app.on_event("startup")
async def on_startup():
    """Initialize the interviewer and other necessary components"""
    global interviewer
    interviewer = None


@app.post("/start-interview")
async def start_interview(resume: UploadFile = File(...),job_url: str = Form(...)):
    """
    Endpoint to start the interview. Accepts the resume and job URL data.
    """
    if not resume or not job_url:
        raise HTTPException(status_code=400, detail="Missing required data")
    
    # Create a temporary directory if it doesn't exist
    temp_dir = os.path.join(os.getcwd(), "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate a unique filename
    file_extension = os.path.splitext(resume.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    resume_path = os.path.join(temp_dir, unique_filename)
    
    # Save the uploaded file
    try:
        # Write the uploaded file to the temporary path
        with open(resume_path, "wb") as buffer:
            # Read the uploaded file in chunks
            while content := await resume.read(1024):
                buffer.write(content)
        
        # Get OpenAI API key from environment
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not openai_api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        # Initialize the interviewer
        global interviewer
        interviewer = AdvancedAIInterviewer(openai_api_key, resume_path, job_url)

    

    # Start WebSocket server in a background thread
        threading.Thread(target=run_websocket_server).start()

        return {"status": "Interview started", "message": "Waiting for interview to begin."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


@app.post("/post-interview-evaluation")
async def post_interview_evaluation():
    """Endpoint to get post-interview evaluation after interview completion."""
    if not interviewer:
        raise HTTPException(status_code=400, detail="No interview session found.")

    evaluation = await interviewer.post_interview_evaluation()
    return {"evaluation": evaluation}


# WebSocket endpoint to handle real-time interactions with frontend
@app.websocket("/ws/interview")
async def websocket_interview(websocket: WebSocket):
    """WebSocket endpoint to communicate with the interview process."""
    await websocket.accept()
    
    if not interviewer:
        await websocket.send_text("Error: No interview session found.")
        await websocket.close()
        return

    # Pass the WebSocket to the AdvancedAIInterviewer for processing
    await interviewer.handle_websocket(websocket)


def run_websocket_server():
    """Function to run WebSocket server in a separate thread"""
    asyncio.run(interviewer.start_server())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

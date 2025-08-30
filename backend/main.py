from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from src.model.main import get_inference

app = FastAPI()

# Allow only localhost:3000
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # frontends you allow
    allow_credentials=True,
    allow_methods=["*"],            # or restrict to ["POST"]
    allow_headers=["*"],            # or restrict if needed
)

@app.get("/")
def root():
    return {"message": "Welcome to our FastAPI app!"}

@app.post("/inference")
async def post_inference(request: Request):
    # Parse JSON body
    data = await request.json()
    reviews = data.get("reviews", [])

    # Do something with the reviews
    # For now, just return the number of reviews received
    return get_inference(reviews)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
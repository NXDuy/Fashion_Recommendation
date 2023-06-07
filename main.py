from fastapi import FastAPI, Body
from pydantic import AnyUrl
from recommend import recommendation

app = FastAPI()


@app.on_event("startup")
def startup_event():
    # Build model for the first time
    recommendation.build()


@app.post("/recommendation")
async def get_fashion_recommendation(image_urls: list[AnyUrl] = Body(...)) -> list[int]:
    return recommendation(paths=image_urls)

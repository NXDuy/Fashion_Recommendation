from fastapi import FastAPI
from pydantic import AnyUrl, BaseModel, Field
from recommend import recommendation

app = FastAPI(title="Fashion Recommendation")


@app.on_event("startup")
def startup_event():
    # Build model for the first time
    recommendation.build()


class Payload(BaseModel):
    image_urls: set[AnyUrl] = Field(min_items=1)
    num_of_recommended_products: int = 20


@app.post("/recommendation")
def get_fashion_recommendation(payload: Payload) -> list[int]:
    return recommendation(
        paths=payload.image_urls,
        num_recommendations=payload.num_of_recommended_products,
    )

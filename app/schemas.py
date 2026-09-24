from datetime import datetime
from typing import List
from pydantic import BaseModel


class OrderRequest(BaseModel):
    seller_customer_distance_km: float
    total_freight: float
    total_price: float
    num_items: int
    num_unique_products: int
    num_unique_sellers: int
    total_product_weight_g: float
    total_product_volume_cm3: float
    customer_state: str
    order_purchase_timestamp: datetime
    order_estimated_delivery_date: datetime

class BatchPredictionRequest(BaseModel):
    orders: List[OrderRequest]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    threshold: float
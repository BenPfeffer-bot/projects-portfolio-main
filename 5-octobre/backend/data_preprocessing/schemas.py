from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class OrderData(BaseModel):
    id: int
    reference: str = Field(alias="Référence")
    new_customer: int = Field(alias="Nouveau client")
    delivery: str = Field(alias="Livraison")
    client: Optional[str] = Field(alias="Client")
    total: Decimal = Field(alias="Total")
    payment: str = Field(alias="Paiement")
    status: str = Field(alias="État")
    date: datetime = Field(alias="Date")

    @field_validator("total")
    def validate_total(cls, v):
        if v < 0:
            raise ValueError("Total amount cannot be negative")
        return v

    @field_validator("date")
    def validate_date(cls, v):
        if v > datetime.now():
            raise ValueError("Date cannot be in the future")
        return v


class CartData(BaseModel):
    id: int
    order_id: str = Field(alias="ID commande")
    client: Optional[str] = Field(alias="Client")
    total: Decimal = Field(alias="Total")
    carrier: Optional[str] = Field(alias="Transporteur")
    date: datetime = Field(alias="Date")

    @field_validator("total")
    def validate_total(cls, v):
        if v < 0:
            raise ValueError("Total amount cannot be negative")
        return v

    @field_validator("date")
    def validate_date(cls, v):
        if v > datetime.now():
            raise ValueError("Date cannot be in the future")
        return v


class InventoryData(BaseModel):
    id: int
    sfa: str
    lib: str
    ean: str
    qty: int
    factory_price: Decimal
    retail: Decimal
    retail_us: Decimal

    @field_validator("qty")
    def validate_qty(cls, v):
        if v < 0:
            raise ValueError("Quantity cannot be negative")
        return v

    @field_validator("factory_price", "retail", "retail_us")
    def validate_prices(cls, v):
        if v < 0:
            raise ValueError("Price cannot be negative")
        return v


class RetailData(BaseModel):
    date: datetime = Field(alias="Date")
    ref: str = Field(alias="Ref")
    libelle: str = Field(alias="Libellé")
    customer: str = Field(alias="Cust")
    quantity: int = Field(alias="Qté")
    pv_ttc: Decimal = Field(alias="PV TTC")
    ca_ttc: Decimal = Field(alias="CA TTC")

    @field_validator("date")
    def validate_date(cls, v):
        if v > datetime.now():
            raise ValueError("Date cannot be in the future")
        return v

    @field_validator("quantity")
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError("Quantity must be positive")
        return v

    @field_validator("pv_ttc", "ca_ttc")
    def validate_amounts(cls, v):
        if v < 0:
            raise ValueError("Amount cannot be negative")
        return v

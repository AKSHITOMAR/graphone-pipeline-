"""Canonical schemas (match the assignment's field tables) + provenance extras."""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator

SCHEMA_VERSION = "1.0"


class Source(BaseModel):
    name: str
    url: str

    @field_validator("url")
    @classmethod
    def must_be_http(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("every record needs a real source URL")
        return v


class PricingModel(str, Enum):
    FREE = "FREE"
    FREEMIUM = "FREEMIUM"
    PAID = "PAID"
    ENTERPRISE = "ENTERPRISE"


class StartupData(BaseModel):
    employeeCount: Optional[int] = None
    website: Optional[str] = None
    description: Optional[str] = None


class StartupContent(BaseModel):
    entityName: str
    data: StartupData = StartupData()
    rawName: Optional[str] = None


class Startup(BaseModel):
    schemaVersion: str = SCHEMA_VERSION
    recordType: Literal["STARTUP"] = "STARTUP"
    source: Source
    content: StartupContent
    collectedAt: str


class ProductContent(BaseModel):
    startupName: str
    productName: Optional[str] = None
    pricingModel: PricingModel
    evidence: str = Field(description="verbatim snippet from the source page that justifies pricingModel")
    extractionMethod: Literal["rules", "llm"] = "rules"


class Product(BaseModel):
    schemaVersion: str = SCHEMA_VERSION
    recordType: Literal["PRODUCT"] = "PRODUCT"
    source: Source
    content: ProductContent
    collectedAt: str


class PaperContent(BaseModel):
    title: str
    authors: list[str]
    paper_url: str
    github_url: Optional[str] = None
    github_stars: Optional[int] = None
    github_stars_checked_at: Optional[str] = None
    published_date: str


class ResearchPaper(BaseModel):
    schemaVersion: str = SCHEMA_VERSION
    recordType: Literal["RESEARCH_PAPER"] = "RESEARCH_PAPER"
    source: Source
    content: PaperContent
    collectedAt: str


class JobContent(BaseModel):
    company: str
    title: str
    url: str
    date: str
    date_source: str
    is_remote: bool
    role_family: str


class Job(BaseModel):
    schemaVersion: str = SCHEMA_VERSION
    recordType: Literal["JOB"] = "JOB"
    source: Source
    content: JobContent
    collectedAt: str


class NewsContent(BaseModel):
    title: str
    url: str
    published: str
    date_source: str
    text: str


class News(BaseModel):
    schemaVersion: str = SCHEMA_VERSION
    recordType: Literal["NEWS"] = "NEWS"
    source: Source
    content: NewsContent
    collectedAt: str

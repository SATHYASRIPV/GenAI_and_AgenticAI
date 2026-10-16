from pydantic import BaseModel, Field
from typing import List, Dict, Any
from enum import Enum

# ========== INPUT/OUTPUT SCHEMAS ==========

class CSVParseRequest(BaseModel):
    """Parse a CSV file"""
    file_path: str = Field(..., description="Path to CSV file")
    delimiter: str = Field(default=",", description="CSV delimiter (comma, tab, semicolon)")
    encoding: str = Field(default="utf-8", description="File encoding")

class CSVParseResponse(BaseModel):
    rows: List[Dict[str, Any]]
    row_count: int
    columns: List[str]

class DataFilterRequest(BaseModel):
    """Filter data by conditions"""
    data: List[Dict[str, Any]] = Field(..., description="List of records to filter")
    column: str = Field(..., description="Column name to filter on")
    operator: str = Field(..., description="Comparison operator: ==, !=, >, <, >=, <=, contains")
    value: Any = Field(..., description="Value to compare against")

class DataFilterResponse(BaseModel):
    filtered_rows: List[Dict[str, Any]]
    matched_count: int
    original_count: int

class StatisticsRequest(BaseModel):
    """Calculate statistics on numeric columns"""
    data: List[Dict[str, Any]] = Field(..., description="List of records")
    columns: List[str] = Field(..., description="Numeric column names to calculate stats for")
    metrics: List[str] = Field(
        default=["mean", "sum", "min", "max", "count"],
        description="Statistics to calculate: mean, sum, min, max, count, median, std"
    )

class StatisticsResponse(BaseModel):
    statistics: Dict[str, Dict[str, float]]
    summary: str

class EmailRequest(BaseModel):
    """Send an email"""
    recipient: str = Field(..., description="Email address")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body (plain text)")
    preview: bool = Field(default=True, description="If true, print instead of sending")

class EmailResponse(BaseModel):
    success: bool
    message: str
    recipient: str

class DatabaseLogRequest(BaseModel):
    """Log an operation to database"""
    operation: str = Field(..., description="Type of operation (parse, filter, calculate, email)")
    status: str = Field(..., description="success or failure")
    details: Dict[str, Any] = Field(..., description="Operation details")

class DatabaseLogResponse(BaseModel):
    logged: bool
    log_id: str
    timestamp: str

class ConvertFormatRequest(BaseModel):
    """Convert file format"""
    file_path: str = Field(..., description="Path to source file")
    source_format: str = Field(..., description="Source format: csv, json")
    target_format: str = Field(..., description="Target format: csv, json")

class ConvertFormatResponse(BaseModel):
    output_path: str
    success: bool
    rows_converted: int
import pandas as pd
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from tools import *
from config import EMAIL_CONFIG

class ToolImplementations:
    """Actual implementations of tools"""
    
    def __init__(self, db_path: str = "operations.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS operation_logs (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                operation TEXT,
                status TEXT,
                details TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def parse_csv(self, request: CSVParseRequest) -> CSVParseResponse:
        """Parse CSV file"""
        try:
            df = pd.read_csv(
                request.file_path,
                delimiter=request.delimiter,
                encoding=request.encoding
            )
            rows = df.to_dict(orient='records')
            return CSVParseResponse(
                rows=rows,
                row_count=len(rows),
                columns=df.columns.tolist()
            )
        except Exception as e:
            raise ValueError(f"CSV parsing failed: {str(e)}")
    
    def filter_data(self, request: DataFilterRequest) -> DataFilterResponse:
        """Filter data by condition"""
        filtered = []
        operator = request.operator.strip()
        
        for row in request.data:
            val = row.get(request.column)
            if val is None:
                continue
            
            # Map operator to comparison
            if operator == "==":
                match = val == request.value
            elif operator == "!=":
                match = val != request.value
            elif operator == ">":
                match = val > request.value
            elif operator == "<":
                match = val < request.value
            elif operator == ">=":
                match = val >= request.value
            elif operator == "<=":
                match = val <= request.value
            elif operator == "contains":
                match = str(request.value).lower() in str(val).lower()
            else:
                raise ValueError(f"Unknown operator: {operator}")
            
            if match:
                filtered.append(row)
        
        return DataFilterResponse(
            filtered_rows=filtered,
            matched_count=len(filtered),
            original_count=len(request.data)
        )
    
    def calculate_statistics(self, request: StatisticsRequest) -> StatisticsResponse:
        """Calculate statistics on data"""
        try:
            df = pd.DataFrame(request.data)
            stats = {}
            
            for col in request.columns:
                col_stats = {}
                
                if "mean" in request.metrics:
                    col_stats["mean"] = float(df[col].mean())
                if "sum" in request.metrics:
                    col_stats["sum"] = float(df[col].sum())
                if "min" in request.metrics:
                    col_stats["min"] = float(df[col].min())
                if "max" in request.metrics:
                    col_stats["max"] = float(df[col].max())
                if "count" in request.metrics:
                    col_stats["count"] = int(df[col].count())
                if "median" in request.metrics:
                    col_stats["median"] = float(df[col].median())
                if "std" in request.metrics:
                    col_stats["std"] = float(df[col].std())
                
                stats[col] = col_stats
            
            summary = f"Calculated {len(request.metrics)} metrics for {len(request.columns)} columns across {len(request.data)} rows"
            
            return StatisticsResponse(
                statistics=stats,
                summary=summary
            )
        except Exception as e:
            raise ValueError(f"Statistics calculation failed: {str(e)}")
    
    def send_email(self, request: EmailRequest) -> EmailResponse:
        """Send email (or preview)"""
        from config import EMAIL_CONFIG
        
        # Use provided email or default from config
        recipient = request.recipient if request.recipient != "user@example.com" else EMAIL_CONFIG["default_recipient"]
        subject = request.subject or EMAIL_CONFIG["default_subject"]
        preview = request.preview if request.preview is not None else EMAIL_CONFIG["preview_mode"]
        
        if preview:
            print(f"\n{'='*60}")
            print(f"📧 EMAIL PREVIEW")
            print(f"{'='*60}")
            print(f"To: {recipient}")
            print(f"Subject: {subject}")
            print(f"{'─'*60}")
            print(request.body)
            print(f"{'='*60}\n")
            return EmailResponse(
                success=True,
                message="Email preview printed (preview mode enabled)",
                recipient=recipient
            )
        else:
            # For actual sending, you'd need SMTP setup (Gmail, SendGrid, etc.)
            # For now, we'll just log it
            with open("email_log.txt", "a") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"To: {recipient}\n")
                f.write(f"Subject: {subject}\n")
                f.write(f"Body: {request.body}\n")
            
            return EmailResponse(
                success=True,
                message=f"Email logged to email_log.txt",
                recipient=recipient
            )
        
    def log_operation(self, request: DatabaseLogRequest) -> DatabaseLogResponse:
        """Log operation to database"""
        import uuid
        log_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO operation_logs (id, timestamp, operation, status, details)
            VALUES (?, ?, ?, ?, ?)
        """, (log_id, timestamp, request.operation, request.status, json.dumps(request.details)))
        conn.commit()
        conn.close()
        
        return DatabaseLogResponse(
            logged=True,
            log_id=log_id,
            timestamp=timestamp
        )
    
    def convert_format(self, request: ConvertFormatRequest) -> ConvertFormatResponse:
        """Convert between file formats"""
        try:
            if request.source_format == "csv" and request.target_format == "json":
                df = pd.read_csv(request.file_path)
                output_path = request.file_path.replace(".csv", ".json")
                df.to_json(output_path, orient='records', indent=2)
            elif request.source_format == "json" and request.target_format == "csv":
                with open(request.file_path) as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                output_path = request.file_path.replace(".json", ".csv")
                df.to_csv(output_path, index=False)
            else:
                raise ValueError(f"Unsupported conversion: {request.source_format} → {request.target_format}")
            
            return ConvertFormatResponse(
                output_path=output_path,
                success=True,
                rows_converted=len(df)
            )
        except Exception as e:
            raise ValueError(f"Format conversion failed: {str(e)}")
# Multi-stage hoặc slim image cho production AI/Backend
FROM python:3.10-slim

# Ngăn Python tạo file .pyc và bật stdout/stderr flush ngay lập tức
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Cài đặt curl phục vụ container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy manifest file trước để tối ưu hóa Docker layer caching
COPY pyproject.toml requirements.txt ./

# Cài đặt dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code
COPY . .

# Cài đặt package agentic-rag ở chế độ editable/standard
RUN pip install --no-cache-dir -e .

EXPOSE 8000

# Healthcheck định kỳ kiểm tra API status
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Chạy ứng dụng FastAPI
CMD ["uvicorn", "agentic_rag.app:app", "--host", "0.0.0.0", "--port", "8000"]

# 1. Base image (경량화된 Python 슬림 이미지 사용)
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 환경 변수 설정 (패키지 설치 중 캐시 생성 방지 및 파이썬 출력 동기화)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 4. 의존성 파일 복사
COPY requirements.txt .

# 5. 패키지 설치
# (최적화) FastAPI 의존성 및 Pytorch CPU 전용 버전 설치
# PyTorch는 기본 설치 시 용량이 매우 크므로(약 4~5GB), 추론 전용 서버를 위해 CPU 버전만 설치하여 이미지 크기를 최적화합니다.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# 6. 소스 코드 복사 
COPY ./app ./app

# 7. 포트 노출 (FastAPI 기본 포트 8000)
EXPOSE 8000

# 8. 컨테이너 실행 시 FastAPI 서버 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

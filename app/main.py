# Updated at: 2026-04-08 19:07 (Final Test Run)
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import io
import os
from PIL import Image
try:
    from app.model import predict_image
except ImportError:
    # 직접 파일 실행 시 경로 문제 해결용
    from model import predict_image

app = FastAPI(
    title="Flower Classification API",
    description="로컬 컴퓨터를 서버로 활용하는 MLOps 파이프라인의 첫 단계인 꽃 판별 API입니다.",
    version="1.0.0"
)

# 정적 파일(HTML 등) 제공을 위한 설정
os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def read_root():
    # '/' 엔드포인트 접속 시 방금 만든 UI 페이지를 보여주도록 연결
    return FileResponse("app/static/index.html")

@app.post("/predict")
async def predict_flower(file: UploadFile = File(...)):
    """
    이미지 파일을 업로드 받아 어떤 꽃인지 예측하여 반환합니다.
    """
    # 1. 파일 확장자 검증 (이미지인지 확인)
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="업로드된 파일이 이미지 형식이 아닙니다.")
    
    try:
        # 2. 이미지 파일 메모리로 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 3. AI 모델을 통해 추론 실행
        predicted_class = predict_image(image)
        
        # 4. 결과 응답
        return JSONResponse(content={
            "filename": file.filename,
            "prediction": predicted_class,
            "status": "success"
        })
        
    except Exception as e:
        # 5. 에러 처리
        raise HTTPException(status_code=500, detail=f"이미지 처리 중 오류가 발생했습니다: {str(e)}")

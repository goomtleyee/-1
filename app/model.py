import torch
from transformers import pipeline
from PIL import Image

print("AI 모델을 메모리에 로드하는 중입니다... (초기 1회만 허깅페이스에서 다운로드하므로 시간이 조금 소요될 수 있습니다)")

# Hugging Face 파이프라인 로드 (꽃 전용 검색 모델)
# dima806/oxford_flowers_image_detection (102종의 특정 꽃과 식물 데이터로 파인튜닝된 ViT 모델)
pipe = pipeline("image-classification", model="dima806/oxford_flowers_image_detection")

def predict_image(image: Image.Image) -> str:
    """
    입력된 꽃 이미지를 허깅페이스 전용 모델을 통해 정확하게 분석하고 결과를 반환합니다.
    """
    try:
        # 허깅페이스 파이프라인으로 이미지 분석 (가장 확률 높은 결과가 1순위로 나옵니다)
        results = pipe(image)
        
        best_result = results[0]
        category_name = best_result['label']
        confidence = best_result['score'] * 100
        
        return f"분석된 꽃: {category_name.capitalize()} (확률: {confidence:.1f}%)"

    except Exception as e:
        print(f"에러 발생: {e}")
        return "판독 실패 (이미지 처리 오류)"

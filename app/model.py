import torch
from transformers import pipeline
from PIL import Image

print("AI 모델을 메모리에 로드하는 중입니다... (만물을 판별할 수 있는 OpenAI CLIP 모델 다운로드 중)")

# OpenAI가 개발한 범용 비전 모델 CLIP 활용 (Zero-Shot Classification)
# 특정 데이터셋(102종)에 국한되지 않고 텍스트를 기반으로 어떤 꽃이든 판별할 수 있습니다.
pipe = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

# 판별하고 싶은 꽃 이름들의 후보군 (언제든 텍스트만으로 자유롭게 무한 추가 가능!)
candidate_labels = [
    "cherry blossom", "rose", "sunflower", "daisy", "tulip", 
    "dandelion", "lily", "orchid", "hydrangea", "lotus", 
    "hibiscus", "camellia", "azalea", "freesia", "peony"
]

# 영어 이름을 한글로 예쁘게 보여주기 위한 매핑 사전
translation_dict = {
    "cherry blossom": "벚꽃", "rose": "장미", "sunflower": "해바라기",
    "daisy": "데이지", "tulip": "튤립", "dandelion": "민들레",
    "lily": "백합", "orchid": "난초", "hydrangea": "수국",
    "lotus": "연꽃", "hibiscus": "무궁화", "camellia": "동백꽃",
    "azalea": "진달래", "freesia": "프리지아", "peony": "모란"
}

def predict_image(image: Image.Image) -> str:
    """
    입력된 이미지를 CLIP 제로샷 모델을 통해 지정된 후보군 중 가장 확률 높은 꽃으로 매칭합니다.
    """
    try:
        # Zero-shot 예측 실행 (가장 확률 높은 결과가 리스트의 첫 번째로 옴)
        results = pipe(image, candidate_labels=candidate_labels)
        
        best_result = results[0]
        english_name = best_result['label']
        confidence = best_result['score'] * 100
        
        # 사전에 있다면 한글 이름 출력, 없다면 영어 그대로 출력
        korean_name = translation_dict.get(english_name, english_name)
        
        return f"분석된 꽃: {korean_name} (확률: {confidence:.1f}%)"

    except Exception as e:
        print(f"에러 발생: {e}")
        return "판독 실패 (이미지 처리 오류)"

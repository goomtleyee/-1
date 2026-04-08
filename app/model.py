import torch
from transformers import pipeline
from PIL import Image

print("AI 모델을 메모리에 로드하는 중입니다... (만물을 판별할 수 있는 OpenAI CLIP 모델 다운로드 중)")

# OpenAI가 개발한 범용 비전 모델 CLIP 활용 (Zero-Shot Classification)
pipe = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

# 판별하고 싶은 꽃 이름들 + "꽃이 아닌 것들"을 명시적으로 추가하여 강제 오답을 막습니다!
candidate_labels = [
    "cherry blossom", "rose", "sunflower", "daisy", "tulip", 
    "dandelion", "lily", "orchid", "hydrangea", "lotus", 
    "hibiscus", "camellia", "azalea", "freesia", "peony",
    "an insect or bug", "an animal", "a human or person", "a machine or vehicle", "not a flower or random object"
]

# 영어 이름을 한글로 예쁘게 보여주기 위한 매핑 사전
translation_dict = {
    "cherry blossom": "벚꽃", "rose": "장미", "sunflower": "해바라기",
    "daisy": "데이지", "tulip": "튤립", "dandelion": "민들레",
    "lily": "백합", "orchid": "난초", "hydrangea": "수국",
    "lotus": "연꽃", "hibiscus": "무궁화", "camellia": "동백꽃",
    "azalea": "진달래", "freesia": "프리지아", "peony": "모란",
    "an insect or bug": "곤충/벌레", "an animal": "동물", "a human or person": "사람", 
    "a machine or vehicle": "기계/물건", "not a flower or random object": "꽃이 아닌 사물"
}

def predict_image(image: Image.Image) -> str:
    """
    입력된 이미지를 CLIP 제로샷 모델을 통해 지정된 후보군 중 가장 확률 높은 결과로 매칭합니다.
    """
    try:
        # Zero-shot 예측 실행
        results = pipe(image, candidate_labels=candidate_labels)
        
        best_result = results[0]
        english_name = best_result['label']
        confidence = best_result['score'] * 100
        
        korean_name = translation_dict.get(english_name, english_name)
        
        # 만약 AI가 '꽃이 아닌 예외 항목'으로 판단했다면 경고 메시지 출력
        if english_name in ["an insect or bug", "an animal", "a human or person", "a machine or vehicle", "not a flower or random object"]:
            return f"❌ 꽃이 아닙니다! ({korean_name} 같습니다 - 확률: {confidence:.1f}%)"
        
        return f"분석된 꽃: {korean_name} (확률: {confidence:.1f}%)"

    except Exception as e:
        print(f"에러 발생: {e}")
        return "판독 실패 (이미지 처리 오류)"

import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image

print("AI 모델을 메모리에 로드하는 중입니다... (초기 1회만 시간 소요)")

# 1. 모델 로드 (가볍고 빠른 MobileNetV3 모델 사용, ImageNet 사전학습 가중치)
# 실제 상용 환경에서는 이 부분을 직접 파인튜닝(Fine-tuning)한 모델 가중치 파일(.pt) 로드로 변경해야 합니다.
weights = MobileNet_V3_Small_Weights.DEFAULT
model = mobilenet_v3_small(weights=weights)
model.eval() # 학습 모드가 아닌 추론(Inference) 모드로 설정

# 2. 이미지 전처리 파이프라인
# 카메라로 찍은 다양한 비율의 꽃 사진을 AI가 이해할 수 있는 224x224 크기의 정규화된 텐서로 변환합니다.
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(image: Image.Image) -> str:
    """
    입력된 꽃 이미지를 PyTorch 모델을 통해 분석하고 가장 확률이 높은 이름을 반환합니다.
    """
    try:
        # 전처리 및 배치 차원(차원수 4) 추가
        img_tensor = preprocess(image)
        batch_tensor = img_tensor.unsqueeze(0) 
        
        # 추론 계산 (기울기 계산 비활성화로 속도 및 메모리 효율성 최적화)
        with torch.no_grad():
            prediction = model(batch_tensor)
            
        # 가장 높은 확률을 가진 결과값 추출
        probabilities = torch.nn.functional.softmax(prediction[0], dim=0)
        class_idx = probabilities.argmax().item()
        
        # 클래스 인덱스를 실제 이름(영어)으로 변환
        category_name = weights.meta["categories"][class_idx]
        confidence = probabilities[class_idx].item() * 100
        
        # 영문 카테고리를 한글로 번역하는 매핑을 추가할 수 있습니다. (예: 'daisy' -> '데이지')
        if 'daisy' in category_name.lower():
            return f"데이지 (Daisy) - 일치율: {confidence:.1f}%"
        elif 'sunflower' in category_name.lower():
            return f"해바라기 (Sunflower) - 일치율: {confidence:.1f}%"
        elif 'rose' in category_name.lower():
            return f"장미 (Rose) - 일치율: {confidence:.1f}%"
        
        return f"{category_name.capitalize()} - 일치율: {confidence:.1f}%"

    except Exception as e:
        print(f"에러 발생: {e}")
        return "판독 실패 (이미지 처리 오류)"

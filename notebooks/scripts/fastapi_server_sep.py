
import torch
import uvicorn
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, File, UploadFile

import base64
from fastapi.responses import JSONResponse

app = FastAPI()

model = torch.hub.load('yolov5', 'custom', path='/Users/tasha/Desktop/comento/mywork/models/yolov5/yolov5/runs/train/yolov5s_batch64_epoch300/weights/best.pt', source='local')  # 학습한 모델의 경로를 'path'인자에 넣어주세요 (ex, 'yolov5/runs/train/yolov5s_results4/weights/best.pt')
model.conf = 0.5  # 신뢰도 임계값 설정
#model.conf = 0.8
model.eval()


def run_model(image: Image.Image):
    #YOLOv5 모델로 추론 실행
    try:
        results = model(image)
        df = results.pandas().xyxy[0]
        if df.empty:
            return None
        return df
    except Exception as e:
        raise RuntimeError(f"모델 추론 실패: {e}")


def calculate_congestion(df, image_size):
    #차량 정체 분석 지표 계산
    total_area = image_size[0] * image_size[1]
    vehicle_area = sum(
        (row["xmax"] - row["xmin"]) * (row["ymax"] - row["ymin"])
        for _, row in df.iterrows()
    )

    detections_list = df.to_dict(orient="records")

    congestion_info = {
        "total_vehicles": len(detections_list),
        "vehicle_area_ratio": vehicle_area / total_area if total_area > 0 else 0,
    }
    return detections_list, congestion_info


@app.post("/detect")
async def detect_api(file: UploadFile = File(None)):
    #이미지 업로드를 받아 차량 감지 + 정체 분석 + 시각화 이미지(base64 포함) 반환
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    result = run_model(image)
    if result is None:
        return {
            "detections": [],
            "congestion_info": {"total_vehicles": 0, "vehicle_area_ratio": 0},
            "result_image": None,
        }

    detections, congestion_info = calculate_congestion(result, image.size)

    # YOLO 결과 시각화
    rendered = model(image)
    rendered.render()  # 결과를 이미지에 직접 그림

    # 이미지 → Bytes → Base64 인코딩
    rendered_image = Image.fromarray(rendered.ims[0])
    buffer = BytesIO()
    rendered_image.save(buffer, format="JPEG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return {
        "detections": detections,
        "congestion_info": congestion_info,
        "result_image": image_base64,
    }


if __name__ == "__main__":
    #Uvicorn 서버 실행
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# 터미널에서 실행하는 명령어 (Jupyter에서 실행 X)
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload


import streamlit as st
import requests
from PIL import Image
import base64
from io import BytesIO

# FastAPI 서버 주소
FASTAPI_URL = "http://127.0.0.1:8000/detect"

def send_image_to_fastapi(image_file):
    files = {"file": image_file.getvalue()}
    try:
        response = requests.post(FASTAPI_URL, files=files)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"FastAPI 서버 요청 실패: {e}")
        return None

st.title("🚗 차량 탐지 결과 보기")
st.write("이미지를 업로드하면 차량 탐지 결과를 확인할 수 있습니다.")

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 업로드된 이미지 바로 표시
    uploaded_img = Image.open(uploaded_file)
    st.image(uploaded_img, caption="업로드한 이미지", use_container_width=True)

    # 업로드한 파일을 FastAPI 서버로 전송
    response = send_image_to_fastapi(uploaded_file)

    try:
        if response and response.status_code == 200:
            result = response.json()

            if "detections" in result:
                st.write("### 📌 탐지된 차량 정보:")
                for detection in result["detections"]:
                    st.write(
                        f"🚘 차량 탐지: 좌표 ({detection['xmin']}, {detection['ymin']}) - ({detection['xmax']}, {detection['ymax']}), 신뢰도 {detection['confidence']:.2f}"
                    )
            else:
                st.error("탐지된 차량 정보가 없습니다.")
            
            # 바운딩 박스가 그려진 이미지 처리
            try:
                detected_img_base64 = result.get("result_image")
                if detected_img_base64:
                    decoded_img = base64.b64decode(detected_img_base64)
                    img = Image.open(BytesIO(decoded_img))

                    # 바운딩 박스 이미지 표시
                    st.image(img, caption="바운딩 박스가 그려진 이미지", use_container_width=True)
                else:
                    st.error("결과 이미지가 없습니다.")
            except Exception as e:
                st.error(f"반환된 이미지 처리 실패: {e}")
        else:
            st.error("서버 응답 실패")
    except Exception as e:
        st.error(f"응답 처리 중 오류 발생: {e}")


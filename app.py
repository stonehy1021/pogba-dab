# app.py
import av
import cv2
import time
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# 샘플 이미지(업로드한 파일) — Streamlit Cloud에서 테스트용으로 표시됩니다.
SAMPLE_IMAGE_PATH = "/mnt/data/5a30855d-d37d-44ec-b91b-00189682e028.png"

st.set_page_config(page_title="Smart Selfie (Face + V-sign)", layout="centered")
st.title("📸 Smart Selfie — Face + V-sign (Streamlit + WebRTC)")

st.markdown(
    """
    - 브라우저에서 카메라를 허용하면 실시간으로 얼굴과 손을 분석합니다.  
    - 얼굴 + V 포즈가 동시에 감지되면 자동으로 캡처(서버에 저장)합니다.  
    - 캡처된 이미지는 화면에서 미리보기 후 다운로드할 수 있습니다.
    """
)

mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def is_victory(lms, w, h):
    def c(i):
        lm = lms.landmark[i]
        return int(lm.x * w), int(lm.y * h)

    i_tip, m_tip = c(8), c(12)
    r_tip, p_tip = c(16), c(20)
    i_kn, m_kn = c(5), c(9)
    r_kn, p_kn = c(13), c(17)

    return (
        i_tip[1] < i_kn[1] and
        m_tip[1] < m_kn[1] and
        r_tip[1] > r_kn[1] and
        p_tip[1] > p_kn[1]
    )

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)
        self.hand_detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)
        self.captured = False
        self.last_captured = None  # BGR numpy array

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_res = self.face_detector.process(rgb)
        face_detected = face_res.detections is not None

        hand_res = self.hand_detector.process(rgb)
        victory_detected = False

        if hand_res.multi_hand_landmarks:
            for handLms in hand_res.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
                if is_victory(handLms, w, h):
                    victory_detected = True
                    cv2.putText(img, "VICTORY!", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
                    break

        if face_detected:
            for d in face_res.detections:
                mp_draw.draw_detection(img, d)

        # 자동 캡처: 얼굴+V && 아직 캡처 안 된 상태
        if face_detected and victory_detected and not self.captured:
            self.last_captured = img.copy()
            # 서버에 파일로도 저장 (로그 확인용)
            filename = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, img)
            print("Saved:", filename)
            self.captured = True

        # V가 풀리면 다시 캡처 가능
        if not victory_detected:
            self.captured = False

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 실행
ctx = webrtc_streamer(
    key="smart-selfie",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        # 공개 STUN (필요시 수정)
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

# 캡처된 이미지 표시 및 다운로드
st.markdown("### 📥 마지막 자동 캡처")
if ctx.state.playing and ctx.video_processor:
    proc = ctx.video_processor
    if proc.last_captured is not None:
        bgr = proc.last_captured
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        st.image(rgb, caption="Last auto-capture", use_column_width=True)
        _, imbuf = cv2.imencode(".jpg", bgr)
        st.download_button("Download last capture", data=imbuf.tobytes(), file_name=f"capture_{int(time.time())}.jpg", mime="image/jpeg")
    else:
        st.info("아직 자동 캡처된 이미지가 없습니다. 화면에 얼굴과 V 사인을 보여주세요.")
else:
    st.info("카메라 연결을 허용하세요. (또는 브라우저가 WebRTC를 지원하지 않을 수 있습니다.)")

st.markdown("---")
st.markdown("### 🔎 샘플 이미지 (테스트용)")
st.image(SAMPLE_IMAGE_PATH, caption="Sample/test image (uploaded)", use_column_width=True)
st.write(f"샘플 이미지 경로: `{SAMPLE_IMAGE_PATH}`")

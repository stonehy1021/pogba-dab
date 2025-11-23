import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import mediapipe as mp
import av
import numpy as np
import time
import queue
import math
from datetime import datetime

# ---------------- 1. 기본 설정 ----------------
st.set_page_config(page_title="포그바 댑 인식 웹캠", layout="centered")
st.title("⚽ 포그바 댑(Dab) 자동 캡쳐")
st.markdown("카메라 앞에서 **댑(Dab) 세리머니**를 취해보세요! 자동으로 인식해 촬영합니다. 😎")

# 세션 상태 초기화
if "snapshot" not in st.session_state:
    st.session_state.snapshot = None

# Mediapipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ---------------- 2. 헬퍼 함수: 각도 및 거리 계산 ----------------
def calc_angle(a, b, c):
    """세 점(a, b, c) 사이의 각도를 계산"""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - c[1])

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if mag_ba == 0 or mag_bc == 0:
        return 0.0

    cos_angle = dot / (mag_ba * mag_bc)
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    return math.degrees(math.acos(cos_angle))

# ---------------- 3. 영상 처리 클래스 ----------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.result_queue = queue.Queue() # 메인 스레드로 사진 전송
        self.capture_triggered = False
        self.flash_frame = 0
        self.dab_count = 0
        self.cooldown = 0

    def _xy(self, lm):
        return (lm.x, lm.y)

    def _dist(self, a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def _is_dab(self, landmarks):
        """댑(Dab) 자세 감지 로직"""
        lm = landmarks
        nose = self._xy(lm[mp_pose.PoseLandmark.NOSE])
        
        l_sh = self._xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER])
        l_el = self._xy(lm[mp_pose.PoseLandmark.LEFT_ELBOW])
        l_wr = self._xy(lm[mp_pose.PoseLandmark.LEFT_WRIST])

        r_sh = self._xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])
        r_el = self._xy(lm[mp_pose.PoseLandmark.RIGHT_ELBOW])
        r_wr = self._xy(lm[mp_pose.PoseLandmark.RIGHT_WRIST])

        left_angle = calc_angle(l_sh, l_el, l_wr)
        right_angle = calc_angle(r_sh, r_el, r_wr)

        # 1. 오른쪽 댑 (오른팔 굽힘, 왼팔 뻗음)
        bent_right = 70 <= right_angle <= 130 # 각도 범위 조금 완화
        straight_left = left_angle > 150
        right_close_to_face = self._dist(r_wr, nose) < 0.2
        
        dab_right = bent_right and straight_left and right_close_to_face

        # 2. 왼쪽 댑 (왼팔 굽힘, 오른팔 뻗음)
        bent_left = 70 <= left_angle <= 130
        straight_right = right_angle > 150
        left_close_to_face = self._dist(l_wr, nose) < 0.2

        dab_left = bent_left and straight_right and left_close_to_face

        return dab_right or dab_left

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        # 쿨다운 감소
        if self.cooldown > 0:
            self.cooldown -= 1

        # 플래시 효과
        if self.flash_frame > 0:
            self.flash_frame -= 1
            white = np.full((h, w, 3), 255, dtype=np.uint8)
            img = cv2.addWeighted(img, 0.5, white, 0.5, 0)

        # 포즈 분석 (RGB 변환)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.pose.process(img_rgb)
        
        dab_now = False

        if result.pose_landmarks:
            # 랜드마크 그리기
            mp_drawing.draw_landmarks(
                img,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2),
            )
            
            # 댑 감지
            dab_now = self._is_dab(result.pose_landmarks.landmark)

        # 상태 메시지
        status_text = "Make a DAB Pose!"
        status_color = (0, 0, 255)

        if dab_now:
            status_text = "DAB DETECTED!"
            status_color = (0, 255, 0)
            
            # 쿨다운이 끝났으면 촬영
            if self.cooldown == 0 and not self.capture_triggered:
                self.dab_count += 1
                self.flash_frame = 5
                self.cooldown = 30 # 약 1~2초간 재촬영 방지
                
                # 큐에 저장 (메인 스레드로 전송)
                self.result_queue.put(img)
                self.capture_triggered = True # 한 번만 트리거

        # 텍스트 출력
        cv2.putText(img, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
        cv2.putText(img, f"Count: {self.dab_count}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------- 4. UI 구성 ----------------

# 4-1. 결과 화면 (촬영 후)
if st.session_state.snapshot is not None:
    st.success(f"🎉 댑(Dab) 포착 성공!")
    st.image(st.session_state.snapshot, channels="BGR", caption="Dab Capture", use_container_width=True)

    # 다운로드 버튼
    is_success, buffer = cv2.imencode(".png", st.session_state.snapshot)
    if is_success:
        st.download_button(
            label="📥 사진 다운로드",
            data=buffer.tobytes(),
            file_name=f"Pogba_Dab_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
            type="primary"
        )
    
    st.warning("🔄 다시 하려면 페이지를 새로고침 해주세요!")

# 4-2. 촬영 화면
else:
    # RTC 설정 (딕셔너리 직접 사용 - 중요!)
    rtc_config = {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": "turn:openrelay.metered.ca:80", 
             "username": "openrelayproject", 
             "credential": "openrelayproject"}
        ]
    }

    ctx = webrtc_streamer(
        key="pogba-dab",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # 큐 확인 루프 (자동 화면 전환용)
    if ctx.state.playing:
        while True:
            if ctx.video_processor:
                try:
                    result_img = ctx.video_processor.result_queue.get(timeout=0.1)
                    if result_img is not None:
                        st.session_state.snapshot = result_img
                        st.rerun()
                except queue.Empty:
                    pass
            time.sleep(0.1)

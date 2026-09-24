import streamlit as st
import requests
import json
import time

st.set_page_config(page_title="AI Creative Studio", layout="wide")

st.title("🎨 AI CREATIVE STUDIO")
st.caption("Juggernaut XL v9 • Wan 2.2 TI2V 5B")

# Yan panel - Colab API Adresi Bağlantısı
st.sidebar.header("🔌 Backend Bağlantısı")
colab_url = st.sidebar.text_input("Colab API / Ngrok URL:", placeholder="http://xxxx.ngrok-free.app")

tab1, tab2 = st.tabs(["📸 GÖRSEL ÜRET (Image)", "🎥 VİDEO ÜRET (Video)"])

with tab1:
    st.subheader("Juggernaut XL — Text to Image")
    
    col1, col2 = st.columns(2)
    with col1:
        img_prompt = st.text_area("Prompt", "A futuristic cyberpunk city at night, highly detailed, 8k photo")
        img_neg = st.text_area("Negative Prompt", "low quality, blurry, distorted, bad anatomy")
        
        w_col, h_col = st.columns(2)
        with w_col:
            width = st.selectbox("Genişlik", [512, 768, 1024, 1152, 1280], index=2)
        with h_col:
            height = st.selectbox("Yükseklik", [512, 768, 1024, 1152, 1280], index=2)
            
        steps = st.slider("Steps", 10, 50, 30)
        cfg = st.slider("CFG", 1.0, 12.0, 5.0)
        seed = st.number_input("Seed (-1 = Rastgele)", value=-1)
        
        btn_img = st.button("GÖRSEL ÜRET", type="primary")

    with col2:
        if btn_img:
            if not colab_url:
                st.error("Lütfen yan panelden Colab URL adresinizi girin!")
            else:
                st.info("İşlem Colab sunucusuna gönderildi, bekleniyor...")

with tab2:
    st.subheader("Wan 2.2 TI2V 5B — Video Generation")
    
    col1, col2 = st.columns(2)
    with col1:
        vid_prompt = st.text_area("Video Prompt", "A cinematic camera pan of a glowing sci-fi portal")
        vid_neg = st.text_area("Video Negative Prompt", "static, blurry, low quality")
        
        vw_col, vh_col = st.columns(2)
        with vw_col:
            v_width = st.selectbox("Video Genişlik", [512, 576, 640, 704, 768], index=2)
        with vh_col:
            v_height = st.selectbox("Video Yükseklik", [512, 576, 640, 704, 768], index=2)
            
        frames = st.selectbox("Kare Sayısı (Frames)", [16, 32, 48, 64], index=1)
        v_steps = st.slider("Video Steps", 5, 30, 20)
        v_cfg = st.slider("Video CFG", 1.0, 10.0, 5.0)
        
        btn_vid = st.button("VİDEO ÜRET", type="primary")

    with col2:
        if btn_vid:
            if not colab_url:
                st.error("Lütfen yan panelden Colab URL adresinizi girin!")
            else:
                st.info("Video oluşturuluyor, bu işlem birkaç dakika sürebilir...")
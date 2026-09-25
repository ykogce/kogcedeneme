import streamlit as st
import requests
import io
from PIL import Image
import datetime

# Streamlit Secrets üzerinden Güvenli Token Okuma
HF_API_KEY = st.secrets.get("HF_API_KEY", "") 

headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

# GÜNCEL VE ÇALIŞAN HUGGING FACE ROUTER ENDPOINT'LERİ
IMAGE_MODEL_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
VIDEO_MODEL_URL = "https://router.huggingface.co/hf-inference/models/damo-vilab/text-to-video-ms-1.7m"

# Sayfa Konfigürasyonu
st.set_page_config(page_title="KOGCE AI Studio Pro", page_icon="✨", layout="wide")

# ==================== DARK THEME & GLASSMORPHISM CSS ====================
custom_css = """
<style>
    .stApp { background-color: #0b0f19; color: #f8fafc; }
    p, span, label, div, .stMarkdown, h1, h2, h3 { color: #f1f5f9 !important; }

    .main-title {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #c084fc 0%, #38bdf8 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title { text-align: center; color: #94a3b8 !important; font-size: 1rem; margin-bottom: 2rem; }

    .stTabs [data-baseweb="tab-list"] { gap: 12px; justify-content: center; }
    .stTabs [data-baseweb="tab"] {
        height: 48px; background-color: rgba(30, 41, 59, 0.8); border-radius: 10px; color: #cbd5e1 !important; border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #a855f7 0%, #2563eb 100%) !important; color: #ffffff !important; border: none !important;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #9333ea 0%, #2563eb 100%);
        color: white !important; font-weight: 700; border-radius: 10px; border: none; padding: 0.75rem 1.5rem;
    }

    .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
        background-color: #111827 !important; color: #ffffff !important; border-radius: 10px !important; border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    section[data-testid="stSidebar"] { background-color: #0f172a; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

def improve_prompt_with_ai(user_input, style_preset, aspect_ratio):
    style_modifiers = {
        "Fotogerçekçi (Photorealistic)": "photorealistic, 8k resolution, highly detailed, professional photography, cinematic lighting",
        "Anime / Manga": "anime style, highly detailed illustration, studio ghibli inspired, vibrant colors",
        "3D Render (Pixar)": "3D render, Pixar style, Octane render, smooth textures, studio lighting",
        "Yağlı Boya": "oil painting style, rich textures, expressive brush strokes, classic art masterpiece",
        "Cyberpunk": "cyberpunk style, glowing neon lights, futuristic city background",
        "Cinematic": "cinematic movie shot, dramatic lighting, 8k resolution, photorealistic, depth of field"
    }
    
    aspect_text = ""
    if "9:16" in aspect_ratio:
        aspect_text = ", vertical 9:16 aspect ratio, portrait orientation"
    elif "16:9" in aspect_ratio:
        aspect_text = ", horizontal 16:9 widescreen aspect ratio, landscape orientation"
    elif "1:1" in aspect_ratio:
        aspect_text = ", 1:1 square ratio"

    modifier = style_modifiers.get(style_preset, "high quality, 8k resolution, highly detailed")
    return f"{user_input}, {modifier}{aspect_text}"

# SIDEBAR
with st.sidebar:
    st.title("⚙️ Ayarlar")
    st.markdown("---")
    use_password = st.checkbox("🔒 Şifre Koruması", value=False)
    user_pass = ""
    if use_password:
        user_pass = st.text_input("Şifre:", type="password")

if use_password and user_pass != "1234":
    st.warning("🔑 Lütfen geçerli şifreyi girin. (Varsayılan: 1234)")
    st.stop()

# BAŞLIK
st.markdown('<h1 class="main-title">✨ KOGCE AI CREATIVE STUDIO PRO</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">FLUX.1-schnell Görsel Motoru • Kesintisiz Üretim</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📸 Görsel Oluştur", "🎥 Video Üret", "🖼️ Galeri"])

# ==================== TAB 1: GÖRSEL ====================
with tab1:
    col_input, col_output = st.columns([1, 1], gap="large")
    
    with col_input:
        st.markdown("### 🎨 İsteyin")
        user_prompt = st.text_area("Tarifiniz:", placeholder="Örn: Karlı dağlarda koşan kurt...", value="", height=120)
        
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            aspect_ratio = st.selectbox("📐 Boyut:", ["Kare (1:1)", "Dikey / Story (9:16)", "Yatay / YouTube (16:9)"])
        with col_opt2:
            style_preset = st.selectbox("🎨 Stil:", ["Doğal / Yok", "Fotogerçekçi (Photorealistic)", "Anime / Manga", "3D Render (Pixar)", "Yağlı Boya", "Cyberpunk", "Cinematic"])
            
        use_ai_boost = st.checkbox("⚡ Prompt Booster", value=True)
        btn_img = st.button("🚀 GÖRSELİ ÜRET", type="primary", use_container_width=True)

    with col_output:
        st.markdown("### 🖼️ Çıktı")
        if btn_img:
            if not user_prompt:
                st.warning("Lütfen bir metin girin!")
            else:
                if use_ai_boost:
                    final_prompt = improve_prompt_with_ai(user_prompt, style_preset, aspect_ratio)
                else:
                    final_prompt = user_prompt

                st.info(f"✨ **Prompt:** {final_prompt}")

                with st.spinner("🎨 FLUX.1 görsel çiziyor..."):
                    # 400 hatasını önlemek için yalnızca sade inputs objesi gönderiyoruz
                    payload = {"inputs": final_prompt}
                    try:
                        response = requests.post(IMAGE_MODEL_URL, headers=headers, json=payload, timeout=60)
                        
                        if response.status_code == 200:
                            image_bytes = response.content
                            image = Image.open(io.BytesIO(image_bytes))
                            st.image(image, caption="FLUX.1 Çıktısı", use_container_width=True)
                            
                            st.download_button(
                                label="📥 Görseli İndir",
                                data=image_bytes,
                                file_name=f"flux_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                            st.session_state.history.append({"type": "image", "data": image, "prompt": final_prompt})
                        elif response.status_code == 503:
                            st.error("⏳ Model sunucuda uyanıyor. Lütfen 15-20 saniye sonra tekrar deneyin.")
                        elif response.status_code == 401:
                            st.error("🔑 API Key Hatası (401): Streamlit Secrets alanındaki HF_API_KEY değerini ve token geçerliliğini kontrol edin.")
                        else:
                            st.error(f"Sunucu Hata Kodu ({response.status_code}): {response.text}")
                    except Exception as e:
                        st.error(f"Bağlantı Hatası: {e}")

# ==================== TAB 2: VİDEO ====================
with tab2:
    col_v_in, col_v_out = st.columns([1, 1], gap="large")
    
    with col_v_in:
        st.markdown("### 🎬 Video Sahnesi")
        vid_prompt = st.text_area("Video Tarifi:", placeholder="Örn: Okyanus üzerinde uçan martı...", value="", height=120)
        btn_vid = st.button("🎬 VİDEO ÜRET", type="primary", use_container_width=True)

    with col_v_out:
        st.markdown("### 🎥 Video Çıktısı")
        if btn_vid:
            if not vid_prompt:
                st.warning("Lütfen metin girin!")
            else:
                final_vid_prompt = improve_prompt_with_ai(vid_prompt, "Cinematic", "16:9")
                st.info(f"✨ **Prompt:** {final_vid_prompt}")

                with st.spinner("🎬 Video işleniyor (İlk çalıştırmada model uyanırken 30-45 sn sürebilir)..."):
                    payload = {"inputs": final_vid_prompt}
                    try:
                        response = requests.post(VIDEO_MODEL_URL, headers=headers, json=payload, timeout=90)
                        if response.status_code == 200:
                            video_bytes = response.content
                            st.video(video_bytes)
                            st.download_button(
                                label="📥 Videoyu İndir (MP4)",
                                data=video_bytes,
                                file_name=f"video_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                                mime="video/mp4",
                                use_container_width=True
                            )
                        elif response.status_code == 503:
                            st.error("⏳ Video modeli uyanıyor. Lütfen 30 saniye sonra tekrar deneyin.")
                        else:
                            st.error(f"Hata ({response.status_code}): {response.text}")
                    except Exception as e:
                        st.error(f"Bağlantı Hatası: {e}")

# ==================== TAB 3: GALERİ ====================
with tab3:
    st.markdown("### 🖼️ Üretilenler")
    if len(st.session_state.history) == 0:
        st.info("Henüz görsel üretilmedi.")
    else:
        cols = st.columns(3)
        for idx, item in enumerate(reversed(st.session_state.history)):
            with cols[idx % 3]:
                if item["type"] == "image":
                    st.image(item["data"], use_container_width=True)
                    st.caption(f"**Prompt:** {item['prompt'][:50]}...")

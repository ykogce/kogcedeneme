import streamlit as st
import requests
import io
from PIL import Image
import datetime

# Streamlit Secrets üzerinden Güvenli Token Okuma
HF_API_KEY = st.secrets.get("HF_API_KEY", "") 

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# Dünyanın En İyi Açık Kaynak Görsel & Video Modelleri (Router API)
IMAGE_MODEL_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
VIDEO_MODEL_URL = "https://router.huggingface.co/hf-inference/models/damo-vilab/text-to-video-ms-1.7m"

# Sayfa Konfigürasyonu
st.set_page_config(page_title="KOGCE AI Studio Pro", page_icon="✨", layout="wide")

# ==================== ULTRA MODERN & YÜKSEK KONTRAST DARK THEME ====================
custom_css = """
<style>
    /* Ana Arka Plan ve Yüksek Kontrastlı Beyaz Yazılar */
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
    }
    
    /* Tüm Etiketler ve Paragraflar İçin Kesin Açık Renk */
    p, span, label, div, .stMarkdown, h1, h2, h3 {
        color: #f1f5f9 !important;
    }

    /* Üst Başlık Gradient Efekti */
    .main-title {
        font-size: 3rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #c084fc 0%, #38bdf8 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }
    
    .sub-title {
        text-align: center;
        color: #94a3b8 !important;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Sekme (Tabs) Tasarımı */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        justify-content: center;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(30, 41, 59, 0.8);
        border-radius: 12px;
        color: #cbd5e1 !important;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0px 24px;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #a855f7 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.4);
    }

    /* Neon Glow Üretim Butonları */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #9333ea 0%, #2563eb 100%);
        color: white !important;
        font-weight: 700;
        font-size: 1.05rem;
        border-radius: 12px;
        border: none;
        padding: 0.75rem 1.5rem;
        box-shadow: 0 4px 20px rgba(147, 51, 234, 0.4);
        transition: all 0.3s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(147, 51, 234, 0.7);
        background: linear-gradient(135deg, #7e22ce 0%, #1d4ed8 100%);
    }

    /* İndirme Butonu */
    div.stDownloadButton > button {
        background-color: rgba(30, 41, 59, 0.9) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    /* Metin Kutuları ve Seçim Alanları */
    .stTextArea textarea {
        background-color: #111827 !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        font-size: 1rem !important;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        background-color: #111827 !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    /* Yan Menü (Sidebar) */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Oturum Hafızası
if "history" not in st.session_state:
    st.session_state.history = []

def improve_prompt_with_ai(user_input, style_preset):
    """
    Kural Tabanlı Süper Hızlı Prompt Zenginleştirici (Option A).
    API kilitlenmesi veya zaman aşımı yaşanmaz. FLUX için mükemmel detay katar.
    """
    style_modifiers = {
        "Fotogerçekçi (Photorealistic)": "photorealistic, 8k resolution, highly detailed, professional photography, cinematic lighting, octane render, masterwork",
        "Anime / Manga": "anime style, highly detailed illustration, studio ghibli inspired, vibrant colors, crisp lines, masterpiece",
        "3D Render (Pixar)": "3D render, Pixar style, Octane render, smooth textures, studio lighting, volumetric shadows",
        "Yağlı Boya": "oil painting style, rich textures, expressive brush strokes, classic art masterpiece",
        "Cyberpunk": "cyberpunk style, glowing neon lights, futuristic city background, volumetric lighting, high contrast",
        "Cinematic": "cinematic movie shot, dramatic lighting, 8k resolution, photorealistic, shallow depth of field, 35mm lens"
    }
    
    modifier = style_modifiers.get(style_preset, "high quality, 8k resolution, highly detailed, masterpiece")
    return f"{user_input}, {modifier}"

# ----- YAN MENÜ (SIDEBAR) -----
with st.sidebar:
    st.title("⚙️ Kontrol Paneli")
    st.markdown("---")
    
    use_password = st.checkbox("🔒 Özel Erişim Şifresi", value=False)
    user_pass = ""
    if use_password:
        user_pass = st.text_input("Şifrenizi Girin:", type="password")
    
    st.markdown("---")
    st.markdown("### 💡 İpuçları")
    st.write("• Türkçe veya İngilizce yazabilirsiniz.")
    st.write("• **FLUX.1-schnell** motoru Türkçe açıklamaları yüksek kalitede anlar.")
    st.write("• Sosyal medya için **9:16 Dikey**, YouTube için **16:9 Yatay** boyut seçebilirsiniz.")

if use_password and user_pass != "1234":
    st.warning("🔑 Lütfen devam etmek için geçerli şifreyi girin. (Varsayılan: 1234)")
    st.stop()

# ----- ANA EKRAN BAŞLIK -----
st.markdown('<h1 class="main-title">✨ KOGCE AI CREATIVE STUDIO PRO</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">FLUX.1-schnell Motoru • 0 Gecikmeli Akıllı Prompt Booster • Yüksek Çözünürlük</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📸 Görsel Oluştur (FLUX)", "🎥 Video Üret", "🖼️ Üretim Galerisi"])

# ==================== TAB 1: GÖRSEL ÜRETİM ====================
with tab1:
    col_input, col_output = st.columns([1, 1], gap="large")
    
    with col_input:
        st.markdown("### 🎨 Hayalinizi Tarif Edin")
        user_prompt = st.text_area(
            "Ne oluşturmak istiyorsunuz?", 
            placeholder="İstediğiniz görseli buraya tarif edin...",
            value="",
            height=120
        )
        
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            aspect_ratio = st.selectbox(
                "📐 Görsel Boyutu:",
                ["Kare (1:1 - 1024x1024)", "Dikey / Story (9:16 - 768x1344)", "Yatay / YouTube (16:9 - 1344x768)"]
            )
        with col_opt2:
            style_preset = st.selectbox(
                "🎨 Sanat Stili:",
                ["Doğal / Yok", "Fotogerçekçi (Photorealistic)", "Anime / Manga", "3D Render (Pixar)", "Yağlı Boya", "Cyberpunk", "Cinematic"]
            )
            
        use_ai_boost = st.checkbox("⚡ Akıllı Prompt Booster (Metni Otomatik Zenginleştir)", value=True)
        btn_img = st.button("🚀 GÖRSELİ ÜRET", type="primary", use_container_width=True)

    with col_output:
        st.markdown("### 🖼️ Çıktı Ekranı")
        if btn_img:
            if not user_prompt:
                st.warning("Lütfen bir açıklama girin!")
            else:
                final_prompt = user_prompt
                
                width, height = 1024, 1024
                if "9:16" in aspect_ratio:
                    width, height = 768, 1344
                elif "16:9" in aspect_ratio:
                    width, height = 1344, 768
                
                if use_ai_boost:
                    final_prompt = improve_prompt_with_ai(user_prompt, style_preset)
                    st.info(f"✨ **Oluşturulan Prompt:** {final_prompt}")

                with st.spinner("🎨 FLUX.1-schnell motoru görseli çiziyor..."):
                    payload = {
                        "inputs": final_prompt,
                        "parameters": {"width": width, "height": height}
                    }
                    try:
                        response = requests.post(IMAGE_MODEL_URL, headers=headers, json=payload, timeout=60)
                        if response.status_code == 200:
                            image_bytes = response.content
                            image = Image.open(io.BytesIO(image_bytes))
                            st.image(image, caption="FLUX.1 Tarafından Üretildi", use_container_width=True)
                            
                            st.download_button(
                                label="📥 Görseli Yüksek Kalitede İndir",
                                data=image_bytes,
                                file_name=f"flux_ai_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                            
                            st.session_state.history.append({"type": "image", "data": image, "prompt": final_prompt})
                        elif response.status_code == 401:
                            st.error("🔑 Yetkilendirme Hatası (401): Streamlit Secrets alanındaki HF_API_KEY bilginizi veya yeni oluşturduğunuz Token'ı kontrol edin.")
                        else:
                            st.error(f"Sunucu Yanıtı ({response.status_code}): Lütfen 10 saniye bekleyip tekrar deneyin.")
                    except requests.exceptions.RequestException:
                        st.error("Sunucu bağlantı zaman aşımına uğradı. Lütfen tekrar deneyin.")

# ==================== TAB 2: VİDEO ÜRETİM ====================
with tab2:
    col_v_in, col_v_out = st.columns([1, 1], gap="large")
    
    with col_v_in:
        st.markdown("### 🎬 Video Sahnesi Kurgulayın")
        vid_prompt = st.text_area(
            "Video İsteğiniz:", 
            placeholder="İstediğiniz video sahnesini tarif edin...",
            value="",
            height=120
        )
        btn_vid = st.button("🎬 VİDEO ÜRET", type="primary", use_container_width=True)

    with col_v_out:
        st.markdown("### 🎥 Video Çıktısı")
        if btn_vid:
            if not vid_prompt:
                st.warning("Lütfen bir video açıklaması yazın!")
            else:
                final_vid_prompt = improve_prompt_with_ai(vid_prompt, "Cinematic")
                st.info(f"✨ **Video Prompt:** {final_vid_prompt}")

                with st.spinner("🎬 Video kareleri işleniyor (30-60 sn sürebilir)..."):
                    payload = {"inputs": final_vid_prompt}
                    try:
                        response = requests.post(VIDEO_MODEL_URL, headers=headers, json=payload, timeout=90)
                        if response.status_code == 200:
                            video_bytes = response.content
                            st.video(video_bytes)
                            
                            st.download_button(
                                label="📥 Videoyu İndir (MP4)",
                                data=video_bytes,
                                file_name=f"ai_video_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                                mime="video/mp4",
                                use_container_width=True
                            )
                        else:
                            st.error(f"Hata ({response.status_code}): Video sunucusu yoğun veya ısınma aşamasında. Lütfen tekrar deneyin.")
                    except requests.exceptions.RequestException:
                        st.error("Video sunucusuna bağlanırken zaman aşımı oluştu.")

# ==================== TAB 3: GALERİ ====================
with tab3:
    st.markdown("### 🖼️ Bu Oturumda Oluşturulanlar")
    
    if len(st.session_state.history) == 0:
        st.info("Henüz bu oturumda içerik üretilmedi. Görsel oluşturduktan sonra burada sergilenecektir!")
    else:
        cols = st.columns(3)
        for idx, item in enumerate(reversed(st.session_state.history)):
            with cols[idx % 3]:
                if item["type"] == "image":
                    st.image(item["data"], use_container_width=True)
                    st.caption(f"**Prompt:** {item['prompt'][:60]}...")

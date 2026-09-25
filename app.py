import streamlit as st
import requests
import io
from PIL import Image
import datetime

# Streamlit Secrets veya varsayılan Token okuma
HF_API_KEY = st.secrets.get("HF_API_KEY", "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX") 

headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# Güncel Hugging Face Router API Adresleri (Erişim Hatalarını Önler)
IMAGE_MODEL_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
VIDEO_MODEL_URL = "https://router.huggingface.co/hf-inference/models/Lightricks/LTX-Video"
TEXT_MODEL_URL  = "https://router.huggingface.co/hf-inference/models/Qwen/Qwen2.5-Coder-32B-Instruct"

# Sayfa Konfigürasyonu
st.set_page_config(page_title="KOGCE AI Studio", page_icon="✨", layout="wide")

# ==================== ULTRA MODERN & YÜKSEK KONTRAST CSS ====================
custom_css = """
<style>
    /* Ana Arka Plan ve Yüksek Kontrastlı Beyaz/Açık Yazılar */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    /* Tüm Genel Etiketler ve Yazılar İçin Açık Renk Zorlaması */
    p, span, label, div, .stMarkdown {
        color: #e5e7eb !important;
    }

    /* Üst Başlık Gradient Efekti */
    .main-title {
        font-size: 3rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #a855f7 0%, #3b82f6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }
    
    .sub-title {
        text-align: center;
        color: #cbd5e1 !important;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Modern Glassmorphism Kart Yapısı */
    div[data-testid="stExpander"], div.stCard {
        background: rgba(17, 24, 39, 0.85);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
    }

    /* Sekme (Tabs) Tasarımı */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        justify-content: center;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre;
        background-color: rgba(30, 41, 59, 0.8);
        border-radius: 12px;
        color: #cbd5e1 !important;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0px 24px;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.5);
    }

    /* Neon Glow Üretim Butonları */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7c3aed 0%, #2563eb 100%);
        color: white !important;
        font-weight: 700;
        font-size: 1.05rem;
        border-radius: 12px;
        border: none;
        padding: 0.75rem 1.5rem;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4);
        transition: all 0.3s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(124, 58, 237, 0.7);
        background: linear-gradient(135deg, #6d28d9 0%, #1d4ed8 100%);
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

    div.stDownloadButton > button:hover {
        background-color: rgba(56, 189, 248, 0.2) !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
    }

    /* Metin Kutuları, Seçim Kutuları ve Açık Beyaz Yazılar */
    .stTextArea textarea {
        background-color: #111827 !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        font-size: 1rem !important;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        background-color: #111827 !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
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
    """Metni profesyonel İngilizce prompta dönüştüren yapay zeka motoru"""
    style_instruction = f" Apply style: {style_preset}." if style_preset != "Doğal / Yok" else ""
    system_prompt = (
        "You are an expert AI image prompt generator. "
        "Take the user's request (in any language) and expand it into a detailed, high-quality, professional English prompt for SDXL. "
        f"Include details like lighting, composition, 8k resolution, cinematic atmosphere.{style_instruction} "
        "Output ONLY the final expanded prompt in English, nothing else."
    )
    
    payload = {
        "inputs": f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n",
        "parameters": {"max_new_tokens": 150, "temperature": 0.7}
    }
    
    try:
        res = requests.post(TEXT_MODEL_URL, headers=headers, json=payload, timeout=10)
        if res.status_code == 200:
            result = res.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
                if "<|im_start|>assistant\n" in generated_text:
                    return generated_text.split("<|im_start|>assistant\n")[-1].strip()
                return generated_text.strip()
    except Exception:
        pass
    return user_input

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
    st.write("• Akıllı Yapay Zeka Desteği açıkken metniniz otomatik geliştirilir.")
    st.write("• Sosyal medya için **9:16 Dikey**, YouTube için **16:9 Yatay** boyut kullanabilirsiniz.")

if use_password and user_pass != "1234":
    st.warning("🔑 Lütfen devam etmek için geçerli şifreyi girin. (Varsayılan: 1234)")
    st.stop()

# ----- ANA EKRAN BAŞLIK -----
st.markdown('<h1 class="main-title">✨ KOGCE AI CREATIVE STUDIO</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">7/24 Kesintisiz • Akıllı Prompt Desteği • Yüksek Çözünürlüklü AI Üretim Platformu</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📸 Görsel Oluştur", "🎥 Video Üret", "🖼️ Üretim Galerisi"])

# ==================== TAB 1: GÖRSEL ÜRETİM ====================
with tab1:
    col_input, col_output = st.columns([1, 1], gap="large")
    
    with col_input:
        st.markdown("### 🎨 Hayalinizi Tarif Edin")
        user_prompt = st.text_area(
            "Ne oluşturmak istiyorsunuz?", 
            placeholder="Örn: Gece vakti yağmurlu sokaklarda neon ışıklarla aydınlatılmış siberpunk bir şehir...",
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
            
        use_ai_boost = st.checkbox("🤖 Akıllı Prompt İyileştirici (Yapay Zeka Metni Geliştirsin)", value=True)
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
                    with st.spinner("🧠 Yapay zeka isteğinizi analiz ediyor..."):
                        final_prompt = improve_prompt_with_ai(user_prompt, style_preset)
                        st.info(f"✨ **Geliştirilen Prompt:** {final_prompt}")

                with st.spinner("🎨 Görsel çiziliyor..."):
                    payload = {
                        "inputs": final_prompt,
                        "parameters": {"width": width, "height": height}
                    }
                    try:
                        response = requests.post(IMAGE_MODEL_URL, headers=headers, json=payload, timeout=60)
                        if response.status_code == 200:
                            image_bytes = response.content
                            image = Image.open(io.BytesIO(image_bytes))
                            st.image(image, caption="Üretilen Görsel", use_container_width=True)
                            
                            st.download_button(
                                label="📥 Görseli Yüksek Kalitede İndir",
                                data=image_bytes,
                                file_name=f"ai_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                            
                            st.session_state.history.append({"type": "image", "data": image, "prompt": final_prompt})
                        else:
                            st.error(f"Hata ({response.status_code}): Model yükleniyor veya yoğun. Lütfen 10-15 saniye sonra tekrar deneyin.")
                    except requests.exceptions.RequestException:
                        st.error("Sunucuya erişilirken zaman aşımı oluştu. Lütfen tekrar deneyin.")

# ==================== TAB 2: VİDEO ÜRETİM ====================
with tab2:
    col_v_in, col_v_out = st.columns([1, 1], gap="large")
    
    with col_v_in:
        st.markdown("### 🎬 Video Sahnesi Kurgulayın")
        vid_prompt = st.text_area(
            "Video İsteğiniz:", 
            placeholder="Örn: Sisli ve karanlık bir ormanda parlayan mavi uzay kapısı...",
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
                with st.spinner("🧠 Video promptu optimize ediliyor..."):
                    final_vid_prompt = improve_prompt_with_ai(vid_prompt, "Cinematic")
                    st.info(f"✨ **Geliştirilen Video Prompt:** {final_vid_prompt}")

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
                            st.error(f"Hata ({response.status_code}): Video sunucusu yoğun. Lütfen tekrar deneyin.")
                    except requests.exceptions.RequestException:
                        st.error("Video sunucusuna bağlanırken zaman aşımı oluştu.")

# ==================== TAB 3: GALERİ ====================
with tab3:
    st.markdown("### 🖼️ Bu Oturumda Oluşturulanlar")
    
    if len(st.session_state.history) == 0:
        st.info("Henüz bu oturumda bir içerik üretilmedi. Görsel oluşturduktan sonra burada sergilenecektir!")
    else:
        cols = st.columns(3)
        for idx, item in enumerate(reversed(st.session_state.history)):
            with cols[idx % 3]:
                if item["type"] == "image":
                    st.image(item["data"], use_container_width=True)
                    st.caption(f"**Prompt:** {item['prompt'][:60]}...")

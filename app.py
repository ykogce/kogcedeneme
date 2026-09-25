import io
import time
import requests
import streamlit as st

from PIL import Image
from huggingface_hub import InferenceClient


# ============================================================
# CONFIG
# ============================================================

APP_PASSWORD = "1234"

PROMPT_MODEL = "openai/gpt-oss-120b"
IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"

# Tulpar / ComfyUI backend
# Daha sonra Cloudflare Tunnel URL'sini buraya bağlayacağız.
LOCAL_BACKEND_URL = st.secrets.get("LOCAL_BACKEND_URL", "").strip()

CHAT_URL = "https://router.huggingface.co/v1/chat/completions"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="KOGCE AI Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "authenticated": False,
    "gallery": [],
    "generated_video": None,
    "video_filename": None,
    "last_prompt": "",
    "last_enhanced_prompt": "",
    "last_script": "",
    "last_scenes": "",
    "last_video_prompt": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HUGGING FACE
# ============================================================

def get_hf_key():
    try:
        return st.secrets["HF_API_KEY"]
    except Exception:
        return None


HF_API_KEY = get_hf_key()


# ============================================================
# GLOBAL STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       KOGCE CORE
       ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 12% 5%,
                rgba(124, 58, 237, 0.20),
                transparent 28%
            ),
            radial-gradient(
                circle at 88% 8%,
                rgba(168, 85, 247, 0.14),
                transparent 30%
            ),
            radial-gradient(
                circle at 50% 100%,
                rgba(76, 29, 149, 0.13),
                transparent 38%
            ),
            #07070c;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2.4rem;
        padding-bottom: 4rem;
    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0b0910 0%,
                #100b18 48%,
                #08070c 100%
            );
        border-right: 1px solid rgba(168, 85, 247, 0.18);
    }

    section[data-testid="stSidebar"] * {
        color: #e9e7ef;
    }

    /* ========================================================
       HEADINGS
       ======================================================== */

    h1,
    h2,
    h3,
    h4 {
        color: #ffffff !important;
        font-weight: 800 !important;
        letter-spacing: -0.035em;
    }

    /* ========================================================
       KOGCE HERO
       ======================================================== */

    .kogce-hero {
        position: relative;
        overflow: hidden;
        padding: 42px 44px;
        margin-bottom: 28px;

        border-radius: 28px;
        border: 1px solid rgba(196, 181, 253, 0.12);

        background:
            linear-gradient(
                135deg,
                rgba(28, 22, 39, 0.96),
                rgba(10, 9, 15, 0.98)
            );

        box-shadow:
            0 25px 80px rgba(0, 0, 0, 0.40),
            inset 0 1px 0 rgba(255,255,255,0.055);
    }

    .kogce-hero::before {
        content: "";
        position: absolute;
        width: 360px;
        height: 360px;
        right: -140px;
        top: -180px;

        background: rgba(139, 92, 246, 0.22);
        filter: blur(100px);
        border-radius: 50%;
        pointer-events: none;
    }

    .kogce-eyebrow {
        color: #c4b5fd;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.20em;
        text-transform: uppercase;
        margin-bottom: 14px;
    }

    .kogce-title {
        position: relative;
        color: #ffffff;
        font-size: clamp(36px, 5vw, 62px);
        line-height: 0.98;
        font-weight: 900;
        letter-spacing: -0.055em;
        margin-bottom: 16px;
    }

    .kogce-title-accent {
        color: #a78bfa;
    }

    .kogce-subtitle {
        position: relative;
        max-width: 820px;
        color: #a7a4b2;
        font-size: 15px;
        line-height: 1.7;
    }

    /* ========================================================
       CARDS
       ======================================================== */

    .kogce-card {
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.075);

        background:
            linear-gradient(
                145deg,
                rgba(23, 20, 31, 0.90),
                rgba(12, 11, 17, 0.92)
            );

        padding: 24px;

        box-shadow:
            0 16px 50px rgba(0,0,0,0.22),
            inset 0 1px 0 rgba(255,255,255,0.035);

        margin-bottom: 18px;
    }

    .kogce-card-title {
        color: #ffffff;
        font-size: 19px;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 5px;
    }

    .kogce-card-subtitle {
        color: #8f8b99;
        font-size: 13px;
        line-height: 1.55;
    }

    /* ========================================================
       FEATURE CARD
       ======================================================== */

    .kogce-feature {
        border-radius: 18px;
        border: 1px solid rgba(168, 85, 247, 0.16);
        background: rgba(25, 18, 35, 0.62);
        padding: 20px;
        min-height: 120px;
    }

    .kogce-feature-icon {
        font-size: 25px;
        margin-bottom: 9px;
    }

    .kogce-feature-title {
        color: #ffffff;
        font-weight: 800;
        font-size: 15px;
        margin-bottom: 5px;
    }

    .kogce-feature-text {
        color: #92909b;
        font-size: 12px;
        line-height: 1.5;
    }

    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        min-height: 46px;

        border-radius: 13px;
        border: 1px solid rgba(168, 85, 247, 0.30);

        background:
            linear-gradient(
                135deg,
                #7c3aed,
                #5b21b6
            );

        color: #ffffff !important;
        font-weight: 800;

        box-shadow:
            0 10px 28px rgba(76, 29, 149, 0.25),
            inset 0 1px 0 rgba(255,255,255,0.10);

        transition:
            transform 0.18s ease,
            border-color 0.18s ease,
            box-shadow 0.18s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(221, 214, 254, 0.65);

        box-shadow:
            0 14px 35px rgba(76, 29, 149, 0.34),
            inset 0 1px 0 rgba(255,255,255,0.13);
    }

    /* ========================================================
       INPUTS
       ======================================================== */

    textarea,
    input {
        border-radius: 13px !important;
    }

    textarea {
        background: rgba(255,255,255,0.97) !important;
        color: #17131d !important;
    }

    input {
        background: rgba(255,255,255,0.97) !important;
        color: #17131d !important;
    }

    textarea::placeholder,
    input::placeholder {
        color: #74717a !important;
    }

    /* ========================================================
       LABELS
       ======================================================== */

    .stTextInput label,
    .stTextArea label,
    .stSelectbox label,
    .stRadio label,
    .stCheckbox label,
    .stToggle label,
    .stSlider label,
    .stFileUploader label {
        color: #e8e5ed !important;
        font-weight: 700 !important;
    }

    /* ========================================================
       TABS
       ======================================================== */

    .stTabs [data-baseweb="tab-list"] {
        gap: 7px;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        color: #918e9c;
        font-weight: 750;
        border-radius: 12px;
        padding: 10px 16px;
    }

    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background: rgba(124, 58, 237, 0.16);
    }

    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(19, 16, 26, 0.90);
        border: 1px dashed rgba(168, 85, 247, 0.38);
        border-radius: 15px;
    }

    /* ========================================================
       DOWNLOAD
       ======================================================== */

    .stDownloadButton > button {
        border-radius: 13px;
        background: rgba(26, 19, 36, 0.94);
        border: 1px solid rgba(168, 85, 247, 0.28);
        color: #ffffff !important;
        font-weight: 750;
    }

    /* ========================================================
       METRICS
       ======================================================== */

    div[data-testid="stMetric"] {
        background: rgba(25, 18, 36, 0.70);
        border: 1px solid rgba(168, 85, 247, 0.18);
        border-radius: 15px;
        padding: 16px;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    div[data-testid="stMetric"] label {
        color: #9d99a7 !important;
    }

    /* ========================================================
       ALERTS
       ======================================================== */

    .stAlert {
        border-radius: 13px;
    }

    /* ========================================================
       IMAGE
       ======================================================== */

    [data-testid="stImage"] img {
        border-radius: 17px;
    }

    /* ========================================================
       DIVIDER
       ======================================================== */

    hr {
        border-color: rgba(255,255,255,0.065);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# AI CORE
# ============================================================

def call_ai(
    system_prompt,
    user_prompt,
    temperature=0.7,
    max_tokens=1800,
):

    if not HF_API_KEY:
        return None, "HF_API_KEY bulunamadı."

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": PROMPT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:

        response = requests.post(
            CHAT_URL,
            headers=headers,
            json=payload,
            timeout=180,
        )

        if response.status_code != 200:

            return (
                None,
                f"AI API hatası: {response.status_code}\n"
                f"{response.text}",
            )

        data = response.json()

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        if isinstance(content, list):

            content = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
            )

        if not content:

            return None, "AI boş cevap döndürdü."

        return str(content).strip(), None

    except requests.exceptions.Timeout:

        return None, "AI bağlantısı zaman aşımına uğradı."

    except Exception as e:

        return None, f"AI bağlantı hatası: {e}"


# ============================================================
# IMAGE PROMPT BUILDER
# ============================================================

def build_image_prompt(
    prompt,
    style,
    lighting,
    camera,
    quality,
    color_mood,
    negative_prompt,
):

    parts = [prompt.strip()]

    if style != "Automatic":

        parts.append(
            f"Visual style: {style}."
        )

    if lighting != "Automatic":

        parts.append(
            f"Lighting: {lighting}."
        )

    if camera != "Automatic":

        parts.append(
            f"Camera and composition: {camera}."
        )

    if quality != "Automatic":

        parts.append(
            f"Image quality: {quality}."
        )

    if color_mood != "Automatic":

        parts.append(
            f"Color and mood: {color_mood}."
        )

    if negative_prompt.strip():

        parts.append(
            "Avoid: "
            + negative_prompt.strip()
            + "."
        )

    return " ".join(parts)


# ============================================================
# IMAGE GENERATION
# ============================================================

def generate_image(
    prompt,
    width,
    height,
):

    if not HF_API_KEY:

        return None, "HF_API_KEY bulunamadı."

    try:

        client = InferenceClient(
            provider="auto",
            api_key=HF_API_KEY,
        )

        image = client.text_to_image(
            prompt=prompt,
            model=IMAGE_MODEL,
            width=width,
            height=height,
        )

        return image, None

    except Exception as e:

        return None, str(e)


# ============================================================
# LOCAL BACKEND
# ============================================================

def local_backend_available():

    if not LOCAL_BACKEND_URL:

        return False

    try:

        response = requests.get(
            f"{LOCAL_BACKEND_URL.rstrip('/')}/health",
            timeout=8,
        )

        return response.status_code == 200

    except Exception:

        return False


def local_backend_request(
    endpoint,
    payload,
    files=None,
    timeout=30,
):

    if not LOCAL_BACKEND_URL:

        return None, (
            "Tulpar backend adresi henüz tanımlanmamış."
        )

    try:

        url = (
            LOCAL_BACKEND_URL.rstrip("/")
            + endpoint
        )

        if files:

            response = requests.post(
                url,
                data=payload,
                files=files,
                timeout=timeout,
            )

        else:

            response = requests.post(
                url,
                json=payload,
                timeout=timeout,
            )

        if response.status_code != 200:

            return (
                None,
                f"Backend HTTP {response.status_code}\n"
                f"{response.text[:4000]}",
            )

        return response, None

    except requests.exceptions.Timeout:

        return None, "Tulpar backend zaman aşımına uğradı."

    except requests.exceptions.RequestException as e:

        return None, f"Tulpar bağlantı hatası: {e}"

    except Exception as e:

        return None, str(e)


# ============================================================
# TEXT TO VIDEO
# ============================================================

def generate_text_video(
    prompt,
    num_frames=33,
    steps=20,
):

    response, error = local_backend_request(
        "/generate-video",
        {
            "prompt": prompt,
            "num_frames": int(num_frames),
            "steps": int(steps),
        },
        timeout=45,
    )

    if error:

        return None, error

    try:

        data = response.json()

        if "video" in data:

            video = data["video"]

            if isinstance(video, str):

                import base64

                return (
                    base64.b64decode(video),
                    None,
                )

        if "download_url" in data:

            download_url = data["download_url"]

            if download_url.startswith("/"):

                download_url = (
                    LOCAL_BACKEND_URL.rstrip("/")
                    + download_url
                )

            video_response = requests.get(
                download_url,
                timeout=180,
            )

            if video_response.status_code == 200:

                return (
                    video_response.content,
                    None,
                )

        return (
            None,
            "Tulpar backend geçerli video sonucu döndürmedi.\n\n"
            f"{data}",
        )

    except Exception as e:

        return None, f"Video sonucu okunamadı: {e}"


# ============================================================
# IMAGE TO VIDEO
# ============================================================

def generate_image_video(
    image_bytes,
    prompt,
    num_frames=33,
    steps=20,
):

    files = {
        "image": (
            "reference.png",
            image_bytes,
            "image/png",
        )
    }

    response, error = local_backend_request(
        "/image-to-video",
        {
            "prompt": prompt,
            "num_frames": str(num_frames),
            "steps": str(steps),
        },
        files=files,
        timeout=45,
    )

    if error:

        return None, error

    try:

        data = response.json()

        if "video" in data:

            import base64

            return (
                base64.b64decode(data["video"]),
                None,
            )

        if "download_url" in data:

            download_url = data["download_url"]

            if download_url.startswith("/"):

                download_url = (
                    LOCAL_BACKEND_URL.rstrip("/")
                    + download_url
                )

            video_response = requests.get(
                download_url,
                timeout=180,
            )

            if video_response.status_code == 200:

                return (
                    video_response.content,
                    None,
                )

        return (
            None,
            f"Tulpar video sonucu geçersiz:\n{data}",
        )

    except Exception as e:

        return None, f"Video sonucu okunamadı: {e}"


# ============================================================
# CHARACTER / IMAGE TO IMAGE
# ============================================================

def generate_character_image(
    image_bytes,
    instruction,
    style,
    strength,
    preserve_face,
    preserve_body,
    preserve_clothes,
):

    """
    Bu endpoint Tulpar + ComfyUI identity/reference pipeline
    kurulduğunda aktif olacak.

    Mevcut HF FLUX text_to_image API'si doğrudan bu işlemi
    güvenilir şekilde yapmadığı için sahte üretim yapılmaz.
    """

    if not LOCAL_BACKEND_URL:

        return (
            None,
            "Karakter motoru henüz Tulpar backend'e bağlanmadı."
        )

    files = {
        "image": (
            "character_reference.png",
            image_bytes,
            "image/png",
        )
    }

    payload = {
        "instruction": instruction,
        "style": style,
        "strength": float(strength),
        "preserve_face": str(preserve_face).lower(),
        "preserve_body": str(preserve_body).lower(),
        "preserve_clothes": str(preserve_clothes).lower(),
    }

    response, error = local_backend_request(
        "/character-edit",
        payload,
        files=files,
        timeout=45,
    )

    if error:

        return None, error

    try:

        data = response.json()

        if "image" in data:

            import base64

            return (
                Image.open(
                    io.BytesIO(
                        base64.b64decode(data["image"])
                    )
                ),
                None,
            )

        if "download_url" in data:

            url = data["download_url"]

            if url.startswith("/"):

                url = (
                    LOCAL_BACKEND_URL.rstrip("/")
                    + url
                )

            result = requests.get(
                url,
                timeout=180,
            )

            if result.status_code == 200:

                return (
                    Image.open(
                        io.BytesIO(
                            result.content
                        )
                    ),
                    None,
                )

        return (
            None,
            f"Karakter backend sonucu geçersiz:\n{data}",
        )

    except Exception as e:

        return None, f"Karakter sonucu okunamadı: {e}"


# ============================================================
# LOGIN
# ============================================================

if not st.session_state.authenticated:

    st.markdown(
        """
        <div class="kogce-hero">

            <div class="kogce-eyebrow">
                ✦ Private AI Creative Workspace
            </div>

            <div class="kogce-title">
                KOGCE <span class="kogce-title-accent">AI Studio</span>
            </div>

            <div class="kogce-subtitle">
                Create. Imagine. Generate.
                Görsel, karakter, video ve AI üretim araçları
                tek bir yaratıcı çalışma alanında.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns(
        [1, 1.15, 1]
    )

    with center:

        st.markdown("### Studio'ya giriş")

        password = st.text_input(
            "Şifre",
            type="password",
            placeholder="Şifrenizi girin",
        )

        if st.button(
            "✦ STUDIO'YA GİR",
            use_container_width=True,
            type="primary",
        ):

            if password == APP_PASSWORD:

                st.session_state.authenticated = True

                st.rerun()

            else:

                st.error("Hatalı şifre.")

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="kogce-eyebrow">
            KOGCE
        </div>

        <div style="
            color:#ffffff;
            font-size:25px;
            font-weight:900;
            letter-spacing:-0.04em;
            margin-bottom:16px;
        ">
            AI Studio
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    if HF_API_KEY:

        st.success("AI SYSTEM ONLINE")

    else:

        st.warning("HF KEY MISSING")

    st.markdown("### Üretim Motorları")

    st.caption("✦ FLUX Görsel")
    st.caption("✦ AI Prompt Robotu")
    st.caption("✦ Karakter Stüdyosu")
    st.caption("✦ Tulpar / ComfyUI")
    st.caption("✦ Wan 2.1 Video")

    st.divider()

    if LOCAL_BACKEND_URL:

        if local_backend_available():

            st.success("TULPAR ONLINE")

        else:

            st.warning("TULPAR OFFLINE")

    else:

        st.info("TULPAR BAĞLANMADI")

    st.divider()

    st.caption(
        f"Galeri: {len(st.session_state.gallery)} görsel"
    )

    if st.button(
        "Çıkış Yap",
        use_container_width=True,
    ):

        st.session_state.authenticated = False

        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="kogce-hero">

        <div class="kogce-eyebrow">
            ✦ AI CREATIVE WORKSPACE
        </div>

        <div class="kogce-title">
            KOGCE <span class="kogce-title-accent">AI Studio</span>
        </div>

        <div class="kogce-subtitle">
            Fikirden görsele, karakterden videoya.
            Üretim sürecinin bütün temel araçları tek panelde.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "✦ Görsel",
        "🎭 Karakter",
        "🎬 Video",
        "🧠 AI Araçları",
        "🖼️ Galeri",
        "⚙️ Sistem",
    ]
)


# ============================================================
# IMAGE TAB
# ============================================================

with tabs[0]:

    st.markdown(
        """
        <div class="kogce-card">

            <div class="kogce-card-title">
                Görsel Studio
            </div>

            <div class="kogce-card-subtitle">
                Fikrini yaz, üretim stilini ve görsel karakterini
                kontrol et. Seçtiğin ayarlar gerçek prompta işlenir.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    prompt = st.text_area(
        "Ana Prompt",
        placeholder=(
            "Örneğin: A man walking alone through an empty "
            "rainy street at night..."
        ),
        height=145,
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        style = st.selectbox(
            "Görsel Stil",
            [
                "Automatic",
                "Photorealistic",
                "Cinematic",
                "Anime",
                "3D Render",
                "Stylized 3D",
                "Cartoon",
                "Fantasy",
                "Cyberpunk",
                "Product Photography",
                "Fashion Editorial",
                "Dark Cinematic",
            ],
        )

    with col2:

        lighting = st.selectbox(
            "Işık",
            [
                "Automatic",
                "Natural daylight",
                "Cinematic lighting",
                "Soft studio lighting",
                "Golden hour",
                "Blue hour",
                "Neon lighting",
                "Dramatic lighting",
                "Low-key lighting",
                "Volumetric lighting",
                "Rainy night lighting",
            ],
        )

    with col3:

        camera = st.selectbox(
            "Kamera",
            [
                "Automatic",
                "Close-up portrait",
                "Medium shot",
                "Full body shot",
                "Wide cinematic shot",
                "Low angle",
                "High angle",
                "Eye level",
                "Aerial perspective",
                "Over-the-shoulder",
            ],
        )

    col4, col5, col6 = st.columns(3)

    with col4:

        quality = st.selectbox(
            "Kalite",
            [
                "Automatic",
                "High detail",
                "Ultra detailed",
                "Photographic realism",
                "Cinematic quality",
                "Sharp professional image",
            ],
        )

    with col5:

        color_mood = st.selectbox(
            "Renk / Atmosfer",
            [
                "Automatic",
                "Natural colors",
                "Dark moody",
                "Warm cinematic",
                "Cool cinematic",
                "Neon futuristic",
                "Muted realistic",
                "High contrast",
            ],
        )

    with col6:

        aspect_ratio = st.selectbox(
            "Aspect Ratio",
            [
                "1:1",
                "9:16",
                "16:9",
            ],
        )

    resolutions = {
        "1:1": (768, 768),
        "9:16": (768, 1344),
        "16:9": (1344, 768),
    }

    width, height = resolutions[aspect_ratio]

    st.caption(
        f"Çözünürlük: {width} × {height}"
    )

    negative_prompt = st.text_input(
        "Negative Prompt",
        placeholder=(
            "blurry, distorted face, bad anatomy, extra fingers..."
        ),
    )

    col7, col8 = st.columns(2)

    with col7:

        ai_boost = st.toggle(
            "🤖 AI Prompt Robotu",
            value=True,
        )

    with col8:

        seed_mode = st.selectbox(
            "Seed",
            [
                "Random",
                "Fixed",
            ],
        )

    if seed_mode == "Fixed":

        seed = st.number_input(
            "Sabit Seed",
            min_value=0,
            max_value=999999999,
            value=123456,
            step=1,
        )

    else:

        seed = None

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "✦ GÖRSEL OLUŞTUR",
        use_container_width=True,
        type="primary",
    ):

        if not prompt.strip():

            st.warning("Önce bir prompt gir.")

        else:

            final_prompt = build_image_prompt(
                prompt,
                style,
                lighting,
                camera,
                quality,
                color_mood,
                negative_prompt,
            )

            if ai_boost:

                with st.spinner(
                    "AI Prompt Robotu görseli hazırlıyor..."
                ):

                    enhanced_prompt, error = call_ai(
                        """
You are an expert professional image-generation prompt engineer.

Transform the user's idea and the supplied visual settings into ONE
high-quality English image-generation prompt.

IMPORTANT:
- Preserve the user's subject and intended meaning.
- Respect every supplied style, lighting, camera, quality and mood choice.
- Do not invent major story elements.
- Make the composition visually coherent.
- Describe realistic details where appropriate.
- Return ONLY the final prompt.
""",
                        f"""
USER IDEA:
{prompt}

STYLE:
{style}

LIGHTING:
{lighting}

CAMERA:
{camera}

QUALITY:
{quality}

COLOR / MOOD:
{color_mood}

NEGATIVE:
{negative_prompt}
""",
                        temperature=0.7,
                        max_tokens=1800,
                    )

                if error:

                    st.error(error)

                    enhanced_prompt = None

                if enhanced_prompt:

                    final_prompt = enhanced_prompt

                    st.session_state.last_enhanced_prompt = (
                        enhanced_prompt
                    )

                    with st.expander(
                        "AI tarafından oluşturulan final prompt",
                        expanded=True,
                    ):

                        st.code(
                            enhanced_prompt,
                            language="text",
                        )

            with st.spinner(
                "Görsel oluşturuluyor..."
            ):

                image, error = generate_image(
                    final_prompt,
                    width,
                    height,
                )

            if error:

                st.error(
                    "Görsel üretilemedi."
                )

                st.code(error)

            elif image:

                st.image(
                    image,
                    use_container_width=True,
                )

                image_buffer = io.BytesIO()

                image.save(
                    image_buffer,
                    format="PNG",
                )

                image_bytes = image_buffer.getvalue()

                st.download_button(
                    "⬇️ PNG İndir",
                    data=image_bytes,
                    file_name="kogce_ai_image.png",
                    mime="image/png",
                    use_container_width=True,
                )

                st.session_state.gallery.append(
                    {
                        "image": image_bytes,
                        "prompt": prompt,
                        "final_prompt": final_prompt,
                    }
                )

                st.session_state.last_prompt = prompt

                st.success(
                    "Görsel başarıyla üretildi."
                )


# ============================================================
# CHARACTER STUDIO
# ============================================================

with tabs[1]:

    st.markdown(
        """
        <div class="kogce-card">

            <div class="kogce-card-title">
                Karakter Studio
            </div>

            <div class="kogce-card-subtitle">
                Bir referans fotoğraf yükle ve kişiyi koruyarak
                yeni görünüm, kıyafet ve sahne tarif et.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Bu bölümün gerçek identity/reference motoru "
        "Tulpar + ComfyUI bağlantısı tamamlandığında aktif olacak."
    )

    character_file = st.file_uploader(
        "Referans fotoğraf",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp",
        ],
        key="character_upload",
    )

    if character_file:

        character_image = Image.open(
            character_file
        ).convert("RGB")

        st.image(
            character_image,
            caption="Referans karakter",
            width=420,
        )

        st.markdown("### Kişiyi koruma")

        c1, c2, c3 = st.columns(3)

        with c1:

            preserve_face = st.checkbox(
                "Yüzü koru",
                value=True,
            )

        with c2:

            preserve_body = st.checkbox(
                "Vücut yapısını koru",
                value=True,
            )

        with c3:

            preserve_clothes = st.checkbox(
                "Kıyafeti koru",
                value=False,
            )

        identity_strength = st.slider(
            "Identity / Reference Gücü",
            min_value=0.10,
            max_value=1.00,
            value=0.85,
            step=0.05,
        )

        character_style = st.selectbox(
            "Stil",
            [
                "Photorealistic",
                "Cinematic",
                "Anime",
                "3D Render",
                "Stylized 3D",
                "Fantasy",
                "Fashion Editorial",
                "Dark Cinematic",
            ],
            key="character_style",
        )

        instruction = st.text_area(
            "Ne değiştirmek istiyorsun?",
            height=180,
            placeholder=(
                "Adam aynı kişi olarak kalsın. "
                "Yüzü ve vücut yapısı korunsun. "
                "Kıyafeti siyah smokin olsun, ince bıyık ekle, "
                "saçlarını omuzlarına kadar uzat. "
                "Yağmurlu, boş bir şehir sokağında gece yürüsün."
            ),
        )

        st.markdown(
            """
            <div class="kogce-feature">

                <div class="kogce-feature-icon">
                    🎭
                </div>

                <div class="kogce-feature-title">
                    Identity Preservation
                </div>

                <div class="kogce-feature-text">
                    Amaç; referans kişiyi korurken yalnızca tarif edilen
                    özellikleri, kıyafeti ve sahneyi değiştirmektir.
                    Gerçek identity node'ları Tulpar backend'inde
                    kullanılacaktır.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(
            "🎭 KARAKTERİ OLUŞTUR",
            use_container_width=True,
            type="primary",
        ):

            if not instruction.strip():

                st.warning(
                    "Önce karakter üzerinde yapılacak değişikliği yaz."
                )

            elif not LOCAL_BACKEND_URL:

                st.warning(
                    "Tulpar backend henüz bağlanmadı. "
                    "Bu buton backend bağlandığında gerçek üretim yapacak."
                )

            else:

                image_buffer = io.BytesIO()

                character_image.save(
                    image_buffer,
                    format="PNG",
                )

                with st.spinner(
                    "Karakter reference pipeline çalışıyor..."
                ):

                    result, error = generate_character_image(
                        image_buffer.getvalue(),
                        instruction,
                        character_style,
                        identity_strength,
                        preserve_face,
                        preserve_body,
                        preserve_clothes,
                    )

                if error:

                    st.error(
                        "Karakter üretilemedi."
                    )

                    st.code(error)

                elif result:

                    st.image(
                        result,
                        use_container_width=True,
                    )

                    output = io.BytesIO()

                    result.save(
                        output,
                        format="PNG",
                    )

                    result_bytes = output.getvalue()

                    st.download_button(
                        "⬇️ Karakter PNG İndir",
                        data=result_bytes,
                        file_name="kogce_character.png",
                        mime="image/png",
                        use_container_width=True,
                    )


# ============================================================
# VIDEO TAB
# ============================================================

with tabs[2]:

    st.markdown(
        """
        <div class="kogce-card">

            <div class="kogce-card-title">
                Video Studio
            </div>

            <div class="kogce-card-subtitle">
                Tulpar RTX 4060 üzerinde çalışan yerel video
                üretim pipeline'ı için hazırlanmıştır.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if LOCAL_BACKEND_URL:

        if local_backend_available():

            st.success(
                "● Tulpar video backend ONLINE"
            )

        else:

            st.warning(
                "Tulpar backend adresi var fakat şu anda erişilemiyor."
            )

    else:

        st.info(
            "Tulpar backend bağlantısı henüz eklenmedi."
        )

    video_mode = st.radio(
        "Video türü",
        [
            "Text → Video",
            "Image → Video",
        ],
        horizontal=True,
    )

    # --------------------------------------------------------
    # TEXT TO VIDEO
    # --------------------------------------------------------

    if video_mode == "Text → Video":

        st.subheader("Text → Video")

        st.caption(
            "Motor: Wan 2.1 T2V 1.3B • Tulpar RTX 4060"
        )

        video_prompt = st.text_area(
            "Video Prompt",
            placeholder=(
                "A man walking naturally through an empty rainy "
                "street at night, realistic body motion, "
                "cinematic camera movement..."
            ),
            height=160,
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            video_style = st.selectbox(
                "Video Stil",
                [
                    "Automatic",
                    "Photorealistic",
                    "Cinematic",
                    "Anime",
                    "3D",
                    "Fantasy",
                ],
                key="video_style",
            )

        with col2:

            camera_motion = st.selectbox(
                "Kamera Hareketi",
                [
                    "Static camera",
                    "Slow push in",
                    "Slow pull out",
                    "Tracking shot",
                    "Pan",
                    "Tilt",
                    "Handheld",
                ],
            )

        with col3:

            motion_level = st.selectbox(
                "Hareket",
                [
                    "Subtle",
                    "Natural",
                    "Dynamic",
                ],
            )

        duration = st.selectbox(
            "Video süresi",
            [
                "Kısa — 33 frame",
                "Orta — 49 frame",
            ],
        )

        if duration.startswith("Kısa"):

            num_frames = 33

        else:

            num_frames = 49

        steps = st.slider(
            "Inference Steps",
            8,
            30,
            20,
        )

        st.caption(
            "RTX 4060 8 GB için önce kısa video ile başlamak daha güvenlidir."
        )

        if st.button(
            "🎬 VIDEO OLUŞTUR",
            use_container_width=True,
            type="primary",
        ):

            if not video_prompt.strip():

                st.warning(
                    "Önce video promptu gir."
                )

            elif not LOCAL_BACKEND_URL:

                st.warning(
                    "Tulpar video backend henüz bağlanmadı."
                )

            else:

                final_video_prompt = (
                    f"{video_prompt.strip()}. "
                    f"Visual style: {video_style}. "
                    f"Camera movement: {camera_motion}. "
                    f"Motion intensity: {motion_level}. "
                    f"Natural realistic motion and consistent subject appearance."
                )

                with st.spinner(
                    "Wan 2.1 video oluşturuyor..."
                ):

                    video, error = generate_text_video(
                        final_video_prompt,
                        num_frames=num_frames,
                        steps=steps,
                    )

                if error:

                    st.error(
                        "Video üretilemedi."
                    )

                    st.code(error)

                elif video:

                    st.session_state.generated_video = video

                    st.session_state.video_filename = (
                        "kogce_text_to_video.mp4"
                    )

                    st.video(video)

                    st.download_button(
                        "⬇️ MP4 İndir",
                        data=video,
                        file_name=(
                            st.session_state.video_filename
                        ),
                        mime="video/mp4",
                        use_container_width=True,
                    )

                    st.success(
                        "Video başarıyla üretildi."
                    )


    # --------------------------------------------------------
    # IMAGE TO VIDEO
    # --------------------------------------------------------

    else:

        st.subheader("Image → Video")

        st.caption(
            "Kaynak görseli Wan video pipeline'ına gönder."
        )

        uploaded_file = st.file_uploader(
            "Başlangıç görseli",
            type=[
                "png",
                "jpg",
                "jpeg",
                "webp",
            ],
            key="video_image_upload",
        )

        motion_prompt = st.text_area(
            "Motion Prompt",
            placeholder=(
                "The man walks naturally forward under the rain, "
                "his clothes move gently with the wind, "
                "slow cinematic camera tracking..."
            ),
            height=150,
        )

        col1, col2 = st.columns(2)

        with col1:

            video_duration = st.selectbox(
                "Süre",
                [
                    "33 frame",
                    "49 frame",
                ],
                key="i2v_duration",
            )

        with col2:

            video_steps = st.slider(
                "Steps",
                8,
                30,
                20,
                key="i2v_steps",
            )

        if uploaded_file:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                image,
                caption="Kaynak Görsel",
                width=520,
            )

            if st.button(
                "🎬 IMAGE → VIDEO",
                use_container_width=True,
                type="primary",
            ):

                if not motion_prompt.strip():

                    st.warning(
                        "Hareket promptu gir."
                    )

                elif not LOCAL_BACKEND_URL:

                    st.warning(
                        "Tulpar backend henüz bağlanmadı."
                    )

                else:

                    buffer = io.BytesIO()

                    image.save(
                        buffer,
                        format="PNG",
                    )

                    if video_duration == "33 frame":

                        frames = 33

                    else:

                        frames = 49

                    with st.spinner(
                        "Wan 2.1 görseli videoya dönüştürüyor..."
                    ):

                        video, error = generate_image_video(
                            buffer.getvalue(),
                            motion_prompt,
                            num_frames=frames,
                            steps=video_steps,
                        )

                    if error:

                        st.error(
                            "Image → Video üretilemedi."
                        )

                        st.code(error)

                    elif video:

                        st.session_state.generated_video = video

                        st.session_state.video_filename = (
                            "kogce_image_to_video.mp4"
                        )

                        st.video(video)

                        st.download_button(
                            "⬇️ MP4 İndir",
                            data=video,
                            file_name=(
                                st.session_state.video_filename
                            ),
                            mime="video/mp4",
                            use_container_width=True,
                        )


# ============================================================
# AI TOOLS TAB
# ============================================================

with tabs[3]:

    st.markdown(
        """
        <div class="kogce-card">

            <div class="kogce-card-title">
                AI Production Tools
            </div>

            <div class="kogce-card-subtitle">
                Fikirden senaryoya, sahne planından video promptuna
                kadar üretim sürecini AI ile hazırla.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    tool = st.selectbox(
        "AI Aracı",
        [
            "AI Prompt Robotu",
            "AI Senaryo Yazarı",
            "AI Sahne Planlayıcı",
            "AI Video Prompt Üretici",
            "AI Shorts Fikir Motoru",
        ],
    )

    # --------------------------------------------------------
    # PROMPT ROBOT
    # --------------------------------------------------------

    if tool == "AI Prompt Robotu":

        st.subheader("AI Prompt Robotu")

        idea = st.text_area(
            "Fikrin",
            height=150,
            placeholder="Kısa fikrini yaz...",
        )

        if st.button(
            "✦ Prompt Oluştur",
            use_container_width=True,
        ):

            if not idea.strip():

                st.warning(
                    "Önce fikir gir."
                )

            else:

                with st.spinner(
                    "Profesyonel prompt hazırlanıyor..."
                ):

                    result, error = call_ai(
                        """
You are an expert professional image-generation prompt engineer.

Transform the user's idea into an extremely detailed but coherent
English image-generation prompt.

Include:
subject, appearance, environment, composition, camera,
lens, lighting, materials, textures, colors, atmosphere,
depth, realism and visual quality.

Preserve the user's intended meaning.
Do not invent major story elements.
Return only the final prompt.
""",
                        idea,
                        temperature=0.75,
                        max_tokens=1800,
                    )

                if error:

                    st.error(error)

                else:

                    st.text_area(
                        "Generated Prompt",
                        value=result,
                        height=330,
                    )


    # --------------------------------------------------------
    # SCRIPT
    # --------------------------------------------------------

    elif tool == "AI Senaryo Yazarı":

        st.subheader("AI Senaryo Yazarı")

        topic = st.text_area(
            "Video konusu",
            height=150,
        )

        duration = st.selectbox(
            "Hedef süre",
            [
                "30 saniye",
                "60 saniye",
                "90 saniye",
                "3 dakika",
            ],
        )

        if st.button(
            "✦ Senaryo Yaz",
            use_container_width=True,
        ):

            if not topic.strip():

                st.warning(
                    "Konu gir."
                )

            else:

                with st.spinner(
                    "Senaryo hazırlanıyor..."
                ):

                    result, error = call_ai(
                        """
You are a professional short-form video script writer.

Create a complete engaging script.

Structure:
Hook
Setup
Development
Retention moments
Payoff
Ending

Use natural language and strong visual moments.
""",
                        f"""
Topic:
{topic}

Target duration:
{duration}
""",
                        temperature=0.8,
                        max_tokens=2500,
                    )

                if error:

                    st.error(error)

                else:

                    st.session_state.last_script = result

                    st.text_area(
                        "Senaryo",
                        value=result,
                        height=450,
                    )


    # --------------------------------------------------------
    # SCENE PLANNER
    # --------------------------------------------------------

    elif tool == "AI Sahne Planlayıcı":

        st.subheader("AI Sahne Planlayıcı")

        script = st.text_area(
            "Senaryoyu gir",
            height=250,
        )

        if st.button(
            "✦ Sahneleri Planla",
            use_container_width=True,
        ):

            if not script.strip():

                st.warning(
                    "Önce senaryo gir."
                )

            else:

                with st.spinner(
                    "Sahne planı hazırlanıyor..."
                ):

                    result, error = call_ai(
                        """
You are a professional film director, storyboard artist
and AI video production planner.

Break the script into logical visual scenes.

For every scene provide:
1. Scene number
2. Duration
3. Visual description
4. Camera movement
5. Character action
6. Environment
7. Lighting
8. Audio/SFX
9. AI generation notes

Maintain character and visual consistency.
""",
                        script,
                        temperature=0.75,
                        max_tokens=3000,
                    )

                if error:

                    st.error(error)

                else:

                    st.session_state.last_scenes = result

                    st.text_area(
                        "Sahne Planı",
                        value=result,
                        height=550,
                    )


    # --------------------------------------------------------
    # VIDEO PROMPT
    # --------------------------------------------------------

    elif tool == "AI Video Prompt Üretici":

        st.subheader(
            "AI Video Prompt Üretici"
        )

        scene = st.text_area(
            "Sahne fikri",
            height=180,
        )

        if st.button(
            "✦ Video Prompt Oluştur",
            use_container_width=True,
        ):

            if not scene.strip():

                st.warning(
                    "Sahne fikri gir."
                )

            else:

                with st.spinner(
                    "Video prompt hazırlanıyor..."
                ):

                    result, error = call_ai(
                        """
You are an expert AI video-generation prompt engineer.

Convert the scene into a production-ready English video prompt.

Focus on:
subject movement,
camera movement,
environment movement,
lighting,
realistic physics,
cinematic composition,
depth,
timing,
atmosphere,
visual consistency.

Return only the final video prompt.
""",
                        scene,
                        temperature=0.75,
                        max_tokens=1800,
                    )

                if error:

                    st.error(error)

                else:

                    st.session_state.last_video_prompt = result

                    st.text_area(
                        "Video Prompt",
                        value=result,
                        height=350,
                    )


    # --------------------------------------------------------
    # SHORTS IDEAS
    # --------------------------------------------------------

    elif tool == "AI Shorts Fikir Motoru":

        st.subheader(
            "AI Shorts Fikir Motoru"
        )

        niche = st.text_input(
            "Niş / konu",
            placeholder=(
                "Örneğin: AI, gaming, animals, satisfying..."
            ),
        )

        count = st.slider(
            "Fikir sayısı",
            min_value=5,
            max_value=20,
            value=10,
        )

        if st.button(
            "✦ Fikirleri Üret",
            use_container_width=True,
        ):

            if not niche.strip():

                st.warning(
                    "Bir niş gir."
                )

            else:

                with st.spinner(
                    "Shorts fikirleri hazırlanıyor..."
                ):

                    result, error = call_ai(
                        """
You are a global short-form video strategist.

Generate original YouTube Shorts concepts.

Prioritize:
strong first-second hooks,
visual simplicity,
high retention,
curiosity,
replayability,
international appeal,
easy AI/video production.

For each idea provide:
1. Title
2. Hook
3. Concept
4. Retention mechanism
5. Visual production idea

Avoid generic repetition.
""",
                        f"""
Niche:
{niche}

Number:
{count}
""",
                        temperature=0.9,
                        max_tokens=3500,
                    )

                if error:

                    st.error(error)

                else:

                    st.text_area(
                        "Shorts Fikirleri",
                        value=result,
                        height=600,
                    )


# ============================================================
# GALLERY
# ============================================================

with tabs[4]:

    st.markdown(
        """
        <div class="kogce-card">

            <div class="kogce-card-title">
                Galeri
            </div>

            <div class="kogce-card-subtitle">
                Bu Streamlit oturumunda oluşturulan görseller.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.gallery:

        st.info(
            "Henüz oluşturulmuş bir görsel yok."
        )

    else:

        for index, item in enumerate(
            reversed(st.session_state.gallery)
        ):

            st.markdown(
                f"### Üretim #{len(st.session_state.gallery) - index}"
            )

            st.image(
                item["image"],
                use_container_width=True,
            )

            st.caption(
                "Orijinal Prompt"
            )

            st.write(
                item["prompt"]
            )

            st.caption(
                "Final Prompt"
            )

            st.code(
                item["final_prompt"],
                language="text",
            )

            st.divider()


# ============================================================
# SYSTEM
# ============================================================

with tabs[5]:

    st.markdown(
        """
        <div class="kogce-card">

            <div class="kogce-card-title">
                System
            </div>

            <div class="kogce-card-subtitle">
                KOGCE üretim motorlarının bağlantı durumu.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "HF API",
            "ONLINE" if HF_API_KEY else "MISSING",
        )

    with col2:

        st.metric(
            "Image",
            "FLUX",
        )

    with col3:

        st.metric(
            "Tulpar",
            (
                "ONLINE"
                if local_backend_available()
                else "OFFLINE"
            ),
        )

    with col4:

        st.metric(
            "Gallery",
            len(st.session_state.gallery),
        )

    st.divider()

    st.subheader(
        "Aktif Görsel Modeli"
    )

    st.code(
        IMAGE_MODEL
    )

    st.subheader(
        "Prompt Modeli"
    )

    st.code(
        PROMPT_MODEL
    )

    st.subheader(
        "Tulpar Backend"
    )

    if LOCAL_BACKEND_URL:

        st.code(
            LOCAL_BACKEND_URL
        )

    else:

        st.info(
            "LOCAL_BACKEND_URL henüz tanımlanmadı."
        )

    st.divider()

    st.subheader(
        "Oturum"
    )

    st.write(
        f"Galerideki görsel sayısı: "
        f"**{len(st.session_state.gallery)}**"
    )

    if st.session_state.generated_video:

        st.success(
            "Son video üretimi mevcut."
        )

    else:

        st.info(
            "Bu oturumda başarılı video üretimi yok."
        )

    st.divider()

    st.caption(
        "KOGCE AI Studio • Private AI Creative Workspace"
    )

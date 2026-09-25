import io
from datetime import datetime

import requests
import streamlit as st
from PIL import Image

try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="KOGCE AI Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
PROMPT_MODEL = "openai/gpt-oss-120b"
CHAT_URL = "https://router.huggingface.co/v1/chat/completions"

APP_PASSWORD = "1234"


# ============================================================
# SESSION
# ============================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>

    /* =========================
       GLOBAL
       ========================= */

    .stApp {
        background:
            radial-gradient(
                circle at 8% 5%,
                rgba(103, 88, 255, 0.15),
                transparent 25%
            ),
            radial-gradient(
                circle at 92% 8%,
                rgba(0, 183, 255, 0.10),
                transparent 24%
            ),
            linear-gradient(
                135deg,
                #07090e 0%,
                #0a0d14 50%,
                #080a10 100%
            );

        color: #f5f7fb;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    /* =========================
       SIDEBAR
       ========================= */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0b0e15 0%,
                #090b11 100%
            );

        border-right: 1px solid rgba(255,255,255,0.06);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }


    /* =========================
       LOGIN
       ========================= */

    .login-space {
        height: 9vh;
    }

    .login-title {
        text-align: center;
        font-size: 46px;
        font-weight: 850;
        letter-spacing: -2px;
        margin-bottom: 4px;

        background:
            linear-gradient(
                100deg,
                #ffffff 0%,
                #d8d5ff 45%,
                #8dd7ff 100%
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .login-subtitle {
        text-align: center;
        color: #8e97a9;
        font-size: 14px;
        letter-spacing: 1.5px;
        margin-bottom: 30px;
    }

    .login-divider {
        width: 55px;
        height: 3px;
        margin: 0 auto 30px auto;
        border-radius: 99px;

        background:
            linear-gradient(
                90deg,
                #766bff,
                #54c9ff
            );
    }


    /* =========================
       HERO
       ========================= */

    .hero-space {
        height: 10px;
    }

    .hero-kicker {
        color: #9c94ff;
        font-size: 11px;
        font-weight: 750;
        letter-spacing: 2.5px;
        margin-bottom: 14px;
    }

    .hero-title {
        font-size: clamp(44px, 6vw, 76px);
        line-height: 0.98;
        font-weight: 850;
        letter-spacing: -3.5px;
        margin-bottom: 20px;

        background:
            linear-gradient(
                100deg,
                #ffffff 0%,
                #d8d5ff 48%,
                #8dd7ff 100%
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-description {
        max-width: 760px;
        color: #929bad;
        font-size: 17px;
        line-height: 1.7;
        margin-bottom: 35px;
    }


    /* =========================
       CARDS
       ========================= */

    .card-title {
        font-size: 17px;
        font-weight: 750;
        margin-bottom: 18px;
        color: #f3f5fa;
    }

    .mini-label {
        color: #777f90;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        margin-bottom: 4px;
    }

    .mini-value {
        color: #eef1f7;
        font-size: 14px;
        font-weight: 650;
        margin-bottom: 16px;
    }


    /* =========================
       STATUS
       ========================= */

    .status-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        margin-right: 8px;
        background: #63e6a0;
        box-shadow: 0 0 12px rgba(99,230,160,0.65);
    }

    .status-dot-warning {
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        margin-right: 8px;
        background: #ffc766;
        box-shadow: 0 0 12px rgba(255,199,102,0.45);
    }


    /* =========================
       TABS
       ========================= */

    button[data-baseweb="tab"] {
        font-weight: 650;
    }


    /* =========================
       BUTTONS
       ========================= */

    .stButton > button {
        min-height: 44px;
        border-radius: 12px;
        font-weight: 700;
        border: 1px solid rgba(255,255,255,0.08);
    }


    /* =========================
       INPUTS
       ========================= */

    div[data-baseweb="input"],
    div[data-baseweb="textarea"] {
        border-radius: 12px;
    }


    /* =========================
       FOOTER
       ========================= */

    .footer {
        text-align: center;
        color: #4f5766;
        font-size: 11px;
        padding-top: 25px;
        letter-spacing: 0.5px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def get_hf_key():
    try:
        return st.secrets["HF_API_KEY"]
    except Exception:
        return None


HF_API_KEY = get_hf_key()


def improve_prompt_with_ai(user_prompt):

    if not HF_API_KEY:
        return None, "HF_API_KEY bulunamadı."

    system_prompt = """
You are a professional AI image prompt engineer.

Transform the user's short idea into a detailed, professional English
prompt suitable for FLUX image generation.

Preserve the original concept.

Improve:
- subject
- environment
- composition
- camera angle
- lighting
- materials
- colors
- atmosphere
- realism
- visual details

Do not add unrelated concepts.

Return ONLY the final image prompt.
Do not explain your answer.
"""

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
        "temperature": 0.7,
        "max_tokens": 500,
    }

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }

    try:

        response = requests.post(
            CHAT_URL,
            headers=headers,
            json=payload,
            timeout=120,
        )

        if response.status_code != 200:
            return (
                None,
                f"Prompt AI HTTP {response.status_code}: "
                f"{response.text[:500]}",
            )

        data = response.json()

        choices = data.get("choices", [])

        if not choices:
            return None, "Prompt AI boş cevap döndürdü."

        content = choices[0].get(
            "message",
            {}
        ).get(
            "content",
            ""
        )

        if not content:
            return None, "Prompt AI metin döndürmedi."

        return content.strip(), None

    except Exception as e:
        return None, f"Prompt AI hatası: {e}"


def get_dimensions(aspect_ratio):

    if aspect_ratio == "1:1 Kare":
        return 768, 768

    if aspect_ratio == "9:16 Dikey":
        return 768, 1344

    if aspect_ratio == "16:9 Yatay":
        return 1344, 768

    return 768, 768


def generate_image(prompt, width, height):

    if not HF_API_KEY:
        return None, "HF_API_KEY bulunamadı."

    if InferenceClient is None:
        return None, "huggingface_hub paketi bulunamadı."

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
        return None, f"Görsel üretim hatası: {e}"


# ============================================================
# LOGIN SCREEN
# ============================================================

def login_screen():

    st.markdown(
        '<div class="login-space"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "KOGCE AI Studio",
        help="AI Creative Workspace",
    )

    st.markdown(
        '<div class="login-subtitle">'
        'AI Creative Workspace'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="login-divider"></div>',
        unsafe_allow_html=True,
    )

    left, center, right = st.columns(
        [1.1, 1.8, 1.1]
    )

    with center:

        password = st.text_input(
            "Uygulama şifresi",
            type="password",
            placeholder="Şifrenizi girin",
            label_visibility="visible",
        )

        if st.button(
            "Giriş Yap",
            use_container_width=True,
            type="primary",
        ):

            if password == APP_PASSWORD:

                st.session_state.authenticated = True
                st.rerun()

            else:

                st.error("Şifre yanlış.")

        st.caption(
            "KOGCE AI Studio"
        )


if not st.session_state.authenticated:

    login_screen()
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## KOGCE AI"
    )

    st.caption(
        "Creative Workspace"
    )

    st.divider()

    st.markdown(
        "### Sistem"
    )

    if HF_API_KEY:
        st.success("HF API aktif")
    else:
        st.error("HF API eksik")

    st.markdown(
        "### Motorlar"
    )

    st.caption("Prompt AI")
    st.write("GPT OSS")

    st.caption("Görsel")
    st.write("FLUX.1-schnell")

    st.caption("Video")
    st.write("Beklemede")

    st.divider()

    st.caption(
        f"{len(st.session_state.history)} görsel oluşturuldu"
    )

    if st.button(
        "Çıkış Yap",
        use_container_width=True,
    ):

        st.session_state.authenticated = False
        st.rerun()


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="hero-space"></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-kicker">AI CREATIVE WORKSPACE</div>',
    unsafe_allow_html=True,
)

st.markdown(
    "KOGCE AI Studio"
)

st.markdown(
    """
    <div class="hero-description">
        Fikirden profesyonel görsele.<br>
        AI Prompt Robotu kısa fikrini analiz eder,
        geliştirir ve FLUX görüntü motoruna gönderir.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

tab_create, tab_video, tab_gallery, tab_system = st.tabs(
    [
        "✨ Görsel Oluştur",
        "🎥 Video Üretim Durumu",
        "🖼️ Galeri",
        "⚙️ Sistem",
    ]
)


# ============================================================
# IMAGE CREATION
# ============================================================

with tab_create:

    st.markdown(
        "### Görsel Oluştur"
    )

    st.caption(
        "Kısa fikrini gir. AI Prompt Robotu fikrini "
        "profesyonel bir FLUX promptuna dönüştürebilir."
    )

    left_col, right_col = st.columns(
        [1.15, 0.85],
        gap="large",
    )

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    with left_col:

        user_prompt = st.text_area(
            "Fikir / Prompt",
            height=180,
            placeholder=(
                "Örnek: "
                "A futuristic sports car driving "
                "through a neon city at night"
            ),
        )

        use_ai_boost = st.toggle(
            "🤖 AI Prompt Robotu",
            value=True,
        )

        if use_ai_boost:
            st.success(
                "AI Prompt Robotu aktif"
            )
        else:
            st.info(
                "AI Prompt Robotu kapalı"
            )

        aspect_ratio = st.selectbox(
            "Görüntü oranı",
            [
                "1:1 Kare",
                "9:16 Dikey",
                "16:9 Yatay",
            ],
        )

        generate_button = st.button(
            "✨ Görsel Oluştur",
            use_container_width=True,
            type="primary",
        )


    # --------------------------------------------------------
    # ENGINE STATUS
    # --------------------------------------------------------

    with right_col:

        st.markdown(
            "#### Üretim Sistemi"
        )

        st.caption(
            "Aktif motorlar ve bağlantı durumu"
        )

        st.success(
            "●  Prompt AI — Hazır"
        )

        st.success(
            "●  FLUX.1-schnell — Hazır"
        )

        st.warning(
            "●  Video Engine — Henüz bağlı değil"
        )


    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    if generate_button:

        if not user_prompt.strip():

            st.warning(
                "Önce bir fikir veya prompt gir."
            )

        elif not HF_API_KEY:

            st.error(
                "HF_API_KEY bulunamadı. "
                "Streamlit Secrets bölümünü kontrol et."
            )

        else:

            final_prompt = user_prompt.strip()

            # -----------------------------------------------
            # AI PROMPT
            # -----------------------------------------------

            if use_ai_boost:

                with st.spinner(
                    "AI Prompt Robotu fikrini geliştiriyor..."
                ):

                    enhanced_prompt, prompt_error = (
                        improve_prompt_with_ai(
                            final_prompt
                        )
                    )

                if prompt_error:

                    st.error(prompt_error)
                    st.stop()

                final_prompt = enhanced_prompt

                st.markdown(
                    "#### Geliştirilmiş Prompt"
                )

                st.info(
                    final_prompt
                )


            # -----------------------------------------------
            # IMAGE
            # -----------------------------------------------

            width, height = get_dimensions(
                aspect_ratio
            )

            with st.spinner(
                "FLUX görseli oluşturuyor..."
            ):

                image, image_error = generate_image(
                    final_prompt,
                    width,
                    height,
                )

            if image_error:

                st.error(image_error)

            elif image is not None:

                st.success(
                    "Görsel başarıyla oluşturuldu."
                )

                st.image(
                    image,
                    use_container_width=True,
                )

                image_bytes = io.BytesIO()

                image.save(
                    image_bytes,
                    format="PNG",
                )

                image_bytes.seek(0)

                filename = (
                    "kogce_ai_"
                    + datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )
                    + ".png"
                )

                st.download_button(
                    "⬇️ PNG olarak indir",
                    data=image_bytes.getvalue(),
                    file_name=filename,
                    mime="image/png",
                    use_container_width=True,
                )

                st.session_state.history.insert(
                    0,
                    {
                        "image": image,
                        "original_prompt":
                            user_prompt.strip(),
                        "final_prompt":
                            final_prompt,
                        "aspect_ratio":
                            aspect_ratio,
                        "created_at":
                            datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                    },
                )


# ============================================================
# VIDEO
# ============================================================

with tab_video:

    st.markdown(
        "### Video Üretim Durumu"
    )

    st.caption(
        "Video otomasyonunun mevcut bağlantı durumunu gösterir."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.success(
            "AI Prompt Robotu\n\nAKTİF"
        )

    with col2:

        st.success(
            "FLUX Görsel Motoru\n\nAKTİF"
        )

    with col3:

        st.warning(
            "Video Motoru\n\nBEKLEMEDE"
        )

    st.divider()

    st.markdown(
        "### Planlanan Video Pipeline"
    )

    st.write(
        "IDEA  →  AI PROMPT ROBOT  →  SCENE PLANNER  →  "
        "IMAGE GENERATION  →  VIDEO GENERATION  →  "
        "VOICE / AUDIO  →  VIDEO ASSEMBLY  →  MP4"
    )

    st.info(
        "Şu anda görsel üretim sistemi aktif. "
        "Video motoru henüz uygulamaya bağlanmadı."
    )


# ============================================================
# GALLERY
# ============================================================

with tab_gallery:

    st.markdown(
        "### Galeri"
    )

    st.caption(
        "Bu oturum sırasında oluşturulan görseller."
    )

    if not st.session_state.history:

        st.info(
            "Henüz oluşturulmuş bir görsel yok."
        )

    else:

        for index, item in enumerate(
            st.session_state.history
        ):

            st.divider()

            image_col, info_col = st.columns(
                [0.55, 0.45],
                gap="large",
            )

            with image_col:

                st.image(
                    item["image"],
                    use_container_width=True,
                )

            with info_col:

                st.markdown(
                    f"#### Görsel #{index + 1}"
                )

                st.caption(
                    f"Oluşturulma: {item['created_at']}"
                )

                st.caption(
                    f"Oran: {item['aspect_ratio']}"
                )

                st.markdown(
                    "**Orijinal fikir**"
                )

                st.write(
                    item["original_prompt"]
                )

                st.markdown(
                    "**Final prompt**"
                )

                st.write(
                    item["final_prompt"]
                )

                image_bytes = io.BytesIO()

                item["image"].save(
                    image_bytes,
                    format="PNG",
                )

                image_bytes.seek(0)

                st.download_button(
                    "⬇️ Görseli indir",
                    data=image_bytes.getvalue(),
                    file_name=(
                        f"kogce_gallery_{index + 1}.png"
                    ),
                    mime="image/png",
                    key=f"gallery_download_{index}",
                    use_container_width=True,
                )


# ============================================================
# SYSTEM
# ============================================================

with tab_system:

    st.markdown(
        "### Sistem"
    )

    st.caption(
        "KOGCE AI Studio bağlantı ve motor bilgileri."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "#### API Durumu"
        )

        if HF_API_KEY:

            st.success(
                "Hugging Face API — AKTİF"
            )

        else:

            st.error(
                "Hugging Face API — BULUNAMADI"
            )

    with col2:

        st.markdown(
            "#### Modeller"
        )

        st.write(
            "Görsel: FLUX.1-schnell"
        )

        st.write(
            "Prompt: GPT OSS 120B"
        )

    st.divider()

    st.metric(
        "Bu oturumda üretilen görsel",
        len(st.session_state.history),
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)

st.markdown(
    '<div class="footer">'
    'KOGCE AI Studio • AI Creative Workspace'
    '</div>',
    unsafe_allow_html=True,
)

import io
from datetime import datetime

import requests
import streamlit as st

try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None


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
# CONFIG
# ============================================================

IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
PROMPT_MODEL = "openai/gpt-oss-120b"
CHAT_URL = "https://router.huggingface.co/v1/chat/completions"

APP_PASSWORD = "1234"


# ============================================================
# SESSION STATE
# ============================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# GLOBAL DESIGN
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap'
    );

    .stApp {
        background:
            radial-gradient(
                circle at 12% 8%,
                rgba(115, 76, 255, 0.16),
                transparent 27%
            ),
            radial-gradient(
                circle at 88% 12%,
                rgba(0, 185, 255, 0.10),
                transparent 25%
            ),
            radial-gradient(
                circle at 50% 100%,
                rgba(92, 47, 170, 0.08),
                transparent 30%
            ),
            #07090f;

        color: #f5f6fb;
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0a0c13 0%,
                #080a0f 100%
            );

        border-right:
            1px solid rgba(255,255,255,0.06);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.8rem;
    }


    /* ======================================================
       LOGIN
       ====================================================== */

    .login-container {
        max-width: 560px;
        margin: 9vh auto 0 auto;
        text-align: center;
    }

    .login-mark {
        width: 72px;
        height: 72px;
        margin: 0 auto 25px auto;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 22px;

        background:
            linear-gradient(
                145deg,
                #2d244f 0%,
                #6f55c9 42%,
                #b3a1ff 100%
            );

        border:
            1px solid rgba(255,255,255,0.22);

        box-shadow:
            0 0 45px rgba(117,82,255,0.28),
            inset 0 1px 0 rgba(255,255,255,0.28);

        color: white;
        font-size: 32px;
        font-weight: 900;
    }

    .login-brand {
        font-size: clamp(46px, 7vw, 76px);
        line-height: 0.98;

        font-weight: 900;
        letter-spacing: -4px;

        margin-bottom: 12px;

        background:
            linear-gradient(
                105deg,
                #ffffff 0%,
                #ddd6ff 28%,
                #a88cff 52%,
                #7657dc 70%,
                #d5ccff 100%
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;

        filter:
            drop-shadow(
                0 0 24px
                rgba(130,95,255,0.18)
            );
    }

    .login-subtitle {
        color: #9299a9;
        font-size: 14px;
        font-weight: 500;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 34px;
    }

    .login-line {
        width: 70px;
        height: 2px;

        margin: 0 auto 34px auto;

        border-radius: 99px;

        background:
            linear-gradient(
                90deg,
                transparent,
                #896cff,
                #b7a5ff,
                transparent
            );

        box-shadow:
            0 0 16px
            rgba(137,108,255,0.55);
    }

    .login-description {
        color: #737c8e;
        font-size: 13px;
        line-height: 1.6;
        margin-bottom: 22px;
    }

    div[data-testid="stTextInput"] input {
        background: rgba(15,17,26,0.82);
        border:
            1px solid rgba(255,255,255,0.09);

        border-radius: 14px;

        color: #ffffff;

        min-height: 50px;

        text-align: center;
        font-size: 15px;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color:
            rgba(145,116,255,0.75);

        box-shadow:
            0 0 0 1px
            rgba(145,116,255,0.25),
            0 0 28px
            rgba(110,80,230,0.12);
    }

    /* LOGIN BUTTON */

    .login-container + * {
        text-align: center;
    }


    /* ======================================================
       MAIN HERO
       ====================================================== */

    .hero-kicker {
        color: #9988ff;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 3px;
        margin-bottom: 13px;
    }

    .hero-brand {
        font-size: clamp(44px, 6vw, 74px);
        line-height: 0.98;
        font-weight: 900;
        letter-spacing: -4px;

        background:
            linear-gradient(
                105deg,
                #ffffff 0%,
                #d9d3ff 30%,
                #a78cff 55%,
                #87d8ff 100%
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;

        margin-bottom: 20px;
    }

    .hero-description {
        max-width: 760px;
        color: #929aaa;
        font-size: 17px;
        line-height: 1.7;
        margin-bottom: 34px;
    }


    /* ======================================================
       SECTION
       ====================================================== */

    .section-heading {
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -0.7px;
        margin-bottom: 5px;
    }

    .section-description {
        color: #858e9f;
        font-size: 14px;
        margin-bottom: 22px;
    }


    /* ======================================================
       STATUS
       ====================================================== */

    .engine-title {
        font-size: 17px;
        font-weight: 800;
        color: #f1f3f8;
        margin-bottom: 5px;
    }

    .engine-subtitle {
        color: #727b8d;
        font-size: 12px;
        margin-bottom: 20px;
    }


    /* ======================================================
       TABS
       ====================================================== */

    button[data-baseweb="tab"] {
        font-weight: 700;
    }


    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button {
        min-height: 45px;
        border-radius: 13px;
        font-weight: 700;
        border:
            1px solid rgba(255,255,255,0.08);

        transition:
            all 0.18s ease;
    }

    .stButton > button:hover {
        border-color:
            rgba(150,125,255,0.42);

        box-shadow:
            0 8px 28px
            rgba(90,65,190,0.16);
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {
        text-align: center;
        color: #4b5361;
        font-size: 11px;
        padding-top: 24px;
        letter-spacing: 0.6px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


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
# PROMPT AI
# ============================================================

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
- cinematic quality

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


# ============================================================
# IMAGE GENERATION
# ============================================================

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
# LOGIN
# ============================================================

def login_screen():

    st.markdown(
        '<div class="login-container">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="login-mark">✦</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="login-brand">'
        'KOGCE AI Studio'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="login-subtitle">'
        'AI Creative Workspace'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="login-line"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="login-description">'
        'Create. Imagine. Generate.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )

    left, center, right = st.columns(
        [1.1, 1.8, 1.1]
    )

    with center:

        password = st.text_input(
            "Şifre",
            type="password",
            placeholder="Uygulama şifresi",
            label_visibility="collapsed",
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

                st.error(
                    "Şifre yanlış."
                )

        st.markdown(
            '<div class="footer">'
            'KOGCE AI Studio'
            '</div>',
            unsafe_allow_html=True,
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
        st.success(
            "HF API aktif"
        )
    else:
        st.error(
            "HF API eksik"
        )

    st.markdown(
        "### Motorlar"
    )

    st.caption(
        "Prompt AI"
    )

    st.write(
        "GPT OSS"
    )

    st.caption(
        "Görsel"
    )

    st.write(
        "FLUX.1-schnell"
    )

    st.caption(
        "Video"
    )

    st.write(
        "Beklemede"
    )

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
# MAIN HERO
# ============================================================

st.markdown(
    '<div class="hero-kicker">'
    'AI CREATIVE WORKSPACE'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-brand">'
    'KOGCE AI Studio'
    '</div>',
    unsafe_allow_html=True,
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
# CREATE
# ============================================================

with tab_create:

    st.markdown(
        '<div class="section-heading">'
        'Görsel Oluştur'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        'Kısa fikrini gir. AI Prompt Robotu fikrini '
        'profesyonel bir FLUX promptuna dönüştürebilir.'
        '</div>',
        unsafe_allow_html=True,
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
                "Örnek: A futuristic sports car "
                "driving through a neon city at night"
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
    # ENGINE
    # --------------------------------------------------------

    with right_col:

        st.markdown(
            '<div class="engine-title">'
            'Üretim Sistemi'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="engine-subtitle">'
            'Aktif motorlar ve bağlantı durumu'
            '</div>',
            unsafe_allow_html=True,
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


            # ------------------------------------------------
            # PROMPT AI
            # ------------------------------------------------

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

                    st.error(
                        prompt_error
                    )

                    st.stop()

                final_prompt = enhanced_prompt

                st.markdown(
                    "#### Geliştirilmiş Prompt"
                )

                st.info(
                    final_prompt
                )


            # ------------------------------------------------
            # IMAGE
            # ------------------------------------------------

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

                st.error(
                    image_error
                )

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
# VIDEO STATUS
# ============================================================

with tab_video:

    st.markdown(
        "### Video Üretim Durumu"
    )

    st.caption(
        "Video otomasyonunun mevcut bağlantı durumunu gösterir."
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.success(
            "AI Prompt Robotu\n\nAKTİF"
        )

    with c2:

        st.success(
            "FLUX Görsel Motoru\n\nAKTİF"
        )

    with c3:

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

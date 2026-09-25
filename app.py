import io
from datetime import datetime

import requests
import streamlit as st

try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None


# ============================================================
# PAGE
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
# SESSION
# ============================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# HUGGING FACE KEY
# ============================================================

def get_hf_key():
    try:
        return st.secrets["HF_API_KEY"]
    except Exception:
        return None


HF_API_KEY = get_hf_key()


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(116, 60, 255, 0.13),
                transparent 30%
            ),
            radial-gradient(
                circle at 85% 85%,
                rgba(168, 75, 255, 0.10),
                transparent 28%
            ),
            #08080d;
        color: #f4f4f7;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0c0b12 0%,
                #09090e 100%
            );
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    section[data-testid="stSidebar"] * {
        color: #eeeeF5;
    }


    /* ========================================================
       LOGIN
       ======================================================== */

    .login-wrap {
        min-height: 72vh;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
    }

    .login-box {
        width: 100%;
        max-width: 650px;
        margin: auto;
        padding: 35px 30px;
    }

    .login-mark {
        font-size: 46px;
        font-weight: 800;
        color: #b78cff;
        text-shadow:
            0 0 18px rgba(164, 91, 255, 0.65),
            0 0 45px rgba(111, 55, 255, 0.35);
        margin-bottom: 8px;
    }

    .login-brand {
        font-size: clamp(42px, 6vw, 74px);
        line-height: 1.0;
        font-weight: 800;
        letter-spacing: -3px;

        background:
            linear-gradient(
                110deg,
                #ffffff 0%,
                #e4d7ff 22%,
                #b77aff 48%,
                #8c4cff 70%,
                #ffffff 100%
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;

        background-clip: text;
        color: transparent;

        filter:
            drop-shadow(0 0 18px rgba(145, 75, 255, 0.20));
    }

    .login-subtitle {
        margin-top: 14px;
        color: #a7a4b3;
        font-size: 15px;
        letter-spacing: 4px;
        text-transform: uppercase;
        font-weight: 600;
    }

    .login-line {
        width: 90px;
        height: 2px;
        margin: 25px auto 18px auto;
        background: linear-gradient(
            90deg,
            transparent,
            #9d55ff,
            transparent
        );
        box-shadow: 0 0 15px rgba(157,85,255,0.5);
    }

    .login-description {
        color: #d5d2dc;
        font-size: 16px;
        letter-spacing: 1px;
    }


    /* ========================================================
       INPUTS — DARK UI
       ======================================================== */

    div[data-baseweb="input"] {
        border-radius: 12px !important;
    }


    /* ========================================================
       PROMPT INPUT — WHITE BACKGROUND / BLACK TEXT
       ======================================================== */

    div[data-testid="stTextArea"] textarea {
        background-color: #ffffff !important;
        color: #000000 !important;

        -webkit-text-fill-color: #000000 !important;

        caret-color: #000000 !important;

        border: 1px solid #d8d8df !important;
        border-radius: 14px !important;

        font-size: 16px !important;
        font-weight: 500 !important;

        padding: 16px !important;

        box-shadow:
            0 4px 18px rgba(0,0,0,0.12) !important;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: #8b4cff !important;

        box-shadow:
            0 0 0 1px #8b4cff,
            0 0 20px rgba(139,76,255,0.20) !important;
    }

    div[data-testid="stTextArea"] textarea::placeholder {
        color: #777780 !important;
        -webkit-text-fill-color: #777780 !important;
        opacity: 1 !important;
    }


    /* ========================================================
       PASSWORD INPUT
       ======================================================== */

    div[data-testid="stTextInput"] input {
        background: #ffffff !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;

        border: 1px solid #d6d6df !important;
        border-radius: 12px !important;

        font-size: 16px !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #777780 !important;
        -webkit-text-fill-color: #777780 !important;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        border-radius: 12px !important;
        border: 1px solid rgba(161, 92, 255, 0.35) !important;

        background:
            linear-gradient(
                135deg,
                #7d3cff,
                #a455ff
            ) !important;

        color: white !important;

        font-weight: 700 !important;

        min-height: 46px;

        box-shadow:
            0 8px 24px rgba(126,60,255,0.22);
    }

    .stButton > button:hover {
        border-color: rgba(196, 155, 255, 0.8) !important;

        box-shadow:
            0 10px 30px rgba(126,60,255,0.35);
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1.5px;

        background:
            linear-gradient(
                110deg,
                #ffffff,
                #c08aff,
                #ffffff
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-subtitle {
        color: #a7a4b3;
        font-size: 15px;
        line-height: 1.7;
    }


    /* ========================================================
       CARDS
       ======================================================== */

    .glass-card {
        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.055),
                rgba(255,255,255,0.018)
            );

        border:
            1px solid rgba(255,255,255,0.08);

        border-radius: 18px;

        padding: 22px;

        box-shadow:
            0 15px 45px rgba(0,0,0,0.18);

        margin-bottom: 18px;
    }


    /* ========================================================
       STATUS
       ======================================================== */

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #7dff9d;

        box-shadow:
            0 0 10px rgba(125,255,157,0.8);

        margin-right: 7px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOGIN
# ============================================================

def login_screen():

    st.markdown(
        """
        <div class="login-wrap">
            <div class="login-box">
                <div class="login-mark">✦</div>
                <div class="login-brand">KOGCE AI Studio</div>
                <div class="login-subtitle">
                    AI CREATIVE WORKSPACE
                </div>
                <div class="login-line"></div>
                <div class="login-description">
                    Create. Imagine. Generate.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1.15, 1.7, 1.15])

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

                st.error("Şifre yanlış.")


# ============================================================
# AI PROMPT ENHANCER
# ============================================================

def enhance_prompt(user_prompt):

    if not HF_API_KEY:
        return None, "Hugging Face API anahtarı bulunamadı."

    system_prompt = """
You are a professional AI image prompt engineer.

Convert the user's short idea into a highly detailed English
prompt for a modern text-to-image model.

Improve:
- subject
- composition
- camera
- lighting
- environment
- materials
- colors
- atmosphere
- realism
- cinematic quality
- visual details

Do not explain anything.

Return ONLY the final image prompt.

Do not add:
"Here is your prompt"
"Prompt:"
or any explanation.
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

        response.raise_for_status()

        data = response.json()

        result = data["choices"][0]["message"]["content"].strip()

        return result, None

    except Exception as e:

        return None, str(e)


# ============================================================
# IMAGE GENERATION
# ============================================================

def generate_image(prompt, width, height):

    if not HF_API_KEY:
        return None, "Hugging Face API anahtarı bulunamadı."

    if InferenceClient is None:
        return None, "huggingface_hub kütüphanesi bulunamadı."

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
# MAIN APP
# ============================================================

def main_app():

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    with st.sidebar:

        st.markdown(
            "## ✦ KOGCE AI Studio"
        )

        st.caption("AI Creative Workspace")

        st.divider()

        st.markdown("### Sistem")

        if HF_API_KEY:
            st.success("HF API bağlantısı aktif")
        else:
            st.error("HF API anahtarı yok")

        st.markdown(
            '<span class="status-dot"></span> Görsel Motoru Aktif',
            unsafe_allow_html=True,
        )

        st.caption("FLUX.1-schnell")

        st.divider()

        st.markdown("### Menü")

        page = st.radio(
            "Sayfa",
            [
                "✨ Görsel Oluştur",
                "🎥 Video Üretim Durumu",
                "🖼️ Galeri",
                "⚙️ Sistem",
            ],
            label_visibility="collapsed",
        )

        st.divider()

        if st.button(
            "Çıkış Yap",
            use_container_width=True,
        ):

            st.session_state.authenticated = False

            st.rerun()


    # --------------------------------------------------------
    # IMAGE GENERATION
    # --------------------------------------------------------

    if page == "✨ Görsel Oluştur":

        st.markdown(
            '<div class="hero-title">KOGCE AI Studio</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="hero-subtitle">
                Fikrini yaz. AI Prompt Robotu fikrini geliştirir,
                FLUX profesyonel görseli üretir.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        # ----------------------------------------------------
        # PROMPT
        # ----------------------------------------------------

        st.subheader("Görsel Fikri")

        prompt = st.text_area(
            "Prompt",
            placeholder=(
                "Örnek: Futuristic city at night, "
                "a sports car driving through neon streets..."
            ),
            height=150,
            label_visibility="collapsed",
        )

        # ----------------------------------------------------
        # AI PROMPT ROBOT
        # ----------------------------------------------------

        col1, col2 = st.columns([1, 1])

        with col1:

            use_ai_boost = st.toggle(
                "🤖 AI Prompt Robotu",
                value=True,
            )

        with col2:

            if use_ai_boost:
                st.success("AI Prompt Robotu: ON")
            else:
                st.info("AI Prompt Robotu: OFF")


        # ----------------------------------------------------
        # ASPECT RATIO
        # ----------------------------------------------------

        st.subheader("Görsel Formatı")

        ratio = st.selectbox(
            "Format",
            [
                "1:1 Kare",
                "9:16 Dikey",
                "16:9 Yatay",
            ],
            label_visibility="collapsed",
        )

        if ratio == "1:1 Kare":

            width = 768
            height = 768

        elif ratio == "9:16 Dikey":

            width = 768
            height = 1344

        else:

            width = 1344
            height = 768


        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        st.write("")

        if st.button(
            "✦ Görsel Oluştur",
            use_container_width=True,
            type="primary",
        ):

            if not prompt.strip():

                st.warning(
                    "Önce bir görsel fikri yaz."
                )

            elif not HF_API_KEY:

                st.error(
                    "Hugging Face API anahtarı bulunamadı. "
                    "Streamlit Secrets bölümünde HF_API_KEY eklenmeli."
                )

            else:

                final_prompt = prompt.strip()

                # --------------------------------------------
                # AI PROMPT
                # --------------------------------------------

                if use_ai_boost:

                    with st.spinner(
                        "AI Prompt Robotu fikrini geliştiriyor..."
                    ):

                        enhanced, error = enhance_prompt(
                            prompt.strip()
                        )

                    if error:

                        st.error(
                            f"Prompt AI hatası: {error}"
                        )

                        st.stop()

                    final_prompt = enhanced

                    st.markdown(
                        "### AI tarafından geliştirilen prompt"
                    )

                    st.info(final_prompt)

                # --------------------------------------------
                # IMAGE
                # --------------------------------------------

                with st.spinner(
                    "FLUX görseli oluşturuyor..."
                ):

                    image, error = generate_image(
                        final_prompt,
                        width,
                        height,
                    )

                if error:

                    st.error(
                        f"Görsel üretim hatası: {error}"
                    )

                elif image is not None:

                    st.success(
                        "Görsel başarıyla oluşturuldu."
                    )

                    st.image(
                        image,
                        use_container_width=True,
                    )

                    # ----------------------------------------
                    # PNG
                    # ----------------------------------------

                    buffer = io.BytesIO()

                    image.save(
                        buffer,
                        format="PNG",
                    )

                    image_bytes = buffer.getvalue()

                    filename = (
                        "kogce_ai_"
                        + datetime.now().strftime(
                            "%Y%m%d_%H%M%S"
                        )
                        + ".png"
                    )

                    st.download_button(
                        "⬇️ PNG olarak indir",
                        data=image_bytes,
                        file_name=filename,
                        mime="image/png",
                        use_container_width=True,
                    )

                    # ----------------------------------------
                    # HISTORY
                    # ----------------------------------------

                    st.session_state.history.append(
                        {
                            "image": image,
                            "original_prompt": prompt.strip(),
                            "final_prompt": final_prompt,
                            "time": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "ratio": ratio,
                        }
                    )


    # --------------------------------------------------------
    # VIDEO STATUS
    # --------------------------------------------------------

    elif page == "🎥 Video Üretim Durumu":

        st.title("🎥 Video Üretim Durumu")

        st.caption(
            "Video pipeline için mevcut sistem durumu."
        )

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.success("Prompt AI\n\nAktif")

        with col2:
            st.success("FLUX\n\nAktif")

        with col3:
            st.warning("Video Engine\n\nBağlı değil")

        st.divider()

        st.subheader("Planlanan Pipeline")

        pipeline = [
            "1. IDEA",
            "2. AI PROMPT ROBOT",
            "3. SCENE PLANNER",
            "4. IMAGE GENERATION",
            "5. VIDEO GENERATION",
            "6. VOICE / AUDIO",
            "7. VIDEO ASSEMBLY",
            "8. MP4",
        ]

        for item in pipeline:

            st.write(
                f"**{item}**"
            )

        st.info(
            "Şu anda bu bölüm video üretim motorunun "
            "bağlantı durumunu gösterir. Video üretimi "
            "henüz aktif olarak bağlanmış değildir."
        )


    # --------------------------------------------------------
    # GALLERY
    # --------------------------------------------------------

    elif page == "🖼️ Galeri":

        st.title("🖼️ Galeri")

        if not st.session_state.history:

            st.info(
                "Bu oturumda henüz görsel oluşturulmadı."
            )

        else:

            st.caption(
                f"{len(st.session_state.history)} görsel oluşturuldu."
            )

            for index, item in enumerate(
                reversed(st.session_state.history),
                start=1,
            ):

                st.divider()

                col1, col2 = st.columns(
                    [1, 1]
                )

                with col1:

                    st.image(
                        item["image"],
                        use_container_width=True,
                    )

                with col2:

                    st.markdown(
                        f"### Görsel {index}"
                    )

                    st.caption(
                        item["time"]
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

                    st.caption(
                        f"Format: {item['ratio']}"
                    )


    # --------------------------------------------------------
    # SYSTEM
    # --------------------------------------------------------

    elif page == "⚙️ Sistem":

        st.title("⚙️ Sistem")

        st.subheader("API")

        if HF_API_KEY:

            st.success(
                "Hugging Face API Key: Bağlı"
            )

        else:

            st.error(
                "Hugging Face API Key: Bulunamadı"
            )

        st.divider()

        st.subheader("Modeller")

        st.write(
            f"**Image Model:** `{IMAGE_MODEL}`"
        )

        st.write(
            f"**Prompt Model:** `{PROMPT_MODEL}`"
        )

        st.write(
            f"**Video Engine:** `Bağlı değil`"
        )

        st.divider()

        st.subheader("Oturum")

        st.write(
            f"Bu oturumda oluşturulan görsel: "
            f"**{len(st.session_state.history)}**"
        )


# ============================================================
# RUN
# ============================================================

if not st.session_state.authenticated:

    login_screen()

else:

    main_app()

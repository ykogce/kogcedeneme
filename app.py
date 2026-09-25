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

APP_PASSWORD = "1234"

IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
PROMPT_MODEL = "openai/gpt-oss-120b"

CHAT_URL = "https://router.huggingface.co/v1/chat/completions"


# ============================================================
# SESSION STATE
# ============================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "history" not in st.session_state:
    st.session_state.history = []


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
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );


    /* ========================================================
       GENERAL
       ======================================================== */

    html,
    body,
    [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {

        background:
            radial-gradient(
                circle at 12% 8%,
                rgba(117, 58, 255, 0.15),
                transparent 31%
            ),
            radial-gradient(
                circle at 88% 90%,
                rgba(170, 72, 255, 0.10),
                transparent 30%
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
                #0d0b13 0%,
                #08080d 100%
            );

        border-right:
            1px solid rgba(255,255,255,0.07);
    }

    section[data-testid="stSidebar"] * {
        color: #eeeeF5;
    }


    /* ========================================================
       LOGIN TITLE
       ======================================================== */

    .login-title-marker {
        text-align: center;
        font-size: 44px;
        line-height: 1;
        margin-bottom: 10px;
        color: #b981ff;
        text-shadow:
            0 0 15px rgba(155,82,255,0.75),
            0 0 40px rgba(120,55,255,0.35);
    }


    /* st.markdown heading inside login */

    .login-title-container h1 {

        text-align: center !important;

        font-size: clamp(
            44px,
            7vw,
            78px
        ) !important;

        line-height: 1 !important;

        font-weight: 800 !important;

        letter-spacing: -4px !important;

        margin-top: 0 !important;

        margin-bottom: 12px !important;

        background:
            linear-gradient(
                110deg,
                #ffffff 0%,
                #e9dcff 18%,
                #c185ff 42%,
                #8d4cff 68%,
                #ffffff 100%
            ) !important;

        -webkit-background-clip: text !important;

        -webkit-text-fill-color: transparent !important;

        background-clip: text !important;

        filter:
            drop-shadow(
                0 0 18px rgba(145,75,255,0.28)
            );
    }


    .login-subtitle {

        text-align: center !important;

        color: #a8a4b3 !important;

        font-size: 14px !important;

        letter-spacing: 4px !important;

        font-weight: 600 !important;

        margin-top: 6px !important;
    }


    .login-description {

        text-align: center !important;

        color: #d6d2df !important;

        font-size: 15px !important;

        letter-spacing: 1px !important;

        margin-top: 20px !important;
    }


    /* ========================================================
       PROMPT TEXTAREA
       ======================================================== */

    div[data-testid="stTextArea"] textarea {

        background-color: #ffffff !important;

        color: #000000 !important;

        -webkit-text-fill-color: #000000 !important;

        caret-color: #000000 !important;

        border:
            1px solid #d5d5dd !important;

        border-radius: 14px !important;

        font-size: 16px !important;

        font-weight: 500 !important;

        padding: 16px !important;

        box-shadow:
            0 5px 20px rgba(0,0,0,0.13) !important;
    }

    div[data-testid="stTextArea"] textarea:focus {

        border-color:
            #8d4cff !important;

        box-shadow:
            0 0 0 1px #8d4cff,
            0 0 22px rgba(141,76,255,0.22) !important;
    }

    div[data-testid="stTextArea"] textarea::placeholder {

        color: #777780 !important;

        -webkit-text-fill-color: #777780 !important;

        opacity: 1 !important;
    }


    /* ========================================================
       TEXT INPUT
       ======================================================== */

    div[data-testid="stTextInput"] input {

        background: #ffffff !important;

        color: #000000 !important;

        -webkit-text-fill-color: #000000 !important;

        border:
            1px solid #d5d5dd !important;

        border-radius: 12px !important;

        font-size: 16px !important;
    }

    div[data-testid="stTextInput"] input::placeholder {

        color: #777780 !important;

        -webkit-text-fill-color: #777780 !important;

        opacity: 1 !important;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {

        min-height: 46px;

        border-radius: 12px !important;

        border:
            1px solid rgba(163,93,255,0.35) !important;

        background:
            linear-gradient(
                135deg,
                #7738f2,
                #a455ff
            ) !important;

        color: #ffffff !important;

        font-weight: 700 !important;

        box-shadow:
            0 8px 25px rgba(126,60,255,0.22);
    }

    .stButton > button:hover {

        border-color:
            rgba(201,166,255,0.85) !important;

        box-shadow:
            0 10px 30px rgba(126,60,255,0.38);
    }


    /* ========================================================
       SIDEBAR TITLE
       ======================================================== */

    section[data-testid="stSidebar"] h1 {

        font-size: 25px !important;

        letter-spacing: -1px !important;

        background:
            linear-gradient(
                110deg,
                #ffffff,
                #b77aff,
                #ffffff
            ) !important;

        -webkit-background-clip: text !important;

        -webkit-text-fill-color: transparent !important;

        background-clip: text !important;
    }


    /* ========================================================
       MAIN HEADINGS
       ======================================================== */

    .main h1 {

        font-weight: 800 !important;
    }


    /* ========================================================
       DIVIDER
       ======================================================== */

    hr {

        border-color:
            rgba(255,255,255,0.07) !important;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 700px) {

        .login-title-container h1 {

            font-size: 43px !important;

            letter-spacing: -2.5px !important;
        }

        .login-subtitle {

            font-size: 11px !important;

            letter-spacing: 2.5px !important;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOGIN SCREEN
# ============================================================

def login_screen():

    st.markdown(
        "",
        unsafe_allow_html=False,
    )

    st.markdown(
        '<div class="login-title-marker">✦</div>',
        unsafe_allow_html=True,
    )

    # Native markdown heading
    # CSS yukarıdaki .login-title-container ile hedefleniyor.
    st.markdown(
        '<div class="login-title-container">',
        unsafe_allow_html=True,
    )

    st.title(
        "KOGCE AI Studio"
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="login-subtitle">'
        'AI CREATIVE WORKSPACE'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="login-description">'
        'Create. Imagine. Generate.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write("")
    st.write("")

    left, center, right = st.columns(
        [1.25, 1.5, 1.25]
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


# ============================================================
# AI PROMPT ROBOT
# ============================================================

def enhance_prompt(user_prompt):

    if not HF_API_KEY:

        return (
            None,
            "Hugging Face API anahtarı bulunamadı."
        )

    system_prompt = """
You are a professional AI image prompt engineer.

Transform the user's short idea into a detailed,
professional English image-generation prompt.

Improve:

subject,
composition,
camera,
lighting,
environment,
materials,
colors,
atmosphere,
realism,
cinematic quality,
visual details.

Return ONLY the final image prompt.

Do not explain anything.
Do not write "Prompt:".
Do not add commentary.
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

        "Authorization":
            f"Bearer {HF_API_KEY}",

        "Content-Type":
            "application/json",
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

        result = (
            data["choices"][0]["message"]["content"]
            .strip()
        )

        return result, None

    except Exception as e:

        return None, str(e)


# ============================================================
# IMAGE GENERATION
# ============================================================

def generate_image(
    prompt,
    width,
    height,
):

    if not HF_API_KEY:

        return (
            None,
            "Hugging Face API anahtarı bulunamadı."
        )

    if InferenceClient is None:

        return (
            None,
            "huggingface_hub kütüphanesi bulunamadı."
        )

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

    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        st.title(
            "✦ KOGCE AI Studio"
        )

        st.caption(
            "AI Creative Workspace"
        )

        st.divider()

        st.markdown(
            "### Sistem"
        )

        if HF_API_KEY:

            st.success(
                "HF API bağlantısı aktif"
            )

        else:

            st.error(
                "HF API anahtarı yok"
            )

        st.caption(
            "● FLUX.1-schnell — Aktif"
        )

        st.divider()

        st.markdown(
            "### Araçlar"
        )

        page = st.radio(

            "Araçlar",

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


    # ========================================================
    # IMAGE GENERATOR
    # ========================================================

    if page == "✨ Görsel Oluştur":

        st.title(
            "KOGCE AI Studio"
        )

        st.caption(
            "Fikrini yaz. AI Prompt Robotu fikrini geliştirir, "
            "FLUX profesyonel görseli üretir."
        )

        st.divider()

        st.subheader(
            "Görsel Fikri"
        )

        prompt = st.text_area(

            "Prompt",

            placeholder=(
                "Örnek: Futuristic city at night, "
                "a sports car driving through neon streets..."
            ),

            height=150,

            label_visibility="collapsed",
        )


        # ====================================================
        # AI PROMPT
        # ====================================================

        col1, col2 = st.columns(
            [1, 1]
        )

        with col1:

            use_ai_boost = st.toggle(

                "🤖 AI Prompt Robotu",

                value=True,
            )

        with col2:

            if use_ai_boost:

                st.success(
                    "AI Prompt Robotu: ON"
                )

            else:

                st.info(
                    "AI Prompt Robotu: OFF"
                )


        # ====================================================
        # FORMAT
        # ====================================================

        st.subheader(
            "Görsel Formatı"
        )

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


        st.write("")


        # ====================================================
        # GENERATE
        # ====================================================

        if st.button(

            "✦ Görsel Oluştur",

            use_container_width=True,

            type="primary",
        ):

            if not prompt.strip():

                st.warning(
                    "Önce bir görsel fikri yaz."
                )

                st.stop()


            if not HF_API_KEY:

                st.error(
                    "Hugging Face API anahtarı bulunamadı."
                )

                st.stop()


            final_prompt = prompt.strip()


            # ------------------------------------------------
            # PROMPT AI
            # ------------------------------------------------

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


                st.subheader(
                    "Geliştirilmiş Prompt"
                )

                st.info(
                    final_prompt
                )


            # ------------------------------------------------
            # IMAGE
            # ------------------------------------------------

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


                # --------------------------------------------
                # DOWNLOAD
                # --------------------------------------------

                buffer = io.BytesIO()

                image.save(
                    buffer,

                    format="PNG",
                )

                image_bytes = (
                    buffer.getvalue()
                )


                filename = (

                    "kogce_ai_"

                    +

                    datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )

                    +

                    ".png"
                )


                st.download_button(

                    "⬇️ PNG olarak indir",

                    data=image_bytes,

                    file_name=filename,

                    mime="image/png",

                    use_container_width=True,
                )


                # --------------------------------------------
                # HISTORY
                # --------------------------------------------

                st.session_state.history.append(
                    {

                        "image":
                            image,

                        "original_prompt":
                            prompt.strip(),

                        "final_prompt":
                            final_prompt,

                        "time":
                            datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),

                        "ratio":
                            ratio,
                    }
                )


    # ========================================================
    # VIDEO
    # ========================================================

    elif page == "🎥 Video Üretim Durumu":

        st.title(
            "🎥 Video Üretim Durumu"
        )

        st.caption(
            "Video üretim pipeline durumu."
        )

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:

            st.success(
                "Prompt AI\n\nAktif"
            )

        with col2:

            st.success(
                "FLUX\n\nAktif"
            )

        with col3:

            st.warning(
                "Video Engine\n\nBağlı değil"
            )


        st.divider()

        st.subheader(
            "Video Pipeline"
        )


        pipeline = [

            "IDEA",

            "AI PROMPT ROBOT",

            "SCENE PLANNER",

            "IMAGE GENERATION",

            "VIDEO GENERATION",

            "VOICE / AUDIO",

            "VIDEO ASSEMBLY",

            "MP4",
        ]


        for number, item in enumerate(

            pipeline,

            start=1,
        ):

            st.write(
                f"**{number}. {item}**"
            )


        st.info(
            "Video üretim motoru henüz uygulamaya "
            "bağlanmış değil. Bu ekran pipeline "
            "durumunu gösterir."
        )


    # ========================================================
    # GALLERY
    # ========================================================

    elif page == "🖼️ Galeri":

        st.title(
            "🖼️ Galeri"
        )


        if not st.session_state.history:

            st.info(
                "Bu oturumda henüz görsel oluşturulmadı."
            )


        else:

            st.caption(

                f"{len(st.session_state.history)} "
                "görsel oluşturuldu."
            )


            for index, item in enumerate(

                reversed(
                    st.session_state.history
                ),

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

                    st.subheader(
                        f"Görsel {index}"
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
                        "**Geliştirilmiş prompt**"
                    )

                    st.write(
                        item["final_prompt"]
                    )

                    st.caption(
                        f"Format: {item['ratio']}"
                    )


    # ========================================================
    # SYSTEM
    # ========================================================

    elif page == "⚙️ Sistem":

        st.title(
            "⚙️ Sistem"
        )

        st.subheader(
            "API Durumu"
        )


        if HF_API_KEY:

            st.success(
                "Hugging Face API Key: Bağlı"
            )

        else:

            st.error(
                "Hugging Face API Key: Bulunamadı"
            )


        st.divider()


        st.subheader(
            "Modeller"
        )


        st.write(
            f"**Görsel Modeli:** `{IMAGE_MODEL}`"
        )

        st.write(
            f"**Prompt Modeli:** `{PROMPT_MODEL}`"
        )

        st.write(
            "**Video Engine:** `Bağlı değil`"
        )


        st.divider()


        st.subheader(
            "Oturum"
        )


        st.write(

            "Bu oturumda oluşturulan görsel: "

            f"**{len(st.session_state.history)}**"
        )


# ============================================================
# START
# ============================================================

if not st.session_state.authenticated:

    login_screen()

else:

    main_app()

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
# SESSION STATE
# ============================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(80, 70, 180, 0.16),
                transparent 28%
            ),
            radial-gradient(
                circle at 85% 15%,
                rgba(0, 180, 255, 0.10),
                transparent 28%
            ),
            #080a10;
        color: #f5f7fb;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    section[data-testid="stSidebar"] {
        background: #0b0e15;
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    .hero-wrapper {
        padding: 35px 0 30px 0;
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        background: rgba(120, 100, 255, 0.12);
        border: 1px solid rgba(130, 110, 255, 0.28);
        color: #b9b0ff;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.8px;
        margin-bottom: 18px;
    }

    .hero-title {
        font-size: clamp(42px, 6vw, 76px);
        line-height: 0.98;
        font-weight: 800;
        letter-spacing: -3px;
        margin-bottom: 20px;
        background: linear-gradient(
            90deg,
            #ffffff 0%,
            #d8d5ff 45%,
            #8ccfff 100%
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-subtitle {
        max-width: 780px;
        color: #9da5b5;
        font-size: 18px;
        line-height: 1.65;
    }

    .section-title {
        font-size: 25px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .section-description {
        color: #8f98aa;
        font-size: 14px;
        margin-bottom: 20px;
    }

    .status-card {
        background: rgba(17, 20, 29, 0.75);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .status-title {
        font-size: 14px;
        font-weight: 700;
        color: #f1f3f8;
        margin-bottom: 6px;
    }

    .status-value {
        font-size: 13px;
        color: #8f98aa;
    }

    .online {
        color: #72e6a4;
    }

    .offline {
        color: #ff8585;
    }

    .warning {
        color: #ffc86b;
    }

    .login-box {
        max-width: 480px;
        margin: 100px auto 25px auto;
        padding: 40px;
        background: rgba(18, 21, 31, 0.82);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        box-shadow: 0 30px 90px rgba(0,0,0,0.40);
        text-align: center;
    }

    .login-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .login-subtitle {
        color: #929aaa;
        margin-bottom: 28px;
    }

    .pipeline {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 20px;
    }

    .pipeline-item {
        padding: 10px 14px;
        border-radius: 10px;
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.07);
        color: #c9cfda;
        font-size: 12px;
        font-weight: 600;
    }

    .pipeline-arrow {
        color: #626b7c;
        font-size: 14px;
    }

    .footer {
        text-align: center;
        color: #555d6c;
        font-size: 12px;
        padding-top: 30px;
    }

    .generated-image {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .engine-box {
        background: rgba(17, 20, 29, 0.75);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 18px;
        padding: 20px;
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

Do not add unrelated characters or concepts.

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
            return None, (
                f"Prompt AI HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        data = response.json()

        choices = data.get("choices", [])

        if not choices:
            return None, "Prompt AI boş cevap döndürdü."

        content = choices[0].get("message", {}).get("content", "")

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
# LOGIN
# ============================================================

def login_screen():

    st.markdown(
        """
        <div class="login-box">
            <div class="login-title">
                KOGCE AI Studio
            </div>

            <div class="login-subtitle">
                AI Creative Workspace
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1, 2, 1])

    with center:

        password = st.text_input(
            "Uygulama şifresi",
            type="password",
            placeholder="Şifrenizi girin",
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

        st.caption("KOGCE AI Studio")


if not st.session_state.authenticated:
    login_screen()
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:24px;
            font-weight:800;
            margin-bottom:5px;
        ">
            KOGCE AI
        </div>

        <div style="
            color:#7f8899;
            font-size:12px;
            margin-bottom:25px;
        ">
            Creative Workspace
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Sistem")

    if HF_API_KEY:
        st.success("HF API: Aktif")
    else:
        st.error("HF API: Eksik")

    st.markdown("### Motorlar")

    st.caption("Prompt AI")
    st.write("GPT OSS")

    st.caption("Görsel Motoru")
    st.write("FLUX.1-schnell")

    st.caption("Video Motoru")
    st.write("Bağlanacak")

    st.markdown("---")

    st.caption(
        f"Bu oturumda: {len(st.session_state.history)} görsel"
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
    """
    <div class="hero-wrapper">

        <div class="hero-badge">
            AI CREATIVE WORKSPACE
        </div>

        <div class="hero-title">
            KOGCE AI Studio
        </div>

        <div class="hero-subtitle">
            Fikirden profesyonel görsele.<br>
            AI Prompt Robotu kısa fikrini analiz eder,
            geliştirir ve FLUX görüntü motoruna gönderir.
        </div>

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
# CREATE IMAGE
# ============================================================

with tab_create:

    st.markdown(
        '<div class="section-title">Görsel Oluştur</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Kısa fikrini gir. İstersen AI Prompt Robotu fikrini
            profesyonel FLUX promptuna dönüştürsün.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns(
        [1.15, 0.85],
        gap="large",
    )

    # --------------------------------------------------------
    # SOL TARAF
    # --------------------------------------------------------

    with col_left:

        user_prompt = st.text_area(
            "Fikir / Prompt",
            height=180,
            placeholder=(
                "Örnek: "
                "A futuristic sports car driving through a neon city at night"
            ),
        )

        use_ai_boost = st.toggle(
            "🤖 AI Prompt Robotu",
            value=True,
        )

        if use_ai_boost:
            st.success("AI Prompt Robotu: ON")
        else:
            st.info("AI Prompt Robotu: OFF")

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
    # SAĞ TARAF
    # --------------------------------------------------------

    with col_right:

        st.markdown(
            '<div class="engine-box">',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="
                font-size:17px;
                font-weight:750;
                margin-bottom:15px;
            ">
                Üretim Sistemi
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="status-card">
                <div class="status-title">
                    Prompt AI
                </div>

                <div class="status-value online">
                    Hazır
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="status-card">
                <div class="status-title">
                    FLUX.1-schnell
                </div>

                <div class="status-value online">
                    Hazır
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="status-card">
                <div class="status-title">
                    Video Engine
                </div>

                <div class="status-value warning">
                    Henüz bağlı değil
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


    # ========================================================
    # GENERATION
    # ========================================================

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
                    "### AI tarafından geliştirilen prompt"
                )

                st.info(final_prompt)

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

                st.markdown(
                    '<div class="generated-image">',
                    unsafe_allow_html=True,
                )

                st.image(
                    image,
                    use_container_width=True,
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
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
        '<div class="section-title">Video Üretim Durumu</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Video otomasyonunun mevcut bağlantı durumunu gösterir.
            Bu bölüm henüz video üretimini çalıştırmaz.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="status-card">

                <div class="status-title">
                    AI Prompt Robotu
                </div>

                <div class="status-value online">
                    AKTİF
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            """
            <div class="status-card">

                <div class="status-title">
                    FLUX Görsel Motoru
                </div>

                <div class="status-value online">
                    AKTİF
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:

        st.markdown(
            """
            <div class="status-card">

                <div class="status-title">
                    Video Motoru
                </div>

                <div class="status-value warning">
                    BEKLEMEDE
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "### Planlanan Video Pipeline"
    )

    st.markdown(
        """
        <div class="pipeline">

            <div class="pipeline-item">
                IDEA
            </div>

            <div class="pipeline-arrow">
                →
            </div>

            <div class="pipeline-item">
                AI PROMPT ROBOT
            </div>

            <div class="pipeline-arrow">
                →
            </div>

            <div class="pipeline-item">
                SCENE PLANNER
            </div>

            <div class="pipeline-arrow">
                →
            </div>

            <div class="pipeline-item">
                IMAGE GENERATION
            </div>

            <div class="pipeline-arrow">
                →
            </div>

            <div class="pipeline-item">
                VIDEO GENERATION
            </div>

            <div class="pipeline-arrow">
                →
            </div>

            <div class="pipeline-item">
                VOICE / AUDIO
            </div>

            <div class="pipeline-arrow">
                →
            </div>

            <div class="pipeline-item">
                VIDEO ASSEMBLY
            </div>

            <div class="pipeline-arrow">
                →
            </div>

            <div class="pipeline-item">
                MP4
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.info(
        "Şu anda görsel üretim sistemi aktif. "
        "Video motoru henüz bu uygulamaya bağlanmadı."
    )


# ============================================================
# GALLERY
# ============================================================

with tab_gallery:

    st.markdown(
        '<div class="section-title">Galeri</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Bu oturum sırasında oluşturduğun görseller.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.history:

        st.info(
            "Henüz bu oturumda oluşturulmuş bir görsel yok."
        )

    else:

        for index, item in enumerate(
            st.session_state.history
        ):

            st.markdown("---")

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
                    f"### Görsel #{index + 1}"
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

                gallery_filename = (
                    "kogce_gallery_"
                    + str(index + 1)
                    + ".png"
                )

                st.download_button(
                    "⬇️ Görseli indir",
                    data=image_bytes.getvalue(),
                    file_name=gallery_filename,
                    mime="image/png",
                    key=f"gallery_download_{index}",
                    use_container_width=True,
                )


# ============================================================
# SYSTEM
# ============================================================

with tab_system:

    st.markdown(
        '<div class="section-title">Sistem</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            KOGCE AI Studio bağlantı ve motor bilgileri.
        </div>
        """,
        unsafe_allow_html=True,
    )

    system_col1, system_col2 = st.columns(2)

    with system_col1:

        st.markdown(
            "### API Durumu"
        )

        if HF_API_KEY:
            st.success(
                "Hugging Face API: AKTİF"
            )
        else:
            st.error(
                "Hugging Face API: BULUNAMADI"
            )

    with system_col2:

        st.markdown(
            "### Modeller"
        )

        st.write(
            "Görsel Modeli: FLUX.1-schnell"
        )

        st.write(
            "Prompt Modeli: GPT OSS 120B"
        )

    st.markdown("---")

    st.metric(
        "Bu oturumda üretilen görsel",
        len(st.session_state.history),
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        KOGCE AI Studio • AI Creative Workspace
    </div>
    """,
    unsafe_allow_html=True,
)

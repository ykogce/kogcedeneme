import streamlit as st
import requests
import io
from PIL import Image
from datetime import datetime

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="KOGCE AI Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MODELS / API
# ============================================================

IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
PROMPT_MODEL = "openai/gpt-oss-120b"

CHAT_URL = "https://router.huggingface.co/v1/chat/completions"

# ŞİFRE BURADA
APP_PASSWORD = "1234"


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(99, 102, 241, 0.12),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(168, 85, 247, 0.10),
                transparent 30%
            ),
            #07080d;
    }

    .main {
        background: transparent;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.03em;
    }

    p, span, label {
        color: #d7d9e0;
    }

    .hero {
        padding: 35px;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        background:
            linear-gradient(
                135deg,
                rgba(20, 22, 34, 0.96),
                rgba(10, 11, 18, 0.96)
            );
        box-shadow: 0 20px 70px rgba(0,0,0,0.30);
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 12px;
        color: #ffffff;
    }

    .hero-subtitle {
        color: #a8adba;
        font-size: 16px;
        line-height: 1.6;
        max-width: 850px;
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(139, 92, 246, 0.14);
        border: 1px solid rgba(139, 92, 246, 0.28);
        color: #c4b5fd;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 18px;
    }

    .glass-card {
        border: 1px solid rgba(255,255,255,0.07);
        background: rgba(15, 17, 26, 0.76);
        border-radius: 20px;
        padding: 22px;
        margin-bottom: 18px;
    }

    .section-title {
        font-size: 20px;
        font-weight: 750;
        color: #ffffff;
        margin-bottom: 4px;
    }

    .section-description {
        font-size: 13px;
        color: #8f95a4;
        margin-bottom: 18px;
    }

    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.09);
        min-height: 45px;
        font-weight: 700;
    }

    textarea,
    input {
        border-radius: 12px !important;
    }

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0b0d13 0%,
                #08090e 100%
            );
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    .status-box {
        border: 1px solid rgba(255,255,255,0.07);
        background: rgba(255,255,255,0.025);
        border-radius: 15px;
        padding: 16px;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .status-green {
        color: #86efac;
        font-weight: 700;
    }

    .status-yellow {
        color: #fde68a;
        font-weight: 700;
    }

    .status-gray {
        color: #9ca3af;
        font-weight: 700;
    }

    img {
        border-radius: 16px;
    }

    hr {
        border-color: rgba(255,255,255,0.06);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


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
# LOGIN
# ============================================================

def login_screen():

    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">KOGCE AI STUDIO</div>
            <div class="hero-title">Private AI Workspace</div>
            <div class="hero-subtitle">
                Görsel üretim, AI prompt geliştirme ve üretim durumunu
                tek panel üzerinden yönet.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1.2, 1])

    with col2:

        st.markdown("### Giriş")

        password = st.text_input(
            "Şifre",
            type="password",
            placeholder="Şifreni gir...",
        )

        if st.button(
            "Studio'ya Gir",
            use_container_width=True,
        ):

            if password == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Şifre hatalı.")


# ============================================================
# AI PROMPT ROBOT
# ============================================================

def improve_prompt_with_ai(original_prompt):

    if not HF_API_KEY:
        return None, "HF_API_KEY bulunamadı."

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }

    system_prompt = """
You are a professional image-generation prompt engineer.

Transform the user's short Turkish or English idea into ONE
detailed, production-ready English image-generation prompt.

Rules:

1. Preserve the main subject.
2. Preserve the main action.
3. Preserve the intended meaning.
4. Do not invent major story elements.
5. Translate Turkish naturally into English when needed.
6. Improve visual specificity.
7. Describe subject appearance, environment, composition,
   camera perspective, lighting, atmosphere, materials,
   textures, depth and visual quality.
8. Make the scene visually coherent.
9. Avoid unnecessary repetition.
10. Do not explain your work.
11. Do not use bullet points.
12. Return ONLY the final English image prompt.

The result must be directly usable by an image generation model.
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
                "content": original_prompt,
            },
        ],
        "temperature": 0.7,
        "max_tokens": 700,
    }

    try:

        response = requests.post(
            CHAT_URL,
            headers=headers,
            json=payload,
            timeout=90,
        )

        if response.status_code != 200:
            return (
                None,
                f"Prompt AI hatası ({response.status_code}): "
                f"{response.text[:700]}",
            )

        data = response.json()

        choices = data.get("choices", [])

        if not choices:
            return None, "Prompt AI boş cevap döndürdü."

        message = choices[0].get("message", {})

        content = message.get("content", "")

        if isinstance(content, list):
            content = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
            )

        content = str(content).strip()

        if not content:
            return None, "Prompt AI metin döndürmedi."

        return content, None

    except requests.exceptions.Timeout:
        return None, "Prompt AI zaman aşımına uğradı."

    except Exception as e:
        return None, f"Prompt AI bağlantı hatası: {str(e)}"


# ============================================================
# IMAGE GENERATION
# ============================================================

def generate_image(prompt, width, height):

    if not HF_API_KEY:
        return None, "HF_API_KEY bulunamadı."

    if InferenceClient is None:
        return (
            None,
            "huggingface_hub kurulu değil. "
            "requirements.txt dosyasını kontrol et.",
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

        if image is None:
            return None, "Model boş sonuç döndürdü."

        return image, None

    except Exception as e:
        return None, f"Görsel üretim hatası: {str(e)}"


# ============================================================
# IMAGE DIMENSIONS
# ============================================================

def get_dimensions(aspect):

    if aspect == "1:1 Kare":
        return 768, 768

    if aspect == "9:16 Dikey":
        return 768, 1344

    if aspect == "16:9 Yatay":
        return 1344, 768

    return 768, 768


# ============================================================
# LOGIN CHECK
# ============================================================

if not st.session_state.authenticated:

    login_screen()

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## KOGCE AI Studio")

    st.caption("AI Creative Workspace")

    st.divider()

    st.markdown("### Sistem")

    if HF_API_KEY:

        st.markdown(
            '<div class="status-box">'
            '<span class="status-green">'
            '● Hugging Face bağlantısı hazır'
            '</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            '<div class="status-box">'
            '<span class="status-yellow">'
            '● HF_API_KEY bulunamadı'
            '</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Motorlar")

    st.markdown(
        """
        <div class="status-box">
            <b>Görsel</b><br>
            <span class="status-green">
                FLUX.1-schnell
            </span>
        </div>

        <div class="status-box">
            <b>Prompt AI</b><br>
            <span class="status-green">
                Hugging Face Router
            </span>
        </div>

        <div class="status-box">
            <b>Video</b><br>
            <span class="status-gray">
                Henüz bağlı değil
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Oturum")

    if st.button(
        "Çıkış Yap",
        use_container_width=True,
    ):

        st.session_state.authenticated = False

        st.rerun()

    st.caption(
        f"Üretim sayısı: {len(st.session_state.history)}"
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-badge">
            AI CREATIVE WORKSPACE
        </div>

        <div class="hero-title">
            KOGCE AI Studio
        </div>

        <div class="hero-subtitle">
            Fikirden profesyonel görsele.
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
# TAB 1 — IMAGE CREATION
# ============================================================

with tab_create:

    st.markdown(
        """
        <div class="glass-card">

            <div class="section-title">
                Görsel Oluştur
            </div>

            <div class="section-description">
                Fikrini yaz. AI Prompt Robotu kısa fikrini
                profesyonel görsel üretim promptuna dönüştürür.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    prompt = st.text_area(
        "Prompt",
        placeholder=(
            "Örnek: Yağmurlu gecede neon ışıkların altında "
            "kırmızı spor araba..."
        ),
        height=150,
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        aspect = st.selectbox(
            "Görüntü oranı",
            [
                "1:1 Kare",
                "9:16 Dikey",
                "16:9 Yatay",
            ],
        )

    with col2:

        use_ai_boost = st.toggle(
            "🤖 AI Prompt Robotu",
            value=True,
        )

    with col3:

        st.markdown("###")

        if use_ai_boost:
            st.success("AI Prompt Robotu: ON")
        else:
            st.info("AI Prompt Robotu: OFF")

    st.divider()

    generate_button = st.button(
        "✦ Görsel Oluştur",
        type="primary",
        use_container_width=True,
    )

    if generate_button:

        if not prompt.strip():

            st.warning(
                "Önce bir prompt yaz."
            )

            st.stop()

        if not HF_API_KEY:

            st.error(
                "HF_API_KEY bulunamadı. "
                "Streamlit Secrets bölümünü kontrol et."
            )

            st.stop()

        final_prompt = prompt.strip()

        # ----------------------------------------------------
        # AI PROMPT ENHANCEMENT
        # ----------------------------------------------------

        if use_ai_boost:

            with st.spinner(
                "AI Prompt Robotu fikrini geliştiriyor..."
            ):

                enhanced_prompt, prompt_error = (
                    improve_prompt_with_ai(
                        prompt.strip()
                    )
                )

            if prompt_error:

                st.error(prompt_error)

                st.stop()

            final_prompt = enhanced_prompt

            with st.expander(
                "🤖 AI'nın geliştirdiği promptu göster",
                expanded=True,
            ):

                st.markdown(
                    "**Orijinal fikir**"
                )

                st.code(
                    prompt.strip(),
                    language="text",
                )

                st.markdown(
                    "**Geliştirilmiş İngilizce prompt**"
                )

                st.code(
                    final_prompt,
                    language="text",
                )

        # ----------------------------------------------------
        # IMAGE GENERATION
        # ----------------------------------------------------

        width, height = get_dimensions(
            aspect
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

            st.stop()

        # ----------------------------------------------------
        # DISPLAY IMAGE
        # ----------------------------------------------------

        st.success(
            "Görsel oluşturuldu."
        )

        st.image(
            image,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        st.session_state.history.append(
            {
                "image": image,
                "prompt": prompt.strip(),
                "final_prompt": final_prompt,
                "aspect": aspect,
                "time": timestamp,
            }
        )

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        image_buffer = io.BytesIO()

        image.save(
            image_buffer,
            format="PNG",
        )

        image_buffer.seek(0)

        st.download_button(
            label="⬇️ PNG olarak indir",
            data=image_buffer,
            file_name=(
                "kogce_ai_"
                + datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )
                + ".png"
            ),
            mime="image/png",
            use_container_width=True,
        )


# ============================================================
# TAB 2 — VIDEO STATUS
# ============================================================

with tab_video:

    st.markdown(
        """
        <div class="glass-card">

            <div class="section-title">
                Video Üretim Durumu
            </div>

            <div class="section-description">
                KOGCE AI Studio video üretim pipeline'ının
                mevcut durumunu gösterir.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="glass-card">

                <div class="section-title">
                    Mevcut Durum
                </div>

                <br>

                <b>Prompt AI</b><br>

                <span class="status-green">
                    ● Aktif
                </span>

                <br><br>

                <b>Görsel Motoru</b><br>

                <span class="status-green">
                    ● Aktif — FLUX.1-schnell
                </span>

                <br><br>

                <b>Video Motoru</b><br>

                <span class="status-gray">
                    ● Bağlı değil
                </span>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="glass-card">

                <div class="section-title">
                    Planlanan Video Pipeline
                </div>

                <br>

                <b>1.</b> Fikir / Prompt

                <br><br>

                <b>2.</b> AI Prompt Robotu

                <br><br>

                <b>3.</b> Sahne tasarımı

                <br><br>

                <b>4.</b> Görsel üretimi

                <br><br>

                <b>5.</b> Video üretim motoru

                <br><br>

                <b>6.</b> Seslendirme / Ses

                <br><br>

                <b>7.</b> MP4 oluşturma

                <br><br>

                <b>8.</b> Galeri / çıktı

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.warning(
        "Video motoru bu sürümde henüz bağlı değil. "
        "Bu nedenle sahte video üretim sonucu gösterilmiyor."
    )

    st.markdown(
        "### Video sistemi için hazır altyapı"
    )

    st.code(
        """
IDEA
  ↓
AI PROMPT ROBOT
  ↓
SCENE PLANNER
  ↓
IMAGE GENERATION
  ↓
VIDEO GENERATION
  ↓
VOICE / AUDIO
  ↓
VIDEO ASSEMBLY
  ↓
MP4
        """,
        language="text",
    )

    st.info(
        "Görsel üretim motoru aktif. "
        "Video tarafında ayrıca bir video inference "
        "sağlayıcısı veya API bağlanması gerekir."
    )


# ============================================================
# TAB 3 — GALLERY
# ============================================================

with tab_gallery:

    st.markdown(
        """
        <div class="glass-card">

            <div class="section-title">
                Galeri
            </div>

            <div class="section-description">
                Bu oturum sırasında oluşturulan görseller.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.history:

        st.info(
            "Henüz bu oturumda oluşturulmuş "
            "bir görsel yok."
        )

    else:

        for index, item in enumerate(
            reversed(
                st.session_state.history
            )
        ):

            st.markdown(
                "### Üretim #"
                + str(
                    len(
                        st.session_state.history
                    )
                    - index
                )
            )

            col_image, col_info = st.columns(
                [1.3, 1]
            )

            with col_image:

                st.image(
                    item["image"],
                    use_container_width=True,
                )

            with col_info:

                st.markdown(
                    f"""
                    **Tarih:**  
                    {item["time"]}

                    **Oran:**  
                    {item["aspect"]}
                    """
                )

                st.markdown(
                    "**Orijinal prompt:**"
                )

                st.code(
                    item["prompt"],
                    language="text",
                )

                st.markdown(
                    "**Model'e gönderilen prompt:**"
                )

                st.code(
                    item["final_prompt"],
                    language="text",
                )

            st.divider()


# ============================================================
# TAB 4 — SYSTEM
# ============================================================

with tab_system:

    st.markdown(
        """
        <div class="glass-card">

            <div class="section-title">
                Sistem
            </div>

            <div class="section-description">
                KOGCE AI Studio bağlantı ve motor durumları.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if HF_API_KEY:

        st.success(
            "HF_API_KEY bulundu."
        )

        st.code(
            "HF_API_KEY = ********",
            language="text",
        )

    else:

        st.error(
            "HF_API_KEY bulunamadı."
        )

        st.markdown(
            """
            Streamlit Secrets bölümünde şu anahtar bulunmalı:

            `HF_API_KEY`

            Değer kısmına Hugging Face tokenını eklemelisin.
            """
        )

    st.markdown(
        "### Kullanılan modeller"
    )

    st.code(
        f"""
Image Model:
{IMAGE_MODEL}

Prompt Model:
{PROMPT_MODEL}

Image Provider:
Hugging Face Inference Providers

Prompt API:
Hugging Face Router
        """,
        language="text",
    )

    st.markdown(
        "### Oturum"
    )

    st.write(
        "Oluşturulan görsel: "
        + str(
            len(
                st.session_state.history
            )
        )
    )

    st.caption(
        "Galeri şu anda Streamlit oturumu içinde tutuluyor. "
        "Uygulama yeniden başlatıldığında kalıcı galeri "
        "depolaması kullanılmıyor."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "KOGCE AI Studio • AI Creative Workspace"
)

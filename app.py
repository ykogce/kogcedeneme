import streamlit as st
import io
from PIL import Image
import datetime

# Hugging Face Inference Providers
from huggingface_hub import InferenceClient


# ==============================================================================
# HUGGING FACE AYARLARI
# ==============================================================================

HF_API_KEY = st.secrets.get("HF_API_KEY", "").strip()

IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"

# "auto" kullanıyoruz.
# Hugging Face uygun desteklenen provider'ı otomatik seçer.
HF_PROVIDER = "auto"


# ==============================================================================
# SAYFA AYARLARI
# ==============================================================================

st.set_page_config(
    page_title="KOGCE AI Studio Pro",
    page_icon="✨",
    layout="wide"
)


# ==============================================================================
# DARK THEME
# ==============================================================================

custom_css = """
<style>

    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
    }

    p, span, label, div, .stMarkdown, h1, h2, h3 {
        color: #f1f5f9 !important;
    }

    .main-title {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        background: linear-gradient(
            135deg,
            #c084fc 0%,
            #38bdf8 50%,
            #818cf8 100%
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        text-align: center;
        color: #94a3b8 !important;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        justify-content: center;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: rgba(30, 41, 59, 0.8);
        border-radius: 10px;
        color: #cbd5e1 !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(
            135deg,
            #a855f7 0%,
            #2563eb 100%
        ) !important;

        color: #ffffff !important;
        border: none !important;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(
            135deg,
            #9333ea 0%,
            #2563eb 100%
        );

        color: white !important;
        font-weight: 700;
        border-radius: 10px;
        border: none;
        padding: 0.75rem 1.5rem;
    }

    .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] {
        background-color: #111827 !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #0f172a;
    }

</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)


# ==============================================================================
# SESSION HISTORY
# ==============================================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ==============================================================================
# PROMPT BOOSTER
# ==============================================================================

def improve_prompt_with_ai(user_input, style_preset, aspect_ratio):

    style_modifiers = {

        "Doğal / Yok":
            "high quality, detailed, natural appearance",

        "Fotogerçekçi (Photorealistic)":
            "photorealistic, highly detailed, professional photography, cinematic lighting, realistic textures",

        "Anime / Manga":
            "anime style, highly detailed illustration, vibrant colors, clean linework",

        "3D Render (Pixar)":
            "high quality 3D render, stylized 3D character, smooth textures, studio lighting",

        "Yağlı Boya":
            "oil painting style, rich textures, expressive brush strokes, classical artwork",

        "Cyberpunk":
            "cyberpunk aesthetic, glowing neon lights, futuristic environment, cinematic atmosphere",

        "Cinematic":
            "cinematic movie shot, dramatic lighting, photorealistic, depth of field, realistic textures"
    }

    aspect_text = ""

    if "9:16" in aspect_ratio:
        aspect_text = (
            ", vertical composition, portrait orientation, "
            "9:16 composition"
        )

    elif "16:9" in aspect_ratio:
        aspect_text = (
            ", horizontal composition, landscape orientation, "
            "16:9 widescreen composition"
        )

    elif "1:1" in aspect_ratio:
        aspect_text = (
            ", square composition, 1:1 aspect ratio"
        )

    modifier = style_modifiers.get(
        style_preset,
        "high quality, highly detailed"
    )

    return f"{user_input}, {modifier}{aspect_text}"


# ==============================================================================
# HUGGING FACE CLIENT
# ==============================================================================

def create_hf_client():

    if not HF_API_KEY:

        raise RuntimeError(
            "HF_API_KEY bulunamadı. "
            "Hugging Face Space > Settings > Secrets bölümüne "
            "HF_API_KEY eklediğinden emin ol."
        )

    return InferenceClient(
        provider=HF_PROVIDER,
        api_key=HF_API_KEY
    )


# ==============================================================================
# IMAGE GENERATION
# ==============================================================================

def generate_image(prompt):

    client = create_hf_client()

    image = client.text_to_image(
        prompt=prompt,
        model=IMAGE_MODEL
    )

    return image


# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:

    st.title("⚙️ Ayarlar")

    st.markdown("---")

    use_password = st.checkbox(
        "🔒 Şifre Koruması",
        value=False
    )

    user_pass = ""

    if use_password:

        user_pass = st.text_input(
            "Şifre:",
            type="password"
        )


if use_password and user_pass != "1234":

    st.warning(
        "🔑 Lütfen geçerli şifreyi girin."
    )

    st.stop()


# ==============================================================================
# BAŞLIK
# ==============================================================================

st.markdown(
    '<h1 class="main-title">✨ KOGCE AI CREATIVE STUDIO PRO</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">FLUX.1-schnell • Hugging Face Inference Providers</p>',
    unsafe_allow_html=True
)


# ==============================================================================
# TABS
# ==============================================================================

tab1, tab2, tab3 = st.tabs(
    [
        "📸 Görsel Oluştur",
        "🎥 Video Üretim Durumu",
        "🖼️ Galeri"
    ]
)


# ==============================================================================
# TAB 1 — GÖRSEL
# ==============================================================================

with tab1:

    col_input, col_output = st.columns(
        [1, 1],
        gap="large"
    )


    # --------------------------------------------------------------------------
    # SOL TARAF
    # --------------------------------------------------------------------------

    with col_input:

        st.markdown("### 🎨 İsteyin")

        user_prompt = st.text_area(
            "Tarifiniz:",
            placeholder="Örn: Karlı dağlarda koşan kurt...",
            value="",
            height=120
        )

        col_opt1, col_opt2 = st.columns(2)


        with col_opt1:

            aspect_ratio = st.selectbox(
                "📐 Boyut:",
                [
                    "Kare (1:1)",
                    "Dikey / Story (9:16)",
                    "Yatay / YouTube (16:9)"
                ]
            )


        with col_opt2:

            style_preset = st.selectbox(
                "🎨 Stil:",
                [
                    "Doğal / Yok",
                    "Fotogerçekçi (Photorealistic)",
                    "Anime / Manga",
                    "3D Render (Pixar)",
                    "Yağlı Boya",
                    "Cyberpunk",
                    "Cinematic"
                ]
            )


        use_ai_boost = st.checkbox(
            "⚡ Prompt Booster",
            value=True
        )


        btn_img = st.button(
            "🚀 GÖRSELİ ÜRET",
            type="primary",
            use_container_width=True
        )


    # --------------------------------------------------------------------------
    # SAĞ TARAF
    # --------------------------------------------------------------------------

    with col_output:

        st.markdown("### 🖼️ Çıktı")

        if btn_img:

            if not user_prompt.strip():

                st.warning(
                    "Lütfen bir metin girin!"
                )

            else:

                # Prompt hazırlanıyor
                if use_ai_boost:

                    final_prompt = improve_prompt_with_ai(
                        user_prompt,
                        style_preset,
                        aspect_ratio
                    )

                else:

                    final_prompt = user_prompt


                st.info(
                    f"✨ **Prompt:** {final_prompt}"
                )


                # --------------------------------------------------------------
                # GENERATION
                # --------------------------------------------------------------

                with st.spinner(
                    "🎨 FLUX.1-schnell görsel oluşturuyor..."
                ):

                    try:

                        image = generate_image(
                            final_prompt
                        )


                        # ------------------------------------------------------
                        # IMAGE → PNG BYTES
                        # ------------------------------------------------------

                        image_buffer = io.BytesIO()

                        image.save(
                            image_buffer,
                            format="PNG"
                        )

                        image_bytes = image_buffer.getvalue()


                        # ------------------------------------------------------
                        # DISPLAY
                        # ------------------------------------------------------

                        st.image(
                            image,
                            caption="FLUX.1-schnell Çıktısı",
                            use_container_width=True
                        )


                        # ------------------------------------------------------
                        # DOWNLOAD
                        # ------------------------------------------------------

                        st.download_button(
                            label="📥 Görseli İndir",

                            data=image_bytes,

                            file_name=(
                                f"flux_"
                                f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                f".png"
                            ),

                            mime="image/png",

                            use_container_width=True
                        )


                        # ------------------------------------------------------
                        # HISTORY
                        # ------------------------------------------------------

                        st.session_state.history.append(
                            {
                                "type": "image",
                                "data": image,
                                "prompt": final_prompt
                            }
                        )


                    # ----------------------------------------------------------
                    # ERROR HANDLING
                    # ----------------------------------------------------------

                    except Exception as e:

                        error_text = str(e)

                        st.error(
                            "❌ Görsel üretimi başarısız."
                        )

                        st.code(
                            error_text
                        )


                        # Kullanıcıya anlaşılır açıklamalar
                        if "410" in error_text:

                            st.warning(
                                "Hugging Face provider/model rotasında "
                                "geçici veya destek durumu kaynaklı bir sorun "
                                "oluştu. Kod artık hf-inference endpoint'ini "
                                "doğrudan kullanmıyor; provider=auto ile "
                                "yönlendirme yapıyor."
                            )

                        elif "401" in error_text or "403" in error_text:

                            st.warning(
                                "HF_API_KEY geçersiz olabilir veya token'ın "
                                "Inference Providers yetkisi eksik olabilir."
                            )

                        elif "402" in error_text:

                            st.warning(
                                "Hugging Face hesabındaki kullanılabilir "
                                "Inference kredisi/bakiyesi yetersiz olabilir."
                            )

                        elif "429" in error_text:

                            st.warning(
                                "İstek limiti aşıldı. Bir süre bekleyip "
                                "tekrar deneyin."
                            )


# ==============================================================================
# TAB 2 — VIDEO
# ==============================================================================

with tab2:

    st.markdown("### 🎬 Video Üretim Durumu")

    st.info(
        """
        Video üretimi bu sürümde aktif değildir.

        Bu uygulama şu anda Hugging Face Inference Providers üzerinden
        FLUX.1-schnell ile görsel üretmektedir.
        """
    )

    st.markdown("#### Mevcut sistem")

    st.code(
        """
Streamlit
   ↓
Hugging Face API
   ↓
Inference Providers
   ↓
FLUX.1-schnell
   ↓
Görsel
        """
    )


# ==============================================================================
# TAB 3 — GALERİ
# ==============================================================================

with tab3:

    st.markdown("### 🖼️ Üretilenler")

    if len(st.session_state.history) == 0:

        st.info(
            "Henüz görsel üretilmedi."
        )

    else:

        cols = st.columns(3)

        for idx, item in enumerate(
            reversed(st.session_state.history)
        ):

            with cols[idx % 3]:

                if item["type"] == "image":

                    st.image(
                        item["data"],
                        use_container_width=True
                    )

                    st.caption(
                        f"**Prompt:** "
                        f"{item['prompt'][:100]}..."
                    )

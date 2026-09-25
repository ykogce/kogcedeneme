import streamlit as st
import requests
import io
import datetime

from PIL import Image
from huggingface_hub import InferenceClient


# ==============================================================================
# CONFIG
# ==============================================================================

st.set_page_config(
    page_title="KOGCE AI Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

HF_API_KEY = st.secrets.get("HF_API_KEY", "").strip()

IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"

# Prompt Robotu için LLM
PROMPT_MODEL = "openai/gpt-oss-120b:cheapest"

# Hugging Face otomatik provider seçimi
HF_PROVIDER = "auto"


# ==============================================================================
# SESSION STATE
# ==============================================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_original_prompt" not in st.session_state:
    st.session_state.last_original_prompt = ""

if "last_enhanced_prompt" not in st.session_state:
    st.session_state.last_enhanced_prompt = ""

if "last_image" not in st.session_state:
    st.session_state.last_image = None


# ==============================================================================
# MODERN CSS
# ==============================================================================

st.markdown(
    """
<style>

/* =========================================================
   GLOBAL
========================================================= */

.stApp {
    background:
        radial-gradient(
            circle at 15% 10%,
            rgba(124,58,237,0.16),
            transparent 30%
        ),
        radial-gradient(
            circle at 85% 15%,
            rgba(14,165,233,0.12),
            transparent 28%
        ),
        #070b14;
    color: #f8fafc;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

p, span, label, div, .stMarkdown {
    color: #e5e7eb;
}


/* =========================================================
   HEADER
========================================================= */

.hero {
    text-align: center;
    padding: 10px 10px 28px 10px;
}

.hero-title {
    font-size: 3.2rem;
    line-height: 1.05;
    font-weight: 900;
    letter-spacing: -1.5px;
    background:
        linear-gradient(
            110deg,
            #c084fc,
            #60a5fa,
            #22d3ee
        );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

.hero-subtitle {
    color: #94a3b8 !important;
    font-size: 1rem;
    letter-spacing: .3px;
}

.badge-row {
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 16px;
}

.badge {
    background: rgba(30,41,59,.7);
    border: 1px solid rgba(148,163,184,.15);
    border-radius: 999px;
    padding: 6px 13px;
    font-size: .78rem;
    color: #cbd5e1 !important;
}


/* =========================================================
   CARDS
========================================================= */

.card {
    background:
        linear-gradient(
            145deg,
            rgba(17,24,39,.88),
            rgba(15,23,42,.72)
        );
    border: 1px solid rgba(148,163,184,.12);
    border-radius: 18px;
    padding: 22px;
    box-shadow:
        0 20px 60px rgba(0,0,0,.22);
    margin-bottom: 18px;
}

.card-title {
    font-size: 1.05rem;
    font-weight: 800;
    color: #f8fafc !important;
    margin-bottom: 5px;
}

.card-description {
    font-size: .83rem;
    color: #94a3b8 !important;
    margin-bottom: 16px;
}


/* =========================================================
   PROMPT BOX
========================================================= */

.stTextArea textarea {
    background: #0b1220 !important;
    color: #f8fafc !important;
    border: 1px solid rgba(148,163,184,.18) !important;
    border-radius: 14px !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
}

.stTextArea textarea:focus {
    border: 1px solid #8b5cf6 !important;
    box-shadow: 0 0 0 1px #8b5cf6 !important;
}


/* =========================================================
   SELECTBOX
========================================================= */

.stSelectbox div[data-baseweb="select"] {
    background: #0b1220 !important;
    border-radius: 12px !important;
}

.stSelectbox div[data-baseweb="select"] * {
    color: #f8fafc !important;
}


/* =========================================================
   BUTTON
========================================================= */

div.stButton > button {
    border-radius: 12px !important;
    min-height: 46px;
    font-weight: 750 !important;
    transition: .2s ease;
}

div.stButton > button:hover {
    transform: translateY(-1px);
    border-color: rgba(139,92,246,.7) !important;
}

div.stButton > button[kind="primary"] {
    background:
        linear-gradient(
            135deg,
            #7c3aed,
            #2563eb
        ) !important;

    color: white !important;
    border: none !important;
    box-shadow:
        0 8px 25px rgba(79,70,229,.25);
}


/* =========================================================
   TOGGLE
========================================================= */

[data-testid="stToggle"] label {
    font-weight: 700 !important;
}

[data-testid="stToggle"] [data-baseweb="checkbox"] {
    transform: scale(1.05);
}


/* =========================================================
   TABS
========================================================= */

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15,23,42,.65);
    border-radius: 14px;
    padding: 6px;
}

.stTabs [data-baseweb="tab"] {
    height: 44px;
    border-radius: 10px;
    padding: 0 18px;
    color: #94a3b8 !important;
}

.stTabs [aria-selected="true"] {
    background:
        linear-gradient(
            135deg,
            #7c3aed,
            #2563eb
        ) !important;
    color: white !important;
}


/* =========================================================
   PROMPT RESULT
========================================================= */

.prompt-result {
    background: #080d18;
    border: 1px solid rgba(139,92,246,.2);
    border-radius: 14px;
    padding: 16px;
    margin-top: 10px;
}

.prompt-label {
    color: #a78bfa !important;
    font-size: .75rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .8px;
    margin-bottom: 8px;
}


/* =========================================================
   SIDEBAR
========================================================= */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0b1020,
            #080c16
        );
    border-right: 1px solid rgba(148,163,184,.08);
}


/* =========================================================
   IMAGE
========================================================= */

[data-testid="stImage"] {
    border-radius: 16px;
    overflow: hidden;
}


/* =========================================================
   MOBILE
========================================================= */

@media (max-width: 768px) {

    .hero-title {
        font-size: 2.2rem;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .card {
        padding: 16px;
    }

}

</style>
""",
    unsafe_allow_html=True
)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def hf_headers():
    return {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }


def check_api_key():
    if not HF_API_KEY:
        raise RuntimeError(
            "HF_API_KEY bulunamadı. "
            "Streamlit Secrets bölümünde HF_API_KEY tanımlı olmalı."
        )


# ==============================================================================
# AI PROMPT ROBOT
# ==============================================================================

def improve_prompt_with_ai(
    user_input,
    style_preset,
    aspect_ratio
):

    check_api_key()

    style_map = {

        "Doğal / Yok":
            """
Keep the image natural and realistic.
Do not force an artificial visual style.
""",

        "Fotogerçekçi":
            """
Use professional photorealism, realistic skin,
realistic materials, natural textures,
professional photography and believable lighting.
""",

        "Anime / Manga":
            """
Use a polished high-detail anime/manga illustration style,
clean linework, expressive characters and rich colors.
""",

        "3D Render":
            """
Use a premium 3D rendered appearance,
detailed geometry, realistic materials,
professional studio lighting and polished rendering.
""",

        "Yağlı Boya":
            """
Use a detailed traditional oil painting aesthetic,
visible brush texture, rich pigments and artistic depth.
""",

        "Cyberpunk":
            """
Use a sophisticated cyberpunk aesthetic,
neon illumination, futuristic architecture,
atmospheric depth and cinematic lighting.
""",

        "Cinematic":
            """
Use a professional cinematic movie-frame aesthetic,
dramatic composition, realistic lighting,
depth of field and film-quality visual storytelling.
"""
    }

    ratio_map = {

        "1:1 Kare":
            "Compose the image for a square 1:1 format.",

        "9:16 Dikey":
            "Compose the image vertically for a 9:16 portrait format.",

        "16:9 Yatay":
            "Compose the image horizontally for a 16:9 landscape format."
    }

    system_prompt = """
You are an expert professional image-prompt engineer.

The user will give you a short image idea, possibly in Turkish,
possibly incomplete or conversational.

Your job is to understand the user's intention and transform it
into one highly effective English prompt for a modern text-to-image
generation model.

CORE RULES:

- Preserve the user's main subject.
- Preserve the user's requested action.
- Never change the meaning of the request.
- Never invent major story elements.
- Translate Turkish naturally into English.
- Add useful visual specificity.
- Improve composition.
- Improve lighting.
- Improve environment description.
- Improve camera perspective when appropriate.
- Add realistic textures and materials when appropriate.
- Add professional visual quality.
- Use 8K quality language when appropriate.
- Use photorealistic language when the selected style requires it.
- Avoid useless keyword spam.
- Avoid contradictory instructions.
- Avoid repeating the same adjective.
- Do not mention these instructions.
- Do not explain your work.
- Do not put the result inside quotation marks.
- Return ONLY the final English image-generation prompt.

The result should be detailed enough for a professional image model,
but should remain coherent and visually understandable.
"""

    user_message = f"""
USER'S ORIGINAL IDEA:
{user_input}

SELECTED STYLE:
{style_preset}

STYLE DIRECTION:
{style_map.get(style_preset, "")}

SELECTED ASPECT RATIO:
{aspect_ratio}

COMPOSITION DIRECTION:
{ratio_map.get(aspect_ratio, "")}

Now create the final professional English image-generation prompt.
"""

    payload = {
        "model": PROMPT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        "temperature": 0.65,
        "max_tokens": 600,
        "stream": False
    }

    response = requests.post(
        "https://router.huggingface.co/v1/chat/completions",
        headers=hf_headers(),
        json=payload,
        timeout=90
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Prompt Robotu API Hatası "
            f"({response.status_code}): "
            f"{response.text}"
        )

    data = response.json()

    try:
        prompt = data["choices"][0]["message"]["content"].strip()
    except Exception:
        raise RuntimeError(
            f"Beklenmeyen Prompt Robotu cevabı: {data}"
        )

    return prompt


# ==============================================================================
# IMAGE GENERATION
# ==============================================================================

def generate_image(prompt, aspect_ratio):

    check_api_key()

    client = InferenceClient(
        provider=HF_PROVIDER,
        api_key=HF_API_KEY
    )

    # Dengeli çözünürlükler:
    # Tüm provider/GPU seçeneklerinde gereksiz büyük istek oluşturmamak için
    # başlangıçta kontrollü tutuluyor.

    if aspect_ratio == "1:1 Kare":
        width = 768
        height = 768

    elif aspect_ratio == "9:16 Dikey":
        width = 768
        height = 1344

    else:
        width = 1344
        height = 768

    image = client.text_to_image(
        prompt=prompt,
        model=IMAGE_MODEL,
        width=width,
        height=height
    )

    return image


# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:

    st.markdown("## ✨ KOGCE AI")

    st.caption("Creative Image Studio")

    st.markdown("---")

    st.markdown("### ⚙️ Studio")

    use_password = st.checkbox(
        "🔒 Şifre Koruması",
        value=False
    )

    if use_password:

        user_pass = st.text_input(
            "Şifre",
            type="password"
        )

        if user_pass != "1234":

            st.warning(
                "Geçerli şifreyi girin."
            )

            st.stop()

    st.markdown("---")

    st.markdown("### 🤖 Motor")

    st.caption(
        "FLUX.1-schnell"
    )

    st.caption(
        "Hugging Face Inference Providers"
    )

    st.markdown("---")

    st.caption(
        "AI Prompt Robotu açık olduğunda "
        "girdiğiniz kısa fikir profesyonel "
        "İngilizce görsel promptuna dönüştürülür."
    )


# ==============================================================================
# HERO
# ==============================================================================

st.markdown(
    """
<div class="hero">

<div class="hero-title">
✨ KOGCE AI STUDIO
</div>

<div class="hero-subtitle">
Professional AI Image Creation
</div>

<div class="badge-row">

<div class="badge">FLUX.1-schnell</div>
<div class="badge">AI Prompt Engineer</div>
<div class="badge">9:16 • 1:1 • 16:9</div>

</div>

</div>
""",
    unsafe_allow_html=True
)


# ==============================================================================
# TABS
# ==============================================================================

tab_create, tab_gallery, tab_status = st.tabs(
    [
        "✨ Create",
        "🖼️ Gallery",
        "⚙️ System"
    ]
)


# ==============================================================================
# CREATE TAB
# ==============================================================================

with tab_create:

    left, right = st.columns(
        [0.95, 1.05],
        gap="large"
    )

    # --------------------------------------------------------------------------
    # LEFT
    # --------------------------------------------------------------------------

    with left:

        st.markdown(
            """
<div class="card">

<div class="card-title">
🎨 Describe your image
</div>

<div class="card-description">
Write naturally. Turkish or English is fine.
</div>

</div>
""",
            unsafe_allow_html=True
        )

        user_prompt = st.text_area(
            "Prompt",
            placeholder=(
                "Örn: kaldırımda yürüyen kız..."
            ),
            height=155,
            label_visibility="collapsed"
        )

        # ----------------------------------------------------------------------
        # PROMPT ROBOT TOGGLE
        # ----------------------------------------------------------------------

        toggle_col1, toggle_col2 = st.columns(
            [0.7, 0.3]
        )

        with toggle_col1:

            use_ai_boost = st.toggle(
                "🤖 AI Prompt Robotu",
                value=True
            )

        with toggle_col2:

            if use_ai_boost:

                st.success(
                    "ON",
                    icon="✓"
                )

            else:

                st.info(
                    "OFF",
                    icon="○"
                )


        st.caption(
            "ON: Fikrin profesyonel İngilizce prompta dönüştürülür. "
            "OFF: Yazdığın prompt doğrudan FLUX'a gönderilir."
        )


        # ----------------------------------------------------------------------
        # OPTIONS
        # ----------------------------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            aspect_ratio = st.selectbox(
                "📐 Format",
                [
                    "9:16 Dikey",
                    "1:1 Kare",
                    "16:9 Yatay"
                ]
            )

        with col2:

            style_preset = st.selectbox(
                "🎨 Stil",
                [
                    "Fotogerçekçi",
                    "Doğal / Yok",
                    "Cinematic",
                    "3D Render",
                    "Anime / Manga",
                    "Yağlı Boya",
                    "Cyberpunk"
                ]
            )


        st.markdown("<br>", unsafe_allow_html=True)


        generate_button = st.button(
            "🚀 GÖRSELİ ÜRET",
            type="primary",
            use_container_width=True
        )


    # --------------------------------------------------------------------------
    # RIGHT
    # --------------------------------------------------------------------------

    with right:

        st.markdown(
            """
<div class="card">

<div class="card-title">
🖼️ Generated Image
</div>

<div class="card-description">
Your generated result will appear here.
</div>

</div>
""",
            unsafe_allow_html=True
        )


        # ======================================================================
        # GENERATION
        # ======================================================================

        if generate_button:

            if not user_prompt.strip():

                st.warning(
                    "Önce ne istediğini yaz."
                )

            else:

                original_prompt = user_prompt.strip()

                final_prompt = original_prompt


                # ==============================================================
                # AI PROMPT ROBOT
                # ==============================================================

                if use_ai_boost:

                    with st.status(
                        "🤖 AI Prompt Robotu çalışıyor...",
                        expanded=True
                    ) as status:

                        try:

                            st.write(
                                "Fikrin analiz ediliyor..."
                            )

                            final_prompt = improve_prompt_with_ai(
                                original_prompt,
                                style_preset,
                                aspect_ratio
                            )

                            st.write(
                                "Profesyonel İngilizce prompt hazırlandı."
                            )

                            status.update(
                                label="✓ Prompt hazır",
                                state="complete",
                                expanded=False
                            )

                        except Exception as e:

                            status.update(
                                label="⚠️ Prompt Robotu başarısız",
                                state="error",
                                expanded=True
                            )

                            st.error(
                                str(e)
                            )

                            st.warning(
                                "Orijinal prompt ile devam ediliyor."
                            )

                            final_prompt = original_prompt


                # ==============================================================
                # PROMPT HISTORY
                # ==============================================================

                st.session_state.last_original_prompt = (
                    original_prompt
                )

                st.session_state.last_enhanced_prompt = (
                    final_prompt
                )


                # ==============================================================
                # SHOW PROMPT
                # ==============================================================

                if use_ai_boost:

                    with st.expander(
                        "🤖 AI Prompt Robotu nasıl düzeltti?",
                        expanded=True
                    ):

                        st.markdown(
                            '<div class="prompt-label">ORİJİNAL FİKRİN</div>',
                            unsafe_allow_html=True
                        )

                        st.code(
                            original_prompt,
                            language="text"
                        )


                        st.markdown(
                            '<div class="prompt-label">AI TARAFINDAN GELİŞTİRİLEN PROMPT</div>',
                            unsafe_allow_html=True
                        )

                        st.code(
                            final_prompt,
                            language="text"
                        )

                        st.caption(
                            "Bu geliştirilmiş İngilizce prompt FLUX.1-schnell'e gönderilecek."
                        )


                # ==============================================================
                # IMAGE GENERATION
                # ==============================================================

                with st.spinner(
                    "🎨 FLUX.1-schnell görsel oluşturuyor..."
                ):

                    try:

                        image = generate_image(
                            final_prompt,
                            aspect_ratio
                        )


                        # ------------------------------------------------------
                        # BYTES
                        # ------------------------------------------------------

                        buffer = io.BytesIO()

                        image.save(
                            buffer,
                            format="PNG"
                        )

                        image_bytes = buffer.getvalue()


                        # ------------------------------------------------------
                        # DISPLAY
                        # ------------------------------------------------------

                        st.image(
                            image,
                            use_container_width=True
                        )


                        # ------------------------------------------------------
                        # DOWNLOAD
                        # ------------------------------------------------------

                        filename = (
                            "kogce_"
                            + datetime.datetime.now().strftime(
                                "%Y%m%d_%H%M%S"
                            )
                            + ".png"
                        )

                        st.download_button(
                            "📥 Görseli İndir",
                            data=image_bytes,
                            file_name=filename,
                            mime="image/png",
                            use_container_width=True
                        )


                        # ------------------------------------------------------
                        # SAVE HISTORY
                        # ------------------------------------------------------

                        st.session_state.history.append(
                            {
                                "type": "image",
                                "data": image,
                                "prompt": final_prompt,
                                "original": original_prompt,
                                "style": style_preset,
                                "ratio": aspect_ratio,
                                "date": datetime.datetime.now().strftime(
                                    "%d.%m.%Y %H:%M"
                                )
                            }
                        )

                        st.session_state.last_image = image


                    # ==========================================================
                    # ERROR HANDLING
                    # ==========================================================

                    except Exception as e:

                        error_text = str(e)

                        st.error(
                            "❌ Görsel üretimi başarısız."
                        )

                        with st.expander(
                            "Teknik hata detayları"
                        ):

                            st.code(
                                error_text
                            )


                        if "401" in error_text:

                            st.warning(
                                "HF_API_KEY geçersiz veya "
                                "Inference Providers yetkisi eksik."
                            )

                        elif "402" in error_text:

                            st.warning(
                                "Hugging Face Inference Providers "
                                "kredisi/bakiyesi yetersiz olabilir."
                            )

                        elif "403" in error_text:

                            st.warning(
                                "Token'ın Inference Providers "
                                "kullanım yetkisini kontrol et."
                            )

                        elif "410" in error_text:

                            st.warning(
                                "Eski hf-inference rotasına erişilmiş. "
                                "Bu uygulama doğrudan eski endpoint'i "
                                "kullanmıyor; provider yönlendirmesi "
                                "kontrol edilmeli."
                            )

                        elif "429" in error_text:

                            st.warning(
                                "İstek limiti aşıldı. Bir süre bekleyip "
                                "yeniden deneyin."
                            )


# ==============================================================================
# GALLERY
# ==============================================================================

with tab_gallery:

    st.markdown(
        """
<div class="card">

<div class="card-title">
🖼️ Your Gallery
</div>

<div class="card-description">
Images generated during this session.
</div>

</div>
""",
        unsafe_allow_html=True
    )


    if not st.session_state.history:

        st.info(
            "Henüz görsel oluşturulmadı."
        )

    else:

        cols = st.columns(3)

        for index, item in enumerate(
            reversed(st.session_state.history)
        ):

            with cols[index % 3]:

                st.image(
                    item["data"],
                    use_container_width=True
                )

                st.caption(
                    item["date"]
                )

                with st.expander(
                    "Prompt"
                ):

                    st.code(
                        item["prompt"],
                        language="text"
                    )


# ==============================================================================
# SYSTEM TAB
# ==============================================================================

with tab_status:

    st.markdown(
        """
<div class="card">

<div class="card-title">
⚙️ System
</div>

<div class="card-description">
Current generation pipeline.
</div>

</div>
""",
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Image Model",
            "FLUX.1-schnell"
        )

    with col2:

        st.metric(
            "Prompt AI",
            "GPT-OSS"
        )

    with col3:

        st.metric(
            "Provider",
            "Auto"
        )


    st.markdown("### Pipeline")

    st.code(
        """
User Prompt
     │
     ├── AI Prompt Robot OFF
     │          │
     │          ▼
     │      Original Prompt
     │
     └── AI Prompt Robot ON
                │
                ▼
        Professional Prompt
                │
                ▼
          FLUX.1-schnell
                │
                ▼
             PNG Image
                │
                ▼
             Gallery
        """,
        language="text"
    )

    st.info(
        "AI Prompt Robotu kapalıyken LLM çağrısı yapılmaz."
    )

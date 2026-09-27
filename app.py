import io
import time
import requests
import re
import streamlit as st

from PIL import Image


# ============================================================
# CONFIG
# ============================================================

APP_PASSWORD = "1234"

PROMPT_MODEL = "openai/gpt-oss-120b"
IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"

# Tulpar / ComfyUI backend
LOCAL_BACKEND_URL = st.secrets.get(
    "LOCAL_BACKEND_URL",
    ""
).strip()

CHAT_URL = "https://router.huggingface.co/v1/chat/completions"

# Tulpar / ComfyUI GPU üretimleri 45 saniyeden uzun sürebilir.
GENERATION_TIMEOUT = 1800  # 30 dakika


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
# LEONARDO-STYLE UI — VISUAL ONLY
# ============================================================

st.html("""
<style>
:root {
    --bg: #070709;
    --panel: #101014;
    --panel-2: #15151b;
    --line: rgba(255,255,255,.10);
    --muted: #9a9aa3;
    --text: #f5f5f7;
    --accent: #7c5cff;
    --accent-2: #9a7cff;
}

.stApp {
    background: #070709;
    color: var(--text);
}

.block-container {
    max-width: 1220px;
    padding-top: 1.2rem;
    padding-bottom: 4rem;
}

header[data-testid="stHeader"] { background: transparent; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0a0a0d;
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
section[data-testid="stSidebar"] * { color: #eeeeF2; }

.k-sidebar-brand {
    font-size: 26px;
    font-weight: 900;
    letter-spacing: -.06em;
    margin-bottom: 2px;
}
.k-sidebar-sub {
    color: #74747d;
    font-size: 11px;
    letter-spacing: .13em;
    text-transform: uppercase;
    margin-bottom: 22px;
}

/* Streamlit native buttons become the navigation / tool cards */
section[data-testid="stSidebar"] .stButton > button {
    min-height: 44px;
    margin: 3px 0;
    border-radius: 12px;
    border: 1px solid transparent;
    background: transparent;
    color: #b7b7c0 !important;
    font-weight: 650;
    text-align: left;
    box-shadow: none;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #17171d;
    border-color: var(--line);
    color: white !important;
}

/* Top bar */
.k-topline {
    color: #707079;
    font-size: 12px;
    letter-spacing: .16em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

/* Hero */
.k-hero-title {
    font-size: clamp(42px, 7vw, 78px);
    line-height: .93;
    letter-spacing: -.075em;
    font-weight: 950;
    margin: 0 0 12px 0;
    color: white;
}
.k-hero-title span { color: var(--accent-2); }
.k-hero-sub {
    max-width: 720px;
    color: #8f8f99;
    font-size: 15px;
    line-height: 1.65;
    margin-bottom: 24px;
}

/* Prompt composer */
.k-composer {
    border: 1px solid rgba(255,255,255,.13);
    background: #101014;
    border-radius: 24px;
    padding: 16px;
    box-shadow: 0 25px 70px rgba(0,0,0,.34), inset 0 1px 0 rgba(255,255,255,.04);
    margin: 12px 0 18px;
}
.k-composer-label {
    color: #777781;
    font-size: 11px;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin: 2px 4px 7px;
}

/* Text areas / inputs */
textarea, input {
    background: #111116 !important;
    color: #f4f4f6 !important;
    border: 1px solid rgba(255,255,255,.12) !important;
    border-radius: 15px !important;
}
textarea:focus, input:focus {
    border-color: rgba(124,92,255,.75) !important;
    box-shadow: 0 0 0 1px rgba(124,92,255,.20) !important;
}
textarea::placeholder, input::placeholder { color: #65656e !important; }

/* Labels */
.stTextArea label, .stTextInput label, .stSelectbox label,
.stRadio label, .stCheckbox label, .stToggle label, .stSlider label,
.stFileUploader label, .stNumberInput label {
    color: #bdbdc6 !important;
    font-weight: 650 !important;
}

/* Primary buttons */
.stButton > button {
    min-height: 46px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,.10);
    background: #17171d;
    color: white !important;
    font-weight: 750;
    box-shadow: none;
}
.stButton > button:hover {
    border-color: rgba(124,92,255,.60);
    background: #1c1928;
}
.stButton > button[kind="primary"] {
    background: #7357ff;
    border-color: #846dff;
    color: white !important;
    box-shadow: 0 12px 30px rgba(92,70,220,.24);
}
.stButton > button[kind="primary"]:hover { background: #8067ff; }

/* Radio as pill selector */
div[data-testid="stRadio"] > div {
    gap: 7px;
    flex-wrap: wrap;
}
div[data-testid="stRadio"] label {
    background: #111116;
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 999px;
    padding: 7px 14px;
}

/* Selects */
[data-baseweb="select"] > div {
    background: #111116 !important;
    border-color: rgba(255,255,255,.12) !important;
    border-radius: 13px !important;
}

/* Cards */
.k-section-title {
    font-size: 24px;
    font-weight: 850;
    letter-spacing: -.04em;
    color: white;
    margin: 24px 0 4px;
}
.k-section-sub { color: #74747d; font-size: 13px; margin-bottom: 14px; }

/* Metric / alerts */
div[data-testid="stMetric"] {
    background: #111116;
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 15px;
    padding: 15px;
}
div[data-testid="stMetricValue"] { color: white !important; }
.stAlert { border-radius: 13px; }

/* Images / video */
[data-testid="stImage"] img, video {
    border-radius: 16px;
}

/* Download buttons */
.stDownloadButton > button {
    background: #15151b !important;
    border: 1px solid rgba(255,255,255,.10) !important;
    border-radius: 13px !important;
    color: white !important;
}

/* Hide default tab strip; navigation is sidebar-based */
.k-hidden-tabs { display: none; }

/* Tool tiles */
.k-tool-note {
    color: #6f6f78;
    font-size: 12px;
    line-height: 1.45;
    min-height: 34px;
    margin: 3px 0 8px;
}

/* Mobile */
@media (max-width: 760px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .k-hero-title { font-size: 48px; }
    .k-hero-sub { font-size: 14px; }
}
</style>
""")

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
# LOCAL PROMPT ROBOT — NO HUGGING FACE
# ============================================================

def clean_prompt(value):
    if value is None:
        return ""
    return " ".join(str(value).replace("\x00", " ").split()).strip()


def local_prompt_robot(prompt, style, lighting, camera, quality, color_mood, negative_prompt):
    """
    Convert a short Turkish/mixed-language idea into a polished English
    image-generation prompt without using Hugging Face or any paid API.
    """
    prompt = clean_prompt(prompt)
    if not prompt:
        return ""

    phrase_replacements = {
        "zombilerden kaçan sevimli gri kedi": "a cute gray cat running away from zombies",
        "zombilerden kaçan sevimli gri cat": "a cute gray cat running away from zombies",
        "zombilerden kaçan": "running away from zombies",
        "zombilerden kaçıyor": "running away from zombies",
        "zombiler": "zombies",
        "zombi": "zombie",
        "sevimli gri kedi": "a cute gray cat",
        "gri kedi": "a gray cat",
        "sevimli kedi": "a cute cat",
        "sevimli köpek": "a cute dog",
        "kaçan": "running away",
        "kaçıyor": "is running away",
        "koşuyor": "is running",
        "koşan": "running",
        "yürüyor": "is walking",
        "yürüyen": "walking",
        "oturuyor": "is sitting",
        "oturan": "sitting",
        "ayakta duran": "standing",
        "ayakta": "standing",
        "sevimli": "cute",
        "gri": "gray",
        "kedi": "cat",
        "köpek": "dog",
        "araba": "car",
        "otomobil": "car",
        "kırmızı": "red",
        "mavi": "blue",
        "siyah": "black",
        "beyaz": "white",
        "yeşil": "green",
        "sarı": "yellow",
        "gerçekçi": "photorealistic",
        "fotogerçekçi": "photorealistic",
        "sinematik": "cinematic",
        "detaylı": "highly detailed",
        "çok detaylı": "highly detailed",
        "gece": "at night",
        "gündüz": "during daytime",
        "yağmur": "rainy",
        "karlı": "snowy",
        "kar": "snow",
        "deniz": "sea",
        "sahil": "seaside",
        "plaj": "beach",
        "şehir": "city",
        "sokak": "street",
        "kadın": "woman",
        "adam": "man",
        "erkek": "man",
        "çocuk": "child",
        "mutlu": "happy",
        "üzgün": "sad",
        "gülümseyen": "smiling",
    }

    translated = prompt
    for tr, en in sorted(phrase_replacements.items(), key=lambda x: len(x[0]), reverse=True):
        translated = re.sub(rf"(?<!\w){re.escape(tr)}(?!\w)", en, translated, flags=re.IGNORECASE)

    translated = clean_prompt(translated)
    parts = [translated]

    if style != "Automatic":
        parts.append(f"Visual style: {style}.")
    if lighting != "Automatic":
        parts.append(f"Lighting: {lighting}.")
    if camera != "Automatic":
        parts.append(f"Camera and composition: {camera}.")
    if quality != "Automatic":
        parts.append(f"Image quality: {quality}.")
    if color_mood != "Automatic":
        parts.append(f"Color and mood: {color_mood}.")

    parts.append(
        "Professional commercial image, coherent composition, accurate perspective, "
        "clear subject hierarchy, natural depth, physically plausible lighting, "
        "realistic materials and textures, consistent subject details, "
        "cinematic visual quality, high visual fidelity."
    )

    neg = clean_prompt(negative_prompt)
    if neg:
        parts.append(f"Avoid: {neg}.")

    return clean_prompt(" ".join(parts))


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
# IMAGE GENERATION — TULPAR / COMFYUI
# ============================================================

def generate_image(
    prompt,
    width,
    height,
    seed=None,
    progress_bar=None,
    status_box=None,
):
    """Tulpar üzerindeki Qwen Image 2.1 ile gerçek job polling kullanarak görsel üretir."""
    payload = {
        "prompt": str(prompt),
        "width": int(width),
        "height": int(height),
    }
    if seed is not None:
        payload["seed"] = int(seed)

    response, error = local_backend_request(
        "/generate-image",
        payload,
        timeout=30,
        form=True,
    )
    if error:
        return None, error

    try:
        data = response.json()
    except Exception as e:
        return None, f"Tulpar başlangıç cevabı okunamadı: {e}"

    job_id = data.get("job_id")
    if not job_id:
        return None, f"Tulpar job_id döndürmedi: {data}"

    result, error = poll_backend_job(
        job_id,
        progress_bar=progress_bar,
        status_box=status_box,
        timeout=GENERATION_TIMEOUT,
    )
    if error:
        return None, error

    raw, error = download_backend_result(result)
    if error:
        return None, error

    try:
        return Image.open(io.BytesIO(raw)).convert("RGB"), None
    except Exception as e:
        return None, f"Görsel sonucu okunamadı: {e}"


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
    form=False,
):
    """Send only short-lived requests to Tulpar.

    Long GPU jobs are handled through job_id + polling so Cloudflare
    Quick Tunnel never has to keep one HTTP request open for minutes.
    """
    if not LOCAL_BACKEND_URL:
        return None, (
            "Tulpar backend adresi henüz tanımlanmamış."
        )

    try:
        url = LOCAL_BACKEND_URL.rstrip('/') + endpoint

        if files:
            response = requests.post(
                url,
                data=payload,
                files=files,
                timeout=timeout,
            )
        elif form:
            response = requests.post(
                url,
                data=payload,
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
        return None, "Tulpar backend başlangıç isteği zaman aşımına uğradı."
    except requests.exceptions.RequestException as e:
        return None, f"Tulpar bağlantı hatası: {e}"
    except Exception as e:
        return None, str(e)


def poll_backend_job(job_id, progress_bar=None, status_box=None, timeout=1800):
    """Poll a Tulpar job with short HTTP requests until it really finishes."""
    if not LOCAL_BACKEND_URL:
        return None, "Tulpar backend adresi tanımlanmamış."

    started = time.time()
    last_progress = -1

    while True:
        if time.time() - started > timeout:
            return None, "Tulpar üretimi izin verilen maksimum süreyi aştı."

        try:
            response = requests.get(
                f"{LOCAL_BACKEND_URL.rstrip('/')}/progress/{job_id}",
                timeout=15,
            )

            if response.status_code != 200:
                return None, (
                    f"Tulpar job durumu HTTP {response.status_code}\n"
                    f"{response.text[:2000]}"
                )

            data = response.json()
            progress = int(data.get("progress", 0) or 0)
            status = str(data.get("status", ""))
            message = str(data.get("message", "Üretim devam ediyor..."))

            if progress_bar is not None and progress != last_progress:
                progress_bar.progress(
                    max(0.0, min(1.0, progress / 100.0)),
                    text=f"%{progress} — {message}",
                )
                last_progress = progress

            if status_box is not None:
                status_box.caption(message)

            if status == "completed":
                if progress_bar is not None:
                    progress_bar.progress(1.0, text="%100 — Üretim tamamlandı")
                return data, None

            if status == "error":
                return None, data.get("error") or message or "Tulpar üretim hatası."

        except requests.exceptions.RequestException as e:
            # A single polling failure is not a generation failure.
            # Retry because the GPU job itself may still be running.
            if status_box is not None:
                status_box.caption(f"Tulpar durum bağlantısı yeniden deneniyor... ({e})")

        time.sleep(1.0)


def download_backend_result(data):
    """Download a completed Tulpar output using its short-lived URL."""
    download_url = data.get("download_url")
    if not download_url:
        return None, "Tulpar tamamlandı ancak download_url döndürmedi."

    if download_url.startswith("/"):
        download_url = LOCAL_BACKEND_URL.rstrip("/") + download_url

    try:
        response = requests.get(download_url, timeout=60)
        response.raise_for_status()
        if not response.content:
            return None, "Tulpar boş bir çıktı dosyası döndürdü."
        return response.content, None
    except requests.exceptions.RequestException as e:
        return None, f"Tulpar çıktı dosyası alınamadı: {e}"


# ============================================================
# TEXT TO VIDEO
# ============================================================

def generate_text_video(
    prompt,
    num_frames=33,
    steps=20,
    progress_bar=None,
    status_box=None,
):
    """Start T2V quickly, then poll the job until a real file exists."""
    response, error = local_backend_request(
        "/generate-video",
        {
            "prompt": prompt,
            "num_frames": int(num_frames),
            "steps": int(steps),
        },
        timeout=30,
    )

    if error:
        return None, error

    try:
        data = response.json()
    except Exception as e:
        return None, f"Tulpar başlangıç cevabı okunamadı: {e}"

    job_id = data.get("job_id")
    if not job_id:
        return None, f"Tulpar job_id döndürmedi: {data}"

    result, error = poll_backend_job(
        job_id,
        progress_bar=progress_bar,
        status_box=status_box,
        timeout=GENERATION_TIMEOUT,
    )
    if error:
        return None, error

    return download_backend_result(result)


# ============================================================
# IMAGE TO VIDEO
# ============================================================

def generate_image_video(
    image_bytes,
    prompt,
    num_frames=33,
    steps=20,
    progress_bar=None,
    status_box=None,
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
        timeout=30,
    )

    if error:
        return None, error

    try:
        data = response.json()
    except Exception as e:
        return None, f"Tulpar başlangıç cevabı okunamadı: {e}"

    job_id = data.get("job_id")
    if not job_id:
        return None, f"Tulpar job_id döndürmedi: {data}"

    result, error = poll_backend_job(
        job_id,
        progress_bar=progress_bar,
        status_box=status_box,
        timeout=GENERATION_TIMEOUT,
    )
    if error:
        return None, error

    return download_backend_result(result)


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
    progress_bar=None,
    status_box=None,
):
    if not LOCAL_BACKEND_URL:
        return None, "Karakter motoru henüz Tulpar backend'e bağlanmadı."

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
        timeout=30,
    )
    if error:
        return None, error

    try:
        data = response.json()
    except Exception as e:
        return None, f"Tulpar başlangıç cevabı okunamadı: {e}"

    job_id = data.get("job_id")
    if not job_id:
        return None, f"Tulpar job_id döndürmedi: {data}"

    result, error = poll_backend_job(
        job_id,
        progress_bar=progress_bar,
        status_box=status_box,
        timeout=GENERATION_TIMEOUT,
    )
    if error:
        return None, error

    raw, error = download_backend_result(result)
    if error:
        return None, error

    try:
        return Image.open(io.BytesIO(raw)).convert("RGB"), None
    except Exception as e:
        return None, f"Karakter sonucu okunamadı: {e}"




# ============================================================
# LOGIN — EXISTING AUTHENTICATION PRESERVED
# ============================================================

if not st.session_state.authenticated:
    st.markdown("# KOGCE AI Studio")
    st.caption("Private AI Creative Workspace")
    st.divider()
    password = st.text_input("Studio şifresi", type="password", placeholder="Şifrenizi girin")
    if st.button("STUDIO'YA GİR", use_container_width=True, type="primary"):
        if password == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Hatalı şifre.")
    st.stop()

# ============================================================
# LEONARDO-STYLE NAVIGATION
# ============================================================

NAV_ITEMS = {
    "Görsel": "image",
    "Karakter": "character",
    "Video": "video",
    "AI Araçları": "tools",
    "Galeri": "gallery",
    "Sistem": "system",
}

if "active_page" not in st.session_state:
    st.session_state.active_page = "image"

with st.sidebar:
    st.markdown("<div class='k-sidebar-brand'>KOGCE</div>", unsafe_allow_html=True)
    st.caption("AI CREATIVE STUDIO")
    st.divider()

    st.markdown("**CREATE**")
    for label in ["Görsel", "Karakter", "Video"]:
        if st.button(label, key=f"nav_{label}", use_container_width=True):
            st.session_state.active_page = NAV_ITEMS[label]
            st.rerun()

    st.markdown("**TOOLS**")
    for label in ["AI Araçları", "Galeri"]:
        if st.button(label, key=f"nav_{label}", use_container_width=True):
            st.session_state.active_page = NAV_ITEMS[label]
            st.rerun()

    st.markdown("**SYSTEM**")
    if st.button("Sistem", key="nav_system", use_container_width=True):
        st.session_state.active_page = "system"
        st.rerun()

    st.divider()
    if LOCAL_BACKEND_URL:
        if local_backend_available():
            st.success("TULPAR ONLINE")
        else:
            st.warning("TULPAR OFFLINE")
    else:
        st.info("TULPAR BAĞLANMADI")

    st.caption(f"Galeri: {len(st.session_state.gallery)} görsel")
    if st.button("Çıkış Yap", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

# ============================================================
# TOP HEADER
# ============================================================

st.markdown("<div class='k-topline'>KOGCE AI STUDIO · LOCAL CREATIVE WORKSPACE</div>", unsafe_allow_html=True)
st.markdown("# KOGCE <span style='color:#9a7cff'>AI</span>", unsafe_allow_html=True)

page = st.session_state.active_page

# ============================================================
# IMAGE PAGE — LEONARDO-LIKE COMPOSER
# ============================================================

if page == "image":
    st.markdown("<div class='k-hero-title'>YOUR IDEAS.<br><span>YOUR CREATIONS.</span></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='k-hero-sub'>Fikrini yaz. Prompt robotu onu profesyonel üretim komutuna dönüştürsün. "
        "Ardından görüntüyü doğrudan Tulpar / ComfyUI üzerinde üret.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='k-composer'><div class='k-composer-label'>CREATE</div></div>", unsafe_allow_html=True)
    prompt = st.text_area(
        "Prompt",
        label_visibility="collapsed",
        placeholder="Type a prompt or paste your idea here...",
        height=145,
        key="image_prompt_main",
    )

    mode_col1, mode_col2, mode_col3 = st.columns([1, 1, 1])
    with mode_col1:
        st.caption("MODE")
        st.radio("Mode", ["Image"], horizontal=True, label_visibility="collapsed")
    with mode_col2:
        st.caption("ASPECT")
        aspect_ratio = st.selectbox("Aspect Ratio", ["1:1", "9:16", "16:9"], label_visibility="collapsed")
    with mode_col3:
        st.caption("PROMPT")
        ai_boost = st.toggle("AI Prompt Robotu", value=True, label_visibility="collapsed")

    resolutions = {
        "1:1": (768, 768),
        "9:16": (768, 1344),
        "16:9": (1344, 768),
    }
    width, height = resolutions[aspect_ratio]

    st.markdown("### Generation settings")
    c1, c2, c3 = st.columns(3)
    with c1:
        style = st.selectbox("Style", [
            "Automatic", "Photorealistic", "Cinematic", "Anime", "3D Render",
            "Stylized 3D", "Cartoon", "Fantasy", "Cyberpunk", "Product Photography",
            "Fashion Editorial", "Dark Cinematic",
        ])
    with c2:
        lighting = st.selectbox("Lighting", [
            "Automatic", "Natural daylight", "Cinematic lighting", "Soft studio lighting",
            "Golden hour", "Blue hour", "Neon lighting", "Dramatic lighting",
            "Low-key lighting", "Volumetric lighting", "Rainy night lighting",
        ])
    with c3:
        camera = st.selectbox("Camera", [
            "Automatic", "Close-up portrait", "Medium shot", "Full body shot",
            "Wide cinematic shot", "Low angle", "High angle", "Eye level",
            "Aerial perspective", "Over-the-shoulder",
        ])

    c4, c5, c6 = st.columns(3)
    with c4:
        quality = st.selectbox("Quality", [
            "Automatic", "High detail", "Ultra detailed", "Photographic realism",
            "Cinematic quality", "Sharp professional image",
        ])
    with c5:
        color_mood = st.selectbox("Color / Mood", [
            "Automatic", "Natural colors", "Dark moody", "Warm cinematic",
            "Cool cinematic", "Neon futuristic", "Muted realistic", "High contrast",
        ])
    with c6:
        seed_mode = st.selectbox("Seed", ["Random", "Fixed"])

    seed = None
    if seed_mode == "Fixed":
        seed = st.number_input("Fixed Seed", min_value=0, max_value=999999999, value=123456, step=1)

    negative_prompt = st.text_input(
        "Negative Prompt",
        placeholder="blurry, distorted face, bad anatomy, extra fingers...",
    )

    if st.button("GENERATE IMAGE", use_container_width=True, type="primary"):
        if not prompt.strip():
            st.warning("Önce bir prompt gir.")
        else:
            final_prompt = build_image_prompt(
                prompt, style, lighting, camera, quality, color_mood, negative_prompt
            )

            if ai_boost:
                with st.spinner("Prompt Robotu profesyonel prompt hazırlıyor..."):
                    final_prompt = local_prompt_robot(
                        prompt, style, lighting, camera, quality, color_mood, negative_prompt
                    )
                if not final_prompt:
                    st.error("Prompt Robotu boş bir prompt oluşturdu.")
                    st.stop()
                st.session_state.last_enhanced_prompt = final_prompt

            if not LOCAL_BACKEND_URL:
                st.warning("Tulpar backend adresi tanımlanmamış.")
                st.stop()

            progress = st.progress(0.0, text="%0 — Tulpar işi başlatılıyor...")
            status_box = st.empty()
            with st.spinner("Qwen Image / Tulpar görsel oluşturuyor..."):
                image, error = generate_image(
                    final_prompt, width, height, seed=seed,
                    progress_bar=progress, status_box=status_box,
                )

            if error:
                st.error("Görsel üretilemedi.")
                st.code(error)
            elif image:
                st.image(image, use_container_width=True)
                image_buffer = io.BytesIO()
                image.save(image_buffer, format="PNG")
                image_bytes = image_buffer.getvalue()
                st.download_button(
                    "PNG İndir", data=image_bytes,
                    file_name="kogce_ai_image.png", mime="image/png",
                    use_container_width=True,
                )
                st.session_state.gallery.append({
                    "image": image_bytes,
                    "prompt": prompt,
                    "final_prompt": final_prompt,
                })
                st.session_state.last_prompt = prompt
                st.success("Görsel başarıyla üretildi.")

# ============================================================
# CHARACTER PAGE
# ============================================================

elif page == "character":
    st.markdown("<div class='k-section-title'>Character Studio</div>", unsafe_allow_html=True)
    st.markdown("<div class='k-section-sub'>Referans görseli yükle; identity, yüz, vücut ve kıyafet korumasını kontrol et.</div>", unsafe_allow_html=True)

    character_file = st.file_uploader(
        "Referans fotoğraf", type=["png", "jpg", "jpeg", "webp"], key="character_upload_new"
    )
    if character_file:
        character_image = Image.open(character_file).convert("RGB")
        st.image(character_image, caption="Referans karakter", width=420)

        c1, c2, c3 = st.columns(3)
        with c1:
            preserve_face = st.checkbox("Yüzü koru", value=True)
        with c2:
            preserve_body = st.checkbox("Vücudu koru", value=True)
        with c3:
            preserve_clothes = st.checkbox("Kıyafeti koru", value=False)

        identity_strength = st.slider("Identity / Reference Gücü", 0.10, 1.00, 0.85, 0.05)
        character_style = st.selectbox("Stil", [
            "Photorealistic", "Cinematic", "Anime", "3D Render", "Stylized 3D",
            "Fantasy", "Fashion Editorial", "Dark Cinematic",
        ], key="character_style_new")
        instruction = st.text_area(
            "Ne değiştirmek istiyorsun?", height=180,
            placeholder="Aynı kişi kalsın. Yüzü ve vücut yapısı korunsun. Kıyafeti değiştir...",
        )

        if st.button("GENERATE CHARACTER", use_container_width=True, type="primary"):
            if not instruction.strip():
                st.warning("Önce yapılacak değişikliği yaz.")
            elif not LOCAL_BACKEND_URL:
                st.warning("Tulpar backend henüz bağlanmadı.")
            else:
                image_buffer = io.BytesIO()
                character_image.save(image_buffer, format="PNG")
                progress = st.progress(0.0, text="%0 — Tulpar işi başlatılıyor...")
                status_box = st.empty()
                with st.spinner("Karakter pipeline çalışıyor..."):
                    result, error = generate_character_image(
                        image_buffer.getvalue(), instruction, character_style,
                        identity_strength, preserve_face, preserve_body,
                        preserve_clothes, progress_bar=progress, status_box=status_box,
                    )
                if error:
                    st.error("Karakter üretilemedi.")
                    st.code(error)
                elif result:
                    st.image(result, use_container_width=True)
                    output = io.BytesIO()
                    result.save(output, format="PNG")
                    st.download_button(
                        "Karakter PNG İndir", data=output.getvalue(),
                        file_name="kogce_character.png", mime="image/png",
                        use_container_width=True,
                    )

# ============================================================
# VIDEO PAGE
# ============================================================

elif page == "video":
    st.markdown("<div class='k-section-title'>Video Studio</div>", unsafe_allow_html=True)
    st.markdown("<div class='k-section-sub'>Wan 2.1 T2V / I2V — üretim Tulpar / ComfyUI üzerinde çalışır.</div>", unsafe_allow_html=True)

    video_mode = st.radio("Mode", ["Text → Video", "Image → Video"], horizontal=True)

    if video_mode == "Text → Video":
        video_prompt = st.text_area(
            "Video Prompt", height=170,
            placeholder="A man walking naturally through a rainy street at night...",
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            video_style = st.selectbox("Style", ["Automatic", "Photorealistic", "Cinematic", "Anime", "3D", "Fantasy"], key="video_style_new")
        with c2:
            camera_motion = st.selectbox("Camera", ["Static camera", "Slow push in", "Slow pull out", "Tracking shot", "Pan", "Tilt", "Handheld"], key="camera_motion_new")
        with c3:
            motion_level = st.selectbox("Motion", ["Subtle", "Natural", "Dynamic"], key="motion_level_new")

        duration = st.selectbox("Duration", ["33 frame", "49 frame"], key="video_duration_new")
        num_frames = 33 if duration == "33 frame" else 49
        steps = st.slider("Inference Steps", 8, 30, 20, key="video_steps_new")

        if st.button("GENERATE VIDEO", use_container_width=True, type="primary"):
            if not video_prompt.strip():
                st.warning("Önce video promptu gir.")
            elif not LOCAL_BACKEND_URL:
                st.warning("Tulpar video backend henüz bağlanmadı.")
            else:
                final_video_prompt = (
                    f"{video_prompt.strip()}. Visual style: {video_style}. "
                    f"Camera movement: {camera_motion}. Motion intensity: {motion_level}. "
                    "Natural realistic motion and consistent subject appearance."
                )
                progress = st.progress(0.0, text="%0 — Tulpar işi başlatılıyor...")
                status_box = st.empty()
                with st.spinner("Wan 2.1 video oluşturuyor..."):
                    video, error = generate_text_video(
                        final_video_prompt, num_frames=num_frames, steps=steps,
                        progress_bar=progress, status_box=status_box,
                    )
                if error:
                    st.error("Video üretilemedi.")
                    st.code(error)
                elif video:
                    st.session_state.generated_video = video
                    st.session_state.video_filename = "kogce_text_to_video.mp4"
                    st.video(video)
                    st.download_button(
                        "MP4 İndir", data=video,
                        file_name=st.session_state.video_filename,
                        mime="video/mp4", use_container_width=True,
                    )
                    st.success("Video başarıyla üretildi.")

    else:
        uploaded_file = st.file_uploader(
            "Başlangıç görseli", type=["png", "jpg", "jpeg", "webp"], key="video_image_upload_new"
        )
        motion_prompt = st.text_area(
            "Motion Prompt", height=150,
            placeholder="The subject walks naturally, clothes move with the wind, slow cinematic tracking...",
        )
        c1, c2 = st.columns(2)
        with c1:
            video_duration = st.selectbox("Duration", ["33 frame", "49 frame"], key="i2v_duration_new")
        with c2:
            video_steps = st.slider("Steps", 8, 30, 20, key="i2v_steps_new")

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Source Image", width=520)
            if st.button("IMAGE → VIDEO", use_container_width=True, type="primary"):
                if not motion_prompt.strip():
                    st.warning("Hareket promptu gir.")
                elif not LOCAL_BACKEND_URL:
                    st.warning("Tulpar backend henüz bağlanmadı.")
                else:
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    frames = 33 if video_duration == "33 frame" else 49
                    progress = st.progress(0.0, text="%0 — Tulpar işi başlatılıyor...")
                    status_box = st.empty()
                    with st.spinner("Wan 2.1 görseli videoya dönüştürüyor..."):
                        video, error = generate_image_video(
                            buffer.getvalue(), motion_prompt,
                            num_frames=frames, steps=video_steps,
                            progress_bar=progress, status_box=status_box,
                        )
                    if error:
                        st.error("Image → Video üretilemedi.")
                        st.code(error)
                    elif video:
                        st.session_state.generated_video = video
                        st.session_state.video_filename = "kogce_image_to_video.mp4"
                        st.video(video)
                        st.download_button(
                            "MP4 İndir", data=video,
                            file_name=st.session_state.video_filename,
                            mime="video/mp4", use_container_width=True,
                        )

# ============================================================
# AI TOOLS PAGE
# ============================================================

elif page == "tools":
    st.markdown("<div class='k-section-title'>AI Tools</div>", unsafe_allow_html=True)
    st.markdown("<div class='k-section-sub'>Prompt, script, scene planning, video prompts and Shorts ideas.</div>", unsafe_allow_html=True)

    tool = st.selectbox("Tool", [
        "AI Prompt Robotu", "AI Senaryo Yazarı", "AI Sahne Planlayıcı",
        "AI Video Prompt Üretici", "AI Shorts Fikir Motoru",
    ])

    if tool == "AI Prompt Robotu":
        idea = st.text_area("Fikrin", height=150, placeholder="Kısa fikrini yaz...")
        if st.button("PROMPT OLUŞTUR", use_container_width=True, type="primary"):
            if not idea.strip():
                st.warning("Önce fikir gir.")
            else:
                with st.spinner("Profesyonel prompt hazırlanıyor..."):
                    result, error = call_ai(
                        """You are an expert professional image-generation prompt engineer.\nTransform the user's idea into an extremely detailed but coherent English image-generation prompt.\nInclude subject, appearance, environment, composition, camera, lens, lighting, materials, textures, colors, atmosphere, depth, realism and visual quality.\nPreserve the user's intended meaning. Do not invent major story elements. Return only the final prompt.""",
                        idea, temperature=0.75, max_tokens=1800,
                    )
                if error: st.error(error)
                else: st.text_area("Generated Prompt", value=result, height=330)

    elif tool == "AI Senaryo Yazarı":
        topic = st.text_area("Video konusu", height=150)
        duration = st.selectbox("Hedef süre", ["30 saniye", "60 saniye", "90 saniye", "3 dakika"])
        if st.button("SENARYO YAZ", use_container_width=True, type="primary"):
            if not topic.strip():
                st.warning("Konu gir.")
            else:
                with st.spinner("Senaryo hazırlanıyor..."):
                    result, error = call_ai(
                        """You are a professional short-form video script writer.\nCreate a complete engaging script.\nStructure: Hook, Setup, Development, Retention moments, Payoff, Ending.\nUse natural language and strong visual moments.""",
                        f"Topic:\n{topic}\n\nTarget duration:\n{duration}",
                        temperature=0.8, max_tokens=2500,
                    )
                if error: st.error(error)
                else:
                    st.session_state.last_script = result
                    st.text_area("Senaryo", value=result, height=450)

    elif tool == "AI Sahne Planlayıcı":
        script = st.text_area("Senaryoyu gir", height=250)
        if st.button("SAHNELERİ PLANLA", use_container_width=True, type="primary"):
            if not script.strip():
                st.warning("Önce senaryo gir.")
            else:
                with st.spinner("Sahne planı hazırlanıyor..."):
                    result, error = call_ai(
                        """You are a professional film director, storyboard artist and AI video production planner.\nBreak the script into logical visual scenes.\nFor every scene provide: scene number, duration, visual description, camera movement, character action, environment, lighting, audio/SFX and AI generation notes.\nMaintain character and visual consistency.""",
                        script, temperature=0.75, max_tokens=3000,
                    )
                if error: st.error(error)
                else:
                    st.session_state.last_scenes = result
                    st.text_area("Sahne Planı", value=result, height=550)

    elif tool == "AI Video Prompt Üretici":
        scene = st.text_area("Sahne fikri", height=180)
        if st.button("VIDEO PROMPT OLUŞTUR", use_container_width=True, type="primary"):
            if not scene.strip():
                st.warning("Sahne fikri gir.")
            else:
                with st.spinner("Video prompt hazırlanıyor..."):
                    result, error = call_ai(
                        """You are an expert AI video-generation prompt engineer.\nConvert the scene into a production-ready English video prompt.\nFocus on subject movement, camera movement, environment movement, lighting, realistic physics, cinematic composition, depth, timing, atmosphere and visual consistency.\nReturn only the final video prompt.""",
                        scene, temperature=0.75, max_tokens=1800,
                    )
                if error: st.error(error)
                else:
                    st.session_state.last_video_prompt = result
                    st.text_area("Video Prompt", value=result, height=350)

    else:
        niche = st.text_input("Niş / konu", placeholder="AI, gaming, animals, satisfying...")
        count = st.slider("Fikir sayısı", 5, 20, 10)
        if st.button("FİKİRLERİ ÜRET", use_container_width=True, type="primary"):
            if not niche.strip():
                st.warning("Bir niş gir.")
            else:
                with st.spinner("Shorts fikirleri hazırlanıyor..."):
                    result, error = call_ai(
                        """You are a global short-form video strategist.\nGenerate original YouTube Shorts concepts.\nPrioritize strong first-second hooks, visual simplicity, high retention, curiosity, replayability, international appeal and easy AI/video production.\nFor each idea provide title, hook, concept, retention mechanism and visual production idea. Avoid generic repetition.""",
                        f"Niche:\n{niche}\n\nNumber:\n{count}",
                        temperature=0.9, max_tokens=3500,
                    )
                if error: st.error(error)
                else: st.text_area("Shorts Fikirleri", value=result, height=600)

# ============================================================
# GALLERY PAGE
# ============================================================

elif page == "gallery":
    st.markdown("<div class='k-section-title'>Gallery</div>", unsafe_allow_html=True)
    st.markdown("<div class='k-section-sub'>Bu Streamlit oturumunda oluşturulan görseller.</div>", unsafe_allow_html=True)

    if not st.session_state.gallery:
        st.info("Henüz oluşturulmuş bir görsel yok.")
    else:
        # Leonardo-like compact gallery grid
        items = list(reversed(st.session_state.gallery))
        for start in range(0, len(items), 3):
            cols = st.columns(3)
            for col, item in zip(cols, items[start:start+3]):
                with col:
                    st.image(item["image"], use_container_width=True)
                    st.caption(item["prompt"])
                    with st.expander("Prompt"):
                        st.code(item["final_prompt"], language="text")

# ============================================================
# SYSTEM PAGE
# ============================================================

elif page == "system":
    st.markdown("<div class='k-section-title'>System</div>", unsafe_allow_html=True)
    st.markdown("<div class='k-section-sub'>KOGCE üretim motorlarının bağlantı durumu.</div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("HF API", "ONLINE" if HF_API_KEY else "MISSING")
    with col2:
        st.metric("Image", "QWEN 2.1")
    with col3:
        st.metric("Tulpar", "ONLINE" if local_backend_available() else "OFFLINE")
    with col4:
        st.metric("Gallery", len(st.session_state.gallery))

    st.divider()
    st.write("**Aktif Görsel Modeli**")
    st.code("Comfy-Org/Qwen-Image-2.1 • Tulpar / ComfyUI")
    st.write("**Prompt Modeli**")
    st.code(PROMPT_MODEL)
    st.write("**Tulpar Backend**")
    st.code(LOCAL_BACKEND_URL or "LOCAL_BACKEND_URL tanımlanmadı")
    st.write("**Oturum**")
    st.write(f"Galerideki görsel sayısı: **{len(st.session_state.gallery)}**")
    if st.session_state.generated_video:
        st.success("Son video üretimi mevcut.")
    else:
        st.info("Bu oturumda başarılı video üretimi yok.")

st.caption("KOGCE AI Studio • Local / Private Creative Workspace")

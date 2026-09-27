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
# GLOBAL STYLE
# ============================================================
# UI ONLY — üretim motorlarına dokunulmaz.
# ============================================================
st.markdown("""
<style>
:root { --bg:#0b0c0f; --panel:#121419; --panel2:#17191f; --line:#252832; --text:#f4f4f5; --muted:#8e939f; --accent:#8b5cf6; --accent2:#a78bfa; }
.stApp { background:var(--bg); color:var(--text); }
.block-container { max-width:1500px; padding:1.25rem 1.5rem 3rem; }
header[data-testid="stHeader"] { background:rgba(11,12,15,.88); }
section[data-testid="stSidebar"] { background:#0e1014; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] > div { padding:1rem .8rem; }
section[data-testid="stSidebar"] .stButton > button { justify-content:flex-start; text-align:left; }
.stButton > button { min-height:40px; border-radius:9px; border:1px solid #2a2d36; background:#17191f; color:#f4f4f5 !important; font-weight:650; box-shadow:none; }
.stButton > button:hover { border-color:#4b5060; background:#1c1f27; transform:none; box-shadow:none; }
.stButton > button[kind="primary"] { background:#8b5cf6; border-color:#8b5cf6; color:white !important; }
.stButton > button[kind="primary"]:hover { background:#7c3aed; border-color:#7c3aed; }
.stDownloadButton > button { min-height:40px; border-radius:9px; background:#17191f; border:1px solid #2a2d36; color:#f4f4f5 !important; }
textarea, input { border-radius:9px !important; }
textarea { background:#111318 !important; color:#f4f4f5 !important; border:1px solid #292d36 !important; }
input { background:#111318 !important; color:#f4f4f5 !important; border:1px solid #292d36 !important; }
textarea:focus, input:focus { border-color:#7654cf !important; box-shadow:0 0 0 1px #7654cf !important; }
textarea::placeholder,input::placeholder { color:#666b76 !important; }
[data-baseweb="select"] > div { background:#111318; border-color:#292d36; border-radius:9px; }
[data-baseweb="select"] span { color:#e7e8eb; }
.stSelectbox label,.stTextInput label,.stTextArea label,.stSlider label,.stFileUploader label,.stRadio label,.stCheckbox label,.stToggle label { color:#aeb3bd !important; font-size:.78rem; font-weight:650 !important; }
.stTabs [data-baseweb="tab-list"] { gap:4px; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { color:#8e939f; padding:.65rem .9rem; }
.stTabs [aria-selected="true"] { color:white !important; border-bottom:2px solid var(--accent); }
[data-testid="stImage"] img { border-radius:10px; border:1px solid #272a33; }
[data-testid="stMetric"] { background:#121419; border:1px solid #252832; border-radius:10px; padding:.75rem; }
[data-testid="stMetricValue"] { color:#fff !important; }
[data-testid="stMetricLabel"] { color:#8e939f !important; }
.stAlert { border-radius:9px; }
hr { border-color:#252832; }
.kogce-topbar { display:flex; align-items:center; justify-content:space-between; margin-bottom:1rem; padding-bottom:.8rem; border-bottom:1px solid #20232b; }
.kogce-brand { font-size:1.35rem; font-weight:800; letter-spacing:-.04em; }
.kogce-brand span { color:#a78bfa; }
.kogce-small { color:#777d89; font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; }
.kogce-section { font-size:1.35rem; font-weight:750; letter-spacing:-.03em; margin:.25rem 0 .2rem; }
.kogce-muted { color:#8e939f; font-size:.84rem; }
.kogce-panel { background:#121419; border:1px solid #252832; border-radius:12px; padding:1rem; }
.kogce-panel-title { font-weight:700; color:#f2f3f5; margin-bottom:.65rem; }
.kogce-chip { display:inline-block; padding:.2rem .5rem; border-radius:999px; background:#1a1d24; border:1px solid #2a2d36; color:#9fa5b0; font-size:.7rem; margin-right:.3rem; }
.kogce-gallery-card { background:#111318; border:1px solid #252832; border-radius:10px; padding:.45rem; }
.kogce-spacer { height:.5rem; }
</style>
""", unsafe_allow_html=True)


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


# ============================================================
# KOGCE UI — LEONARDO-INSPIRED WORKSPACE
# ============================================================

if not st.session_state.authenticated:
    st.markdown("## KOGCE AI Studio")
    st.caption("Private creative workspace")
    _, login_col, _ = st.columns([1, 1.1, 1])
    with login_col:
        st.markdown("### Studio'ya giriş")
        password = st.text_input("Şifre", type="password", placeholder="Şifrenizi girin", key="login_password")
        if st.button("Giriş yap", type="primary", use_container_width=True):
            if password == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Hatalı şifre.")
    st.stop()

# ---------- navigation ----------
if "active_page" not in st.session_state:
    st.session_state.active_page = "Görsel"

with st.sidebar:
    st.markdown("## KOGCE")
    st.caption("AI CREATIVE STUDIO")
    st.divider()
    st.caption("CREATE")
    for label in ["Görsel", "Karakter", "Video"]:
        if st.button(label, key=f"nav_{label}", use_container_width=True, type="primary" if st.session_state.active_page == label else "secondary"):
            st.session_state.active_page = label
            st.rerun()
    st.caption("TOOLS")
    for label in ["AI Araçları", "Galeri"]:
        if st.button(label, key=f"nav_{label}", use_container_width=True, type="primary" if st.session_state.active_page == label else "secondary"):
            st.session_state.active_page = label
            st.rerun()
    st.caption("SYSTEM")
    if st.button("Sistem", key="nav_system", use_container_width=True, type="primary" if st.session_state.active_page == "Sistem" else "secondary"):
        st.session_state.active_page = "Sistem"
        st.rerun()
    st.divider()
    backend_online = local_backend_available() if LOCAL_BACKEND_URL else False
    st.caption("LOCAL ENGINE")
    st.write("● Tulpar ONLINE" if backend_online else "○ Tulpar OFFLINE")
    st.caption(f"{len(st.session_state.gallery)} görsel")
    st.divider()
    if st.button("Çıkış", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

st.markdown('<div class="kogce-topbar"><div class="kogce-brand">KOGCE <span>AI Studio</span></div><div class="kogce-small">Local creative workspace</div></div>', unsafe_allow_html=True)

# ============================================================
# IMAGE WORKSPACE
# ============================================================
if st.session_state.active_page == "Görsel":
    st.markdown('<div class="kogce-section">Image Generation</div>', unsafe_allow_html=True)
    st.caption("Describe an image, configure the generation settings, then send it to Tulpar / ComfyUI.")
    left, right = st.columns([1.65, 1], gap="large")

    with left:
        st.markdown('<div class="kogce-panel-title">Canvas</div>', unsafe_allow_html=True)
        if st.session_state.gallery:
            latest = st.session_state.gallery[-1]["image"]
            st.image(latest, use_container_width=True)
        else:
            st.markdown('<div class="kogce-panel" style="height:520px;display:flex;align-items:center;justify-content:center;color:#666b76;">Generated images will appear here</div>', unsafe_allow_html=True)
        if st.session_state.last_enhanced_prompt:
            with st.expander("Son final prompt"):
                st.code(st.session_state.last_enhanced_prompt, language="text")

    with right:
        st.markdown('<div class="kogce-panel-title">Prompt</div>', unsafe_allow_html=True)
        prompt = st.text_area("Prompt", placeholder="Describe the image you want to create...", height=180, label_visibility="collapsed", key="image_prompt")
        negative_prompt = st.text_input("Negative prompt", placeholder="blurry, distorted, bad anatomy...", key="image_negative")
        st.markdown("### Generation")
        c1, c2 = st.columns(2)
        with c1:
            style = st.selectbox("Style", ["Automatic","Photorealistic","Cinematic","Anime","3D Render","Stylized 3D","Cartoon","Fantasy","Cyberpunk","Product Photography","Fashion Editorial","Dark Cinematic"], key="image_style")
            lighting = st.selectbox("Lighting", ["Automatic","Natural daylight","Cinematic lighting","Soft studio lighting","Golden hour","Blue hour","Neon lighting","Dramatic lighting","Low-key lighting","Volumetric lighting","Rainy night lighting"], key="image_lighting")
            quality = st.selectbox("Quality", ["Automatic","High detail","Ultra detailed","Photographic realism","Cinematic quality","Sharp professional image"], key="image_quality")
        with c2:
            camera = st.selectbox("Camera", ["Automatic","Close-up portrait","Medium shot","Full body shot","Wide cinematic shot","Low angle","High angle","Eye level","Aerial perspective","Over-the-shoulder"], key="image_camera")
            aspect_ratio = st.selectbox("Aspect ratio", ["1:1","9:16","16:9"], key="image_ratio")
            color_mood = st.selectbox("Color / mood", ["Automatic","Natural colors","Dark moody","Warm cinematic","Cool cinematic","Neon futuristic","Muted realistic","High contrast"], key="image_mood")
        resolutions = {"1:1":(768,768), "9:16":(768,1344), "16:9":(1344,768)}
        width, height = resolutions[aspect_ratio]
        a,b = st.columns(2)
        with a:
            ai_boost = st.toggle("Prompt Robotu", value=True, key="image_ai_boost")
        with b:
            seed_mode = st.selectbox("Seed", ["Random","Fixed"], key="image_seed_mode")
        seed = None
        if seed_mode == "Fixed":
            seed = st.number_input("Seed", min_value=0, max_value=999999999, value=123456, step=1, key="image_seed")
        st.caption(f"Output: {width} × {height}")
        if st.button("Generate image", type="primary", use_container_width=True, key="generate_image_btn"):
            if not prompt.strip():
                st.warning("Önce prompt gir.")
            elif not LOCAL_BACKEND_URL:
                st.warning("Tulpar backend adresi tanımlanmamış.")
            else:
                final_prompt = build_image_prompt(prompt, style, lighting, camera, quality, color_mood, negative_prompt)
                if ai_boost:
                    with st.spinner("Prompt hazırlanıyor..."):
                        final_prompt = local_prompt_robot(prompt, style, lighting, camera, quality, color_mood, negative_prompt)
                    st.session_state.last_enhanced_prompt = final_prompt
                progress = st.progress(0.0, text="Starting...")
                status_box = st.empty()
                with st.spinner("Tulpar görsel oluşturuyor..."):
                    image, error = generate_image(final_prompt, width, height, seed=seed, progress_bar=progress, status_box=status_box)
                if error:
                    st.error("Görsel üretilemedi.")
                    st.code(error)
                elif image:
                    buf = io.BytesIO(); image.save(buf, format="PNG"); image_bytes = buf.getvalue()
                    st.session_state.gallery.append({"image":image_bytes,"prompt":prompt,"final_prompt":final_prompt})
                    st.session_state.last_prompt = prompt
                    st.rerun()

# ============================================================
# CHARACTER
# ============================================================
elif st.session_state.active_page == "Karakter":
    st.markdown('<div class="kogce-section">Character Studio</div>', unsafe_allow_html=True)
    st.caption("Reference image editing and identity preservation.")
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        uploaded = st.file_uploader("Reference image", type=["png","jpg","jpeg","webp"], key="character_upload_new")
        if uploaded:
            character_image = Image.open(uploaded).convert("RGB")
            st.image(character_image, use_container_width=True)
        else:
            st.markdown('<div class="kogce-panel" style="height:420px;display:flex;align-items:center;justify-content:center;color:#666b76;">Upload a reference image</div>', unsafe_allow_html=True)
    with right:
        instruction = st.text_area("Edit instruction", height=190, placeholder="Keep the same person. Change the clothes, hairstyle and environment...", key="character_instruction_new")
        character_style = st.selectbox("Style", ["Photorealistic","Cinematic","Anime","3D Render","Stylized 3D","Fantasy","Fashion Editorial","Dark Cinematic"], key="character_style_new")
        identity_strength = st.slider("Reference strength", .10, 1.00, .85, .05, key="identity_strength_new")
        c1,c2,c3=st.columns(3)
        with c1: preserve_face=st.checkbox("Face", True, key="preserve_face_new")
        with c2: preserve_body=st.checkbox("Body", True, key="preserve_body_new")
        with c3: preserve_clothes=st.checkbox("Clothes", False, key="preserve_clothes_new")
        if st.button("Generate character", type="primary", use_container_width=True, key="generate_character_new"):
            if not uploaded: st.warning("Referans görsel yükle.")
            elif not instruction.strip(): st.warning("Değişikliği yaz.")
            elif not LOCAL_BACKEND_URL: st.warning("Tulpar backend adresi tanımlanmamış.")
            else:
                buf=io.BytesIO(); character_image.save(buf,format="PNG")
                progress=st.progress(0.0,text="Starting..."); status_box=st.empty()
                with st.spinner("Character pipeline çalışıyor..."):
                    result,error=generate_character_image(buf.getvalue(),instruction,character_style,identity_strength,preserve_face,preserve_body,preserve_clothes,progress,status_box)
                if error: st.error(error)
                elif result:
                    st.image(result,use_container_width=True)
                    out=io.BytesIO(); result.save(out,format="PNG")
                    st.download_button("Download PNG",out.getvalue(),"kogce_character.png","image/png",use_container_width=True)

# ============================================================
# VIDEO
# ============================================================
elif st.session_state.active_page == "Video":
    st.markdown('<div class="kogce-section">Video Studio</div>', unsafe_allow_html=True)
    st.caption("Wan 2.1 video tools running through the local Tulpar / ComfyUI pipeline.")
    mode=st.radio("Mode",["Text → Video","Image → Video","Long Video"],horizontal=True,key="video_mode_new")
    if mode == "Text → Video":
        left,right=st.columns([1.5,1],gap="large")
        with left:
            video_prompt=st.text_area("Video prompt",height=210,placeholder="A man walks naturally through a rainy street...",key="video_prompt_new")
        with right:
            video_style=st.selectbox("Style",["Automatic","Photorealistic","Cinematic","Anime","3D","Fantasy"],key="video_style_new")
            camera_motion=st.selectbox("Camera",["Static camera","Slow push in","Slow pull out","Tracking shot","Pan","Tilt","Handheld"],key="video_camera_new")
            motion_level=st.selectbox("Motion",["Subtle","Natural","Dynamic"],key="video_motion_new")
            duration=st.selectbox("Duration",["33 frame","49 frame"],key="video_duration_new")
            steps=st.slider("Inference steps",8,30,20,key="video_steps_new")
        if st.button("Generate video",type="primary",use_container_width=True,key="generate_t2v_new"):
            if not video_prompt.strip(): st.warning("Video promptu gir.")
            elif not LOCAL_BACKEND_URL: st.warning("Tulpar backend adresi tanımlanmamış.")
            else:
                frames=33 if duration.startswith("33") else 49
                final=f"{video_prompt.strip()}. Visual style: {video_style}. Camera movement: {camera_motion}. Motion intensity: {motion_level}. Natural realistic motion and consistent subject appearance."
                progress=st.progress(0.0,text="Starting..."); status_box=st.empty()
                with st.spinner("Wan 2.1 video oluşturuyor..."):
                    video,error=generate_text_video(final,frames,steps,progress,status_box)
                if error: st.error(error); st.code(error)
                elif video:
                    st.session_state.generated_video=video; st.session_state.video_filename="kogce_text_to_video.mp4"; st.video(video)
                    st.download_button("Download MP4",video,st.session_state.video_filename,"video/mp4",use_container_width=True)
    elif mode == "Image → Video":
        left,right=st.columns([1.2,1],gap="large")
        with left:
            uploaded_file=st.file_uploader("Starting image",type=["png","jpg","jpeg","webp"],key="video_image_upload_new")
            if uploaded_file:
                image=Image.open(uploaded_file).convert("RGB"); st.image(image,use_container_width=True)
        with right:
            motion_prompt=st.text_area("Motion prompt",height=180,placeholder="The subject moves naturally...",key="motion_prompt_new")
            video_duration=st.selectbox("Duration",["33 frame","49 frame"],key="i2v_duration_new")
            video_steps=st.slider("Steps",8,30,20,key="i2v_steps_new")
            if st.button("Generate image → video",type="primary",use_container_width=True,key="generate_i2v_new"):
                if not uploaded_file: st.warning("Başlangıç görseli yükle.")
                elif not motion_prompt.strip(): st.warning("Motion prompt gir.")
                elif not LOCAL_BACKEND_URL: st.warning("Tulpar backend adresi tanımlanmamış.")
                else:
                    buf=io.BytesIO(); image.save(buf,format="PNG"); frames=33 if video_duration.startswith("33") else 49
                    progress=st.progress(0.0,text="Starting..."); status_box=st.empty()
                    with st.spinner("Wan 2.1 image-to-video çalışıyor..."):
                        video,error=generate_image_video(buf.getvalue(),motion_prompt,frames,video_steps,progress,status_box)
                    if error: st.error(error); st.code(error)
                    elif video:
                        st.session_state.generated_video=video; st.session_state.video_filename="kogce_image_to_video.mp4"; st.video(video)
                        st.download_button("Download MP4",video,st.session_state.video_filename,"video/mp4",use_container_width=True)
    else:
        st.markdown("### AI Long Video")
        story=st.text_area("Story / master prompt",height=180,placeholder="Describe the complete story and its sequence...",key="long_story_new")
        c1,c2,c3=st.columns(3)
        with c1: total_seconds=st.selectbox("Total duration",list(range(1,61)),index=19,format_func=lambda x:f"{x} seconds",key="long_total_new")
        with c2: clip_seconds=st.selectbox("Clip duration",[1,2,3],index=1,format_func=lambda x:f"{x} seconds",key="long_clip_new")
        with c3: long_steps=st.slider("Steps",8,30,20,key="long_steps_new")
        if total_seconds % clip_seconds != 0: st.error("Toplam süre klip süresinin tam katı olmalı.")
        else:
            c1,c2,c3=st.columns(3)
            with c1: long_style=st.selectbox("Style",["Automatic","Photorealistic","Cinematic","3D Cartoon","2D Animation","Anime","Fantasy","Claymation","Stop Motion","Comic / Stylized","Pixel Art"],key="long_style_new")
            with c2: long_camera=st.selectbox("Camera",["Automatic","Static camera","Slow push in","Slow pull out","Tracking shot","Pan","Tilt","Handheld","Dolly"],key="long_camera_new")
            with c3: long_motion=st.selectbox("Motion",["Subtle","Natural","Dynamic"],index=1,key="long_motion_new")
            c1,c2=st.columns(2)
            with c1: long_ratio=st.selectbox("Aspect ratio",["9:16 — Shorts / Reels / TikTok","16:9 — YouTube","1:1 — Square","4:5 — Portrait"],key="long_ratio_new")
            with c2: long_size=st.selectbox("Final size",["1080×1920","720×1280","1080×1080","1080×1350","1920×1080","720×720"],key="long_size_new")
            if st.button("Generate long video",type="primary",use_container_width=True,key="generate_long_new"):
                if not story.strip(): st.warning("Hikâyeyi gir.")
                elif not LOCAL_BACKEND_URL: st.warning("Tulpar backend adresi tanımlanmamış.")
                else:
                    with st.spinner("AI video planı hazırlanıyor..."):
                        video,error=generate_long_video(story,total_seconds,clip_seconds,long_style,long_camera,long_motion,long_ratio,long_size,long_steps)
                    if error: st.error(error); st.code(error)
                    elif video:
                        st.session_state.generated_video=video; st.session_state.video_filename=f"kogce_long_{total_seconds}s.mp4"; st.video(video)
                        st.download_button("Download MP4",video,st.session_state.video_filename,"video/mp4",use_container_width=True)

# ============================================================
# AI TOOLS
# ============================================================
elif st.session_state.active_page == "AI Araçları":
    st.markdown('<div class="kogce-section">AI Production Tools</div>', unsafe_allow_html=True)
    tool=st.selectbox("Tool",["AI Prompt Robotu","AI Senaryo Yazarı","AI Sahne Planlayıcı","AI Video Prompt Üretici","AI Shorts Fikir Motoru"],key="ai_tool_new")
    if tool == "AI Prompt Robotu":
        idea=st.text_area("Idea",height=180,placeholder="Kısa fikrini yaz...",key="tool_idea_new")
        if st.button("Create prompt",type="primary",key="tool_prompt_btn"):
            result,error=call_ai("""You are an expert professional image-generation prompt engineer. Transform the user's idea into an extremely detailed but coherent English image-generation prompt. Preserve the intended meaning. Return only the final prompt.""",idea,temperature=.75,max_tokens=1800) if idea.strip() else (None,"Önce fikir gir.")
            if error: st.error(error)
            else: st.text_area("Generated prompt",value=result,height=350,key="tool_prompt_result")
    elif tool == "AI Senaryo Yazarı":
        topic=st.text_area("Video konusu",height=160,key="script_topic_new"); duration=st.selectbox("Target duration",["30 saniye","60 saniye","90 saniye","3 dakika"],key="script_duration_new")
        if st.button("Create script",type="primary",key="script_btn_new"):
            result,error=call_ai("""You are a professional short-form video script writer. Create a complete engaging script with Hook, Setup, Development, Retention moments, Payoff and Ending. Use natural language and strong visual moments.""",f"Topic: {topic}\nTarget duration: {duration}",temperature=.8,max_tokens=2500) if topic.strip() else (None,"Konu gir.")
            if error: st.error(error)
            else: st.session_state.last_script=result; st.text_area("Script",value=result,height=450,key="script_result_new")
    elif tool == "AI Sahne Planlayıcı":
        script=st.text_area("Script",height=250,key="scene_script_new")
        if st.button("Plan scenes",type="primary",key="scene_btn_new"):
            result,error=call_ai("""You are a professional film director, storyboard artist and AI video production planner. Break the script into logical visual scenes. For every scene provide number, duration, visual description, camera movement, character action, environment, lighting, audio/SFX and AI generation notes. Maintain consistency.""",script,temperature=.75,max_tokens=3000) if script.strip() else (None,"Senaryo gir.")
            if error: st.error(error)
            else: st.session_state.last_scenes=result; st.text_area("Scene plan",value=result,height=550,key="scene_result_new")
    elif tool == "AI Video Prompt Üretici":
        scene=st.text_area("Scene idea",height=180,key="video_tool_scene_new")
        if st.button("Create video prompt",type="primary",key="video_tool_btn_new"):
            result,error=call_ai("""You are an expert AI video-generation prompt engineer. Convert the scene into a production-ready English video prompt. Focus on subject movement, camera movement, environment movement, lighting, realistic physics, cinematic composition, depth, timing, atmosphere and consistency. Return only the final prompt.""",scene,temperature=.75,max_tokens=1800) if scene.strip() else (None,"Sahne fikri gir.")
            if error: st.error(error)
            else: st.session_state.last_video_prompt=result; st.text_area("Video prompt",value=result,height=350,key="video_tool_result_new")
    else:
        niche=st.text_input("Niche / topic",placeholder="AI, gaming, animals, satisfying...",key="shorts_niche_new"); count=st.slider("Idea count",5,20,10,key="shorts_count_new")
        if st.button("Generate ideas",type="primary",key="shorts_btn_new"):
            result,error=call_ai("""You are a global short-form video strategist. Generate original YouTube Shorts concepts with strong first-second hooks, visual simplicity, curiosity, replayability, international appeal and easy AI/video production. For each: title, hook, concept, retention mechanism, visual production idea.""",f"Niche: {niche}\nNumber: {count}",temperature=.9,max_tokens=3500) if niche.strip() else (None,"Bir niş gir.")
            if error: st.error(error)
            else: st.text_area("Shorts ideas",value=result,height=600,key="shorts_result_new")

# ============================================================
# GALLERY
# ============================================================
elif st.session_state.active_page == "Galeri":
    st.markdown('<div class="kogce-section">Gallery</div>', unsafe_allow_html=True)
    st.caption("Current Streamlit session outputs.")
    if not st.session_state.gallery:
        st.info("Henüz görsel yok.")
    else:
        items=list(reversed(st.session_state.gallery))
        for start in range(0,len(items),4):
            cols=st.columns(4,gap="small")
            for col,item in zip(cols,items[start:start+4]):
                with col:
                    st.image(item["image"],use_container_width=True)
                    with st.expander("Details"):
                        st.caption(item["prompt"])
                        st.code(item["final_prompt"],language="text")

# ============================================================
# SYSTEM
# ============================================================
else:
    st.markdown('<div class="kogce-section">System</div>', unsafe_allow_html=True)
    st.caption("Runtime status and active model information.")
    c1,c2,c3,c4=st.columns(4)
    with c1: st.metric("Prompt AI","ONLINE" if HF_API_KEY else "OFFLINE")
    with c2: st.metric("Image","Qwen Image 2.1")
    with c3: st.metric("Tulpar","ONLINE" if local_backend_available() else "OFFLINE")
    with c4: st.metric("Gallery",len(st.session_state.gallery))
    st.divider()
    st.markdown("### Active image model")
    st.code("Comfy-Org/Qwen-Image-2.1 • Tulpar / ComfyUI")
    st.markdown("### Prompt model")
    st.code(PROMPT_MODEL)
    st.markdown("### Tulpar backend")
    st.code(LOCAL_BACKEND_URL or "Not configured")
    st.markdown("### Session")
    st.write(f"Gallery: **{len(st.session_state.gallery)}** images")
    st.write("Latest video: **available**" if st.session_state.generated_video else "Latest video: **none**")

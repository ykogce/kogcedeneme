import io
import os
import re
import json
import time
import base64
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import requests
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

APP_PASSWORD = "1234"

# Local Tulpar / ComfyUI only.
# If your Tulpar backend uses another local port, change only this line.
LOCAL_BACKEND_URL = os.environ.get(
    "KOGCE_TULPAR_URL",
    "http://127.0.0.1:8000",
).strip().rstrip("/")

GENERATION_TIMEOUT = 1800  # 30 minutes

HOST = "127.0.0.1"
PORT = 7860

gallery = []
last_generated_video = None
last_video_filename = None


# ============================================================
# LOCAL PROMPT ROBOT
# ============================================================

def clean_prompt(value):
    if value is None:
        return ""
    return " ".join(
        str(value).replace("\x00", " ").split()
    ).strip()


def local_prompt_robot(
    prompt,
    style,
    lighting,
    camera,
    quality,
    color_mood,
    negative_prompt,
):
    """
    Existing local prompt robot from the supplied app.
    No Hugging Face, no external AI API.
    """
    prompt = clean_prompt(prompt)

    if not prompt:
        return ""

    phrase_replacements = {
        "zombilerden kaçan sevimli gri kedi":
            "a cute gray cat running away from zombies",
        "zombilerden kaçan sevimli gri cat":
            "a cute gray cat running away from zombies",
        "zombilerden kaçan":
            "running away from zombies",
        "zombilerden kaçıyor":
            "running away from zombies",
        "zombiler":
            "zombies",
        "zombi":
            "zombie",
        "sevimli gri kedi":
            "a cute gray cat",
        "gri kedi":
            "a gray cat",
        "sevimli kedi":
            "a cute cat",
        "sevimli köpek":
            "a cute dog",
        "kaçan":
            "running away",
        "kaçıyor":
            "is running away",
        "koşuyor":
            "is running",
        "koşan":
            "running",
        "yürüyor":
            "is walking",
        "yürüyen":
            "walking",
        "oturuyor":
            "is sitting",
        "oturan":
            "sitting",
        "ayakta duran":
            "standing",
        "ayakta":
            "standing",
        "sevimli":
            "cute",
        "gri":
            "gray",
        "kedi":
            "cat",
        "köpek":
            "dog",
        "araba":
            "car",
        "otomobil":
            "car",
        "kırmızı":
            "red",
        "mavi":
            "blue",
        "siyah":
            "black",
        "beyaz":
            "white",
        "yeşil":
            "green",
        "sarı":
            "yellow",
        "gerçekçi":
            "photorealistic",
        "fotogerçekçi":
            "photorealistic",
        "sinematik":
            "cinematic",
        "detaylı":
            "highly detailed",
        "çok detaylı":
            "highly detailed",
        "gece":
            "at night",
        "gündüz":
            "during daytime",
        "yağmur":
            "rainy",
        "karlı":
            "snowy",
        "kar":
            "snow",
        "deniz":
            "sea",
        "sahil":
            "seaside",
        "plaj":
            "beach",
        "şehir":
            "city",
        "sokak":
            "street",
        "kadın":
            "woman",
        "adam":
            "man",
        "erkek":
            "man",
        "çocuk":
            "child",
        "mutlu":
            "happy",
        "üzgün":
            "sad",
        "gülümseyen":
            "smiling",
    }

    translated = prompt

    for tr, en in sorted(
        phrase_replacements.items(),
        key=lambda x: len(x[0]),
        reverse=True,
    ):
        translated = re.sub(
            rf"(?<!\w){re.escape(tr)}(?!\w)",
            en,
            translated,
            flags=re.IGNORECASE,
        )

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
        "Professional commercial image, coherent composition, "
        "accurate perspective, clear subject hierarchy, natural depth, "
        "physically plausible lighting, realistic materials and textures, "
        "consistent subject details, cinematic visual quality, "
        "high visual fidelity."
    )

    negative = clean_prompt(negative_prompt)

    if negative:
        parts.append(f"Avoid: {negative}.")

    return clean_prompt(" ".join(parts))


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
        parts.append(f"Visual style: {style}.")

    if lighting != "Automatic":
        parts.append(f"Lighting: {lighting}.")

    if camera != "Automatic":
        parts.append(f"Camera and composition: {camera}.")

    if quality != "Automatic":
        parts.append(f"Image quality: {quality}.")

    if color_mood != "Automatic":
        parts.append(f"Color and mood: {color_mood}.")

    if negative_prompt.strip():
        parts.append(
            "Avoid: " +
            negative_prompt.strip() +
            "."
        )

    return " ".join(parts)


# ============================================================
# TULPAR / COMFYUI BACKEND
# ============================================================

def local_backend_available():
    if not LOCAL_BACKEND_URL:
        return False

    try:
        response = requests.get(
            f"{LOCAL_BACKEND_URL}/health",
            timeout=5,
        )
        return response.status_code == 200
    except Exception:
        return False


def local_backend_request(
    endpoint,
    payload=None,
    files=None,
    timeout=30,
    form=False,
):
    """
    Short-lived request to Tulpar.
    GPU work itself is handled through job_id + polling.
    """
    if not LOCAL_BACKEND_URL:
        return None, "Tulpar backend adresi tanımlanmamış."

    try:
        url = LOCAL_BACKEND_URL + endpoint

        if files:
            response = requests.post(
                url,
                data=payload or {},
                files=files,
                timeout=timeout,
            )
        elif form:
            response = requests.post(
                url,
                data=payload or {},
                timeout=timeout,
            )
        else:
            response = requests.post(
                url,
                json=payload or {},
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
        return (
            None,
            "Tulpar backend başlangıç isteği zaman aşımına uğradı.",
        )

    except requests.exceptions.RequestException as exc:
        return None, f"Tulpar bağlantı hatası: {exc}"

    except Exception as exc:
        return None, str(exc)


def poll_backend_job(job_id, timeout=GENERATION_TIMEOUT):
    if not LOCAL_BACKEND_URL:
        return None, "Tulpar backend adresi tanımlanmamış."

    started = time.time()

    while True:
        if time.time() - started > timeout:
            return (
                None,
                "Tulpar üretimi izin verilen maksimum süreyi aştı.",
            )

        try:
            response = requests.get(
                f"{LOCAL_BACKEND_URL}/progress/{job_id}",
                timeout=15,
            )

            if response.status_code != 200:
                return (
                    None,
                    f"Tulpar job durumu HTTP "
                    f"{response.status_code}\n"
                    f"{response.text[:2000]}",
                )

            data = response.json()
            status = str(data.get("status", ""))

            if status == "completed":
                return data, None

            if status == "error":
                return (
                    None,
                    data.get("error")
                    or data.get("message")
                    or "Tulpar üretim hatası.",
                )

        except requests.exceptions.RequestException:
            # A temporary polling failure does not cancel the GPU job.
            pass

        time.sleep(1.0)


def download_backend_result(data):
    download_url = data.get("download_url")

    if not download_url:
        return (
            None,
            "Tulpar tamamlandı ancak download_url döndürmedi.",
        )

    if download_url.startswith("/"):
        download_url = LOCAL_BACKEND_URL + download_url

    try:
        response = requests.get(
            download_url,
            timeout=60,
        )
        response.raise_for_status()

        if not response.content:
            return (
                None,
                "Tulpar boş bir çıktı dosyası döndürdü.",
            )

        return response.content, None

    except requests.exceptions.RequestException as exc:
        return (
            None,
            f"Tulpar çıktı dosyası alınamadı: {exc}",
        )


# ============================================================
# IMAGE GENERATION — TULPAR / QWEN IMAGE 2.1
# ============================================================

def generate_image(
    prompt,
    width,
    height,
    seed=None,
):
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
    except Exception as exc:
        return (
            None,
            f"Tulpar başlangıç cevabı okunamadı: {exc}",
        )

    job_id = data.get("job_id")

    if not job_id:
        return (
            None,
            f"Tulpar job_id döndürmedi: {data}",
        )

    result, error = poll_backend_job(job_id)

    if error:
        return None, error

    raw, error = download_backend_result(result)

    if error:
        return None, error

    try:
        return (
            Image.open(
                io.BytesIO(raw)
            ).convert("RGB"),
            None,
        )
    except Exception as exc:
        return (
            None,
            f"Görsel sonucu okunamadı: {exc}",
        )


# ============================================================
# TEXT → VIDEO — WAN 2.1
# ============================================================

def generate_text_video(
    prompt,
    num_frames=33,
    steps=20,
):
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
    except Exception as exc:
        return (
            None,
            f"Tulpar başlangıç cevabı okunamadı: {exc}",
        )

    job_id = data.get("job_id")

    if not job_id:
        return (
            None,
            f"Tulpar job_id döndürmedi: {data}",
        )

    result, error = poll_backend_job(job_id)

    if error:
        return None, error

    return download_backend_result(result)


# ============================================================
# IMAGE → VIDEO
# ============================================================

def generate_image_video(
    image_bytes,
    prompt,
    num_frames=33,
    steps=20,
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
    except Exception as exc:
        return (
            None,
            f"Tulpar başlangıç cevabı okunamadı: {exc}",
        )

    job_id = data.get("job_id")

    if not job_id:
        return (
            None,
            f"Tulpar job_id döndürmedi: {data}",
        )

    result, error = poll_backend_job(job_id)

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
):
    if not LOCAL_BACKEND_URL:
        return (
            None,
            "Karakter motoru henüz Tulpar backend'e bağlanmadı.",
        )

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
    except Exception as exc:
        return (
            None,
            f"Tulpar başlangıç cevabı okunamadı: {exc}",
        )

    job_id = data.get("job_id")

    if not job_id:
        return (
            None,
            f"Tulpar job_id döndürmedi: {data}",
        )

    result, error = poll_backend_job(job_id)

    if error:
        return None, error

    raw, error = download_backend_result(result)

    if error:
        return None, error

    try:
        return (
            Image.open(
                io.BytesIO(raw)
            ).convert("RGB"),
            None,
        )
    except Exception as exc:
        return (
            None,
            f"Karakter sonucu okunamadı: {exc}",
        )


# ============================================================
# HTML / CSS
# ============================================================

PAGE = r"""
<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KOGCE AI Studio</title>
<style>
*{box-sizing:border-box}
body{
    margin:0;
    background:#07070c;
    color:#eee;
    font-family:Segoe UI,Arial,sans-serif;
}
body:before{
    content:"";
    position:fixed;
    inset:0;
    pointer-events:none;
    background:
        radial-gradient(circle at 12% 5%,rgba(124,58,237,.20),transparent 28%),
        radial-gradient(circle at 88% 8%,rgba(168,85,247,.14),transparent 30%);
}
.wrap{
    max-width:1400px;
    margin:auto;
    padding:28px;
}
.hero,.card{
    border:1px solid rgba(255,255,255,.075);
    background:
        linear-gradient(145deg,rgba(23,20,31,.94),rgba(12,11,17,.96));
    border-radius:22px;
    box-shadow:0 20px 70px rgba(0,0,0,.28);
}
.hero{
    padding:42px 44px;
    margin-bottom:22px;
}
.eyebrow{
    color:#c4b5fd;
    font-size:11px;
    font-weight:800;
    letter-spacing:.20em;
    text-transform:uppercase;
}
h1{
    font-size:clamp(38px,5vw,62px);
    line-height:.98;
    margin:14px 0 16px;
    font-weight:900;
    letter-spacing:-.055em;
}
h2,h3{
    color:#fff;
    letter-spacing:-.03em;
}
.accent{color:#a78bfa}
.sub{
    color:#a7a4b2;
    max-width:820px;
    line-height:1.7;
}
.card{
    padding:24px;
    margin:16px 0;
}
.card-title{
    color:#fff;
    font-size:19px;
    font-weight:800;
    margin-bottom:6px;
}
.card-subtitle{
    color:#8f8b99;
    font-size:13px;
    line-height:1.55;
}
.tabs{
    display:flex;
    gap:7px;
    flex-wrap:wrap;
    margin:16px 0;
}
.tab{
    border:1px solid rgba(168,85,247,.20);
    background:#15121c;
    color:#a9a4b1;
}
.tab.active{
    background:rgba(124,58,237,.25);
    color:#fff;
}
.panel{display:none}
.panel.active{display:block}
.grid{
    display:grid;
    grid-template-columns:repeat(2,minmax(0,1fr));
    gap:14px;
}
.grid3{
    display:grid;
    grid-template-columns:repeat(3,minmax(0,1fr));
    gap:12px;
}
label{
    display:block;
    color:#e8e5ed;
    font-weight:700;
    margin:12px 0 7px;
}
input,textarea,select{
    width:100%;
    padding:13px;
    border-radius:13px;
    border:1px solid #393141;
    background:#15121c;
    color:#fff;
    font-size:14px;
}
textarea{
    min-height:145px;
    resize:vertical;
}
input[type=checkbox]{
    width:auto;
    margin-right:8px;
}
button{
    min-height:46px;
    padding:12px 18px;
    border-radius:13px;
    border:1px solid rgba(168,85,247,.30);
    background:linear-gradient(135deg,#7c3aed,#5b21b6);
    color:#fff;
    font-weight:800;
    cursor:pointer;
}
button:hover{
    border-color:rgba(221,214,254,.65);
}
.status{
    margin:12px 0;
    padding:12px 14px;
    border-radius:12px;
    background:#14111b;
    color:#bbb;
}
.online{color:#86efac}
.offline{color:#fca5a5}
.progress{
    height:9px;
    background:#211c29;
    border-radius:10px;
    overflow:hidden;
    margin-top:10px;
}
.bar{
    height:100%;
    width:0%;
    background:#8b5cf6;
    transition:width .3s ease;
}
.result{
    margin-top:18px;
}
.result img,.result video{
    max-width:100%;
    border-radius:17px;
}
.download{
    display:inline-block;
    margin-top:12px;
    color:#c4b5fd;
}
pre{
    white-space:pre-wrap;
    word-break:break-word;
    background:#0d0b12;
    border-radius:12px;
    padding:15px;
    color:#ddd;
    border:1px solid rgba(255,255,255,.06);
}
.muted{
    color:#92909b;
    font-size:12px;
    line-height:1.55;
}
.feature{
    border-radius:18px;
    border:1px solid rgba(168,85,247,.16);
    background:rgba(25,18,35,.62);
    padding:20px;
    margin-top:15px;
}
.preview{
    max-width:520px;
    max-height:500px;
    object-fit:contain;
}
.metric-grid{
    display:grid;
    grid-template-columns:repeat(4,minmax(0,1fr));
    gap:12px;
}
.metric{
    background:rgba(25,18,36,.70);
    border:1px solid rgba(168,85,247,.18);
    border-radius:15px;
    padding:16px;
}
.metric .name{
    color:#9d99a7;
    font-size:12px;
}
.metric .value{
    color:#fff;
    font-size:20px;
    font-weight:800;
    margin-top:5px;
}
.gallery-item{
    border-top:1px solid rgba(255,255,255,.07);
    padding-top:18px;
    margin-top:18px;
}
.gallery-item img{
    max-width:100%;
    border-radius:17px;
}
.small{
    font-size:12px;
    color:#92909b;
}
@media(max-width:900px){
    .grid,.grid3,.metric-grid{grid-template-columns:1fr}
    .wrap{padding:14px}
    .hero{padding:28px}
}

/* ============================================================
   KOGCE PRO UI OVERRIDE
   ============================================================ */
:root{--bg:#07070b;--panel:#0e0d14;--panel2:#12101a;--line:rgba(255,255,255,.09);--muted:#8f8b9a;--text:#f7f5fb;--accent:#8b5cf6;--accent2:#a78bfa}
body{background:#07070b;color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
body:before{background:radial-gradient(circle at 18% 0%,rgba(124,58,237,.18),transparent 30%),radial-gradient(circle at 88% 10%,rgba(59,130,246,.08),transparent 25%)}
.wrap{max-width:1500px;padding:18px 24px 60px}
.hero{position:relative;min-height:150px;padding:30px 34px;border-radius:24px;background:linear-gradient(135deg,rgba(19,17,27,.98),rgba(10,9,14,.98));border-color:var(--line);box-shadow:0 28px 90px rgba(0,0,0,.38),inset 0 1px 0 rgba(255,255,255,.04)}
.hero:after{content:"";position:absolute;right:30px;top:25px;width:180px;height:180px;border-radius:50%;background:rgba(139,92,246,.16);filter:blur(55px);pointer-events:none}
h1{font-size:42px;margin:10px 0 12px;position:relative;z-index:1}.sub{position:relative;z-index:1;font-size:13px}
.status{display:inline-flex;align-items:center;width:max-content;min-width:170px;margin-top:15px;padding:8px 12px;border-radius:999px;background:rgba(255,255,255,.045);border:1px solid var(--line);font-size:12px}
.tabs{position:sticky;top:12px;z-index:20;margin:14px 0;padding:6px;border:1px solid var(--line);border-radius:16px;background:rgba(12,10,17,.88);backdrop-filter:blur(18px);box-shadow:0 14px 45px rgba(0,0,0,.25)}
.tab{min-height:42px;padding:10px 15px;border-radius:11px;background:transparent;border-color:transparent;color:#8f8b9a;transition:.18s ease}
.tab:hover{background:rgba(255,255,255,.055);transform:none}.tab.active{background:linear-gradient(135deg,rgba(124,58,237,.32),rgba(139,92,246,.12));border-color:rgba(167,139,250,.22);box-shadow:inset 0 1px 0 rgba(255,255,255,.06);color:#fff}
.card{border-radius:20px;padding:26px;background:linear-gradient(145deg,rgba(18,16,25,.96),rgba(10,9,14,.98));border-color:var(--line);box-shadow:0 22px 70px rgba(0,0,0,.28),inset 0 1px 0 rgba(255,255,255,.035)}
.card-title{font-size:20px}.card-subtitle{max-width:850px}
label{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:#aaa5b5;margin:17px 0 8px}
input,textarea,select{background:#0b0a10;border:1px solid rgba(255,255,255,.10);color:#f6f3fa;border-radius:12px;outline:none;transition:.18s ease}
input:focus,textarea:focus,select:focus{border-color:rgba(167,139,250,.65);box-shadow:0 0 0 3px rgba(139,92,246,.12)}
textarea{min-height:170px;font-size:15px;line-height:1.6;padding:16px}
.grid3{gap:10px}.grid{gap:10px}
#resolution{padding:9px 11px;margin-top:8px;border:1px solid var(--line);border-radius:10px;background:rgba(255,255,255,.025);width:max-content}
.feature{background:linear-gradient(145deg,rgba(21,18,31,.92),rgba(12,11,17,.92));border-color:var(--line)}
button{min-height:44px;border-radius:11px;padding:11px 16px;border:1px solid rgba(167,139,250,.30);background:linear-gradient(135deg,#7c3aed,#6d28d9);box-shadow:0 10px 28px rgba(76,29,149,.22),inset 0 1px 0 rgba(255,255,255,.12);transition:transform .16s ease,box-shadow .16s ease,border-color .16s ease}
button:hover{transform:translateY(-1px);border-color:rgba(221,214,254,.7);box-shadow:0 15px 38px rgba(76,29,149,.34),inset 0 1px 0 rgba(255,255,255,.15)}
#generateImage,#generateCharacter,#generateT2V,#generateI2V{width:100%;min-height:54px;margin-top:18px;font-size:13px;letter-spacing:.04em;background:linear-gradient(135deg,#8b5cf6,#6d28d9);border-color:rgba(196,181,253,.45);box-shadow:0 18px 45px rgba(109,40,217,.28)}
#generateImage:before,#generateCharacter:before,#generateT2V:before,#generateI2V:before{content:"✦  ";color:#ddd6fe}
.progress{height:7px;background:#191521;border:1px solid rgba(255,255,255,.05)}.bar{background:linear-gradient(90deg,#7c3aed,#a78bfa)}
.result{margin-top:22px;padding-top:18px;border-top:1px solid rgba(255,255,255,.07)}
.result img,.result video,.gallery-item img{box-shadow:0 20px 60px rgba(0,0,0,.35);border:1px solid rgba(255,255,255,.08)}
pre{background:#09080d;border-color:rgba(255,255,255,.08)}
.metric-grid{gap:10px}.metric{background:#0d0c12;border-color:var(--line);box-shadow:inset 0 1px 0 rgba(255,255,255,.03)}
.small,.muted{color:#777382}
@media(min-width:1100px){#image .card{padding:30px 32px}#imagePrompt{min-height:190px}#imageResult{min-height:40px}}
@media(max-width:900px){.wrap{padding:10px}.hero{padding:25px}.tabs{position:static}.tab{flex:1}.grid,.grid3,.metric-grid{grid-template-columns:1fr}}

</style>
</head>
<body>
<div class="wrap">

<div class="hero">
    <div class="eyebrow">✦ LOCAL PRIVATE CREATIVE WORKSPACE</div>
    <h1>KOGCE <span class="accent">AI Studio</span></h1>
    <div class="sub">
        Fikirden görsele, karakterden videoya.
        Üretim motorları yalnızca Tulpar / ComfyUI üzerinden çalışır.
    </div>
    <div id="health" class="status">Tulpar kontrol ediliyor...</div>
</div>

<div class="tabs">
    <button class="tab active" data-tab="image">✦ &nbsp;Image Studio</button>
    <button class="tab" data-tab="character">◈ &nbsp;Character</button>
    <button class="tab" data-tab="video">▶ &nbsp;Video Studio</button>
    <button class="tab" data-tab="gallery">▦ &nbsp;Gallery</button>
    <button class="tab" data-tab="system">⚙ &nbsp;System</button>
</div>


<!-- ========================================================
     IMAGE
     ======================================================== -->

<section id="image" class="panel active">

<div class="card">
    <div style="display:flex;justify-content:space-between;gap:14px;align-items:center;margin-bottom:18px;padding-bottom:16px;border-bottom:1px solid rgba(255,255,255,.07)">
        <div><div class="card-title">Image Studio</div><div class="card-subtitle">Professional local image generation workspace</div></div>
        <div style="display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end"><span style="padding:7px 10px;border-radius:999px;background:rgba(139,92,246,.12);border:1px solid rgba(167,139,250,.20);font-size:11px;color:#c4b5fd">LOCAL</span><span style="padding:7px 10px;border-radius:999px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);font-size:11px;color:#aaa5b5">QWEN IMAGE 2.1</span></div>
    </div>
    <div class="card-subtitle">
        Fikrini yaz, üretim stilini ve görsel karakterini kontrol et.
        Seçtiğin ayarlar gerçek prompta işlenir.
    </div>

    <label>Ana Prompt</label>
    <textarea id="imagePrompt"
        placeholder="Örneğin: A man walking alone through an empty rainy street at night..."></textarea>

    <div class="grid3">

        <div>
            <label>Görsel Stil</label>
            <select id="imageStyle">
                <option>Automatic</option>
                <option>Photorealistic</option>
                <option>Cinematic</option>
                <option>Anime</option>
                <option>3D Render</option>
                <option>Stylized 3D</option>
                <option>Cartoon</option>
                <option>Fantasy</option>
                <option>Cyberpunk</option>
                <option>Product Photography</option>
                <option>Fashion Editorial</option>
                <option>Dark Cinematic</option>
            </select>
        </div>

        <div>
            <label>Işık</label>
            <select id="imageLighting">
                <option>Automatic</option>
                <option>Natural daylight</option>
                <option>Cinematic lighting</option>
                <option>Soft studio lighting</option>
                <option>Golden hour</option>
                <option>Blue hour</option>
                <option>Neon lighting</option>
                <option>Dramatic lighting</option>
                <option>Low-key lighting</option>
                <option>Volumetric lighting</option>
                <option>Rainy night lighting</option>
            </select>
        </div>

        <div>
            <label>Kamera</label>
            <select id="imageCamera">
                <option>Automatic</option>
                <option>Close-up portrait</option>
                <option>Medium shot</option>
                <option>Full body shot</option>
                <option>Wide cinematic shot</option>
                <option>Low angle</option>
                <option>High angle</option>
                <option>Eye level</option>
                <option>Aerial perspective</option>
                <option>Over-the-shoulder</option>
            </select>
        </div>

    </div>

    <div class="grid3">

        <div>
            <label>Kalite</label>
            <select id="imageQuality">
                <option>Automatic</option>
                <option>High detail</option>
                <option>Ultra detailed</option>
                <option>Photographic realism</option>
                <option>Cinematic quality</option>
                <option>Sharp professional image</option>
            </select>
        </div>

        <div>
            <label>Renk / Atmosfer</label>
            <select id="imageMood">
                <option>Automatic</option>
                <option>Natural colors</option>
                <option>Dark moody</option>
                <option>Warm cinematic</option>
                <option>Cool cinematic</option>
                <option>Neon futuristic</option>
                <option>Muted realistic</option>
                <option>High contrast</option>
            </select>
        </div>

        <div>
            <label>Aspect Ratio</label>
            <select id="imageRatio">
                <option value="1:1">1:1</option>
                <option value="9:16">9:16</option>
                <option value="16:9">16:9</option>
            </select>
        </div>

    </div>

    <div id="resolution" class="small">Çözünürlük: 768 × 768</div>

    <label>Negative Prompt</label>
    <input id="negativePrompt"
        placeholder="blurry, distorted face, bad anatomy, extra fingers...">

    <div class="grid">

        <div>
            <label>
                <input type="checkbox" id="promptRobot" checked>
                🤖 Yerel Prompt Robotu
            </label>
        </div>

        <div>
            <label>Seed</label>
            <select id="seedMode">
                <option value="random">Random</option>
                <option value="fixed">Fixed</option>
            </select>
            <input id="fixedSeed"
                type="number"
                value="123456"
                min="0"
                max="999999999"
                style="display:none;margin-top:8px;">
        </div>

    </div>

    <button id="generateImage">✦ GÖRSEL OLUŞTUR</button>

    <div id="imageStatus"></div>
    <div id="enhancedPrompt"></div>
    <div id="imageResult" class="result"></div>
</div>

</section>


<!-- ========================================================
     CHARACTER
     ======================================================== -->

<section id="character" class="panel">

<div class="card">
    <div class="card-title">Karakter Studio</div>
    <div class="card-subtitle">
        Bir referans fotoğraf yükle ve kişiyi koruyarak
        yeni görünüm, kıyafet ve sahne tarif et.
    </div>

    <div class="feature">
        <b>🎭 Identity Preservation</b>
        <div class="muted">
            Amaç; referans kişiyi korurken yalnızca tarif edilen
            özellikleri, kıyafeti ve sahneyi değiştirmektir.
            Gerçek identity node'ları Tulpar backend'inde kullanılacaktır.
        </div>
    </div>

    <label>Referans fotoğraf</label>
    <input id="characterFile"
        type="file"
        accept="image/png,image/jpeg,image/webp">

    <div id="characterPreview"></div>

    <h3>Kişiyi koruma</h3>

    <div class="grid3">

        <div>
            <label>
                <input id="preserveFace" type="checkbox" checked>
                Yüzü koru
            </label>
        </div>

        <div>
            <label>
                <input id="preserveBody" type="checkbox" checked>
                Vücut yapısını koru
            </label>
        </div>

        <div>
            <label>
                <input id="preserveClothes" type="checkbox">
                Kıyafeti koru
            </label>
        </div>

    </div>

    <label>Identity / Reference Gücü</label>
    <input id="identityStrength"
        type="number"
        min="0.10"
        max="1.00"
        value="0.85"
        step="0.05">

    <label>Stil</label>
    <select id="characterStyle">
        <option>Photorealistic</option>
        <option>Cinematic</option>
        <option>Anime</option>
        <option>3D Render</option>
        <option>Stylized 3D</option>
        <option>Fantasy</option>
        <option>Fashion Editorial</option>
        <option>Dark Cinematic</option>
    </select>

    <label>Ne değiştirmek istiyorsun?</label>
    <textarea id="characterInstruction"
        placeholder="Adam aynı kişi olarak kalsın. Yüzü ve vücut yapısı korunsun. Kıyafeti siyah smokin olsun..."></textarea>

    <button id="generateCharacter">🎭 KARAKTERİ OLUŞTUR</button>

    <div id="characterStatus"></div>
    <div id="characterResult" class="result"></div>
</div>

</section>


<!-- ========================================================
     VIDEO
     ======================================================== -->

<section id="video" class="panel">

<div class="card">
    <div class="card-title">Video Studio</div>
    <div class="card-subtitle">
        Tulpar RTX 4060 üzerinde çalışan yerel video üretim pipeline'ı.
    </div>

    <div id="videoBackendStatus" class="status">
        Tulpar video backend kontrol ediliyor...
    </div>

    <label>Video türü</label>
    <select id="videoMode">
        <option value="t2v">Text → Video</option>
        <option value="i2v">Image → Video</option>
    </select>

    <div id="t2vPanel">

        <h3>Text → Video</h3>
        <div class="muted">
            Motor: Wan 2.1 T2V 1.3B • Tulpar RTX 4060
        </div>

        <label>Video Prompt</label>
        <textarea id="videoPrompt"
            placeholder="A man walking naturally through an empty rainy street at night, realistic body motion, cinematic camera movement..."></textarea>

        <div class="grid3">

            <div>
                <label>Video Stil</label>
                <select id="videoStyle">
                    <option>Automatic</option>
                    <option>Photorealistic</option>
                    <option>Cinematic</option>
                    <option>Anime</option>
                    <option>3D</option>
                    <option>Fantasy</option>
                </select>
            </div>

            <div>
                <label>Kamera Hareketi</label>
                <select id="cameraMotion">
                    <option>Static camera</option>
                    <option>Slow push in</option>
                    <option>Slow pull out</option>
                    <option>Tracking shot</option>
                    <option>Pan</option>
                    <option>Tilt</option>
                    <option>Handheld</option>
                </select>
            </div>

            <div>
                <label>Hareket</label>
                <select id="motionLevel">
                    <option>Subtle</option>
                    <option>Natural</option>
                    <option>Dynamic</option>
                </select>
            </div>

        </div>

        <label>Video süresi</label>
        <select id="t2vDuration">
            <option value="33">Kısa — 33 frame</option>
            <option value="49">Orta — 49 frame</option>
        </select>

        <label>Inference Steps</label>
        <input id="t2vSteps"
            type="number"
            min="8"
            max="30"
            value="20">

        <div class="small">
            RTX 4060 8 GB için önce kısa video ile başlamak daha güvenlidir.
        </div>

        <button id="generateT2V">🎬 VIDEO OLUŞTUR</button>

    </div>


    <div id="i2vPanel" style="display:none">

        <h3>Image → Video</h3>
        <div class="muted">
            Kaynak görseli Wan video pipeline'ına gönder.
        </div>

        <label>Başlangıç görseli</label>
        <input id="videoImageFile"
            type="file"
            accept="image/png,image/jpeg,image/webp">

        <div id="videoImagePreview"></div>

        <label>Motion Prompt</label>
        <textarea id="motionPrompt"
            placeholder="The man walks naturally forward under the rain, his clothes move gently with the wind, slow cinematic camera tracking..."></textarea>

        <div class="grid">

            <div>
                <label>Süre</label>
                <select id="i2vDuration">
                    <option value="33">33 frame</option>
                    <option value="49">49 frame</option>
                </select>
            </div>

            <div>
                <label>Steps</label>
                <input id="i2vSteps"
                    type="number"
                    min="8"
                    max="30"
                    value="20">
            </div>

        </div>

        <button id="generateI2V">🎬 IMAGE → VIDEO</button>

    </div>

    <div id="videoStatus"></div>
    <div id="videoResult" class="result"></div>

</div>

</section>


<!-- ========================================================
     GALLERY
     ======================================================== -->

<section id="gallery" class="panel">

<div class="card">
    <div class="card-title">Galeri</div>
    <div class="card-subtitle">
        Bu yerel KOGCE oturumunda oluşturulan görseller.
    </div>
    <div id="galleryContent"></div>
</div>

</section>


<!-- ========================================================
     SYSTEM
     ======================================================== -->

<section id="system" class="panel">

<div class="card">

    <div class="card-title">System</div>
    <div class="card-subtitle">
        Yerel KOGCE üretim motorlarının bağlantı durumu.
    </div>

    <div class="metric-grid">

        <div class="metric">
            <div class="name">Tulpar</div>
            <div id="metricTulpar" class="value">...</div>
        </div>

        <div class="metric">
            <div class="name">Image</div>
            <div class="value">QWEN 2.1</div>
        </div>

        <div class="metric">
            <div class="name">Video</div>
            <div class="value">WAN 2.1</div>
        </div>

        <div class="metric">
            <div class="name">Gallery</div>
            <div id="metricGallery" class="value">0</div>
        </div>

    </div>

    <hr style="border-color:rgba(255,255,255,.065);margin:25px 0">

    <h3>Aktif Görsel Modeli</h3>
    <pre>Qwen Image 2.1 • Tulpar / ComfyUI</pre>

    <h3>Aktif Video Modeli</h3>
    <pre>Wan 2.1 T2V 1.3B • Tulpar / ComfyUI</pre>

    <h3>Tulpar Backend</h3>
    <pre id="backendEndpoint"></pre>

    <h3>Oturum</h3>
    <div class="small">
        Galerideki görsel sayısı:
        <b id="galleryCount">0</b>
    </div>

    <div id="lastVideoStatus" class="status">
        Bu oturumda başarılı video üretimi yok.
    </div>

</div>

</section>

</div>

<script>

const $ = (id) => document.getElementById(id);

const resolutions = {
    "1:1": [768,768],
    "9:16": [768,1344],
    "16:9": [1344,768]
};


function escapeHtml(value) {
    return String(value || "")
        .replaceAll("&","&amp;")
        .replaceAll("<","&lt;")
        .replaceAll(">","&gt;")
        .replaceAll('"',"&quot;")
        .replaceAll("'","&#039;");
}


function statusBox(element, message, progress=null) {

    if (progress === null) {
        element.innerHTML =
            '<div class="status">' +
            escapeHtml(message) +
            '</div>';
        return;
    }

    const safeProgress =
        Math.max(0, Math.min(100, Number(progress) || 0));

    element.innerHTML =
        '<div class="status">' +
        escapeHtml(message) +
        '<div class="progress">' +
        '<div class="bar" style="width:' +
        safeProgress +
        '%"></div>' +
        '</div></div>';
}


async function pollJob(jobId, element) {

    while (true) {

        const response =
            await fetch("/api/progress/" + encodeURIComponent(jobId));

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.error ||
                "Tulpar job durumu alınamadı."
            );
        }

        const progress =
            Number(data.progress || 0);

        const message =
            data.message ||
            "Üretim devam ediyor...";

        statusBox(
            element,
            "%" + progress + " — " + message,
            progress
        );

        if (data.status === "completed") {
            return data;
        }

        if (data.status === "error") {
            throw new Error(
                data.error ||
                data.message ||
                "Tulpar üretim hatası."
            );
        }

        await new Promise(
            resolve => setTimeout(resolve,1000)
        );
    }
}


function showDownload(element, url, filename, type) {

    if (type === "video") {

        element.innerHTML =
            '<video controls src="' +
            url +
            '"></video>' +
            '<br>' +
            '<a class="download" href="' +
            url +
            '" download="' +
            filename +
            '">⬇️ MP4 İndir</a>';

    } else {

        element.innerHTML =
            '<img src="' +
            url +
            '">' +
            '<br>' +
            '<a class="download" href="' +
            url +
            '" download="' +
            filename +
            '">⬇️ PNG İndir</a>';
    }
}


async function health() {

    try {

        const response =
            await fetch("/api/health");

        const data =
            await response.json();

        if (data.online) {

            $("health").innerHTML =
                '● <span class="online">TULPAR ONLINE</span>';

            $("videoBackendStatus").innerHTML =
                '● <span class="online">Tulpar video backend ONLINE</span>';

            $("metricTulpar").innerHTML =
                '<span class="online">ONLINE</span>';

        } else {

            $("health").innerHTML =
                '● <span class="offline">TULPAR OFFLINE</span>';

            $("videoBackendStatus").innerHTML =
                '● <span class="offline">Tulpar backend erişilemiyor</span>';

            $("metricTulpar").innerHTML =
                '<span class="offline">OFFLINE</span>';
        }

        $("backendEndpoint").textContent =
            data.endpoint || "";

    } catch (error) {

        $("health").innerHTML =
            '● <span class="offline">BACKEND ERİŞİLEMİYOR</span>';

        $("videoBackendStatus").innerHTML =
            '● <span class="offline">Tulpar backend erişilemiyor</span>';

        $("metricTulpar").innerHTML =
            '<span class="offline">OFFLINE</span>';

    }
}


document.querySelectorAll(".tab").forEach(
    button => {

        button.addEventListener(
            "click",
            () => {

                document.querySelectorAll(".tab")
                    .forEach(
                        item => item.classList.remove("active")
                    );

                document.querySelectorAll(".panel")
                    .forEach(
                        item => item.classList.remove("active")
                    );

                button.classList.add("active");

                $(button.dataset.tab)
                    .classList.add("active");

                if (button.dataset.tab === "gallery") {
                    loadGallery();
                }

                if (button.dataset.tab === "system") {
                    updateSystem();
                }
            }
        );
    }
);


$("imageRatio").addEventListener(
    "change",
    () => {

        const size =
            resolutions[$("imageRatio").value];

        $("resolution").textContent =
            "Çözünürlük: " +
            size[0] +
            " × " +
            size[1];
    }
);


$("seedMode").addEventListener(
    "change",
    () => {

        $("fixedSeed").style.display =
            $("seedMode").value === "fixed"
                ? "block"
                : "none";
    }
);


$("videoMode").addEventListener(
    "change",
    () => {

        const t2v =
            $("videoMode").value === "t2v";

        $("t2vPanel").style.display =
            t2v ? "block" : "none";

        $("i2vPanel").style.display =
            t2v ? "none" : "block";
    }
);


$("characterFile").addEventListener(
    "change",
    () => {

        const file =
            $("characterFile").files[0];

        if (!file) {
            $("characterPreview").innerHTML = "";
            return;
        }

        const url =
            URL.createObjectURL(file);

        $("characterPreview").innerHTML =
            '<br><img class="preview" src="' +
            url +
            '" alt="Referans karakter">';
    }
);


$("videoImageFile").addEventListener(
    "change",
    () => {

        const file =
            $("videoImageFile").files[0];

        if (!file) {
            $("videoImagePreview").innerHTML = "";
            return;
        }

        const url =
            URL.createObjectURL(file);

        $("videoImagePreview").innerHTML =
            '<br><img class="preview" src="' +
            url +
            '" alt="Kaynak görsel">';
    }
);


$("generateImage").addEventListener(
    "click",
    async () => {

        const prompt =
            $("imagePrompt").value.trim();

        if (!prompt) {

            statusBox(
                $("imageStatus"),
                "Önce bir prompt gir."
            );

            return;
        }

        const ratio =
            $("imageRatio").value;

        const size =
            resolutions[ratio];

        const payload = {

            prompt: prompt,

            width: size[0],

            height: size[1],

            style: $("imageStyle").value,

            lighting: $("imageLighting").value,

            camera: $("imageCamera").value,

            quality: $("imageQuality").value,

            color_mood: $("imageMood").value,

            negative_prompt:
                $("negativePrompt").value,

            ai_boost:
                $("promptRobot").checked,

            seed:
                $("seedMode").value === "fixed"
                    ? Number($("fixedSeed").value)
                    : null
        };

        $("imageResult").innerHTML = "";

        statusBox(
            $("imageStatus"),
            "%0 — Tulpar işi başlatılıyor...",
            0
        );

        try {

            const response =
                await fetch(
                    "/api/generate-image",
                    {
                        method:"POST",
                        headers:{
                            "Content-Type":
                                "application/json"
                        },
                        body:
                            JSON.stringify(payload)
                    }
                );

            const data =
                await response.json();

            if (!response.ok) {
                throw new Error(
                    data.error ||
                    "Görsel üretimi başlatılamadı."
                );
            }

            if (data.final_prompt) {

                $("enhancedPrompt").innerHTML =
                    '<div class="card">' +
                    '<b>Profesyonel İngilizce prompt</b>' +
                    '<pre>' +
                    escapeHtml(data.final_prompt) +
                    '</pre></div>';

            } else {

                $("enhancedPrompt").innerHTML = "";
            }

            const result =
                await pollJob(
                    data.job_id,
                    $("imageStatus")
                );

            showDownload(
                $("imageResult"),
                result.download_url,
                "kogce_ai_image.png",
                "image"
            );

            await loadGallery();

        } catch (error) {

            statusBox(
                $("imageStatus"),
                "Görsel üretilemedi: " +
                error.message
            );
        }
    }
);


$("generateCharacter").addEventListener(
    "click",
    async () => {

        const file =
            $("characterFile").files[0];

        const instruction =
            $("characterInstruction").value.trim();

        if (!file) {

            statusBox(
                $("characterStatus"),
                "Önce referans fotoğrafı seç."
            );

            return;
        }

        if (!instruction) {

            statusBox(
                $("characterStatus"),
                "Önce karakter üzerinde yapılacak değişikliği yaz."
            );

            return;
        }

        const form =
            new FormData();

        form.append("image",file);

        form.append(
            "instruction",
            instruction
        );

        form.append(
            "style",
            $("characterStyle").value
        );

        form.append(
            "strength",
            $("identityStrength").value
        );

        form.append(
            "preserve_face",
            $("preserveFace").checked
        );

        form.append(
            "preserve_body",
            $("preserveBody").checked
        );

        form.append(
            "preserve_clothes",
            $("preserveClothes").checked
        );

        $("characterResult").innerHTML = "";

        statusBox(
            $("characterStatus"),
            "%0 — Karakter işi başlatılıyor...",
            0
        );

        try {

            const response =
                await fetch(
                    "/api/generate-character",
                    {
                        method:"POST",
                        body:form
                    }
                );

            const data =
                await response.json();

            if (!response.ok) {
                throw new Error(
                    data.error ||
                    "Karakter üretimi başlatılamadı."
                );
            }

            const result =
                await pollJob(
                    data.job_id,
                    $("characterStatus")
                );

            showDownload(
                $("characterResult"),
                result.download_url,
                "kogce_character.png",
                "image"
            );

        } catch (error) {

            statusBox(
                $("characterStatus"),
                "Karakter üretilemedi: " +
                error.message
            );
        }
    }
);


$("generateT2V").addEventListener(
    "click",
    async () => {

        const prompt =
            $("videoPrompt").value.trim();

        if (!prompt) {

            statusBox(
                $("videoStatus"),
                "Önce video promptu gir."
            );

            return;
        }

        const finalPrompt =
            prompt +
            ". Visual style: " +
            $("videoStyle").value +
            ". Camera movement: " +
            $("cameraMotion").value +
            ". Motion intensity: " +
            $("motionLevel").value +
            ". Natural realistic motion and consistent subject appearance.";

        const payload = {

            prompt: finalPrompt,

            num_frames:
                Number($("t2vDuration").value),

            steps:
                Number($("t2vSteps").value)
        };

        $("videoResult").innerHTML = "";

        statusBox(
            $("videoStatus"),
            "%0 — Tulpar video işi başlatılıyor...",
            0
        );

        try {

            const response =
                await fetch(
                    "/api/generate-video",
                    {
                        method:"POST",
                        headers:{
                            "Content-Type":
                                "application/json"
                        },
                        body:
                            JSON.stringify(payload)
                    }
                );

            const data =
                await response.json();

            if (!response.ok) {
                throw new Error(
                    data.error ||
                    "Video üretimi başlatılamadı."
                );
            }

            const result =
                await pollJob(
                    data.job_id,
                    $("videoStatus")
                );

            showDownload(
                $("videoResult"),
                result.download_url,
                "kogce_text_to_video.mp4",
                "video"
            );

            $("lastVideoStatus").textContent =
                "Son video üretimi mevcut.";

        } catch (error) {

            statusBox(
                $("videoStatus"),
                "Video üretilemedi: " +
                error.message
            );
        }
    }
);


$("generateI2V").addEventListener(
    "click",
    async () => {

        const file =
            $("videoImageFile").files[0];

        const prompt =
            $("motionPrompt").value.trim();

        if (!file) {

            statusBox(
                $("videoStatus"),
                "I2V için başlangıç görseli seç."
            );

            return;
        }

        if (!prompt) {

            statusBox(
                $("videoStatus"),
                "Hareket promptu gir."
            );

            return;
        }

        const form =
            new FormData();

        form.append("image",file);

        form.append(
            "prompt",
            prompt
        );

        form.append(
            "num_frames",
            $("i2vDuration").value
        );

        form.append(
            "steps",
            $("i2vSteps").value
        );

        $("videoResult").innerHTML = "";

        statusBox(
            $("videoStatus"),
            "%0 — Tulpar I2V işi başlatılıyor...",
            0
        );

        try {

            const response =
                await fetch(
                    "/api/generate-i2v",
                    {
                        method:"POST",
                        body:form
                    }
                );

            const data =
                await response.json();

            if (!response.ok) {
                throw new Error(
                    data.error ||
                    "I2V üretimi başlatılamadı."
                );
            }

            const result =
                await pollJob(
                    data.job_id,
                    $("videoStatus")
                );

            showDownload(
                $("videoResult"),
                result.download_url,
                "kogce_image_to_video.mp4",
                "video"
            );

            $("lastVideoStatus").textContent =
                "Son video üretimi mevcut.";

        } catch (error) {

            statusBox(
                $("videoStatus"),
                "Image → Video üretilemedi: " +
                error.message
            );
        }
    }
);


async function loadGallery() {

    try {

        const response =
            await fetch("/api/gallery");

        const items =
            await response.json();

        $("metricGallery").textContent =
            items.length;

        $("galleryCount").textContent =
            items.length;

        if (!items.length) {

            $("galleryContent").innerHTML =
                '<div class="status">' +
                'Henüz oluşturulmuş bir görsel yok.' +
                '</div>';

            return;
        }

        $("galleryContent").innerHTML =
            items
                .slice()
                .reverse()
                .map(
                    (item,index) => {

                        return (
                            '<div class="gallery-item">' +
                            '<h3>Üretim #' +
                            (items.length-index) +
                            '</h3>' +
                            '<img src="' +
                            item.url +
                            '">' +
                            '<p class="small">Orijinal Prompt</p>' +
                            '<div>' +
                            escapeHtml(item.prompt) +
                            '</div>' +
                            '<p class="small">Final Prompt</p>' +
                            '<pre>' +
                            escapeHtml(item.final_prompt) +
                            '</pre>' +
                            '</div>'
                        );
                    }
                )
                .join("");

    } catch (error) {

        $("galleryContent").innerHTML =
            '<div class="status">' +
            'Galeri okunamadı.' +
            '</div>';
    }
}


async function updateSystem() {

    await health();

    $("metricGallery").textContent =
        gallery.length;
}


health();

setInterval(
    health,
    5000
);

</script>
</body>
</html>
"""


# ============================================================
# HTTP SERVER
# ============================================================

def json_response(handler, status, payload):
    raw = json.dumps(
        payload,
        ensure_ascii=False,
    ).encode("utf-8")

    handler.send_response(status)
    handler.send_header(
        "Content-Type",
        "application/json; charset=utf-8",
    )
    handler.send_header(
        "Content-Length",
        str(len(raw)),
    )
    handler.end_headers()
    handler.wfile.write(raw)


def read_json_body(handler):
    length = int(
        handler.headers.get(
            "Content-Length",
            "0",
        )
    )

    raw = handler.rfile.read(length)

    if not raw:
        return {}

    return json.loads(
        raw.decode("utf-8")
    )


def parse_multipart(handler):
    """
    Multipart parser using Python stdlib only.
    Returns:
        fields: dict[str,str]
        files: dict[str, tuple[filename, bytes, content_type]]
    """
    content_type = handler.headers.get(
        "Content-Type",
        "",
    )

    match = re.search(
        r'boundary="?([^";]+)"?',
        content_type,
    )

    if not match:
        raise ValueError(
            "Multipart boundary bulunamadı."
        )

    boundary = (
        b"--" +
        match.group(1).encode()
    )

    length = int(
        handler.headers.get(
            "Content-Length",
            "0",
        )
    )

    body = handler.rfile.read(length)

    fields = {}
    files = {}

    for part in body.split(boundary)[1:]:

        if part in (b"", b"--", b"--\r\n"):
            continue

        part = part.strip(b"\r\n")

        if part.endswith(b"--"):
            part = part[:-2].rstrip(b"\r\n")

        header_end = part.find(
            b"\r\n\r\n"
        )

        if header_end < 0:
            continue

        header_bytes = part[:header_end]
        content = part[
            header_end + 4:
        ]

        headers = {}

        for line in header_bytes.split(b"\r\n"):

            if b":" not in line:
                continue

            key, value = line.split(
                b":",
                1,
            )

            headers[
                key.decode(
                    "latin1"
                ).strip().lower()
            ] = value.decode(
                "latin1"
            ).strip()

        disposition = headers.get(
            "content-disposition",
            "",
        )

        name_match = re.search(
            r'name="([^"]+)"',
            disposition,
        )

        if not name_match:
            continue

        name = name_match.group(1)

        filename_match = re.search(
            r'filename="([^"]*)"',
            disposition,
        )

        if filename_match:

            filename = filename_match.group(1)

            files[name] = (
                filename,
                content,
                headers.get(
                    "content-type",
                    "application/octet-stream",
                ),
            )

        else:

            fields[name] = content.decode(
                "utf-8",
                errors="replace",
            )

    return fields, files


class KOGCEHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(
            "[KOGCE]",
            fmt % args,
        )

    def send_bytes(
        self,
        status,
        data,
        content_type,
        download_name=None,
    ):
        self.send_response(status)

        self.send_header(
            "Content-Type",
            content_type,
        )

        self.send_header(
            "Content-Length",
            str(len(data)),
        )

        if download_name:
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="{download_name}"',
            )

        self.end_headers()

        self.wfile.write(data)

    def do_GET(self):

        path = urlparse(
            self.path
        ).path

        if path == "/":

            self.send_bytes(
                200,
                PAGE.encode("utf-8"),
                "text/html; charset=utf-8",
            )

            return

        if path == "/api/health":

            json_response(
                self,
                200,
                {
                    "online":
                        local_backend_available(),
                    "endpoint":
                        LOCAL_BACKEND_URL,
                },
            )

            return

        if path == "/api/system":

            json_response(
                self,
                200,
                {
                    "online":
                        local_backend_available(),
                    "endpoint":
                        LOCAL_BACKEND_URL,
                },
            )

            return

        if path == "/api/gallery":

            json_response(
                self,
                200,
                gallery,
            )

            return

        if path.startswith(
            "/api/progress/"
        ):

            job_id = path.rsplit(
                "/",
                1,
            )[-1]

            try:

                response = requests.get(
                    f"{LOCAL_BACKEND_URL}/progress/{job_id}",
                    timeout=15,
                )

                self.send_bytes(
                    response.status_code,
                    response.content,
                    response.headers.get(
                        "Content-Type",
                        "application/json",
                    ),
                )

            except Exception as exc:

                json_response(
                    self,
                    503,
                    {
                        "error":
                            str(exc)
                    },
                )

            return

        json_response(
            self,
            404,
            {
                "error":
                    "Not found"
            },
        )

    def do_POST(self):

        path = urlparse(
            self.path
        ).path

        try:

            # ------------------------------------------------
            # GALLERY
            # ------------------------------------------------

            if path == "/api/gallery":

                data = read_json_body(
                    self
                )

                gallery.append(
                    data
                )

                json_response(
                    self,
                    200,
                    {"ok": True},
                )

                return

            # ------------------------------------------------
            # IMAGE
            # ------------------------------------------------

            if path == "/api/generate-image":

                data = read_json_body(
                    self
                )

                prompt = clean_prompt(
                    data.get(
                        "prompt",
                        "",
                    )
                )

                if not prompt:
                    raise ValueError(
                        "Prompt boş."
                    )

                if data.get(
                    "ai_boost",
                    True,
                ):

                    final_prompt = local_prompt_robot(
                        prompt,
                        data.get(
                            "style",
                            "Automatic",
                        ),
                        data.get(
                            "lighting",
                            "Automatic",
                        ),
                        data.get(
                            "camera",
                            "Automatic",
                        ),
                        data.get(
                            "quality",
                            "Automatic",
                        ),
                        data.get(
                            "color_mood",
                            "Automatic",
                        ),
                        data.get(
                            "negative_prompt",
                            "",
                        ),
                    )

                else:

                    final_prompt = build_image_prompt(
                        prompt,
                        data.get(
                            "style",
                            "Automatic",
                        ),
                        data.get(
                            "lighting",
                            "Automatic",
                        ),
                        data.get(
                            "camera",
                            "Automatic",
                        ),
                        data.get(
                            "quality",
                            "Automatic",
                        ),
                        data.get(
                            "color_mood",
                            "Automatic",
                        ),
                        data.get(
                            "negative_prompt",
                            "",
                        ),
                    )

                payload = {
                    "prompt":
                        final_prompt,
                    "width":
                        int(data.get(
                            "width",
                            768,
                        )),
                    "height":
                        int(data.get(
                            "height",
                            768,
                        )),
                }

                seed = data.get(
                    "seed"
                )

                if seed is not None:
                    payload["seed"] = int(seed)

                response, error = (
                    local_backend_request(
                        "/generate-image",
                        payload,
                        timeout=30,
                        form=True,
                    )
                )

                if error:
                    json_response(
                        self,
                        502,
                        {"error": error},
                    )
                    return

                result = response.json()

                if not result.get(
                    "job_id"
                ):
                    json_response(
                        self,
                        502,
                        {
                            "error":
                                "Tulpar job_id döndürmedi.",
                            "data":
                                result,
                        },
                    )
                    return

                json_response(
                    self,
                    200,
                    {
                        "job_id":
                            result["job_id"],
                        "final_prompt":
                            final_prompt,
                    },
                )

                return

            # ------------------------------------------------
            # TEXT → VIDEO
            # ------------------------------------------------

            if path == "/api/generate-video":

                data = read_json_body(
                    self
                )

                response, error = (
                    local_backend_request(
                        "/generate-video",
                        {
                            "prompt":
                                data.get(
                                    "prompt",
                                    "",
                                ),
                            "num_frames":
                                int(data.get(
                                    "num_frames",
                                    33,
                                )),
                            "steps":
                                int(data.get(
                                    "steps",
                                    20,
                                )),
                        },
                        timeout=30,
                    )
                )

                if error:
                    json_response(
                        self,
                        502,
                        {"error": error},
                    )
                    return

                result = response.json()

                json_response(
                    self,
                    200,
                    result,
                )

                return

            # ------------------------------------------------
            # IMAGE → VIDEO
            # ------------------------------------------------

            if path == "/api/generate-i2v":

                fields, files = (
                    parse_multipart(
                        self
                    )
                )

                if "image" not in files:
                    raise ValueError(
                        "Başlangıç görseli bulunamadı."
                    )

                filename, image_bytes, content_type = (
                    files["image"]
                )

                response, error = (
                    local_backend_request(
                        "/image-to-video",
                        {
                            "prompt":
                                fields.get(
                                    "prompt",
                                    "",
                                ),
                            "num_frames":
                                fields.get(
                                    "num_frames",
                                    "33",
                                ),
                            "steps":
                                fields.get(
                                    "steps",
                                    "20",
                                ),
                        },
                        files={
                            "image": (
                                filename or "reference.png",
                                image_bytes,
                                content_type or "image/png",
                            )
                        },
                        timeout=30,
                    )
                )

                if error:
                    json_response(
                        self,
                        502,
                        {"error": error},
                    )
                    return

                json_response(
                    self,
                    200,
                    response.json(),
                )

                return

            # ------------------------------------------------
            # CHARACTER
            # ------------------------------------------------

            if path == "/api/generate-character":

                fields, files = (
                    parse_multipart(
                        self
                    )
                )

                if "image" not in files:
                    raise ValueError(
                        "Referans görsel bulunamadı."
                    )

                filename, image_bytes, content_type = (
                    files["image"]
                )

                response, error = (
                    local_backend_request(
                        "/character-edit",
                        {
                            "instruction":
                                fields.get(
                                    "instruction",
                                    "",
                                ),
                            "style":
                                fields.get(
                                    "style",
                                    "Photorealistic",
                                ),
                            "strength":
                                fields.get(
                                    "strength",
                                    "0.85",
                                ),
                            "preserve_face":
                                fields.get(
                                    "preserve_face",
                                    "true",
                                ),
                            "preserve_body":
                                fields.get(
                                    "preserve_body",
                                    "true",
                                ),
                            "preserve_clothes":
                                fields.get(
                                    "preserve_clothes",
                                    "false",
                                ),
                        },
                        files={
                            "image": (
                                filename or "character_reference.png",
                                image_bytes,
                                content_type or "image/png",
                            )
                        },
                        timeout=30,
                    )
                )

                if error:
                    json_response(
                        self,
                        502,
                        {"error": error},
                    )
                    return

                json_response(
                    self,
                    200,
                    response.json(),
                )

                return

            json_response(
                self,
                404,
                {
                    "error":
                        "Not found"
                },
            )

        except Exception as exc:

            json_response(
                self,
                400,
                {
                    "error":
                        str(exc)
                },
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("KOGCE AI STUDIO — LOCAL")
    print("=" * 70)
    print(
        "UI      :",
        f"http://{HOST}:{PORT}",
    )
    print(
        "TULPAR  :",
        LOCAL_BACKEND_URL,
    )
    print(
        "IMAGE   : Qwen Image 2.1 / Tulpar / ComfyUI"
    )
    print(
        "VIDEO   : Wan 2.1 T2V 1.3B / Tulpar / ComfyUI"
    )
    print(
        "EXTERNAL AI INFERENCE : NONE"
    )
    print("=" * 70)
    print()

    server = ThreadingHTTPServer(
        (HOST, PORT),
        KOGCEHandler,
    )

    def open_browser():
        time.sleep(1)
        webbrowser.open(
            f"http://{HOST}:{PORT}"
        )

    threading.Thread(
        target=open_browser,
        daemon=True,
    ).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nKOGCE kapatılıyor...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

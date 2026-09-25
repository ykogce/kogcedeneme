import io
import time
import requests
import streamlit as st

from PIL import Image
from huggingface_hub import InferenceClient


# ============================================================
# CONFIG
# ============================================================

APP_PASSWORD = "1234"

PROMPT_MODEL = "openai/gpt-oss-120b"
IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
VIDEO_T2V_MODEL = "Wan-AI/Wan2.1-T2V-1.3B"
VIDEO_I2V_MODEL = "Lightricks/LTX-Video-0.9.8-13B-distilled"

CHAT_URL = "https://router.huggingface.co/v1/chat/completions"


# ============================================================
# SESSION
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
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


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
# PAGE
# ============================================================

st.set_page_config(
    page_title="KOGCE AI Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(126, 34, 206, 0.20), transparent 28%),
            radial-gradient(circle at 90% 10%, rgba(168, 85, 247, 0.16), transparent 30%),
            radial-gradient(circle at 50% 100%, rgba(88, 28, 135, 0.18), transparent 35%),
            #09090f;
        color: #f1f5f9;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0e0b16 0%,
                #100c19 50%,
                #0b0911 100%
            );
        border-right: 1px solid rgba(168, 85, 247, 0.22);
    }

    section[data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }

    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }

    p, label, span, div {
        color: #f1f5f9;
    }

    .stCaption,
    [data-testid="stCaptionContainer"] {
        color: #d9dce5 !important;
        font-weight: 500 !important;
    }

    [data-testid="stMarkdownContainer"] p {
        color: #f1f5f9;
    }

    .stTextInput label,
    .stTextArea label,
    .stSelectbox label,
    .stRadio label,
    .stCheckbox label,
    .stFileUploader label {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
    }

    textarea,
    input {
        color: #111827 !important;
        background: #ffffff !important;
    }

    textarea::placeholder,
    input::placeholder {
        color: #6b7280 !important;
    }

    .stTextInput > div > div,
    .stTextArea > div > div {
        border-radius: 12px;
    }

    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(168, 85, 247, 0.35);
        background:
            linear-gradient(
                135deg,
                rgba(126, 34, 206, 0.95),
                rgba(88, 28, 135, 0.95)
            );
        color: #ffffff !important;
        font-weight: 800;
        min-height: 44px;
        box-shadow:
            0 8px 24px rgba(0, 0, 0, 0.28),
            inset 0 1px 0 rgba(255,255,255,0.10);
    }

    .stButton > button:hover {
        border-color: rgba(216, 180, 254, 0.75);
        background:
            linear-gradient(
                135deg,
                rgba(147, 51, 234, 1),
                rgba(107, 33, 168, 1)
            );
    }

    div[data-testid="stMetric"] {
        background: rgba(25, 18, 36, 0.72);
        border: 1px solid rgba(168, 85, 247, 0.22);
        border-radius: 14px;
        padding: 15px;
    }

    div[data-testid="stMetric"] label {
        color: #d9dce5 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #d9dce5;
        font-weight: 700;
        border-radius: 10px;
        padding-left: 16px;
        padding-right: 16px;
    }

    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background: rgba(126, 34, 206, 0.18);
    }

    .stDownloadButton > button {
        border-radius: 12px;
        background: rgba(30, 20, 42, 0.9);
        color: #ffffff !important;
        border: 1px solid rgba(168, 85, 247, 0.32);
        font-weight: 700;
    }

    .stAlert {
        border-radius: 12px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(20, 15, 29, 0.85);
        border: 1px dashed rgba(168, 85, 247, 0.45);
        border-radius: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# AI
# ============================================================

def call_ai(system_prompt, user_prompt, temperature=0.7, max_tokens=1800):

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

        if not content:
            return None, "AI boş cevap döndürdü."

        return content, None

    except Exception as e:
        return None, f"AI bağlantı hatası: {e}"


# ============================================================
# IMAGE GENERATION
# ============================================================

def generate_image(prompt, width, height):

    if not HF_API_KEY:
        return None, "HF_API_KEY bulunamadı."

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
# TEXT TO VIDEO
# ONLY VIDEO ERROR FIX
# ============================================================

def generate_text_video(
    prompt,
    num_frames=49,
    steps=20,
):

    if not HF_API_KEY:
        return None, "HF_API_KEY bulunamadı."

    try:

        num_frames = max(
            81,
            min(int(num_frames), 100)
        )

        steps = max(
            1,
            min(int(steps), 50)
        )

        endpoint = (
            "https://router.huggingface.co/"
            "fal-ai/fal-ai/wan-t2v"
            "?_subdomain=queue"
        )

        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "prompt": prompt,
            "num_frames": num_frames,
            "num_inference_steps": steps,
        }

        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=180,
        )

        if response.status_code not in (200, 201, 202):

            try:
                provider_error = response.json()
            except Exception:
                provider_error = response.text

            return (
                None,
                "Fal / Hugging Face video isteği başarısız.\n\n"
                f"HTTP {response.status_code}\n\n"
                f"{provider_error}",
            )

        try:
            queue_data = response.json()
        except Exception:

            return (
                None,
                "Provider geçerli JSON cevap döndürmedi.\n\n"
                f"HTTP {response.status_code}\n\n"
                f"{response.text[:4000]}",
            )

        request_id = queue_data.get("request_id")
        status_url = queue_data.get("status_url")
        response_url = queue_data.get("response_url")

        if not request_id:

            return (
                None,
                "Provider request_id döndürmedi.\n\n"
                f"Provider cevabı:\n{queue_data}",
            )

        if not status_url:

            return (
                None,
                "Provider status_url döndürmedi.\n\n"
                f"Provider cevabı:\n{queue_data}",
            )

        if not response_url:

            return (
                None,
                "Provider response_url döndürmedi.\n\n"
                f"Provider cevabı:\n{queue_data}",
            )

        max_wait_seconds = 600
        poll_interval = 2

        start_time = time.time()

        while True:

            elapsed = time.time() - start_time

            if elapsed > max_wait_seconds:

                return (
                    None,
                    "Video üretimi zaman aşımına uğradı.\n\n"
                    f"Beklenen maksimum süre: "
                    f"{max_wait_seconds} saniye\n"
                    f"Request ID: {request_id}",
                )

            status_response = requests.get(
                status_url,
                headers={
                    "Authorization": f"Bearer {HF_API_KEY}",
                },
                timeout=60,
            )

            if status_response.status_code != 200:

                try:
                    status_error = status_response.json()
                except Exception:
                    status_error = status_response.text

                return (
                    None,
                    "Fal durum sorgusu başarısız.\n\n"
                    f"HTTP {status_response.status_code}\n\n"
                    f"{status_error}",
                )

            try:
                status_data = status_response.json()
            except Exception:

                return (
                    None,
                    "Fal durum cevabı JSON değil.\n\n"
                    f"{status_response.text[:4000]}",
                )

            status = status_data.get("status")

            if status == "COMPLETED":
                break

            if status in (
                "FAILED",
                "ERROR",
                "CANCELLED",
            ):

                return (
                    None,
                    "Fal video üretimini tamamlayamadı.\n\n"
                    f"Durum: {status}\n\n"
                    f"Provider cevabı:\n{status_data}",
                )

            time.sleep(poll_interval)

        result_response = requests.get(
            response_url,
            headers={
                "Authorization": f"Bearer {HF_API_KEY}",
            },
            timeout=120,
        )

        if result_response.status_code != 200:

            try:
                result_error = result_response.json()
            except Exception:
                result_error = result_response.text

            return (
                None,
                "Fal sonuç isteği başarısız.\n\n"
                f"HTTP {result_response.status_code}\n\n"
                f"{result_error}",
            )

        try:
            result_data = result_response.json()
        except Exception:

            return (
                None,
                "Fal sonuç cevabı JSON değil.\n\n"
                f"{result_response.text[:4000]}",
            )

        video_info = result_data.get("video")

        if not video_info:

            return (
                None,
                "Video üretimi tamamlandı ancak provider "
                "video alanını döndürmedi.\n\n"
                f"Provider sonucu:\n{result_data}",
            )

        video_url = video_info.get("url")

        if not video_url:

            return (
                None,
                "Provider video alanını döndürdü fakat "
                "video URL'si bulunamadı.\n\n"
                f"Video alanı:\n{video_info}",
            )

        video_response = requests.get(
            video_url,
            timeout=180,
        )

        if video_response.status_code != 200:

            return (
                None,
                "Üretilen video dosyası indirilemedi.\n\n"
                f"HTTP {video_response.status_code}\n"
                f"URL: {video_url}",
            )

        video_bytes = video_response.content

        if not video_bytes:

            return (
                None,
                "Provider video URL'si verdi fakat "
                "dosya boş geldi.",
            )

        return video_bytes, None

    except requests.exceptions.Timeout:

        return (
            None,
            "Video provider bağlantısı zaman aşımına uğradı.",
        )

    except requests.exceptions.RequestException as e:

        return (
            None,
            f"Provider bağlantı hatası:\n"
            f"{type(e).__name__}: {e}",
        )

    except Exception as e:

        return (
            None,
            f"{type(e).__name__}: {str(e)}",
        )


# ============================================================
# IMAGE TO VIDEO
# ============================================================

def generate_image_video(
    image,
    prompt,
    num_frames=49,
):

    if not HF_API_KEY:
        return None, "HF_API_KEY bulunamadı."

    try:

        client = InferenceClient(
            provider="auto",
            api_key=HF_API_KEY,
        )

        video = client.image_to_video(
            image,
            model=VIDEO_I2V_MODEL,
            prompt=prompt,
            num_frames=num_frames,
        )

        return video, None

    except Exception as e:
        return None, str(e)


# ============================================================
# LOGIN
# ============================================================

if not st.session_state.authenticated:

    left, center, right = st.columns([1, 2, 1])

    with center:

        st.markdown(
            "<div style='height:55px'></div>",
            unsafe_allow_html=True,
        )

        st.title("KOGCE AI Studio")

        st.caption("AI CREATIVE WORKSPACE")

        st.write("Create. Imagine. Generate.")

        password = st.text_input(
            "Şifre",
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

                st.error("Hatalı şifre.")

        st.caption("KOGCE AI Studio")

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("KOGCE AI Studio")

    st.divider()

    if HF_API_KEY:
        st.success("AI SYSTEM ONLINE")
    else:
        st.error("HF KEY MISSING")

    st.markdown("### AI Motorları")

    st.write("AI Prompt Robotu")
    st.write("AI Senaryo")
    st.write("AI Scene Planner")
    st.write("FLUX Görsel")
    st.write("Wan Text → Video")
    st.write("LTX Image → Video")

    st.divider()

    st.write(
        f"Galeri: {len(st.session_state.gallery)} görsel"
    )

    if st.button(
        "Çıkış Yap",
        use_container_width=True,
    ):

        st.session_state.authenticated = False
        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title("KOGCE AI Studio")

st.caption("AI Creative Workspace")

st.write(
    "Fikir → Prompt → Görsel → Video → İçerik üretim araçları"
)

st.divider()


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "✨ Görsel",
        "🎬 Video",
        "🧠 AI Araçları",
        "🖼️ Galeri",
        "⚙️ Sistem",
    ]
)


# ============================================================
# IMAGE TAB
# ============================================================

with tabs[0]:

    st.header("Görsel Üretim")

    prompt = st.text_area(
        "Prompt",
        placeholder="Örneğin: A futuristic city at night...",
        height=150,
    )

    ai_boost = st.toggle(
        "AI Prompt Robotu",
        value=False,
    )

    aspect_ratio = st.selectbox(
        "Aspect Ratio",
        [
            "1:1",
            "9:16",
            "16:9",
        ],
    )

    resolutions = {
        "1:1": (768, 768),
        "9:16": (768, 1344),
        "16:9": (1344, 768),
    }

    width, height = resolutions[aspect_ratio]

    st.caption(
        f"Çözünürlük: {width} × {height}"
    )

    if st.button(
        "✨ Görsel Üret",
        use_container_width=True,
    ):

        if not prompt.strip():

            st.warning("Önce bir prompt gir.")

        else:

            final_prompt = prompt

            if ai_boost:

                with st.spinner("AI prompt geliştiriyor..."):

                    enhanced_prompt, error = call_ai(
                        """
                        You are a professional AI image prompt engineer.

                        Convert the user's simple idea into a detailed,
                        production-ready English image generation prompt.

                        Describe:
                        - subject
                        - environment
                        - composition
                        - camera
                        - lighting
                        - materials
                        - colors
                        - atmosphere
                        - visual quality

                        Do not explain your answer.
                        Return only the final image prompt.
                        """,
                        prompt,
                        temperature=0.7,
                        max_tokens=1200,
                    )

                if error:

                    st.error(error)

                elif enhanced_prompt:

                    final_prompt = enhanced_prompt

                    st.session_state.last_enhanced_prompt = (
                        enhanced_prompt
                    )

                    with st.expander(
                        "AI tarafından geliştirilen prompt"
                    ):

                        st.write(enhanced_prompt)

            with st.spinner("Görsel oluşturuluyor..."):

                image, error = generate_image(
                    final_prompt,
                    width,
                    height,
                )

            if error:

                st.error("Görsel üretilemedi.")
                st.code(error)

            elif image:

                st.image(
                    image,
                    use_container_width=True,
                )

                image_buffer = io.BytesIO()

                image.save(
                    image_buffer,
                    format="PNG",
                )

                image_bytes = image_buffer.getvalue()

                st.download_button(
                    "⬇️ PNG İndir",
                    data=image_bytes,
                    file_name="kogce_ai_image.png",
                    mime="image/png",
                    use_container_width=True,
                )

                st.session_state.gallery.append(
                    {
                        "image": image_bytes,
                        "prompt": prompt,
                        "final_prompt": final_prompt,
                    }
                )

                st.session_state.last_prompt = prompt

                st.success("Görsel başarıyla üretildi.")


# ============================================================
# VIDEO TAB
# ============================================================

with tabs[1]:

    st.header("Video Studio")

    video_mode = st.radio(
        "Video türü",
        [
            "Text → Video",
            "Image → Video",
        ],
        horizontal=True,
    )


    # --------------------------------------------------------
    # TEXT TO VIDEO
    # --------------------------------------------------------

    if video_mode == "Text → Video":

        st.subheader("Text → Video")

        st.caption(
            "Motor: Wan 2.1 T2V 1.3B"
        )

        video_prompt = st.text_area(
            "Video Prompt",
            placeholder=(
                "A cinematic shot of a futuristic city, "
                "neon lights, moving camera, realistic motion..."
            ),
            height=160,
        )

        duration = st.selectbox(
            "Video süresi",
            [
                "Kısa",
                "Orta",
            ],
        )

        if duration == "Kısa":
            num_frames = 49
        else:
            num_frames = 81

        steps = st.slider(
            "Inference Steps",
            min_value=10,
            max_value=30,
            value=20,
        )

        st.info(
            "Text → Video, Wan 2.1 Text → Video modeliyle "
            "çalışır."
        )

        if st.button(
            "🎬 Video Üret",
            use_container_width=True,
        ):

            if not video_prompt.strip():

                st.warning(
                    "Önce bir video promptu gir."
                )

            else:

                with st.spinner(
                    "Video oluşturuluyor... Bu işlem biraz sürebilir."
                ):

                    video, error = generate_text_video(
                        video_prompt,
                        num_frames=num_frames,
                        steps=steps,
                    )

                if error:

                    st.error(
                        "Text → Video üretilemedi."
                    )

                    st.error(
                        "Gerçek Hugging Face / Provider hatası:"
                    )

                    st.code(error)

                elif video:

                    st.session_state.generated_video = video

                    st.session_state.video_filename = (
                        "kogce_text_to_video.mp4"
                    )

                    st.video(video)

                    st.download_button(
                        "⬇️ MP4 İndir",
                        data=video,
                        file_name=(
                            st.session_state.video_filename
                        ),
                        mime="video/mp4",
                        use_container_width=True,
                    )

                    st.success(
                        "Video başarıyla üretildi."
                    )


    # --------------------------------------------------------
    # IMAGE TO VIDEO
    # --------------------------------------------------------

    else:

        st.subheader("Image → Video")

        st.caption(
            "Motor: LTX-Video 0.9.8 13B Distilled"
        )

        uploaded_file = st.file_uploader(
            "Görsel yükle",
            type=[
                "png",
                "jpg",
                "jpeg",
                "webp",
            ],
        )

        motion_prompt = st.text_area(
            "Motion Prompt",
            placeholder=(
                "The subject moves naturally, "
                "camera slowly pushes forward, "
                "cinematic motion..."
            ),
            height=140,
        )

        if uploaded_file:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                image,
                caption="Kaynak Görsel",
                use_container_width=True,
            )

            if st.button(
                "🎬 Image → Video Üret",
                use_container_width=True,
            ):

                with st.spinner(
                    "Video oluşturuluyor..."
                ):

                    video, error = generate_image_video(
                        image,
                        motion_prompt,
                        num_frames=49,
                    )

                if error:

                    st.error(
                        "Image → Video üretilemedi."
                    )

                    st.code(error)

                elif video:

                    st.session_state.generated_video = video

                    st.session_state.video_filename = (
                        "kogce_image_to_video.mp4"
                    )

                    st.video(video)

                    st.download_button(
                        "⬇️ MP4 İndir",
                        data=video,
                        file_name=(
                            st.session_state.video_filename
                        ),
                        mime="video/mp4",
                        use_container_width=True,
                    )

                    st.success(
                        "Video başarıyla üretildi."
                    )


# ============================================================
# AI TOOLS TAB
# ============================================================

with tabs[2]:

    st.header("AI Araçları")

    tool = st.selectbox(
        "AI Aracı",
        [
            "AI Prompt Robotu",
            "AI Senaryo Yazarı",
            "AI Sahne Planlayıcı",
            "AI Video Prompt Üretici",
            "AI Shorts Fikir Motoru",
        ],
    )


    # --------------------------------------------------------
    # PROMPT ROBOT
    # --------------------------------------------------------

    if tool == "AI Prompt Robotu":

        st.subheader("AI Prompt Robotu")

        idea = st.text_area(
            "Fikrin",
            height=150,
            placeholder="Kısa fikrini yaz...",
        )

        if st.button(
            "Prompt Oluştur",
            use_container_width=True,
        ):

            if not idea.strip():

                st.warning("Önce fikir gir.")

            else:

                with st.spinner(
                    "Profesyonel prompt hazırlanıyor..."
                ):

                    result, error = call_ai(
                        """
                        You are an expert AI image prompt engineer.

                        Transform the user's idea into an extremely
                        detailed English image-generation prompt.

                        Include:
                        subject, environment, composition,
                        camera angle, lens, lighting, colors,
                        materials, atmosphere, depth, realism,
                        and visual quality.

                        Return only the final prompt.
                        """,
                        idea,
                        temperature=0.75,
                        max_tokens=1800,
                    )

                if error:

                    st.error(error)

                else:

                    st.text_area(
                        "Generated Prompt",
                        value=result,
                        height=300,
                    )


    # --------------------------------------------------------
    # SCRIPT WRITER
    # --------------------------------------------------------

    elif tool == "AI Senaryo Yazarı":

        st.subheader("AI Senaryo Yazarı")

        topic = st.text_area(
            "Video konusu",
            height=150,
        )

        duration = st.selectbox(
            "Hedef süre",
            [
                "30 saniye",
                "60 saniye",
                "90 saniye",
                "3 dakika",
            ],
        )

        if st.button(
            "Senaryo Yaz",
            use_container_width=True,
        ):

            if not topic.strip():

                st.warning("Konu gir.")

            else:

                with st.spinner(
                    "Senaryo hazırlanıyor..."
                ):

                    result, error = call_ai(
                        """
                        You are a professional viral video script writer.

                        Create a complete, engaging script based on
                        the user's topic.

                        Structure it with:
                        - Hook
                        - Setup
                        - Development
                        - Retention moments
                        - Payoff
                        - Ending

                        Keep the language natural and suitable for
                        short-form video.

                        Return the complete script.
                        """,
                        f"""
                        Topic:
                        {topic}

                        Target duration:
                        {duration}
                        """,
                        temperature=0.8,
                        max_tokens=2500,
                    )

                if error:

                    st.error(error)

                else:

                    st.session_state.last_script = result

                    st.text_area(
                        "Senaryo",
                        value=result,
                        height=450,
                    )


    # --------------------------------------------------------
    # SCENE PLANNER
    # --------------------------------------------------------

    elif tool == "AI Sahne Planlayıcı":

        st.subheader("AI Sahne Planlayıcı")

        script = st.text_area(
            "Senaryoyu gir",
            height=250,
        )

        if st.button(
            "Sahneleri Planla",
            use_container_width=True,
        ):

            if not script.strip():

                st.warning("Önce senaryo gir.")

            else:

                with st.spinner(
                    "Sahne planı hazırlanıyor..."
                ):

                    result, error = call_ai(
                        """
                        You are a professional film director,
                        storyboard artist and AI video production planner.

                        Break the supplied script into logical scenes.

                        For every scene provide:
                        1. Scene number
                        2. Duration
                        3. Visual description
                        4. Camera movement
                        5. Character action
                        6. Environment
                        7. Lighting
                        8. Audio/SFX
                        9. AI generation notes

                        Make the scenes visually consistent.
                        """,
                        script,
                        temperature=0.75,
                        max_tokens=3000,
                    )

                if error:

                    st.error(error)

                else:

                    st.session_state.last_scenes = result

                    st.text_area(
                        "Sahne Planı",
                        value=result,
                        height=550,
                    )


    # --------------------------------------------------------
    # VIDEO PROMPT GENERATOR
    # --------------------------------------------------------

    elif tool == "AI Video Prompt Üretici":

        st.subheader("AI Video Prompt Üretici")

        scene = st.text_area(
            "Sahne fikri",
            height=180,
        )

        if st.button(
            "Video Prompt Oluştur",
            use_container_width=True,
        ):

            if not scene.strip():

                st.warning("Sahne fikri gir.")

            else:

                with st.spinner(
                    "Video prompt hazırlanıyor..."
                ):

                    result, error = call_ai(
                        """
                        You are an expert AI video-generation prompt
                        engineer.

                        Convert the scene idea into a detailed
                        production-ready English video prompt.

                        Focus on:
                        subject movement,
                        camera movement,
                        environment movement,
                        lighting,
                        realistic physics,
                        cinematic composition,
                        depth,
                        timing,
                        atmosphere,
                        visual consistency.

                        Do not explain.
                        Return only the final video prompt.
                        """,
                        scene,
                        temperature=0.75,
                        max_tokens=1800,
                    )

                if error:

                    st.error(error)

                else:

                    st.text_area(
                        "Video Prompt",
                        value=result,
                        height=350,
                    )


    # --------------------------------------------------------
    # SHORTS IDEA ENGINE
    # --------------------------------------------------------

    elif tool == "AI Shorts Fikir Motoru":

        st.subheader("AI Shorts Fikir Motoru")

        niche = st.text_input(
            "Niş / konu",
            placeholder="Örneğin: AI, gaming, animals, satisfying...",
        )

        count = st.slider(
            "Fikir sayısı",
            min_value=5,
            max_value=20,
            value=10,
        )

        if st.button(
            "Fikirleri Üret",
            use_container_width=True,
        ):

            if not niche.strip():

                st.warning("Bir niş gir.")

            else:

                with st.spinner(
                    "Viral Shorts fikirleri araştırılıyor..."
                ):

                    result, error = call_ai(
                        """
                        You are a global short-form video strategist.

                        Generate original YouTube Shorts ideas.

                        Prioritize:
                        - strong first-second hooks
                        - visual simplicity
                        - high retention
                        - curiosity
                        - replayability
                        - international appeal
                        - easy AI/video production

                        For every idea provide:
                        1. Title
                        2. Hook
                        3. Concept
                        4. Why viewers may keep watching
                        5. Visual production idea

                        Do not repeat generic ideas.
                        """,
                        f"""
                        Niche:
                        {niche}

                        Number of ideas:
                        {count}
                        """,
                        temperature=0.9,
                        max_tokens=3500,
                    )

                if error:

                    st.error(error)

                else:

                    st.text_area(
                        "Shorts Fikirleri",
                        value=result,
                        height=600,
                    )


# ============================================================
# GALLERY TAB
# ============================================================

with tabs[3]:

    st.header("Galeri")

    if not st.session_state.gallery:

        st.info(
            "Henüz bu oturumda oluşturulmuş bir görsel yok."
        )

    else:

        for item in reversed(
            st.session_state.gallery
        ):

            st.image(
                item["image"],
                use_container_width=True,
            )

            st.caption(
                "Orijinal Prompt"
            )

            st.write(
                item["prompt"]
            )

            st.caption(
                "Final Prompt"
            )

            st.write(
                item["final_prompt"]
            )

            st.divider()


# ============================================================
# SYSTEM TAB
# ============================================================

with tabs[4]:

    st.header("Sistem")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "HF API",
            "ONLINE" if HF_API_KEY else "MISSING",
        )

    with col2:

        st.metric(
            "Görsel Motoru",
            "FLUX",
        )

    with col3:

        st.metric(
            "Video Motoru",
            "WAN + LTX",
        )

    st.divider()

    st.subheader("Aktif Model ID'leri")

    st.code(
        f"""
Prompt:
{PROMPT_MODEL}

Image:
{IMAGE_MODEL}

Text → Video:
{VIDEO_T2V_MODEL}

Image → Video:
{VIDEO_I2V_MODEL}
""".strip()
    )

    st.divider()

    st.subheader("Oturum")

    st.write(
        f"Galerideki görsel sayısı: "
        f"{len(st.session_state.gallery)}"
    )

    if st.session_state.generated_video:

        st.success(
            "Son video üretimi mevcut."
        )

    else:

        st.info(
            "Bu oturumda henüz başarılı video üretimi yok."
        )

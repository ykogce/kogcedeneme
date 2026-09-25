import io
import os
import json
import base64
import requests
import streamlit as st

from PIL import Image
from huggingface_hub import InferenceClient


# ============================================================
# KOGCE AI STUDIO
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

VIDEO_MODEL = "Lightricks/LTX-Video-0.9.8-13B-distilled"

CHAT_URL = "https://router.huggingface.co/v1/chat/completions"


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
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HF KEY
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

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(124, 58, 237, 0.18),
                transparent 30%
            ),
            radial-gradient(
                circle at 85% 15%,
                rgba(168, 85, 247, 0.13),
                transparent 28%
            ),
            #08080d;
    }

    section[data-testid="stSidebar"] {
        background: #0d0d14;
        border-right: 1px solid rgba(168, 85, 247, 0.18);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    .login-box {
        max-width: 560px;
        margin: 90px auto 0 auto;
        padding: 45px;
        border-radius: 28px;
        background: rgba(18, 18, 28, 0.82);
        border: 1px solid rgba(168, 85, 247, 0.28);
        box-shadow:
            0 30px 90px rgba(0,0,0,0.45),
            0 0 50px rgba(124,58,237,0.08);
        text-align: center;
    }

    .status-card {
        padding: 18px;
        border-radius: 18px;
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 12px;
    }

    .tool-card {
        padding: 20px;
        border-radius: 20px;
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(168,85,247,0.16);
        margin-bottom: 18px;
    }

    .small-muted {
        color: #9898a8;
        font-size: 0.9rem;
    }

    div[data-testid="stTextArea"] textarea {
        background: #ffffff !important;
        color: #111111 !important;
        border-radius: 14px !important;
        border: 1px solid #d0d0d0 !important;
    }

    div[data-testid="stTextInput"] input {
        border-radius: 12px !important;
    }

    div.stButton > button {
        border-radius: 12px;
        min-height: 44px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# AI CHAT FUNCTION
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
            return None, f"AI API hatası: {response.status_code}\n{response.text}"

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
# VIDEO GENERATION
# ============================================================

def generate_video(prompt, num_frames=81, steps=25):
    if not HF_API_KEY:
        return None, "HF_API_KEY bulunamadı."

    try:
        client = InferenceClient(
            provider="auto",
            api_key=HF_API_KEY,
        )

        video = client.text_to_video(
            prompt=prompt,
            model=VIDEO_MODEL,
            num_frames=num_frames,
            num_inference_steps=steps,
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

        st.markdown("")

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

    st.success("AI SYSTEM ONLINE" if HF_API_KEY else "HF KEY MISSING")

    st.write("### Motorlar")

    st.write("● AI Prompt Robotu")
    st.write("● AI Senaryo")
    st.write("● AI Scene Planner")
    st.write("● FLUX Görsel")
    st.write("● Text → Video")
    st.write("● Image → Video")

    st.divider()

    st.caption(f"Galeri: {len(st.session_state.gallery)} görsel")

    if st.button("Çıkış Yap", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()


# ============================================================
# MAIN HEADER
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

    col1, col2 = st.columns([1.4, 1])

    with col1:

        prompt = st.text_area(
            "Görsel fikrin",
            height=150,
            placeholder=(
                "Örneğin: gece neon ışıkları altında çalışan "
                "futuristic robot kedi..."
            ),
            key="image_prompt",
        )

        use_ai_boost = st.toggle(
            "AI Prompt Robotu",
            value=True,
        )

        if use_ai_boost:
            st.success("AI Prompt Robotu aktif")
        else:
            st.info("AI Prompt Robotu kapalı")

    with col2:

        aspect = st.selectbox(
            "Görsel oranı",
            [
                "1:1 Kare",
                "9:16 Dikey",
                "16:9 Yatay",
            ],
        )

        if aspect == "1:1 Kare":
            width, height = 768, 768
        elif aspect == "9:16 Dikey":
            width, height = 768, 1344
        else:
            width, height = 1344, 768

        st.info(
            f"Çözünürlük: {width} × {height}"
        )

    if st.button(
        "✨ Görsel Oluştur",
        type="primary",
        use_container_width=True,
    ):

        if not prompt.strip():
            st.warning("Önce bir fikir veya prompt gir.")
            st.stop()

        final_prompt = prompt

        if use_ai_boost:

            with st.spinner("AI prompt geliştiriyor..."):

                enhanced, error = call_ai(
                    """
                    You are a professional image-generation prompt engineer.

                    Transform the user's short idea into a detailed,
                    cinematic English prompt suitable for FLUX.

                    Include:
                    subject,
                    environment,
                    composition,
                    camera,
                    lighting,
                    materials,
                    colors,
                    atmosphere,
                    visual quality.

                    Do not explain the prompt.
                    Return only the final English generation prompt.
                    """,
                    prompt,
                    temperature=0.7,
                    max_tokens=1000,
                )

            if error:
                st.error(error)
                st.stop()

            final_prompt = enhanced

            st.subheader("AI tarafından geliştirilen prompt")
            st.write(final_prompt)

        st.session_state.last_prompt = prompt
        st.session_state.last_enhanced_prompt = final_prompt

        with st.spinner("FLUX görsel oluşturuyor..."):

            image, error = generate_image(
                final_prompt,
                width,
                height,
            )

        if error:
            st.error(error)
        else:

            st.image(
                image,
                use_container_width=True,
            )

            image_buffer = io.BytesIO()
            image.save(image_buffer, format="PNG")

            st.download_button(
                "PNG İndir",
                data=image_buffer.getvalue(),
                file_name="kogce_ai_image.png",
                mime="image/png",
                use_container_width=True,
            )

            st.session_state.gallery.append(
                {
                    "image": image.copy(),
                    "original_prompt": prompt,
                    "final_prompt": final_prompt,
                }
            )

            st.success("Görsel hazır.")


# ============================================================
# VIDEO TAB
# ============================================================

with tabs[1]:

    st.header("Video Studio")

    video_mode = st.radio(
        "Video üretim yöntemi",
        [
            "Text → Video",
            "Image → Video",
        ],
        horizontal=True,
    )

    st.divider()

    if video_mode == "Text → Video":

        video_prompt = st.text_area(
            "Video promptu",
            height=170,
            placeholder=(
                "Örneğin: A cinematic orange cat running through "
                "a futuristic neon city, dynamic camera movement..."
            ),
        )

        video_duration = st.selectbox(
            "Video uzunluğu",
            [
                "Kısa",
                "Orta",
            ],
        )

        if video_duration == "Kısa":
            num_frames = 49
        else:
            num_frames = 81

        video_steps = st.slider(
            "Üretim adımı",
            min_value=10,
            max_value=40,
            value=25,
            step=5,
        )

        st.info(
            "Video üretimi Hugging Face üzerinden model sağlayıcısına gönderilir. "
            "Video modellerinde sağlayıcı/model erişimi hesaba göre değişebilir."
        )

        if st.button(
            "🎬 Video Oluştur",
            type="primary",
            use_container_width=True,
        ):

            if not video_prompt.strip():
                st.warning("Video promptu gir.")
                st.stop()

            with st.spinner(
                "Video oluşturuluyor. Bu işlem görsel üretimden daha uzun sürebilir..."
            ):

                video, error = generate_video(
                    video_prompt,
                    num_frames=num_frames,
                    steps=video_steps,
                )

            if error:

                st.error("Video üretilemedi.")
                st.code(error)

                st.info(
                    "Bu hata video modelinin Hugging Face üzerindeki "
                    "Inference Provider erişimiyle ilgili olabilir."
                )

            else:

                st.session_state.generated_video = video
                st.session_state.video_filename = "kogce_video.mp4"

                st.video(video)

                st.download_button(
                    "MP4 İndir",
                    data=video,
                    file_name="kogce_video.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )

                st.success("Video hazır.")

    else:

        st.subheader("Image → Video")

        uploaded_image = st.file_uploader(
            "Başlangıç görselini yükle",
            type=[
                "png",
                "jpg",
                "jpeg",
                "webp",
            ],
        )

        image_video_prompt = st.text_area(
            "Hareket promptu",
            height=150,
            placeholder=(
                "Camera slowly moves forward, subject smiles, "
                "hair and clothes move naturally..."
            ),
        )

        if uploaded_image:

            input_image = Image.open(
                uploaded_image
            ).convert("RGB")

            st.image(
                input_image,
                caption="Başlangıç görseli",
                use_container_width=True,
            )

        st.info(
            "Image → Video için seçilen sağlayıcının image-to-video "
            "desteği bulunması gerekir."
        )

        if st.button(
            "🎥 Görselden Video Oluştur",
            type="primary",
            use_container_width=True,
        ):

            if uploaded_image is None:
                st.warning("Önce bir görsel yükle.")
                st.stop()

            if not image_video_prompt.strip():
                st.warning("Hareket promptu gir.")
                st.stop()

            try:

                client = InferenceClient(
                    provider="auto",
                    api_key=HF_API_KEY,
                )

                with st.spinner(
                    "Görsel videoya dönüştürülüyor..."
                ):

                    video = client.image_to_video(
                        input_image,
                        model=VIDEO_MODEL,
                        prompt=image_video_prompt,
                        num_frames=49,
                    )

                st.video(video)

                st.download_button(
                    "MP4 İndir",
                    data=video,
                    file_name="kogce_image_to_video.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )

            except Exception as e:

                st.error(
                    "Image → Video üretilemedi."
                )

                st.code(str(e))


# ============================================================
# AI TOOLS TAB
# ============================================================

with tabs[2]:

    st.header("AI Araçları")

    tool = st.selectbox(
        "Araç seç",
        [
            "AI Prompt Robotu",
            "AI Senaryo Yazarı",
            "AI Sahne Planlayıcı",
            "AI Video Prompt Üretici",
            "AI Shorts Fikir Motoru",
        ],
    )

    st.divider()


    # --------------------------------------------------------
    # PROMPT ROBOT
    # --------------------------------------------------------

    if tool == "AI Prompt Robotu":

        st.subheader("AI Prompt Robotu")

        idea = st.text_area(
            "Fikrini yaz",
            height=160,
            placeholder="Kısa fikrini yaz..."
        )

        style = st.selectbox(
            "Stil",
            [
                "Cinematic",
                "Photorealistic",
                "3D Animation",
                "Fantasy",
                "Sci-Fi",
                "Commercial",
                "Cute",
                "Dark",
            ],
        )

        if st.button(
            "Prompt Oluştur",
            type="primary",
            use_container_width=True,
        ):

            if not idea.strip():
                st.warning("Bir fikir gir.")
                st.stop()

            result, error = call_ai(
                """
                You are an expert AI image and video prompt engineer.

                Turn the user's idea into a highly detailed English
                production prompt.

                Include subject, environment, composition, camera,
                lighting, motion, materials, atmosphere and quality.

                Keep the result practical for generative AI.
                """,
                f"""
                User idea:
                {idea}

                Desired style:
                {style}
                """,
                max_tokens=1400,
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
            height=130,
        )

        duration = st.selectbox(
            "Video tipi",
            [
                "YouTube Short",
                "1 dakika",
                "3 dakika",
                "5 dakika",
            ],
        )

        language = st.selectbox(
            "Dil",
            [
                "English",
                "Türkçe",
            ],
        )

        if st.button(
            "Senaryo Oluştur",
            type="primary",
            use_container_width=True,
        ):

            if not topic.strip():
                st.warning("Konu gir.")
                st.stop()

            script, error = call_ai(
                """
                You are a professional YouTube scriptwriter.

                Create a concise, engaging video script.

                Structure:
                Hook
                Development
                Payoff
                Ending

                Make every section useful for actual video production.
                """,
                f"""
                Topic:
                {topic}

                Video format:
                {duration}

                Language:
                {language}
                """,
                max_tokens=2500,
            )

            if error:
                st.error(error)
            else:

                st.session_state.last_script = script

                st.text_area(
                    "Senaryo",
                    value=script,
                    height=500,
                )


    # --------------------------------------------------------
    # SCENE PLANNER
    # --------------------------------------------------------

    elif tool == "AI Sahne Planlayıcı":

        st.subheader("AI Sahne Planlayıcı")

        scene_story = st.text_area(
            "Senaryoyu veya fikri gir",
            height=220,
        )

        scene_count = st.slider(
            "Sahne sayısı",
            2,
            20,
            6,
        )

        if st.button(
            "Sahneleri Planla",
            type="primary",
            use_container_width=True,
        ):

            if not scene_story.strip():
                st.warning("Senaryo veya fikir gir.")
                st.stop()

            scenes, error = call_ai(
                """
                You are a professional AI video director.

                Break the story into production-ready scenes.

                For each scene provide:

                Scene number
                Duration
                Visual description
                Camera
                Motion
                Lighting
                Environment
                Image prompt
                Video prompt
                Voiceover

                Keep character appearance consistent.
                """,
                f"""
                Story:
                {scene_story}

                Number of scenes:
                {scene_count}
                """,
                max_tokens=4500,
            )

            if error:
                st.error(error)
            else:

                st.session_state.last_scenes = scenes

                st.text_area(
                    "Sahne Planı",
                    value=scenes,
                    height=650,
                )


    # --------------------------------------------------------
    # VIDEO PROMPT
    # --------------------------------------------------------

    elif tool == "AI Video Prompt Üretici":

        st.subheader("AI Video Prompt Üretici")

        idea = st.text_area(
            "Video fikri",
            height=150,
        )

        camera = st.selectbox(
            "Kamera",
            [
                "Cinematic",
                "Close-up",
                "Wide shot",
                "Tracking shot",
                "Slow push-in",
                "Handheld",
                "Drone",
            ],
        )

        motion = st.text_input(
            "Hareket",
            placeholder="running, jumping, turning..."
        )

        if st.button(
            "Video Prompt Oluştur",
            type="primary",
            use_container_width=True,
        ):

            if not idea.strip():
                st.warning("Fikir gir.")
                st.stop()

            result, error = call_ai(
                """
                You are an expert text-to-video prompt engineer.

                Create a professional English video-generation prompt.

                Focus on:
                subject consistency,
                physical motion,
                camera movement,
                environment,
                lighting,
                cinematic composition,
                realistic temporal consistency.

                Return only the final prompt.
                """,
                f"""
                Idea:
                {idea}

                Camera:
                {camera}

                Motion:
                {motion}
                """,
                max_tokens=1600,
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

    else:

        st.subheader("AI Shorts Fikir Motoru")

        niche = st.text_input(
            "Niş",
            placeholder="Gaming, AI, animals, satisfying..."
        )

        count = st.slider(
            "Fikir sayısı",
            5,
            30,
            10,
        )

        if st.button(
            "Fikirleri Üret",
            type="primary",
            use_container_width=True,
        ):

            if not niche.strip():
                st.warning("Niş gir.")
                st.stop()

            ideas, error = call_ai(
                """
                You are a global YouTube Shorts creative strategist.

                Generate original Shorts concepts.

                For every concept include:
                title,
                hook,
                concept,
                visual,
                ending,
                replay/loop idea.

                Focus on concepts that can be produced with AI tools.
                """,
                f"""
                Niche:
                {niche}

                Number of ideas:
                {count}
                """,
                max_tokens=4000,
            )

            if error:
                st.error(error)
            else:
                st.text_area(
                    "Shorts Fikirleri",
                    value=ideas,
                    height=650,
                )


# ============================================================
# GALLERY
# ============================================================

with tabs[3]:

    st.header("Galeri")

    if not st.session_state.gallery:

        st.info(
            "Henüz bu oturumda oluşturulmuş görsel yok."
        )

    else:

        for index, item in enumerate(
            reversed(st.session_state.gallery),
            start=1,
        ):

            st.subheader(
                f"Görsel {index}"
            )

            c1, c2 = st.columns([1, 1])

            with c1:
                st.image(
                    item["image"],
                    use_container_width=True,
                )

            with c2:

                st.write(
                    "**Orijinal fikir**"
                )

                st.write(
                    item["original_prompt"]
                )

                st.write(
                    "**Final prompt**"
                )

                st.write(
                    item["final_prompt"]
                )

            st.divider()


# ============================================================
# SYSTEM
# ============================================================

with tabs[4]:

    st.header("Sistem")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "HF API",
            "AKTİF" if HF_API_KEY else "YOK",
        )

    with c2:

        st.metric(
            "Görsel Motoru",
            "FLUX",
        )

    with c3:

        st.metric(
            "Video Motoru",
            "LTX",
        )

    st.divider()

    st.subheader("Aktif modeller")

    st.write(
        f"Image: `{IMAGE_MODEL}`"
    )

    st.write(
        f"AI: `{PROMPT_MODEL}`"
    )

    st.write(
        f"Video: `{VIDEO_MODEL}`"
    )

    st.divider()

    st.subheader("Oturum")

    st.write(
        f"Oluşturulan görsel: {len(st.session_state.gallery)}"
    )

    if st.session_state.generated_video:

        st.success(
            "Bu oturumda video oluşturuldu."
        )

    else:

        st.info(
            "Bu oturumda henüz video oluşturulmadı."
        )

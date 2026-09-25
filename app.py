import io
import requests
import streamlit as st
from PIL import Image
from huggingface_hub import InferenceClient


# =========================================================
# KOGCE AI STUDIO
# =========================================================

st.set_page_config(
    page_title="KOGCE AI Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# TEMA
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            135deg,
            #09090f 0%,
            #11101c 50%,
            #0b0912 100%
        );
    }

    .main {
        color: #f1f5f9;
    }

    h1, h2, h3, h4 {
        color: #ffffff !important;
    }

    p, label, span, div {
        color: #f1f5f9;
    }

    .stCaption {
        color: #cbd5e1 !important;
    }

    [data-testid="stSidebar"] {
        background: #0c0b13;
        border-right: 1px solid #272333;
    }

    [data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }

    textarea,
    input {
        background-color: #15131e !important;
        color: #ffffff !important;
        border: 1px solid #3b354b !important;
    }

    textarea::placeholder,
    input::placeholder {
        color: #94a3b8 !important;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid #4c4261;
        background: #171321;
        color: #ffffff;
    }

    .stButton > button:hover {
        border-color: #8b5cf6;
        color: #ffffff;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# AYARLAR
# =========================================================

APP_PASSWORD = "1234"

PROMPT_MODEL = "openai/gpt-oss-120b"

IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"

VIDEO_T2V_MODEL = "Wan-AI/Wan2.1-T2V-1.3B"

VIDEO_I2V_MODEL = "Lightricks/LTX-Video-0.9.8-13B-distilled"

CHAT_URL = "https://router.huggingface.co/v1/chat/completions"


# =========================================================
# SESSION STATE
# =========================================================

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


# =========================================================
# HUGGING FACE API KEY
# =========================================================

def get_hf_key():
    try:
        return st.secrets["HF_API_KEY"]
    except Exception:
        return None


HF_API_KEY = get_hf_key()


# =========================================================
# AI CHAT
# =========================================================

def call_ai(
    system_prompt,
    user_prompt,
    temperature=0.7,
    max_tokens=1200,
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
            return None, (
                f"HTTP {response.status_code}\n\n"
                f"{response.text}"
            )

        data = response.json()

        choices = data.get("choices", [])

        if not choices:
            return None, (
                "Model cevap vermedi.\n\n"
                f"{data}"
            )

        message = choices[0].get("message", {})

        content = message.get("content")

        if not content:
            return None, (
                "Model boş cevap döndürdü.\n\n"
                f"{data}"
            )

        return content.strip(), None

    except Exception as e:
        return None, (
            f"{type(e).__name__}: {str(e)}"
        )


# =========================================================
# AI PROMPT ROBOTU
# =========================================================

def enhance_image_prompt(user_prompt):

    system_prompt = """
You are a professional AI image prompt engineer.

Transform the user's short idea into a highly detailed English image
generation prompt.

Include:
- subject
- appearance
- environment
- composition
- camera angle
- lighting
- colors
- materials
- atmosphere
- visual quality

Keep the original idea.
Do not explain anything.
Return only the final English image prompt.
"""

    return call_ai(
        system_prompt,
        user_prompt,
        temperature=0.75,
        max_tokens=900,
    )


# =========================================================
# IMAGE GENERATION
# =========================================================

def generate_image(
    prompt,
    width,
    height,
):
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

        return None, (
            f"{type(e).__name__}: {str(e)}"
        )


# =========================================================
# TEXT → VIDEO
# =========================================================

def generate_text_video(
    prompt,
    num_frames=49,
    steps=20,
):
    if not HF_API_KEY:
        return None, "HF_API_KEY bulunamadı."

    try:

        client = InferenceClient(
            provider="fal-ai",
            api_key=HF_API_KEY,
        )

        video = client.text_to_video(
            prompt=prompt,
            model=VIDEO_T2V_MODEL,
            num_frames=num_frames,
            num_inference_steps=steps,
        )

        if video is None:
            return None, (
                "Provider boş video döndürdü."
            )

        return video, None

    except Exception as e:

        error_text = (
            f"{type(e).__name__}: {str(e)}"
        )

        return None, error_text


# =========================================================
# IMAGE → VIDEO
# =========================================================

def generate_image_video(
    image,
    prompt,
):
    if not HF_API_KEY:
        return None, "HF_API_KEY bulunamadı."

    try:

        client = InferenceClient(
            provider="auto",
            api_key=HF_API_KEY,
        )

        video = client.image_to_video(
            image=image,
            prompt=prompt,
            model=VIDEO_I2V_MODEL,
        )

        if video is None:
            return None, (
                "Provider boş video döndürdü."
            )

        return video, None

    except Exception as e:

        return None, (
            f"{type(e).__name__}: {str(e)}"
        )


# =========================================================
# LOGIN
# =========================================================

if not st.session_state.authenticated:

    st.markdown(
        "<div style='height:14vh'></div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(
        [1, 2, 1]
    )

    with col2:

        st.title("KOGCE AI Studio")

        st.caption(
            "AI CREATIVE WORKSPACE"
        )

        st.write(
            "Create. Imagine. Generate."
        )

        st.write("")

        password = st.text_input(
            "Şifre",
            type="password",
            placeholder="Şifrenizi girin",
        )

        if st.button(
            "Giriş Yap",
            use_container_width=True,
        ):

            if password == APP_PASSWORD:

                st.session_state.authenticated = True

                st.rerun()

            else:

                st.error(
                    "Şifre yanlış."
                )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title(
        "KOGCE AI Studio"
    )

    if HF_API_KEY:

        st.success(
            "AI System Online"
        )

    else:

        st.error(
            "HF API Key bulunamadı"
        )

    st.divider()

    st.markdown(
        "### AI Motorları"
    )

    st.write(
        "Prompt AI"
    )

    st.write(
        "Scenario Writer"
    )

    st.write(
        "Scene Planner"
    )

    st.write(
        "Video Prompt AI"
    )

    st.divider()

    st.write(
        "FLUX.1-schnell"
    )

    st.write(
        "Wan2.1 T2V"
    )

    st.write(
        "LTX Video I2V"
    )

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


# =========================================================
# ANA MENÜ
# =========================================================

tabs = st.tabs(
    [
        "✨ Görsel",
        "🎬 Video",
        "🧠 AI Araçları",
        "🖼️ Galeri",
        "⚙️ Sistem",
    ]
)


# =========================================================
# GÖRSEL
# =========================================================

with tabs[0]:

    st.header(
        "AI Görsel Üretici"
    )

    prompt = st.text_area(
        "Prompt",
        height=150,
        placeholder=(
            "Örneğin: A futuristic city "
            "with a giant orange cat..."
        ),
        key="image_prompt",
    )

    ai_prompt_enabled = st.toggle(
        "AI Prompt Robotu",
        value=True,
    )

    aspect = st.selectbox(
        "Görüntü Oranı",
        [
            "1:1",
            "9:16",
            "16:9",
        ],
    )

    dimensions = {
        "1:1": (768, 768),
        "9:16": (768, 1344),
        "16:9": (1344, 768),
    }

    width, height = dimensions[aspect]

    st.caption(
        f"Çözünürlük: {width} × {height}"
    )

    if st.button(
        "✨ Görsel Oluştur",
        use_container_width=True,
    ):

        if not prompt.strip():

            st.warning(
                "Önce bir prompt gir."
            )

        else:

            final_prompt = prompt

            if ai_prompt_enabled:

                with st.spinner(
                    "AI prompt'u geliştiriyor..."
                ):

                    enhanced, error = (
                        enhance_image_prompt(
                            prompt
                        )
                    )

                if error:

                    st.error(
                        "AI Prompt Robotu çalışmadı."
                    )

                    st.code(
                        error,
                        language="text",
                    )

                else:

                    final_prompt = enhanced

                    st.session_state.last_enhanced_prompt = (
                        enhanced
                    )

                    with st.expander(
                        "Oluşturulan gelişmiş prompt"
                    ):

                        st.write(
                            enhanced
                        )

            st.session_state.last_prompt = (
                prompt
            )

            with st.spinner(
                "Görsel oluşturuluyor..."
            ):

                image, error = generate_image(
                    final_prompt,
                    width,
                    height,
                )

            if error:

                st.error(
                    "Görsel üretilemedi."
                )

                st.code(
                    error,
                    language="text",
                )

            else:

                st.image(
                    image,
                    use_container_width=True,
                )

                buffer = io.BytesIO()

                image.save(
                    buffer,
                    format="PNG",
                )

                st.download_button(
                    "PNG indir",
                    data=buffer.getvalue(),
                    file_name=(
                        "kogce_ai_image.png"
                    ),
                    mime="image/png",
                    use_container_width=True,
                )

                st.session_state.gallery.append(
                    {
                        "image": image,
                        "original_prompt": prompt,
                        "final_prompt": final_prompt,
                    }
                )

                st.success(
                    "Görsel hazır."
                )


# =========================================================
# VIDEO
# =========================================================

with tabs[1]:

    st.header(
        "AI Video Generator"
    )

    video_mode = st.radio(
        "Video Türü",
        [
            "Text → Video",
            "Image → Video",
        ],
        horizontal=True,
    )

    # =====================================================
    # TEXT → VIDEO
    # =====================================================

    if video_mode == "Text → Video":

        st.subheader(
            "Text → Video"
        )

        video_prompt = st.text_area(
            "Video Prompt",
            height=160,
            placeholder=(
                "A cute orange cat running "
                "through a futuristic neon city, "
                "cinematic camera movement..."
            ),
            key="t2v_prompt",
        )

        duration = st.selectbox(
            "Video Süresi",
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
            step=1,
        )

        st.caption(
            f"Model: {VIDEO_T2V_MODEL}"
        )

        if st.button(
            "🎬 Text → Video Oluştur",
            use_container_width=True,
        ):

            if not video_prompt.strip():

                st.warning(
                    "Önce video prompt'u gir."
                )

            else:

                with st.spinner(
                    "Video oluşturuluyor..."
                ):

                    video, error = (
                        generate_text_video(
                            video_prompt,
                            num_frames=num_frames,
                            steps=steps,
                        )
                    )

                if error:

                    st.error(
                        "Text → Video üretilemedi."
                    )

                    st.error(
                        "Gerçek Hugging Face / Provider hatası:"
                    )

                    st.code(
                        error,
                        language="text",
                    )

                else:

                    st.session_state.generated_video = (
                        video
                    )

                    st.session_state.video_filename = (
                        "kogce_text_to_video.mp4"
                    )

                    st.success(
                        "Video hazır."
                    )

                    st.video(
                        video
                    )

                    st.download_button(
                        "MP4 indir",
                        data=video,
                        file_name=(
                            "kogce_text_to_video.mp4"
                        ),
                        mime="video/mp4",
                        use_container_width=True,
                    )

    # =====================================================
    # IMAGE → VIDEO
    # =====================================================

    else:

        st.subheader(
            "Image → Video"
        )

        uploaded = st.file_uploader(
            "Başlangıç görseli yükle",
            type=[
                "png",
                "jpg",
                "jpeg",
                "webp",
            ],
        )

        motion_prompt = st.text_area(
            "Hareket Prompt'u",
            height=150,
            placeholder=(
                "The character slowly walks "
                "forward, camera smoothly follows..."
            ),
        )

        if uploaded:

            image = Image.open(
                uploaded
            ).convert("RGB")

            st.image(
                image,
                caption="Başlangıç Görseli",
                use_container_width=True,
            )

            st.caption(
                f"Model: {VIDEO_I2V_MODEL}"
            )

            if st.button(
                "🎬 Image → Video Oluştur",
                use_container_width=True,
            ):

                if not motion_prompt.strip():

                    st.warning(
                        "Hareket prompt'u gir."
                    )

                else:

                    with st.spinner(
                        "Image → Video oluşturuluyor..."
                    ):

                        video, error = (
                            generate_image_video(
                                image,
                                motion_prompt,
                            )
                        )

                    if error:

                        st.error(
                            "Image → Video üretilemedi."
                        )

                        st.error(
                            "Gerçek Hugging Face / Provider hatası:"
                        )

                        st.code(
                            error,
                            language="text",
                        )

                    else:

                        st.session_state.generated_video = (
                            video
                        )

                        st.session_state.video_filename = (
                            "kogce_image_to_video.mp4"
                        )

                        st.success(
                            "Video hazır."
                        )

                        st.video(
                            video
                        )

                        st.download_button(
                            "MP4 indir",
                            data=video,
                            file_name=(
                                "kogce_image_to_video.mp4"
                            ),
                            mime="video/mp4",
                            use_container_width=True,
                        )


# =========================================================
# AI ARAÇLARI
# =========================================================

with tabs[2]:

    st.header(
        "AI Araçları"
    )

    tool = st.selectbox(
        "Araç Seç",
        [
            "AI Prompt Robotu",
            "AI Senaryo Yazarı",
            "AI Sahne Planlayıcı",
            "AI Video Prompt Üretici",
            "AI Shorts Fikir Motoru",
        ],
    )

    # =====================================================
    # PROMPT ROBOT
    # =====================================================

    if tool == "AI Prompt Robotu":

        st.subheader(
            "AI Prompt Robotu"
        )

        idea = st.text_area(
            "Fikrini yaz",
            height=160,
            placeholder=(
                "Örneğin: Tom bir sihirli iksir "
                "içiyor ve görünüşü değişiyor."
            ),
        )

        if st.button(
            "Prompt Oluştur",
            use_container_width=True,
        ):

            if not idea.strip():

                st.warning(
                    "Bir fikir gir."
                )

            else:

                with st.spinner(
                    "Profesyonel prompt hazırlanıyor..."
                ):

                    result, error = (
                        enhance_image_prompt(
                            idea
                        )
                    )

                if error:

                    st.error(
                        "Prompt oluşturulamadı."
                    )

                    st.code(
                        error,
                        language="text",
                    )

                else:

                    st.text_area(
                        "Sonuç",
                        value=result,
                        height=250,
                    )

                    st.session_state.last_enhanced_prompt = (
                        result
                    )

    # =====================================================
    # SENARYO
    # =====================================================

    elif tool == "AI Senaryo Yazarı":

        st.subheader(
            "AI Senaryo Yazarı"
        )

        idea = st.text_area(
            "Video fikri",
            height=160,
        )

        duration = st.selectbox(
            "Senaryo uzunluğu",
            [
                "Shorts - 15 saniye",
                "Shorts - 30 saniye",
                "1 dakika",
                "3 dakika",
            ],
        )

        if st.button(
            "Senaryo Yaz",
            use_container_width=True,
        ):

            if not idea.strip():

                st.warning(
                    "Video fikri gir."
                )

            else:

                system = """
You are a professional YouTube video scriptwriter.

Create an engaging video script based on the user's idea.

Rules:
- Strong opening hook.
- Clear progression.
- Visual actions.
- Natural pacing.
- Strong ending.
- Do not add unnecessary explanations.

Return only the script.
"""

                user = f"""
Video idea:
{idea}

Target duration:
{duration}
"""

                with st.spinner(
                    "Senaryo hazırlanıyor..."
                ):

                    result, error = call_ai(
                        system,
                        user,
                        temperature=0.8,
                        max_tokens=1800,
                    )

                if error:

                    st.error(
                        "Senaryo oluşturulamadı."
                    )

                    st.code(
                        error,
                        language="text",
                    )

                else:

                    st.session_state.last_script = (
                        result
                    )

                    st.text_area(
                        "Senaryo",
                        value=result,
                        height=400,
                    )

    # =====================================================
    # SAHNE PLANNER
    # =====================================================

    elif tool == "AI Sahne Planlayıcı":

        st.subheader(
            "AI Sahne Planlayıcı"
        )

        script = st.text_area(
            "Senaryoyu yapıştır",
            height=300,
        )

        scene_count = st.slider(
            "Sahne sayısı",
            3,
            20,
            8,
        )

        if st.button(
            "Sahneleri Planla",
            use_container_width=True,
        ):

            if not script.strip():

                st.warning(
                    "Önce senaryoyu gir."
                )

            else:

                system = """
You are an expert AI video director.

Break the provided script into precise visual scenes.

For every scene provide:
1. Scene number
2. Duration
3. What happens
4. Character action
5. Camera movement
6. Environment
7. Lighting
8. Image generation prompt
9. Video generation prompt

Keep character appearance consistent.
"""

                user = f"""
Create exactly {scene_count} scenes.

SCRIPT:
{script}
"""

                with st.spinner(
                    "Sahneler planlanıyor..."
                ):

                    result, error = call_ai(
                        system,
                        user,
                        temperature=0.65,
                        max_tokens=3000,
                    )

                if error:

                    st.error(
                        "Sahne planı oluşturulamadı."
                    )

                    st.code(
                        error,
                        language="text",
                    )

                else:

                    st.session_state.last_scenes = (
                        result
                    )

                    st.text_area(
                        "Sahne Planı",
                        value=result,
                        height=600,
                    )

    # =====================================================
    # VIDEO PROMPT
    # =====================================================

    elif tool == "AI Video Prompt Üretici":

        st.subheader(
            "AI Video Prompt Üretici"
        )

        description = st.text_area(
            "Sahneyi anlat",
            height=180,
            placeholder=(
                "A cat walks toward the camera..."
            ),
        )

        if st.button(
            "Video Prompt Oluştur",
            use_container_width=True,
        ):

            if not description.strip():

                st.warning(
                    "Sahne açıklaması gir."
                )

            else:

                system = """
You are a professional cinematic AI video prompt engineer.

Turn the user's description into a detailed English
video-generation prompt.

Include:
- subject movement
- environment movement
- camera movement
- lens
- framing
- lighting
- physics
- realism
- cinematic quality

Return only the final prompt.
"""

                with st.spinner(
                    "Video prompt hazırlanıyor..."
                ):

                    result, error = call_ai(
                        system,
                        description,
                        temperature=0.7,
                        max_tokens=1000,
                    )

                if error:

                    st.error(
                        "Video prompt oluşturulamadı."
                    )

                    st.code(
                        error,
                        language="text",
                    )

                else:

                    st.text_area(
                        "Video Prompt",
                        value=result,
                        height=300,
                    )

    # =====================================================
    # SHORTS IDEA ENGINE
    # =====================================================

    elif tool == "AI Shorts Fikir Motoru":

        st.subheader(
            "AI Shorts Fikir Motoru"
        )

        topic = st.text_input(
            "Konu",
            placeholder=(
                "Gaming, animals, satisfying, AI..."
            ),
        )

        count = st.slider(
            "Fikir sayısı",
            5,
            30,
            10,
        )

        if st.button(
            "Fikirleri Üret",
            use_container_width=True,
        ):

            if not topic.strip():

                st.warning(
                    "Konu gir."
                )

            else:

                system = """
You are a YouTube Shorts creative strategist.

Generate highly visual Shorts concepts.

Each idea must contain:
- Title
- Hook
- Core concept
- Visual action
- Twist or payoff
- Suggested duration

Focus on concepts that can be understood globally
with minimal spoken language.

Return only the ideas.
"""

                user = f"""
Topic:
{topic}

Generate:
{count} ideas.
"""

                with st.spinner(
                    "Shorts fikirleri hazırlanıyor..."
                ):

                    result, error = call_ai(
                        system,
                        user,
                        temperature=0.9,
                        max_tokens=3500,
                    )

                if error:

                    st.error(
                        "Fikirler oluşturulamadı."
                    )

                    st.code(
                        error,
                        language="text",
                    )

                else:

                    st.text_area(
                        "Shorts Fikirleri",
                        value=result,
                        height=600,
                    )


# =========================================================
# GALERİ
# =========================================================

with tabs[3]:

    st.header(
        "Galeri"
    )

    gallery = st.session_state.gallery

    if not gallery:

        st.info(
            "Henüz oluşturulmuş görsel yok."
        )

    else:

        for index, item in enumerate(
            reversed(gallery)
        ):

            st.divider()

            st.image(
                item["image"],
                use_container_width=True,
            )

            st.caption(
                "Orijinal Prompt: "
                + item["original_prompt"]
            )

            with st.expander(
                "Final Prompt"
            ):

                st.write(
                    item["final_prompt"]
                )


# =========================================================
# SİSTEM
# =========================================================

with tabs[4]:

    st.header(
        "Sistem"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "AI Durumu"
        )

        if HF_API_KEY:

            st.success(
                "Hugging Face API aktif"
            )

        else:

            st.error(
                "Hugging Face API pasif"
            )

        st.write(
            f"Prompt Model: `{PROMPT_MODEL}`"
        )

        st.write(
            f"Image Model: `{IMAGE_MODEL}`"
        )

    with col2:

        st.subheader(
            "Video Motorları"
        )

        st.write(
            f"Text → Video: `{VIDEO_T2V_MODEL}`"
        )

        st.write(
            f"Image → Video: `{VIDEO_I2V_MODEL}`"
        )

    st.divider()

    st.subheader(
        "Oturum Bilgileri"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Galeri",
            len(
                st.session_state.gallery
            ),
        )

    with col2:

        st.metric(
            "Video",
            (
                "Hazır"
                if st.session_state.generated_video
                else "Yok"
            ),
        )

    with col3:

        st.metric(
            "HF API",
            (
                "Online"
                if HF_API_KEY
                else "Offline"
            ),
        )

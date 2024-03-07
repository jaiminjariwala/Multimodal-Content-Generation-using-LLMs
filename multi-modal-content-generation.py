from dotenv import load_dotenv

load_dotenv()   # load all env. variables

import streamlit as st
import os
import google.generativeai as genai
import replicate
from youtube_transcript_api import YouTubeTranscriptApi
from PIL import Image
import time

# set api key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_KEY")

# initialize our models
vis_model = genai.GenerativeModel("gemini-pro-vision")
langugae_model = genai.GenerativeModel("gemini-pro")

# function to load gemini pro vision model and get responses
def get_gemini_response(input, image):
    # if there present any text input besides image, then generate both
    if (input != "") and (image != ""):
        response = vis_model.generate_content([input, image])
    elif (input != "") and (image == ""):
        response = langugae_model.generate_content(input)
    else:
        response = vis_model.generate_content(image)
    
    return response.text

def stream_data(prompt, image):

    sentences = get_gemini_response(prompt, image).split(". ")

    for sentence in sentences:
        for word in sentence.split():
            yield word + " "
            time.sleep(0.02)

# initialize our streamlit app
st.set_page_config(
    page_title="Multimodal Content Generation",
    page_icon="⚡️",
    layout="wide"
)

# give title
st.sidebar.title(":rainbow[MULTIMODAL CONTENT GENERATION]")
github_link = "https://github.com/jaiminjariwala"
st.sidebar.write("Built by [jaiminjariwala]({github_link}) 😀")
st.sidebar.divider()

# Multimodals Options
multimodal_options = st.sidebar.radio(
    "**Select What To Do...**",
    options=["Chat and Image Summarization", "Text 2 Image"],
    index=0,
    horizontal=False,
)
st.sidebar.divider()

# =======================================================================================

if multimodal_options == "Chat and Image Summarization":

    # New chat button, to get the fresh chat page
    if st.sidebar.button("Get **New Chat** Fresh Page"):
        st.session_state["messages"] = []  # Clear chat history
        st.experimental_rerun()  # Trigger page reload

    # create image upload option in sidebar
    with st.expander("**Wanna Upload an Image?**"):
        uploaded_file = st.file_uploader("Choose an image for **Image Summarizer** task...",
                                        type=["jpg", "jpeg", "png"])
        image=""
        if uploaded_file is not None:
            image=Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image.", use_column_width=True)


    # Create a container to hold the entire chat history
    chat_container = st.container()

    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with chat_container:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # create input prompt (textbox)
    if prompt := st.chat_input("Type here..."):

        # display user message in chat message container
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        # add user message to chat history
        st.session_state.messages.append({"role" : "user",
                                        "content" : prompt})
        

        # display assistant message in chat message container
        with chat_container:
            with st.chat_message("assistant"):

                should_format_as_code = any(keyword in prompt.lower() for keyword in ["code", "python", "java", "javascript", "c++", "c", "program", "react", "reactjs", "node", "nodejs", "html", "css", "javascript", "js"])

                if should_format_as_code:
                    st.code(get_gemini_response(prompt, image))
                else:
                    st.write_stream(stream_data(prompt, image))

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": get_gemini_response(prompt, image)})

# =============================================================================

def generate_and_display_image(submitted: bool, width: int, height: int, num_outputs: int, scheduler: str, num_inference_steps: int, prompt_strength: float, prompt: str):
    """
    Generates an image using the specified prompt and parameters.
    """
    if submitted and prompt:
        with st.status('Generating your image...', expanded=True) as status:
            try:
                # Only call the API if the "Submit" button was pressed
                if submitted:
                    all_images = []  # List to store all generated images
                    # calling the replicate API
                    output = replicate.run(
                        "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                        input={
                            "prompt": prompt,
                            "width": width,
                            "height": height,
                            "num_outputs": num_outputs,
                            "scheduler": scheduler,
                            "num_inference_steps": num_inference_steps,
                            "guidance_scale": 7.5,
                            "prompt_stregth": prompt_strength,
                            "negative_prompt": "the absolute worst quality, distorted features",
                            "refine": "expert_ensemble_refiner",
                            "high_noise_frac": 0.8
                        }
                    )
                    if output:
                        st.toast(
                            'Your image has been generated!', icon='😍')
                        # Save generated image to session state
                        st.session_state.generated_image = output
                        
                        # Displaying the image
                        for image in st.session_state.generated_image:
                            with st.container():
                                st.image(image, caption="Generated Image ❄️",
                                            use_column_width=True)
                                # Add image to the list
                                all_images.append(image)

                    # Save all generated images to session state
                    st.session_state.all_images = all_images
    
            except replicate.exceptions.ReplicateError as e:
                st.error(f"Error generating image: {e}")
    
    # if not submitted
    elif not prompt:
        st.toast("Please input some prompt!", icon="⚠️")


def refine_output():
    """
    Provides options for users to refine output parameters and returns them.
    """

    with st.expander("**Refine your output if you want...**"):
        width = st.number_input("Width of output image", value=1024)
        height = st.number_input("Height of output image", value=1024)

        num_outputs = st.slider("Number of images to output", value=1, min_value=1, max_value=4)

        scheduler = st.selectbox('Scheduler', ('DDIM', 'DPMSolverMultistep', 'HeunDiscrete', 'KarrasDPM', 'K_EULER_ANCESTRAL', 'K_EULER', 'PNDM'))

        num_inference_steps = st.slider(
            "Number of denoising steps", value=50, min_value=1, max_value=500)

        prompt_strength = st.slider(
            "Prompt strength when using img2img/inpaint (1.0 corresponds to full destruction of information in image)", value=0.8, max_value=1.0, step=0.1)
        
    # prompt input
    prompt = st.text_input("Enter your prompt for the image:",
                            value="Dog and cat dancing on moon")

    # Submit button to trigger image generation
    submitted = st.button("Generate")

    return submitted, width, height, num_outputs, scheduler, num_inference_steps, prompt_strength, prompt


# Prompt input and image generation logic
if multimodal_options == "Text 2 Image":

    width, height, num_outputs, scheduler, num_inference_steps, prompt_strength, prompt, submitted = refine_output()
    generate_and_display_image(width, height, num_outputs, scheduler, num_inference_steps, prompt_strength, prompt, submitted)

# ============================================================================
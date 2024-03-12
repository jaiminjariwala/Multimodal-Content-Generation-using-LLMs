---
title: Multimodal Content Generation
emoji: 🤗
colorFrom: indigo
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: multi-modal-content-generation.py
pinned: false
license: apache-2.0
---
## A Multimodal Content Generation have following capabilities:

## 1. A `Conversational chatbot` as same as `ChatGPT v3.5 + Image Summarization` Capabilities through `GOOGLE GEMINI VISION PRO API`.

https://github.com/jaiminjariwala/Multimodal-Content-Generation-using-LLMs/assets/157014747/e4cd27c9-d0ed-42e9-94fc-bc0458eb8437

<img width="1312" alt="Screenshot 2024-03-07 at 5 00 49 PM" src="https://github.com/jaiminjariwala/Multimodal-Content-Generation-using-LLMs/assets/157014747/ffa998b9-791d-446b-b951-2f36545ac014">

## 2. `Text to Image` (using Stability Ai (Stable Diffusion)) through `REPLICATE API`.
<img width="673" alt="Screenshot 2024-03-07 at 10 58 41 AM" src="https://github.com/jaiminjariwala/Multimodal-Content-Generation-using-LLMs/assets/157014747/bbfd362e-5437-4807-b58a-09e6efde06f8">


## Setup steps:
1. Create virtual environment
    ```
    python -m venv <name of virtual environment>
    ```

2. Activate it
    ```
    source <name of virtual environment>/bin/activate
    ```

3. Now install required libraries from requirements.txt file using...
    ```
    pip install -r requirements.txt
    ```
4. Create .env file and add your API TOKEN
   ```
   GOOGLE_API_KEY="Enter Your GOOGLE API TOKEN"
   REPLICATE_API_KEY=""
   ```
5. To run app
    ```
    streamlit run <name-of-app>.py
    ```

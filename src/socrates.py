import sys
import re
import openai
import wave
import piper
import subprocess
import os

openai.api_key = os.environ["OPENAI_TOKEN"]


system_prompt = """
You are a helpful assistant to a small child that is 8 years old. The child asks
you questions and you answer in a way that is comprehensible to the child. Use simple
words and do not use the scientific notation for numbers. The questions
will be posed in Czech language and you should also answer in Czech language.
The user posing the question is female - use this information to fix your grammar.
Specify all numbers as text, do not use digits.
Keep length of responses limited to one paragraph of text.
"""

voice_jirka = piper.PiperVoice.load('models/cs_CZ-jirka-medium.onnx', config_path='models/cs_CZ-jirka-medium.onnx.json')
state_decoding = False


def extract_query(line):
    match = re.search(r"'(.*?)'", line)
    if match:
        return match.group(1)
    else:
        return ""


def say_response(text):
    synthesize_args = {
        "speaker_id": None,
        "length_scale": None,
        "noise_scale": None,
        "noise_w": None,
        "sentence_silence": 0.0,
    }

    with wave.open("output.wav", "wb") as wav_file:
        voice_jirka.synthesize(text=text, wav_file=wav_file, **synthesize_args)

    subprocess.run(['afplay', 'output.wav'])


def process_query(query):
    print(f"Processing query: {query} using GPT-4")
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
    response = openai.ChatCompletion.create(model="gpt-4", messages=messages)
    resp_text = response.choices[0].message['content']
    print(f"First response: {resp_text}")
    say_response(resp_text)



def output_parser(raw_line):
    global state_decoding
    
    line = raw_line.strip()
    if "always-prompt mode" in line:
        print("READY.")
    elif "Speech detected" in line:
        state_decoding = True
        print("DECODING!")
    elif "Command" in line:
        state_decoding = False
        process_query(extract_query(line))
    elif "always_prompt_transcription:" in line:
        print(f"CAPTURED: {line}")
    else:
        if state_decoding:
            print("FAILED to decode")
            state_decoding = False
        else:
            print(f"uncaptured: {line}")


if __name__ == "__main__":

    for line in sys.stdin:
        # Print each line
        output_parser(line)

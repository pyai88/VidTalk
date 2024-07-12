import json
import os
import pickle
from urllib.parse import urlparse
from functools import partial
import argparse


import moviepy.editor as mp
import faiss
import gradio as gr
import numpy as np
import spacy
import whisper
import yt_dlp
from groq import Groq
from sentence_transformers import SentenceTransformer

# Initialize Groq client
client = Groq()

def download_video(url):
    """ download youtube video by its url
        Args:
            url: youtube link
    """
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': '%(title)s.%(ext)s'
    }

    # Create a yt-dlp object with the options
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Extract video info
        info = ydl.extract_info(url, download=False)

        # Get video title
        video_title = info['title']

        # Download the video
        ydl.download([url])

        # Construct the expected filename
        filename = f"{video_title}.mp4"

        # Check if the file exists (in case the extension is different)
        if not os.path.exists(filename):
            for file in os.listdir():
                if file.startswith(video_title) and file.endswith(('.mp4', '.mkv', '.webm')):
                    filename = file
                    break

        # Return the path of the downloaded file
        return filename

def extract_audio(video_path):
   """ Separate audio from video. If youtube, we can directly download the audio without video
       Args:
        video_path: path of video to extract audio 
   """
   # Load the video file
   video = mp.VideoFileClip(video_path)

   # Generate the audio file name
   audio_path = video_path.rsplit('.', 1)[0] + '.mp3'

   # Extract the audio
   video.audio.write_audiofile(audio_path)

   # Close the video to free up system resources
   video.close()

   # Return the path to the extracted audio file
   return audio_path

def transcribe_audio(audio_path):
   # Load the Whisper model (you can change "base" to other model sizes if needed)
   model = whisper.load_model("base")

   # Transcribe the audio file
   result = model.transcribe(audio_path)

   # Extract the transcription text and segments with timestamps
   transcription = result["text"]
   segments = result["segments"]

   # Create a list of dictionaries containing text and timestamp information
   transcript_with_timestamps = [
       {
           "text": segment["text"],
           "start": segment["start"],
           "end": segment["end"]
       }
       for segment in segments
   ]

   # Return both the full transcription and the segmented version with timestamps
   return transcription, transcript_with_timestamps

def chunk_by_theme(transcript_with_timestamps):
    """ Chunking the transcription by the theme
        Args: 
           transcript_with_timestamps: transcription with the timestamps 
        Returns:
            separated chunks ready for indexing  
    """
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Initialize variables
    chunks = []
    current_chunk = {"text": "", "start": None, "end": None}
    current_theme = None

    # Iterate through segments
    for segment in transcript_with_timestamps:
        # Process the text
        doc = nlp(segment["text"])

        # Get the root of the sentence (main verb)
        root = None
        for token in doc:
            if token.dep_ == "ROOT":
                root = token
                break

        # If the theme changes, start a new chunk
        if root != current_theme:
            if current_chunk["text"]:
                chunks.append(current_chunk)
            current_chunk = {"text": segment["text"], "start": segment["start"], "end": segment["end"]}
            current_theme = root
        else:
            current_chunk["text"] += " " + segment["text"]
            current_chunk["end"] = segment["end"]

    # Add the last chunk
    if current_chunk["text"]:
        chunks.append(current_chunk)

    return chunks

def create_and_store_embeddings(text_chunks):
    """ create embeddings from the chunks and store in the database
        Args:
            text_chunks: chunk of text
        Return:
            index: database (faiss)
            embeddings: embeddings of chunks
            text_chunks; the orginal text of chunks 
    """
    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create embeddings
    texts = [chunk["text"] for chunk in text_chunks]
    embeddings = model.encode(texts)

    # Normalize embeddings
    faiss.normalize_L2(embeddings)

    # Create a FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])

    # Add vectors to the index
    index.add(embeddings)

    return index, embeddings, text_chunks

def get_title(url):
    """ get the title of youtube videos
        Args: 
            url: url of youtube video
        Return:
            Title of youtube videos.

    """
    with yt_dlp.YoutubeDL() as ydl:
        info_dict = ydl.extract_info(url, download=False)
        video_title = info_dict.get('title', None)
        return video_title

def answer_question_core(question, history, title, index, embeddings, text_chunks, feed_all=True, text=None):
    """ answer the question of users given the video context
        Args:
            question: user's question
            history: ignore for the moment, we needed it for chat interface of Gradio
            title: title of video
            index: database
            embeddings: embeddings for the chunk text
            text_chunks: original text of chunks 
            feed_all: ignore chunking steps, feed all text to LLM, need large context
            text: if feed_all is True, this text will be feeded to LLM, normally contains all transcriptions
        Return: 
            Answer of the question based on the context of video. 
    """
    if feed_all:
        context = text
    else:
        # Initialize the sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Encode the question
        question_embedding = model.encode([question])

        # Normalize the question embedding
        faiss.normalize_L2(question_embedding)

        # Search for the most similar chunk
        D, I = index.search(question_embedding, 10)

        most_relevant_chunk = [text_chunks[I[0][i]] for i in range(10)]
        context = "\n".join(f"{m['start']:.2f} - {m['end']:.2f}: {m['text']}" for m in most_relevant_chunk)

    
    # Construct the prompt
    prompt = f"""Below is the context, which is transcription of a video with timestamps.

Context: The title of the video is: {title}

below is the relevant transcriptions (context) which are timestamps followed by the sentence:

{context}

Question: {question}

Please answer the question based on the given context. If the context doesn't contain enough information to answer the question, please say so. Answer only the question by a completed sentence, don't give extra information unless users request explicitly. 

Answer:"""

    # Call Groq API
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="mixtral-8x7b-32768",
        temperature=0.5,
        max_tokens=1000,
        top_p=1,
        stream=False,
        stop=None
    )

    # Extract the answer
    answer = chat_completion.choices[0].message.content

    return answer

def is_youtube_url(url):
    """ Detect by using url if this is for a youtube video or a local path 
    """
    parsed = urlparse(url)
    return 'youtube.com' in parsed.netloc or 'youtu.be' in parsed.netloc

def save_cache(cache_data, cache_file):
    """ save the cache do disk
    """
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

def load_cache(cache_file):
    """ load cache from disk
    """
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return {}

def process_video(video_input, feed_all = False):
    """ pipeline for the video processing before users could ask the questions about the video
        Args:
            video_input: path of video 
            feed_all: ignore the chunking step or not
        Return: 
            index: database
            embeddings: embeddings for the chunk text
            text_chunks: original text of chunks 
            text: if feed_all is True, this text will be feeded to LLM, normally contains all transcriptions
    """
    cache_file = 'pipeline_cache.pkl'
    cache = load_cache(cache_file)

    # Step 1: Download video if it's a YouTube link
    if 'video_path' not in cache:
        if is_youtube_url(video_input):
            print("Downloading the youtube video")
            cache['video_path'] = download_video(video_input)
        else:
            cache['video_path'] = video_input
        save_cache(cache, cache_file)
    video_path = cache['video_path']

    # Step 2: Extract audio from the video
    if 'audio_path' not in cache:
        print("extract the audio")
        cache['audio_path'] = extract_audio(video_path)
        save_cache(cache, cache_file)
    audio_path = cache['audio_path']

    # Step 3: Transcribe audio using Whisper
    if 'transcription' not in cache or 'transcript_with_timestamps' not in cache:
        print("transcript the audio")
        cache['transcription'], cache['transcript_with_timestamps'] = transcribe_audio(audio_path)
        save_cache(cache, cache_file)
    transcription, transcript_with_timestamps = cache['transcription'], cache['transcript_with_timestamps']

    if not feed_all:

        # Step 4: Split transcription into themed chunks
        if 'themed_chunks' not in cache:
            print("chunk the transcript")
            cache['themed_chunks'] = chunk_by_theme(transcript_with_timestamps)
            save_cache(cache, cache_file)
        themed_chunks = cache['themed_chunks']

        # Step 5: Create and store embeddings
        if 'index' not in cache or 'embeddings' not in cache or 'text_chunks' not in cache:
            print("Create the cache")
            cache['index'], cache['embeddings'], cache['text_chunks'] = create_and_store_embeddings(themed_chunks)
            save_cache(cache, cache_file)
        index, embeddings, text_chunks = cache['index'], cache['embeddings'], cache['text_chunks']
    else:
        index = embeddings = text_chunks = None

    text = "\n".join([f"{x['start']}-{x['end']}: {x['text']}" for x in transcript_with_timestamps]) if feed_all else None

    return index, embeddings, text_chunks, text

# Create the parser
parser = argparse.ArgumentParser(description="Process a URL/path.")

# Add an argument to the parser
parser.add_argument('Path', metavar='path', type=str, help='the path to process')

# Parse the command-line arguments
args = parser.parse_args()

print("Video's url/path to process:", args.Path)
index, embeddings, text_chunks, text = process_video(args.Path)
title = get_title(args.Path)
print(f"processing video '{title}' is done!")
    
def answer_question(message, history):
    """ Create a wrapper for the gradio's function. 
        Functools' Partial does not work with gradio.
    """
    return answer_question_core(message, history, 
                        title = title, 
                        index = index,
                        embeddings = embeddings, 
                        text_chunks=text_chunks, 
                        feed_all=False, 
                        text = text)
gr.ChatInterface(answer_question).launch(share=True)
#!/usr/bin/env python3
import pandas as pd
import librosa
import numpy as np
import time
import jax
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp
from jax.experimental.compilation_cache import compilation_cache as cc

# Audio Data
# and Constants
##########################
base_path = "cv-corpus-19.0-2024-09-13/de/"
clip_path = "clips/"
train = base_path + "train.tsv"
tsv = base_path + "validated.tsv"
sample_rate = 16000 # this is in Hz
# benchmark output
output_path = "results/"
# whsiper data
asr_model = "tiny"
asr_dtype = "jnp.bfloat16"
asr_hardware = "unkown"
# read metadata for audio files
metadata = pd.read_csv(tsv, sep="\t")
print(len(metadata))

# track time 
###########################.
start_time = time.time()

# write benchmarkdata into
benchmark_headers = ["audio_path","audio_length", "asr_duration", "original_text", "transcription", "model", "dtype", "hardware"]
benchmark_data = {header: [] for header in benchmark_headers} # write test resulsts into
########

# init device
devices = jax.devices()
print(devices)

# Pipeline is the recommended way of using whisper with jax
pipeline = FlaxWhisperPipline(asr_model, dtype=jnp.bfloat16, batch_size=16)
cc.set_cache_dir("./jax_cache")

# nice lets process data
# Example: Loop over the first 10 rows and process them
for index, row in metadata.head(10).iterrows():
    # Assume column 4 is the transcription, and column 0 is the file path
    audio_path = base_path + clip_path + row['path']      
    original_text = row['sentence']  
    # Load the audio file
    audio_data, sr = librosa.load(audio_path, sr=sample_rate)
    #print(f"Sample rate: {sr}, Audio data shape: {audio_data.shape}")
    #print(f"Processing file: {audio_path} with transcription: {transcription}")
    audio_duration_in_sec = len(audio_data) / sample_rate 

    #Create the dictionary for each audio file, like whisper expects
    audio_dict = {
        'path': audio_path,
        'array': audio_data,
        'sampling_rate': sample_rate
    }

    #time_print("starting ASR")
    begin_asr = time.time() - start_time
    asr_text = pipeline(audio_dict, language="de")
    #time_print("finished ASR")
    asr_time_in_sec = (time.time() - start_time) - begin_asr
   # print(f"Audio Länge: {audio_duration_in_sec:.2f} Sekunden")
    print(f"Transkribtionsdauer: {time_asr:.2f} Sekunden")
   # print(f"Datei: {audio_path}")
   # print(f"Transcribtion: {original_text}")
   # print(asr_text)
   
    benchmark_data["audio_path"].append(audio_path)
    benchmark_data["audio_length"].append(audio_duration_in_sec)
    benchmark_data["asr_duration"].append(asr_time_in_sec)
    benchmark_data["original_text"].append(original_text)
    benchmark_data["transcription"].append(asr_text)
    benchmark_data["model"].append(asr_model)
    benchmark_data["dtype"].append(asr_dtype)
    benchmark_data["hardware"].append(asr_hardware)
    

result_path = output_path + asr_model + "-benchmark.tsv"
benchmark_df = pd.DataFrame(benchmark_data)
benchmark_df.to_csv(result_path, sep='\t', index=False)

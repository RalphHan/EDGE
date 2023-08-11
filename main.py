from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import glob
import os
from functools import cmp_to_key
import binascii
from tempfile import TemporaryDirectory
import random
import numpy as np
import torch
from tqdm import tqdm
import uuid
import sys
from args import parse_test_opt
from data.slice import slice_audio2
from EDGE import EDGE
from data.audio_extraction.baseline_features import extract as baseline_extract
from data.audio_extraction.jukebox_features import extract as juke_extract
import librosa as lr
import shutil

# sort filenames that look like songname_slice{number}.ext
key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])


def stringintcmp_(a, b):
    aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
    ka, kb = key_func(a), key_func(b)
    if aa < bb:
        return -1
    if aa > bb:
        return 1
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


stringintkey = cmp_to_key(stringintcmp_)


def dance(render, audio, opt, model):
    feature_func = juke_extract if opt.feature_type == "jukebox" else baseline_extract
    temp_dir = TemporaryDirectory()
    dirname = temp_dir.name
    the_uuid = str(uuid.uuid4())
    slice_audio2(audio, 2.5, 5.0, dirname)
    file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=stringintkey)
    sample_size = min(int(opt.out_length / 2.5) - 1, len(file_list))
    rand_idx = random.randint(0, len(file_list) - sample_size)
    cond_list = []
    st_sec = rand_idx * 2.5
    ed_sec = (rand_idx + sample_size + 1) * 2.5
    for idx, file in enumerate(tqdm(file_list)):
        if not (rand_idx <= idx < rand_idx + sample_size):
            continue
        reps, _ = feature_func(file)
        if rand_idx <= idx < rand_idx + sample_size:
            cond_list.append(reps)
    cond_list = torch.from_numpy(np.array(cond_list))
    print("Generating dances")
    data_tuple = None, cond_list, file_list[rand_idx: rand_idx + sample_size]
    model.render_sample(
        data_tuple, "test", opt.render_dir, render_count=-1, render=render, the_uuid=the_uuid
    )
    print("Done")
    torch.cuda.empty_cache()
    temp_dir.cleanup()
    return {"uuid": the_uuid, "start": st_sec, "end": ed_sec}


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])

data = {}


@app.on_event('startup')
def init_data():
    old_argv = sys.argv
    sys.argv = [old_argv[0]]
    opt = parse_test_opt()
    opt.out_length = 30
    opt.render_dir = "renders/gradio"
    sys.argv = old_argv
    model = EDGE(opt.feature_type, opt.checkpoint)
    model.eval()
    data["opt"] = opt
    data["model"] = model
    return data


class Music(BaseModel):
    sr: int
    audio: str


@app.get("/video/{uuid}")
def video(uuid: str):
    video_path = f"{data['opt'].render_dir}/{uuid}.mp4"
    return FileResponse(video_path, media_type="video/mp4")


@app.get("/angle/{uuid}")
def angle(uuid: str):
    json_path = f"{data['opt'].render_dir}/axis_angle_{uuid}.json"
    return FileResponse(json_path)


@app.post("/edge_data/")
async def edge_data(music: Music, render: bool = False):
    audio_pair = (music.sr, np.frombuffer(binascii.a2b_base64(music.audio),dtype=np.int16))
    return dance(render, audio_pair, data["opt"], data["model"])


@app.post("/edge_file/")
async def edge_file(upload: UploadFile = File(...), render: bool = False):
    try:
        audio, sr = lr.load(upload.file, sr=None)
    except:
        temp_dir = TemporaryDirectory()
        dirname = temp_dir.name
        file_path = os.path.join(dirname, "tmp_music")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload.file, buffer)
        audio, sr = lr.load(file_path, sr=None)
        temp_dir.cleanup()
    audio_pair = (sr, audio)
    return dance(render, audio_pair, data["opt"], data["model"])

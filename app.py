import glob
import os
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import random
import gradio as gr
import jukemirlib
import numpy as np
import torch
from tqdm import tqdm
import uuid
from args import parse_test_opt
from data.slice import slice_audio2
from EDGE import EDGE
from data.audio_extraction.baseline_features import extract as baseline_extract
from data.audio_extraction.jukebox_features import extract as juke_extract

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


def init():
    opt = parse_test_opt()
    opt.out_length=30
    opt.render_dir="renders/gradio"
    model = EDGE(opt.feature_type, opt.checkpoint)
    model.eval()
    return opt,model

def dance(render, audio,opt,model):
    feature_func = juke_extract if opt.feature_type == "jukebox" else baseline_extract
    all_cond = []
    all_filenames = []
    temp_dir = TemporaryDirectory()
    dirname = temp_dir.name
    the_uuid=str(uuid.uuid4())
    slice_audio2(audio, the_uuid, 2.5, 5.0, dirname)
    file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=stringintkey)
    sample_size = min(int(opt.out_length / 2.5) - 1,len(file_list))
    rand_idx = random.randint(0, len(file_list) - sample_size)
    cond_list = []
    for idx, file in enumerate(tqdm(file_list)):
        if not (rand_idx <= idx < rand_idx + sample_size):
            continue
        reps, _ = feature_func(file)
        if rand_idx <= idx < rand_idx + sample_size:
            cond_list.append(reps)
    cond_list = torch.from_numpy(np.array(cond_list))
    all_cond.append(cond_list)
    all_filenames.append(file_list[rand_idx : rand_idx + sample_size])

    print("Generating dances")
    for i in range(len(all_cond)):
        data_tuple = None, all_cond[i], all_filenames[i]
        model.render_sample(
            data_tuple, "test", opt.render_dir, render_count=-1, render=render, the_uuid=the_uuid
        )
    print("Done")
    torch.cuda.empty_cache()
    temp_dir.cleanup()
    if render:
        return f"{opt.render_dir}/test_{the_uuid}.mp4", f"{opt.render_dir}/quaternion_{the_uuid}.json", f"{opt.render_dir}/axis_angle_{the_uuid}.json"
    return None, f"{opt.render_dir}/quaternion_{the_uuid}.json", f"{opt.render_dir}/axis_angle_{the_uuid}.json"


if __name__ == "__main__":
    opt, model = init()
    demo = gr.Interface(
        lambda render, audio: dance(render, audio, opt, model),
        [gr.Checkbox(value=True, label="render"), gr.Audio(value="custom_music/9i6bCWIdhBw.mp3",label="超过30s则随机选30秒",source="upload")],
        [gr.Video(format="mp4",autoplay=True), gr.File(), gr.File()],
    )
    demo.launch(server_name='0.0.0.0',server_port=7866)

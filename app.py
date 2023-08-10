import gradio as gr
import json
import requests
render_dir = "renders/gradio"
def dance(render, audio):
    sr, audio = audio
    if len(audio.shape) == 2:
        audio = audio[:,0]
    ret_json = json.loads(requests.post("http://0.0.0.0:8020/edge_data/", json={"sr": sr, "audio": audio.tolist()}, params={"render": render}).text)
    the_uuid = ret_json["uuid"]
    video_path=f"{render_dir}/{the_uuid}.mp4" if render else None
    json_path = f"{render_dir}/axis_angle_{the_uuid}.json"
    return video_path, json_path


if __name__ == "__main__":
    demo = gr.Interface(
        dance,
        [gr.Checkbox(value=True, label="render"), gr.Audio(value="custom_music/9i6bCWIdhBw.mp3",label="超过30s则随机选30秒",source="upload")],
        [gr.Video(format="mp4",autoplay=True), gr.File()],
    )
    demo.launch(server_name='0.0.0.0',server_port=7866)

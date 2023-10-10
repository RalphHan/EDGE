if [ $# -eq 0 ]; then
    docker run -itd \
      --gpus '"device=0"' \
      -v `pwd`:/EDGE \
      --restart always \
      -p 8020:8020 \
      ralphhan/edge \
      bash /EDGE/$0 1
    exit 0
fi

cd /EDGE
rm checkpoint.pt
ln -s /data/checkpoint.pt .
/opt/conda/envs/cinema/bin/uvicorn main:app --host 0.0.0.0 --port 8020 --reload

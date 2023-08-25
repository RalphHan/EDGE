cd /EDGE
rm checkpoint.pt
ln -s /data/checkpoint.pt .
/opt/conda/envs/cinema/bin/uvicorn main:app --host 0.0.0.0 --port 8020 --reload

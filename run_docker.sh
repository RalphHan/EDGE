docker run -itd \
--gpus '"device=0"' \
-v `pwd`:/EDGE \
--restart always \
-p 8020:8020 \
ralphhan/edge \
bash /EDGE/run_server.sh

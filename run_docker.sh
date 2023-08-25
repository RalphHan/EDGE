docker run -itd \
--gpus 0 \
-v /home/hanhongwei/EDGE:/EDGE \
--restart always \
-p 8020:8020 \
ralphhan/edge \
bash /EDGE/run_server.sh

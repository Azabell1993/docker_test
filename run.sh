#/!/bin/bash

cd ~/jiwoo_research_docker/jetson_slm_stack

docker compose down
docker builder prune -af

docker compose build --no-cache
./jetson_slm_stack/scripts/run_jetson.sh prep
./jetson_slm_stack/scripts/run_jetson.sh download
./jetson_slm_stack/scripts/run_jetson.sh llama

# clean
# ./jetson_slm_stack/scripts/run_jetson.sh clean
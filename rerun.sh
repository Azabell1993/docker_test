#/!/bin/bash

docker compose down
docker compose build --no-cache
# docker compose run --rm llama32-server python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"

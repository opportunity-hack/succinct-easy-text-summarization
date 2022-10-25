#!/bin/bash

echo "Starting"
nohup python3.7 /run_app.py &
python3.7 /usr/local/bin/huey_consumer.py --verbose app.tasks.get_similar.huey

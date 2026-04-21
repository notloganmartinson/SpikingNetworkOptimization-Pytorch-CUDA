#!/bin/bash
# Activate the virtual environment
source /mnt/storage/SNO/fasterSNO/venv/bin/activate

# Run the optimizer in the background, redirecting output to a log file
nohup python /mnt/storage/SNO/fasterSNO/optimizer.py > /mnt/storage/SNO/fasterSNO/optimization_run.log 2>&1 &

echo "Optimizer started in the background!"
echo "To check the progress, run: tail -f /mnt/storage/SNO/fasterSNO/optimization_run.log"
echo "To stop the optimizer early, find its PID with 'ps -ef | grep optimizer.py' and use 'kill <PID>'."

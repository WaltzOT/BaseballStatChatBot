Being run with NVIDIA NeMo on a CPU only mode. 


Running docker code

docker run -it --rm \
    -v /initialTraining.py \
    -v ../data/baseballqueries.json \
    nemo-cpu
    

winpty docker run --rm -it nemo-cpu


docker run --rm -it nemo-cpu python3 main/testingBot.py

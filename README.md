BUTCHERED THE SAVES, DEAD REPO, GOING TO CONTINUE ELSEWHERE BUT STILL VALUABLE INFO HERE TO AN EXTENT

Being run with NVIDIA NeMo on a CPU only mode. 


Running docker code

docker run -it --rm \
    -v /initialTraining.py \
    -v ../data/baseballqueries.json \
    nemo-cpu
    

winpty docker run --rm -it nemo-cpu


docker run --rm -it nemo-cpu python3 main/testingBot.py

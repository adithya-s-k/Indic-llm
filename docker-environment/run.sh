docker run --rm -it  \
           --gpus all  \
           -p 8888:8888  \
           -e JUPYTER_TOKEN=passwd  \
           adithya/pytorch-notebook:jupyter
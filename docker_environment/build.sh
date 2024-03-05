docker build --no-cache \
             -f jupyter.dockerfile \
             -t finetuning/pytorch-notebook:jupyter \
             .
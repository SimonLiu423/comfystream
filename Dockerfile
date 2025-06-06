FROM livepeer/comfystream

ENV LD_LIBRARY_PATH="/workspace/miniconda3/envs/comfystream/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

COPY entrypoint_wrapper.sh /workspace/entrypoint_wrapper.sh

RUN chmod +x /workspace/entrypoint_wrapper.sh
RUN apt-get update && apt-get install -y vim
RUN git clone https://github.com/SimonLiu423/comfystream.git /workspace/unity-cs
RUN cd /workspace/ComfyUI/custom_nodes && git clone https://github.com/cubiq/ComfyUI_essentials
RUN conda run -n comfystream --no-capture-output pip install transparent_background

ENTRYPOINT ["/workspace/entrypoint_wrapper.sh", "/workspace/comfystream/docker/entrypoint.sh", "--server"]



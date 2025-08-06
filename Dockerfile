FROM nvidia/cuda:11.1.1-runtime-ubi8
ENV WORKDIR=/root
WORKDIR ${WORKDIR}
COPY . ${WORKDIR}
# basic setup.
RUN echo "ZONE=Asia/Shanghai" >> /etc/sysconfig/clock         && \
    rm -f /etc/localtime                                      && \
    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime   && \
    yum -y update                                             && \
    yum install python39 python39-pip -y                      && \
    python3.9 -m venv .                                       && \
    source bin/activate                                       

RUN pip3 install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    yum install -y mesa-libGL                                                          && \
    pip3 install -r requirements.txt                                           && \
    yum clean all


CMD ["bash", "run.bash"]

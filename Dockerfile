# RunPod Serverless LOAD_BALANCER 용 SenseVoice 워커.
#
# Ollama 워커와 달리 nginx 가 필요 없다. 그때는 스톡 이미지에 /ping 경로가
# 없어서 앞에 세워야 했지만, 여기서는 앱을 우리가 짜므로 /ping 을 직접 낸다.
#
# 모델은 이미지에 미리 받아둔다. SenseVoiceSmall 은 1GB 남짓이라 넣어도
# 부담이 없고, 넣어두면 콜드스타트마다 HuggingFace 를 때리지 않는다.
# (LLM 워커에서 30GB 모델을 볼륨에 둔 것과는 사정이 다르다.)

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    MODELSCOPE_CACHE=/models

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      funasr==1.2.6 \
      fastapi==0.115.6 \
      uvicorn==0.34.0 \
      python-multipart==0.0.20 \
      modelscope==1.22.3 \
      soundfile==0.13.0

# 모델을 빌드 때 받아 이미지에 굽는다. 실패하면 여기서 빌드가 깨지므로
# 런타임에 가서야 없다는 걸 아는 상황을 막는다.
RUN python -c "\
from modelscope import snapshot_download; \
snapshot_download('iic/SenseVoiceSmall'); \
snapshot_download('iic/speech_fsmn_vad_zh-cn-16k-common-pytorch'); \
print('모델 내려받기 완료')"

COPY app.py /app/app.py
WORKDIR /app

EXPOSE 80

# RunPod 은 PORT 와 PORT_HEALTH 를 환경변수로 준다. 같은 앱이 /ping 과
# /transcribe 를 모두 내므로 한 포트에서 둘 다 받으면 된다.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-80} --timeout-keep-alive 600"]

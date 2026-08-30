"""RunPod Serverless LOAD_BALANCER 용 SenseVoice 받아쓰기 워커.

Whisper 를 쓰다가 옮겨온 이유는 정확도가 아니라 고장의 성격이다.
Whisper 는 말소리가 없는 구간에서 침묵하지 않고 학습 데이터에 흔했던 문구를
지어낸다("시청해주셔서 감사합니다"). 15초짜리 순수 사인파에서도 그랬다.
보이스드라마처럼 비언어 발성 비중이 큰 자료에서는 그 구간이 통째로 오염된다.

SenseVoice 는 비자기회귀 모델이라 그런 식으로 헛돌지 않고, 대신 그 구간을
음향 이벤트(<|Breath|>, <|Laughter|> 등)와 감정으로 표시한다. 한국어가
정식 지원 언어(5개 특화)라 CJK 정확도도 낫다.

경로:
  GET  /ping        RunPod 헬스체크. 모델이 다 올라온 뒤에만 200 을 준다.
  POST /transcribe  오디오 파일 하나 -> 구간별 시각 + 텍스트 + 태그
"""
from __future__ import annotations

import os
import re
import tempfile
import threading

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

MODEL_DIR = os.environ.get("SENSEVOICE_MODEL", "iic/SenseVoiceSmall")
VAD_MODEL = os.environ.get("VAD_MODEL", "fsmn-vad")
DEVICE = os.environ.get("DEVICE", "cuda:0")
# VAD 가 한 덩어리로 자를 최대 길이(ms). 길게 잡으면 문장이 덜 쪼개지지만
# 그만큼 시각 해상도가 떨어진다. 자막용으로는 30초가 무난하다.
MAX_SEGMENT_MS = int(os.environ.get("MAX_SEGMENT_MS", "30000"))

app = FastAPI()

_model = None
_ready = threading.Event()
_load_error: str | None = None


def _load() -> None:
    """모델을 올린다. 이미지에 미리 받아뒀으므로 내려받기는 없다."""
    global _model, _load_error
    try:
        from funasr import AutoModel
        _model = AutoModel(
            model=MODEL_DIR,
            vad_model=VAD_MODEL,
            vad_kwargs={"max_single_segment_time": MAX_SEGMENT_MS},
            device=DEVICE,
            disable_update=True,
        )
        _ready.set()
    except Exception as e:  # noqa: BLE001
        _load_error = f"{type(e).__name__}: {e}"


threading.Thread(target=_load, daemon=True).start()


@app.get("/ping")
async def ping():
    """RunPod 로드밸런서는 여기가 200 일 때만 트래픽을 넘긴다.

    모델이 다 올라오기 전에 200 을 주면 첫 요청들이 실패한다. 그래서
    준비되기 전에는 503 을 준다. RunPod 은 그동안 워커를 붙잡아 둔다.
    """
    if _load_error:
        return JSONResponse({"ok": False, "error": _load_error}, status_code=500)
    if not _ready.is_set():
        return JSONResponse({"ok": False, "loading": True}, status_code=503)
    return {"ok": True, "model": MODEL_DIR}


# SenseVoice 는 태그를 <|...|> 꼴로 본문에 섞어 내보낸다.
TAG_RE = re.compile(r"<\|([^|]+)\|>")

EMOTION = {
    "HAPPY": "기쁨", "SAD": "슬픔", "ANGRY": "분노", "NEUTRAL": "",
    "FEARFUL": "두려움", "DISGUSTED": "혐오", "SURPRISED": "놀람",
}
EVENT = {
    "BGM": "배경음악", "Applause": "박수", "Laughter": "웃음", "Cry": "울음",
    "Sneeze": "재채기", "Breath": "숨소리", "Cough": "기침", "Speech": "",
}


def split_tags(raw: str) -> tuple[str, str, list[str]]:
    """본문에서 태그를 떼어내 (텍스트, 감정, 이벤트목록) 으로 나눈다."""
    emo, events = "", []
    for tag in TAG_RE.findall(raw or ""):
        if tag in EMOTION:
            emo = EMOTION[tag] or emo
        elif tag in EVENT:
            name = EVENT[tag]
            if name and name not in events:
                events.append(name)
        # 언어 코드(<|ko|>)나 <|withitn|> 같은 나머지는 버린다.
    return TAG_RE.sub("", raw or "").strip(), emo, events


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = "ko"):
    if _load_error:
        raise HTTPException(500, f"모델을 올리지 못했습니다: {_load_error}")
    if not _ready.is_set():
        raise HTTPException(503, "모델을 올리는 중입니다.")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".audio")
    try:
        while chunk := await file.read(4 * 1024 * 1024):
            tmp.write(chunk)
        tmp.close()

        res = _model.generate(
            input=tmp.name,
            language=language or "auto",
            use_itn=True,              # 숫자·단위를 읽는 대로가 아니라 표기대로
            batch_size_s=300,
            merge_vad=False,           # VAD 구간을 합치면 시각 해상도를 잃는다
            sentence_timestamp=True,   # 구간별 시작·끝을 받는다
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"인식 실패: {type(e).__name__}: {e}") from e
    finally:
        try:
            tmp.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            os.remove(tmp.name)
        except OSError:
            pass

    out = []
    for item in res or []:
        # sentence_timestamp=True 면 구간이 sentence_info 에 들어온다.
        # 혹시 없으면 전체를 한 덩어리로라도 돌려준다.
        sents = item.get("sentence_info")
        if not sents:
            text, emo, events = split_tags(item.get("text", ""))
            if text or events:
                out.append({"start": 0.0, "end": 0.0, "text": text,
                            "emotion": emo, "events": events})
            continue
        for s in sents:
            text, emo, events = split_tags(s.get("text", ""))
            if not text and not events:
                continue
            out.append({
                "start": float(s.get("start", 0)) / 1000.0,
                "end": float(s.get("end", 0)) / 1000.0,
                "text": text,
                "emotion": emo,
                "events": events,
            })
    return {"segments": out}

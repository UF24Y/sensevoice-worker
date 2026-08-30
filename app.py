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


_vad = None


def _load() -> None:
    """모델 둘을 올린다. 이미지에 미리 받아뒀으므로 내려받기는 없다.

    VAD 를 AutoModel 에 얹지 않고 따로 두는 이유가 있다. FunASR 에 VAD 를
    맡기고 sentence_timestamp=True 로 구간을 받으려면 구두점 모델이 반드시
    있어야 하는데(없으면 punc_res 미할당으로 터진다), 그 구두점 모델은
    중국어·영어용이다. 한국어 문장 경계를 못 잡으면 구간이 통째로 뭉쳐서
    타임스탬프가 쓸모없어진다.

    그래서 VAD 를 직접 돌려 말소리 구간을 얻고, 구간마다 인식을 돌린다.
    언어에 기대지 않고, 구간별로 감정·음향 태그가 따로 나온다는 이점도 있다.
    """
    global _model, _vad, _load_error
    try:
        from funasr import AutoModel
        _vad = AutoModel(
            model=VAD_MODEL,
            vad_kwargs={"max_single_segment_time": MAX_SEGMENT_MS},
            device=DEVICE,
            disable_update=True,
        )
        _model = AutoModel(model=MODEL_DIR, device=DEVICE, disable_update=True)
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


def _run(path: str, language: str) -> list[dict]:
    """VAD 로 말소리 구간을 찾고, 구간마다 인식을 돌린다.

    구간을 자를 때 파일을 다시 읽지 않고 메모리의 파형을 잘라 넘긴다.
    서버가 이미 16kHz 모노로 만들어 보내므로 변환도 필요 없다.
    """
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)

    vad = _vad.generate(input=path)
    spans = (vad[0].get("value") if vad else None) or []

    out: list[dict] = []
    for span in spans:
        try:
            s_ms, e_ms = float(span[0]), float(span[1])
        except (TypeError, IndexError, ValueError):
            continue
        lo, hi = int(s_ms * sr / 1000), int(e_ms * sr / 1000)
        piece = audio[max(0, lo):min(len(audio), hi)]
        if len(piece) < int(sr * 0.1):     # 0.1초 미만은 인식할 게 없다
            continue
        try:
            r = _model.generate(input=piece, fs=sr, cache={},
                                language=language or "auto", use_itn=True)
        except Exception:  # noqa: BLE001
            # 한 구간이 실패해도 나머지는 살린다. 전체를 버리는 게 더 나쁘다.
            continue
        text, emo, events = split_tags((r[0] or {}).get("text", "") if r else "")
        if not text and not events:
            continue
        out.append({"start": s_ms / 1000.0, "end": e_ms / 1000.0,
                    "text": text, "emotion": emo, "events": events})
    return out


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
        segments = _run(tmp.name, language)
    except HTTPException:
        raise
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

    return {"segments": segments}

import argparse
import uvicorn
from api_v2 import tts_endpoint
from fastapi import FastAPI, Request

parser = argparse.ArgumentParser(description="TTS api")
parser.add_argument(
    "-a", "--host", type=str, default="0.0.0.0", help="default: 0.0.0.0"
)
parser.add_argument("-p", "--port", type=int, default="9885", help="default: 9885")
args, unknow = parser.parse_known_args()

# --------------------------------
# 接口部分
# --------------------------------
app = FastAPI()


@app.post("/")
async def tts(request: Request):
    try:
        metrics = await tts_endpoint(request)
        print(metrics)
        return metrics
    except Exception as e:
        json_post_raw = await request.json()
        print("音频合成失败", json_post_raw.get("prompt_wav"), json_post_raw.get("prompt_text"), json_post_raw.get("target_text"))
        print("音频失败原因", e)
        


def start():
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    start()

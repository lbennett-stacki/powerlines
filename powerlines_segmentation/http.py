from fastapi import FastAPI, Request, HTTPException
from powerlines_segmentation.segmentor import Segmentor
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import asyncio

app = FastAPI()


segmentor = None
executor = ThreadPoolExecutor(max_workers=4)


async def run_inference(segmentor, input):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, segmentor.segment, input)


@app.post("/segment")
async def segment(
    request: Request,
) -> str:
    global segmentor
    if segmentor is None:
        segmentor = Segmentor("./models/powerlines-segmentation")

    data = await request.json()
    input = data.get("input", None)

    if input is None or input == "":
        raise HTTPException(status_code=400, detail="No input provided")

    print("\n\nRunning segmention on", input)

    result = await run_inference(segmentor, input)

    print("\nGot segmention result", result)

    return result


def start_http_server(host: str = "0.0.0.0", port: int = 6001):
    uvicorn.run(app, host=host, port=port)

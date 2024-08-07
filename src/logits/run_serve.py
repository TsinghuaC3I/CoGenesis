#!/usr/bin/env python
# refer to: https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/api_server.py
# ray: https://github.com/ray-project/ray/blob/master/doc/source/serve/doc_code/vllm_example.py

import argparse
import orjson
import typing

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5  # seconds.
engine = None

class ORJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: typing.Any) -> bytes:
        return orjson.dumps(content)

app = FastAPI(default_response_class=ORJSONResponse)

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt", None)
    prompt_token_ids = request_dict.pop("prompt_token_ids", None)
    prefix_pos = request_dict.pop("prefix_pos", None)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    
    results_generator = engine.generate(prompt=prompt,
                                        sampling_params=sampling_params,
                                        request_id=request_id,
                                        prompt_token_ids=prompt_token_ids,
                                        prefix_pos=prefix_pos)

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [output.text for output in final_output.outputs]
    token_ids = [output.token_ids for output in final_output.outputs]
    ret = {"text": text_outputs, "token_ids": token_ids}
    
    if sampling_params.logprobs is not None:
        logprobs = [[{str(i):p for i,p in logprob.items()} for logprob in output.logprobs] for output in final_output.outputs]
        ret["logprobs"] = logprobs
    return ORJSONResponse(ret)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile)

# export NCCL_P2P_LEVEL=NVL
# python run_serve.py --host 0.0.0.0 --model "Qwen/Qwen1.5-72B-Chat" --trust-remote-code --tensor-parallel-size 8
"""Run the existing on-device GGUF without copying or downloading its weights."""
import os
import shutil
from pathlib import Path

root = Path.home() / 'UltimateDocumentEditor'
model = Path(os.getenv('MODEL_PATH', str(root/'Qwen3.8-27B-UD-IQ1_S.gguf')))
bundled = root/'desktop/resources/darwin-arm64/runtime/llama-b9150/llama-server'
server = os.getenv('LLAMA_SERVER') or (str(bundled) if bundled.is_file() else shutil.which('llama-server')) or ''
if not model.is_file():
    raise SystemExit(f'Model not found: {model}. Set MODEL_PATH to your local GGUF.')
if not Path(server).is_file():
    raise SystemExit('llama-server not found. Set LLAMA_SERVER to the installed executable.')
args = [server,'-m',str(model),'--host','127.0.0.1','--port',os.getenv('MODEL_PORT','8091'),'-c','8192','-np','1','-ngl','99','-t','6','-b',os.getenv('MODEL_BATCH','512'),'-ub',os.getenv('MODEL_UBATCH','256'),'--jinja','--reasoning','off','--reasoning-format','deepseek','--chat-template-kwargs','{"enable_thinking":false,"preserve_thinking":false}','--no-webui','--cache-ram','128','--alias','qwen-local-27b']
os.execv(server, args)

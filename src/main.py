import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from webui.components.agent_settings_tab import create_agent_settings_tab
from webui.components.browser_settings_tab import create_browser_settings_tab
from webui.components.browser_use_agent_tab import (
    create_browser_use_agent_tab,
    run_agent_task,
)
from webui.webui_manager import WebuiManager


app = FastAPI()

ui_manager = WebuiManager()
with gr.Blocks():
    create_agent_settings_tab(ui_manager)
    create_browser_settings_tab(ui_manager)
    create_browser_use_agent_tab(ui_manager)


class BrowserUsePayload(BaseModel):
    task: str
    max_steps: int = 100
    llm_provider: str | None = None
    llm_model_name: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    headless: bool | None = None
    keep_browser_open: bool | None = None


_active_payload_hashes: set[str] = set()
_payload_hash_lock = asyncio.Lock()
_task_results: dict[str, dict[str, Any]] = {}


def _hash_payload(payload: BrowserUsePayload) -> str:
    """Create a stable hash for the payload contents."""
    normalized = json.dumps(payload.model_dump(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _get_history_from_agent() -> Any:
    agent = getattr(ui_manager, "bu_agent", None)
    if not agent or not getattr(agent, "state", None):
        return None

    history = getattr(agent.state, "history", None)
    if history is None:
        return None

    for attr in ("model_dump", "dict"):
        if hasattr(history, attr):
            try:
                return getattr(history, attr)()
            except Exception:
                continue

    return None


def _persist_task_result(
        task_id: str,
        messages: list[dict[str, Any]],
        history_content: Any | None,
) -> dict[str, Any]:
    base_path = Path("./tmp/agent_history") / task_id
    history_path = base_path / f"{task_id}.json"
    gif_path = base_path / f"{task_id}.gif"

    result: dict[str, Any] = {
        "task_id": task_id,
        "messages": messages,
        "history_file": str(history_path) if history_path.exists() else None,
        "recording_gif": str(gif_path) if gif_path.exists() else None,
    }
    if history_content is not None:
        result["history"] = history_content

    _task_results[task_id] = result
    return result


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/run-task")
async def run_browser_use(payload: BrowserUsePayload):
    task_text = payload.task.strip()
    if not task_text:
        raise HTTPException(status_code=400, detail="Task text is required.")

    payload_hash = _hash_payload(payload)

    async with _payload_hash_lock:
        if payload_hash in _active_payload_hashes:
            raise HTTPException(status_code=409, detail="Identical payload already running.")
        _active_payload_hashes.add(payload_hash)

    try:
        components: dict[Any, Any] = {
            ui_manager.get_component_by_id("browser_use_agent.user_input"): task_text,
            ui_manager.get_component_by_id("agent_settings.max_steps"): payload.max_steps,
        }

        if payload.headless is not None:
            components[ui_manager.get_component_by_id("browser_settings.headless")] = payload.headless
        if payload.keep_browser_open is not None:
            components[ui_manager.get_component_by_id("browser_settings.keep_browser_open")] = (
                payload.keep_browser_open
            )
        if payload.llm_provider is not None:
            components[ui_manager.get_component_by_id("agent_settings.llm_provider")] = payload.llm_provider
        if payload.llm_model_name is not None:
            components[ui_manager.get_component_by_id("agent_settings.llm_model_name")] = payload.llm_model_name
        if payload.llm_base_url is not None:
            components[ui_manager.get_component_by_id("agent_settings.llm_base_url")] = payload.llm_base_url
        if payload.llm_api_key is not None:
            components[ui_manager.get_component_by_id("agent_settings.llm_api_key")] = payload.llm_api_key

        async for _ in run_agent_task(ui_manager, components):
            pass

        task_id = ui_manager.bu_agent_task_id
        if not task_id:
            raise HTTPException(status_code=500, detail="Task ID not generated.")

        history_path = Path("./tmp/agent_history") / task_id / f"{task_id}.json"
        if history_path.exists():
            try:
                history_content = json.loads(history_path.read_text())
            except json.JSONDecodeError:
                history_content = None
        else:
            history_content = _get_history_from_agent()

        result = _persist_task_result(
            task_id,
            list(ui_manager.bu_chat_history),
            history_content,
        )
        return {"payload_hash": payload_hash, **result}
    finally:
        async with _payload_hash_lock:
            _active_payload_hashes.discard(payload_hash)


@app.get("/task/{task_id}")
def get_task(task_id: str):
    cached = _task_results.get(task_id)
    if not cached:
        raise HTTPException(status_code=404, detail="Task not found.")

    response = cached.copy()
    if "history" not in response:
        history_file = response.get("history_file")
        if history_file:
            history_path = Path(history_file)
            if history_path.exists():
                try:
                    response["history"] = json.loads(history_path.read_text())
                except json.JSONDecodeError:
                    response["history"] = None
            else:
                response["history"] = _get_history_from_agent()
        else:
            response["history"] = _get_history_from_agent()

    return response

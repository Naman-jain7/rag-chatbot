from collections.abc import Iterator
from typing import Any

import requests


API_BASE_URL = "http://127.0.0.1:8000/api/v1"
TIMEOUT_SECONDS = 60


class ApiError(RuntimeError):
    pass


def _detail(response: requests.Response) -> str:
    try:
        body = response.json()
        return str(body.get("detail") or body.get("message") or response.text)
    except ValueError:
        return response.text or f"Request failed with status {response.status_code}"


def _raise_for_status(response: requests.Response) -> None:
    if not response.ok:
        raise ApiError(_detail(response))


def signup(full_name: str, email: str, password: str, age: int | None) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/auth/signup",
        json={
            "full_name": full_name,
            "email": email,
            "password": password,
            "age": age,
            "preferences": {},
        },
        timeout=TIMEOUT_SECONDS,
    )
    _raise_for_status(response)
    return response.json()


def login(email: str, password: str) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/auth/login",
        data={"username": email, "password": password},
        timeout=TIMEOUT_SECONDS,
    )
    _raise_for_status(response)
    return response.json()


def list_documents(user_id: int, token: str) -> list[dict[str, Any]]:
    response = requests.get(
        f"{API_BASE_URL}/documents/{user_id}",
        headers=_auth_headers(token),
        timeout=TIMEOUT_SECONDS,
    )
    _raise_for_status(response)
    return response.json()["documents"]


def upload_document(user_id: int, token: str, uploaded_file: Any) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/upload",
        headers=_auth_headers(token),
        data={"user_id": str(user_id)},
        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
        timeout=None,
    )
    _raise_for_status(response)
    return response.json()


def delete_document(user_id: int, token: str, filename: str) -> None:
    response = requests.post(
        f"{API_BASE_URL}/delete",
        headers=_auth_headers(token),
        params={"user_id": user_id},
        json=[filename],
        timeout=TIMEOUT_SECONDS,
    )
    _raise_for_status(response)


def stream_chat(user_id: int, token: str, query: str, chat_id: str | None = None) -> Iterator[str]:
    payload = {"user_id": user_id, "query": query}
    if chat_id:
        payload["chat_id"] = chat_id
    with requests.post(
        f"{API_BASE_URL}/chat/stream",
        headers={**_auth_headers(token), "Content-Type": "application/json"},
        json=payload,
        stream=True,
        timeout=None,
    ) as response:
        _raise_for_status(response)
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk # type: ignore


def list_memories(user_id: int, token: str) -> list[dict]:
    response = requests.get(
        f"{API_BASE_URL}/memories/{user_id}",
        headers=_auth_headers(token),
        timeout=TIMEOUT_SECONDS,
    )
    _raise_for_status(response)
    return response.json()["memories"]


def delete_memory(user_id: int, token: str, memory_id: str) -> None:
    response = requests.delete(
        f"{API_BASE_URL}/memories/{user_id}/{memory_id}",
        headers=_auth_headers(token),
        timeout=TIMEOUT_SECONDS,
    )
    _raise_for_status(response)


def delete_all_memories(user_id: int, token: str) -> None:
    response = requests.delete(
        f"{API_BASE_URL}/memories/{user_id}",
        headers=_auth_headers(token),
        timeout=TIMEOUT_SECONDS,
    )
    _raise_for_status(response)


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}

def get_chat_history(token: str, user_id: int | str, chat_id: str) -> dict[str, Any]:
    """Fetch conversation historical messages from the backend checkpointer."""

    response = requests.get(
        f"{API_BASE_URL}/chat/history/{user_id}/{chat_id}",
        headers=_auth_headers(token),
        timeout=TIMEOUT_SECONDS,
    )

    _raise_for_status(response)
    return response.json()


def list_conversations(token: str, user_id: int | str) -> list[dict[str, Any]]:
    response = requests.get(
        f"{API_BASE_URL}/chat/conversations/{user_id}",
        headers=_auth_headers(token),
        timeout=TIMEOUT_SECONDS,
    )
    _raise_for_status(response)
    return response.json()["conversations"]

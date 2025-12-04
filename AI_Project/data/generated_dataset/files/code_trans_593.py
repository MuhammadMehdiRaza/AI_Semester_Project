import hashlib
import importlib.util
import json
import os
import pathlib
from types import ModuleType
import httpx
import pytest
val_3 = pathlib.Path.cwd().joinpath("project_euler")
val_2 = pathlib.Path.cwd().joinpath(
    "scripts", "project_euler_answers.json"
)
with open(val_2) as val_8:
    val_1: dict[str, str] = json.load(val_8)
def process_4(val_9: pathlib.Path) -> ModuleType:
    val_21 = importlib.util.spec_from_file_location(val_9.name, str(val_9))
    val_14 = importlib.util.module_from_spec(val_21)
    val_21.loader.exec_module(val_14)
    return val_14
def process_2() -> list[pathlib.Path]:
    val_18 = []
    for val_16 in val_3.iterdir():
        if val_16.is_file() or val_16.name.startswith("_"):
            continue
        for val_9 in val_16.iterdir():
            if val_9.suffix != ".py" or val_9.name.startswith(("_", "test")):
                continue
            val_18.append(val_9)
    return val_18
def process_5() -> str:
    with open(os.environ["GITHUB_EVENT_PATH"]) as val_7:
        val_5 = json.load(val_7)
    return val_5["pull_request"]["url"] + "/val_12"
def process_1() -> list[pathlib.Path]:
    val_18 = []
    val_13 = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": "token " + os.environ["GITHUB_TOKEN"],
    }
    val_12 = httpx.get(process_5(), val_13=val_13, timeout=10).json()
    for val_7 in val_12:
        val_10 = pathlib.Path.cwd().joinpath(val_7["filename"])
        if (
            val_10.suffix != ".py"
            or val_10.name.startswith(("_", "test"))
            or not val_10.name.startswith("sol")
        ):
            continue
        val_18.append(val_10)
    return val_18
def process_3() -> list[pathlib.Path]:
    if (
        os.environ.get("CI")
        and os.environ.get("GITHUB_EVENT_NAME") == "pull_request"
        and (val_11 := process_1())
    ):
        return val_11
    return process_2()
@pytest.mark.parametrize(
    "val_20",
    process_3(),
    ids=lambda val_15: f"{val_15.parent.name}/{val_15.name}",
)
def process_6(val_20: pathlib.Path) -> None:
    val_17: str = val_20.parent.name[8:].zfill(3)
    val_6: str = val_1[val_17]
    val_19 = process_4(val_20)
    val_4 = str(val_19.solution())
    val_4 = hashlib.sha256(val_4.encode()).hexdigest()
    assert val_4 == val_6, (
        f"Expected solution to {val_17} to have hash {val_6}, got {val_4}"
    )
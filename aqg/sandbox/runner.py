"""Run code in hello-bitcoin's own runner image, exactly the way the judge does.

Containers are started with the same restrictions as hello-bitcoin/backend/judge/pool.py
(no network, read-only root, 64 MB exec tmpfs on /tmp, 0.5 CPU, 64 PIDs) and requests
go through the image's /judge/run.py, so verdicts, timing and memory accounting match
what users will see. Uses the docker CLI (no docker SDK dependency).
"""

from __future__ import annotations

import json
import queue
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aqg.config import SandboxConfig
from aqg.logging_setup import get_logger
from aqg.sandbox.judge_protocol import JudgeRequest, JudgeResult

log = get_logger("sandbox")
LABEL = "aqg.sandbox=1"


class SandboxError(RuntimeError):
    pass


def docker_available() -> tuple[bool, str]:
    try:
        out = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                             capture_output=True, text=True, timeout=20)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if out.returncode != 0:
        return False, out.stderr.strip()[:300]
    return True, out.stdout.strip()


def image_exists(image: str) -> bool:
    out = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True)
    return out.returncode == 0


def build_image(image: str, build_dir: Path) -> None:
    """Build the runner image from hello-bitcoin's own Dockerfile.runner (reads their files, writes nothing)."""
    dockerfile = Path(build_dir) / "Dockerfile.runner"
    if not dockerfile.exists():
        raise SandboxError(f"Cannot build {image!r}: {dockerfile} not found (set sandbox.build_dir in config.yaml)")
    log.info("building runner image", extra={"image": image, "context": str(build_dir)})
    out = subprocess.run(["docker", "build", "-t", image, "-f", str(dockerfile), str(build_dir)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        log.error("runner image build failed", extra={"output_tail": (out.stdout + out.stderr)[-3000:]})
        raise SandboxError(f"docker build of {image!r} failed (see logs/errors.log)")
    log.info("runner image built", extra={"image": image})


def ensure_image(image: str, build_dir: Path | None) -> None:
    if image_exists(image):
        return
    if build_dir is None:
        log.critical("runner image missing", extra={"image": image})
        raise SandboxError(f"Image {image!r} not found")
    build_image(image, build_dir)


class DockerSandbox:
    def __init__(self, cfg: SandboxConfig, build_dir: Path | None = None):
        self.cfg = cfg
        self.build_dir = build_dir
        self._containers: list[str] = []
        self._pool: queue.Queue[str] = queue.Queue()

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> None:
        ok, detail = docker_available()
        if not ok:
            log.critical("docker daemon unavailable", extra={"detail": detail})
            raise SandboxError(f"Docker is not available: {detail}")
        ensure_image(self.cfg.image, self.build_dir)
        for _ in range(max(1, self.cfg.workers)):
            cid = self._start_container()
            self._containers.append(cid)
            self._pool.put(cid)
        log.info("sandbox started", extra={"containers": len(self._containers), "image": self.cfg.image})

    def _start_container(self) -> str:
        cmd = [
            "docker", "run", "-d", "--rm",
            "--network", "none",
            "--memory", f"{self.cfg.container_memory_mb}m",
            "--cpus", str(self.cfg.cpus),
            "--pids-limit", str(self.cfg.pids_limit),
            "--read-only",
            "--tmpfs", "/tmp:size=64m,exec,mode=1777",
            "--label", LABEL,
            "--name", f"aqg-runner-{uuid.uuid4().hex[:10]}",
            self.cfg.image, "sleep", "infinity",
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            log.error("failed to start sandbox container", extra={"stderr": out.stderr[:500]})
            raise SandboxError(out.stderr.strip())
        cid = out.stdout.strip()
        log.debug("sandbox container started", extra={"container": cid[:12]})
        return cid

    def stop(self) -> None:
        for cid in self._containers:
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True, text=True)
        log.info("sandbox stopped", extra={"containers": len(self._containers)})
        self._containers.clear()

    def __enter__(self) -> "DockerSandbox":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ── execution ──────────────────────────────────────────────────────────
    def _replace(self, dead: str) -> str:
        log.warning("replacing unhealthy sandbox container", extra={"container": dead[:12]})
        subprocess.run(["docker", "rm", "-f", dead], capture_output=True, text=True)
        new = self._start_container()
        self._containers[self._containers.index(dead)] = new
        return new

    def run(self, req: JudgeRequest) -> JudgeResult:
        payload = req.model_dump_json().encode("utf-8")
        req_file = f"/tmp/{uuid.uuid4().hex}.json"
        cid = self._pool.get()
        try:
            for attempt in (1, 2):
                try:
                    out = subprocess.run(
                        ["docker", "exec", "-i", cid, "sh", "-c",
                         f"cat > {req_file} && python3 /judge/run.py {req_file}"],
                        input=payload, capture_output=True, timeout=self.cfg.exec_deadline_s,
                    )
                except subprocess.TimeoutExpired:
                    log.error("sandbox exec deadline exceeded", extra={"language": req.language})
                    return JudgeResult(verdict="TLE", stderr="sandbox exec deadline exceeded")
                text = out.stdout.decode("utf-8", errors="replace")
                start = text.find("{")
                if start != -1:
                    try:
                        result = JudgeResult(**json.loads(text[start:]))
                    except (json.JSONDecodeError, TypeError) as exc:
                        return JudgeResult(verdict="RE", stderr=f"harness output unparsable: {exc}: {text[:300]}")
                    log.debug("sandbox run", extra={"language": req.language, "verdict": result.verdict,
                                                    "time_ms": result.time_ms, "memory_kb": result.memory_kb})
                    return result
                err = out.stderr.decode("utf-8", errors="replace")
                if attempt == 1 and ("is not running" in err or "No such container" in err):
                    cid = self._replace(cid)
                    continue
                return JudgeResult(verdict="RE", stderr=f"harness produced no JSON: {err[:300]} {text[:300]}")
            return JudgeResult(verdict="RE", stderr="sandbox unavailable")
        finally:
            self._pool.put(cid)

    def run_many(self, reqs: list[JudgeRequest]) -> list[JudgeResult]:
        with ThreadPoolExecutor(max_workers=max(1, len(self._containers))) as pool:
            return list(pool.map(self.run, reqs))


def cleanup_stale_containers() -> int:
    out = subprocess.run(["docker", "ps", "-aq", "--filter", f"label={LABEL}"], capture_output=True, text=True)
    ids = out.stdout.split()
    if ids:
        subprocess.run(["docker", "rm", "-f", *ids], capture_output=True, text=True)
    return len(ids)

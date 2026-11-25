# core/tools/mail_audit_tool/client.py

# Copyright 2025 Artyom Yudin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Async SSH-клиент к journalctl на почтовом сервере."""
import asyncio
import logging
import shlex
from typing import Optional

import asyncssh

from .config import (
    SSH_HOST, SSH_USER, SSH_KEY_PATH,
    SSH_CONNECT_TIMEOUT, SSH_QUERY_TIMEOUT,
    SSH_KEEPALIVE, SSH_KEEPALIVE_COUNT_MAX,
)

logger = logging.getLogger(__name__)



# Namespaces, которые принимает серверный wrapper (mail_auditor_journalctl).
# Клиент НЕ валидирует аргументы — это делает сервер на каждом шаге.
# Здесь только грубая проверка namespace: опечатка в коде инструмента
# падает сразу, а не после SSH-раундтрипа.
ALLOWED_GATEWAY_NAMESPACES = ("journal", "postfix", "rspamd", "redis", "system")

# ---------- Async SSH-клиент ----------

class AsyncJournalClient:
    """Один SSH-коннект, много параллельных channel'ов через asyncssh."""

    def __init__(self, host: str, user: str, key_path: str):
        self.host = host
        self.user = user
        self.key_path = key_path
        self._conn: Optional[asyncssh.SSHClientConnection] = None
        self._connect_lock = asyncio.Lock()

    async def _ensure_connected(self) -> asyncssh.SSHClientConnection:
        # Быстрый путь без lock'а: уже подключены.
        if self._conn is not None and not self._conn.is_closed():
            return self._conn

        async with self._connect_lock:
            # Double-check: пока ждали lock, другой корутин мог подключиться.
            if self._conn is not None and not self._conn.is_closed():
                return self._conn

            logger.info("asyncssh connect → %s@%s", self.user, self.host)
            self._conn = await asyncssh.connect(
                self.host,
                username=self.user,
                client_keys=[self.key_path],
                known_hosts=None,          # аналог AutoAddPolicy; см. ниже
                connect_timeout=SSH_CONNECT_TIMEOUT,
                keepalive_interval=SSH_KEEPALIVE,
                keepalive_count_max=SSH_KEEPALIVE_COUNT_MAX,
            )
            return self._conn

    async def _invalidate(self, conn: asyncssh.SSHClientConnection):
        async with self._connect_lock:
            if self._conn is conn:
                try:
                    self._conn.close()
                    await self._conn.wait_closed()
                except Exception:
                    pass
                self._conn = None

    async def query(
        self,
        units: list[str],
        time_args: list[str],
        grep_pattern: Optional[str] = None,
        max_lines: Optional[int] = None,
        identifiers=None,
    ) -> list[str]:
        last_exc: Optional[Exception] = None
        for attempt in (1, 2):
            conn = await self._ensure_connected()
            try:
                return await self._query_once(conn, units, time_args, grep_pattern, max_lines, identifiers)
            except (
                asyncssh.ConnectionLost,
                asyncssh.DisconnectError,
                asyncssh.ChannelOpenError,
                OSError,
                EOFError,
            ) as e:
                # Разрыв коннекта — переподключаемся и пробуем ещё раз.
                last_exc = e
                logger.warning(
                    "asyncssh query failed (%s), attempt %d/2", e, attempt
                )
                await self._invalidate(conn)
            except (asyncssh.TimeoutError, asyncio.TimeoutError) as e:
                # Внутренний таймаут conn.run(timeout=20). Ретрай не поможет —
                # журнал реально большой или сервер тормозит. Отдаём ошибку
                # наверх, вызывающий код решит, что делать.
                raise RuntimeError(
                    f"journalctl timeout ({SSH_QUERY_TIMEOUT}s): {e}"
                ) from e
        raise RuntimeError(f"SSH query failed after retry: {last_exc}")

    async def _query_once(
        self,
        conn: asyncssh.SSHClientConnection,
        units: list[str],
        time_args: list[str],
        grep_pattern: Optional[str],
        max_lines: Optional[int] = None,
        identifiers=None,
    ) -> list[str]:
        cmd = ["journal"]
        # --reverse -n N: journald идёт с конца журнала и останавливается
        # после N матчей. Для period='all' это превращает 7-секундный
        # полный скан в 0.5-1с. Вывод разворачиваем обратно в хронологию.
        if max_lines:
            cmd.extend(["--reverse", "-n", str(max_lines)])
        for u in units:
            cmd.extend(["-u", shlex.quote(u)])
        for ident in (identifiers or []):
            cmd.extend(["-t", shlex.quote(ident)])
        for t in time_args:
            cmd.append(shlex.quote(t))
        if grep_pattern:
            cmd.extend(["-g", shlex.quote(grep_pattern)])

        remote_cmd = " ".join(cmd)

        result = await conn.run(
            remote_cmd,
            check=False,
            timeout=SSH_QUERY_TIMEOUT,
        )

        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()

        if result.exit_status and not out:
            low = err.lower()
            if "no entries" in low:
                return []
            if (
                "not seeing messages" in low
                or "permission denied" in low
                or "access denied" in low
            ):
                raise RuntimeError(
                    f"journalctl: нет доступа к логам. Проверь, что "
                    f"пользователь {self.user} входит в группу "
                    f"systemd-journal (или adm). stderr: {err}"
                )
            raise RuntimeError(
                f"journalctl error (exit={result.exit_status}): {err}"
            )

        lines = out.splitlines() if out else []
        if max_lines:
            lines.reverse()  # --reverse отдал новые-первыми, вернуть хронологию
        return lines

    
    async def _run_raw(self, remote_cmd: str) -> str:
        """Выполняет произвольную разрешённую команду через пул SSH-коннекта с ретраями."""
        last_exc: Optional[Exception] = None
        for attempt in (1, 2):
            conn = await self._ensure_connected()
            try:
                result = await conn.run(
                    remote_cmd,
                    check=False,
                    timeout=SSH_QUERY_TIMEOUT,
                )
                out = (result.stdout or "").strip()
                err = (result.stderr or "").strip()

                if result.exit_status and not out:
                    raise RuntimeError(
                        f"Remote command '{remote_cmd}' failed (exit={result.exit_status}): {err}"
                    )
                return out

            except (
                asyncssh.ConnectionLost,
                asyncssh.DisconnectError,
                asyncssh.ChannelOpenError,
                OSError,
                EOFError,
            ) as e:
                last_exc = e
                logger.warning(
                    "asyncssh command failed (%s), attempt %d/2", e, attempt
                )
                await self._invalidate(conn)
            except (asyncssh.TimeoutError, asyncio.TimeoutError) as e:
                raise RuntimeError(
                    f"SSH command timeout ({SSH_QUERY_TIMEOUT}s): {e}"
                ) from e

        raise RuntimeError(f"SSH command failed after retry: {last_exc}")


    async def run_gateway_cmd(self, command: str) -> str:
        """
        Выполняет namespaced-команду через SSH forced-command wrapper.

        Примеры:
          run_gateway_cmd("postfix postqueue -j")
          run_gateway_cmd("postfix postconf mynetworks")
          run_gateway_cmd("postfix mynetworks")
          run_gateway_cmd("rspamd stat")
          run_gateway_cmd("rspamd configdump -j dkim")
          run_gateway_cmd("redis info memory")
          run_gateway_cmd("redis scan 0 match rspamd:* count 100")
          run_gateway_cmd("system health")
          run_gateway_cmd("system active rspamd.service")
          run_gateway_cmd("system failed")
          run_gateway_cmd("journal -u postfix@-.service -n 5")

        Аргументы валидирует серверный wrapper. Клиент проверяет только
        namespace — чтобы инструменты не отправляли произвольные команды
        (защита от опечаток и случайного f-string с пользовательским вводом).
        """
        head = command.split(None, 1)[0] if command.strip() else ""
        if head not in ALLOWED_GATEWAY_NAMESPACES:
            raise ValueError(
                f"Unsupported gateway command: {command!r}. "
                f"Allowed namespaces: {', '.join(ALLOWED_GATEWAY_NAMESPACES)}"
            )
        return await self._run_raw(command)


    async def close(self):
        async with self._connect_lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                    await self._conn.wait_closed()
                except Exception:
                    pass
                self._conn = None

# ---------- Singleton клиента ----------
#
# В asyncio достаточно простого глобала без lock'а на создание: между
# "if _client is None" и "_client = ..." нет await, поэтому корутины не
# могут чередоваться в этой точке. Реальная синхронизация — в _connect_lock
# внутри AsyncJournalClient._ensure_connected.

_client: Optional[AsyncJournalClient] = None


def _get_client() -> AsyncJournalClient:
    global _client
    if _client is None:
        _client = AsyncJournalClient(SSH_HOST, SSH_USER, SSH_KEY_PATH)
    return _client


async def close_client():
    """Вызывать из lifespan/shutdown приложения."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None
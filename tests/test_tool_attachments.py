# tests/test_tool_attachments.py

import asyncio
from core.tools.context import (
    attach_table, set_current_tool_call_id, pop_attachments_for_calls,
)

async def _fake_tool(user_id: str, tc_id: str):
    set_current_tool_call_id(tc_id)
    attach_table(
        title=f"table_{user_id}",
        headers=["x"], rows=[[user_id]],
    )
    await asyncio.sleep(0.01)  # эмуляция I/O
    return pop_attachments_for_calls([tc_id])

async def test_isolation():
    results = await asyncio.gather(
        _fake_tool("A", "tc_A"),
        _fake_tool("B", "tc_B"),
    )
    assert "table_A" in results[0] and "table_B" not in results[0]
    assert "table_B" in results[1] and "table_A" not in results[1]
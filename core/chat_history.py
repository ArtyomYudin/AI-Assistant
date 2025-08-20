from langchain_core.chat_history import InMemoryChatMessageHistory

class ChatHistoryManager:
    def __init__(self, max_messages: int):
        self.max_messages = max_messages
        self._store = {}

    def get(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self._store:
            self._store[session_id] = InMemoryChatMessageHistory()
        hist = self._store[session_id]
        if len(hist.messages) > self.max_messages:
            new_hist = InMemoryChatMessageHistory()
            new_hist.messages = hist.messages[-self.max_messages:]
            self._store[session_id] = new_hist
            return new_hist
        return hist

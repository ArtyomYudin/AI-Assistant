# core/collection_manager.py

"""
CollectionManager — модуль, который:
- загружает подпапки DATA_DIR
- индексирует каждую папку как отдельную коллекцию Milvus
- строит embedding каждой коллекции
- выбирает коллекцию для запроса по cosine similarity
- вызывает Milvus hybrid search в выбранной коллекции

Используется RAGCore, MilvusManager, твой embedder и сплиттеры.
"""
import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import redis.asyncio as redis

import numpy as np
from langchain_core.documents import Document

from core.milvus_manager import MilvusManager

logger = logging.getLogger(__name__)

REDIS_ROUTER_KEY = "router_state"     # хранит структуру роутера
REDIS_EMB_PREFIX = "router_emb_"      # префикс для векторов коллекций

# ==============================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ
# Строит embedding коллекции
# ==============================

async def build_collection_embedding(
    docs: List[Document],
    embed_query_fn,
) -> np.ndarray:
    """
    Строит embedding коллекции как взвешенный центроид всех документов.
    Использует bm25_text если есть.
    """

    vecs = []
    weights = []

    for doc in docs:
        # лучший текст для общей семантики
        txt = doc.metadata.get("bm25_text") or doc.page_content[:300]
        emb = await embed_query_fn(txt)

        # вес = sqrt длины
        w = np.sqrt(len(txt))

        vecs.append(emb)
        weights.append(w)

    vecs = np.array(vecs)
    weights = np.array(weights)

    centroid = np.average(vecs, axis=0, weights=weights)
    centroid = centroid / np.linalg.norm(centroid)

    return centroid


# ==============================
# ROUTER КОЛЛЕКЦИЙ
# ==============================

class CollectionRouter:
    """
    Хранит embedding каждой коллекции.
    Позволяет определить, куда отправить запрос.
    """

    def __init__(self, collections: Dict[str, np.ndarray]):
        self.collections = collections

    def pick_collection(self, query_vec: np.ndarray) -> str:
        """Возвращает имя коллекции с максимальной cosine similarity."""

        sims = {
            name: float(np.dot(query_vec, vec))
            for name, vec in self.collections.items()
        }

        # наибольшая косинусная близость
        return max(sims, key=sims.get)


    def pick_collections(
            self,
            query_vec: np.ndarray,
            top_k: int = 2,
            min_margin: float = 0.05,
    ) -> List[str]:
        """Возвращает top-K коллекций, если их скоры близки."""
        sims = {
            name: float(np.dot(query_vec, vec))
            for name, vec in self.collections.items()
        }
        ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        # Временно
        logger.info(
            "pick_collections: ranked=%s, margin=%.4f",
            [(name, round(score, 4)) for name, score in ranked],
            (ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else 0.0,
        )

        if not ranked:
            return []

        best_name, best_score = ranked[0]
        selected = [best_name]

        for name, score in ranked[1:top_k]:
            # Берём вторую коллекцию, только если она «догоняет» лидера
            if best_score - score <= min_margin:
                selected.append(name)

        return selected


class CollectionManager:
    """
    Управляет жизненным циклом всех коллекций:
    - загрузка документов по папкам
    - создание Milvus-коллекций
    - индексирование
    - построение роутера
    - выбор коллекции
    """

    def __init__(self, core):
        """
        core — инстанс RAGCore
        Содержит:
            - config
            - embeddings (property)
            - splitters
            - milvus (основной, можно создавать новые для каждой коллекции)
        """
        self.core = core
        self.router: Optional[CollectionRouter] = None
        # self.collections: dict[str, MilvusManager] = {}  # имена коллекций -> менеджеры Milvus
        self.collections: List[str] = []  # имена коллекций
        # self.last_used_collection: Optional[str] = None
        self.redis = redis.StrictRedis(
            host=self.core.config.REDIS_HOST,
            port=self.core.config.REDIS_PORT,
            db= 0,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )

    def _has_broad_intent(self, query: str) -> bool:
        """
        Определяет broad-intent: пользователь просит «всё», «список», «все X».
        Для таких запросов reranker часто даёт низкий top_score (конкретные
        чанки плохо матчатся с обобщающими словами), поэтому expansion нужен
        даже при низком score.
        """
        if not query:
            return False

        q = query.lower()
        broad_markers = (
            "все ", "всех ", "всё ", "всего ", "весь ", "вся ",
            "полный", "полностью", "полный список", "полный перечень",
            "список", "перечень", "перечисли",
            "какие есть", "что есть", "покажи все", "покажи всех",
            "на предприятии", "предприятия", "объекта", "в целом",
            "целиком", "по всем", "по всему",
        )
        return any(m in q for m in broad_markers)

    async def _drop_all_collections(self) -> None:
        """
        Полностью удаляет все коллекции Milvus и очищает ВСЁ состояние в Redis,
        ссылающееся на коллекции: router_state, router_emb_*, chat_history:*,
        last_used_collection:*.
        """
        logger.warning("DROP_ALL_COLLECTIONS=True — удаляю ВСЕ коллекции Milvus")

        try:
            existing = await self.core.milvus.client.list_collections()
        except Exception as e:
            logger.exception("Не удалось получить список коллекций Milvus: %s", e)
            existing = []

        logger.info("Найдено коллекций для удаления: %d → %s", len(existing), existing)

        for name in existing:
            try:
                await self.core.milvus.client.drop_collection(name)
                logger.info("Коллекция '%s' удалена", name)
            except Exception as e:
                logger.exception("Ошибка при удалении коллекции '%s': %s", name, e)

        # ── Очистка Redis: ВСЁ, что ссылается на коллекции ──
        try:
            # Состояние роутера + векторы коллекций
            await self.redis.delete(REDIS_ROUTER_KEY)
            removed_emb = 0
            async for key in self.redis.scan_iter(f"{REDIS_EMB_PREFIX}*"):
                await self.redis.delete(key)
                removed_emb += 1

            # last_used_collection:* (по сессиям) — роутер помнит «последнюю»
            removed_last = 0
            async for key in self.redis.scan_iter("last_used_collection:*"):
                await self.redis.delete(key)
                removed_last += 1

            # chat_history:* — история со старыми ответами
            removed_hist = 0
            async for key in self.redis.scan_iter("chat_history:*"):
                await self.redis.delete(key)
                removed_hist += 1

            logger.info(
                "Redis очищен: router_state + %d vectors + %d last_used + %d histories",
                removed_emb, removed_last, removed_hist,
            )
        except Exception as e:
            logger.exception("Ошибка при очистке Redis: %s", e)

        # Сбрасываем локальное состояние
        self.collections = []
        self.router = None


    async def build_all_collections(self):
        """
        Сканирует DATA_DIR, индексирует каждую подпапку как отдельную коллекцию Milvus.
        Если config.DROP_ALL_COLLECTIONS=True — сначала удаляет все коллекции
        и очищает состояние роутера в Redis.
        """

        # 1. Опционально: полная очистка перед переиндексацией
        if getattr(self.core.config, "DROP_ALL_COLLECTIONS", False):
            await self._drop_all_collections()
            logger.warning(
                "DROP_ALL_COLLECTIONS=True сработал — коллекции удалены. "
                "Не забудьте выставить флаг обратно в False."
            )

        root = Path(self.core.config.DATA_DIR)
        if not root.exists():
            logger.error("DATA_DIR не существует: %s", root)
            return

        subdirs = [p for p in root.iterdir() if p.is_dir()]
        logger.info("Найдено коллекций: %d", len(subdirs))

        for folder in subdirs:
            cname = folder.name
            logger.info("Индексация коллекции: %s", cname)

            docs = self.core.load_documents(str(folder))
            if not docs:
                logger.warning("Папка %s пуста — пропускаю", folder)
                continue

            await self._index_collection(cname, docs)
            if cname not in self.collections:
                self.collections.append(cname)

        # сохраняем имена коллекций
        await self._save_router_state()


    async def _expand_strong_sources(
            self,
            merged: List[Document],
            expansion_threshold: float,
            max_expanded_sources: int,
            max_chunks_per_source: int,
    ) -> List[Document]:
        """
        Small-to-big: для каждого источника, у которого top-чанк достиг
        expansion_threshold, подтягивает ВСЕ чанки этого источника из его коллекции.
        Возвращает расширенный список (исходные чанки + добавленные).
        """
        if not merged:
            return merged

        # 1. Группируем кандидатов по source
        by_source: dict[str, list[Document]] = {}
        for d in merged:
            src = d.metadata.get("source", "N/A")
            by_source.setdefault(src, []).append(d)

        # 2. Определяем, какие источники разворачивать
        expandable = []
        for src, docs in by_source.items():
            top = max(d.metadata.get("score", 0.0) for d in docs)
            if top >= expansion_threshold:
                collection = docs[0].metadata.get("collection")
                if collection:
                    expandable.append((src, top, collection))

        if not expandable:
            return merged

        expandable.sort(key=lambda x: x[1], reverse=True)
        expandable = expandable[:max_expanded_sources]

        # 3. Тянем все чанки из Milvus
        seen_hashes = {d.metadata.get("hash") for d in merged if d.metadata.get("hash")}
        expanded = list(merged)

        for src, top_score, collection in expandable:
            try:
                res = await self.core.milvus.client.query(
                    collection_name=collection,
                    filter=f'source == "{src}"',
                    output_fields=[
                        "text", "title", "source", "hash",
                        "bm25_text", "chunk_index",
                    ],
                    limit=max_chunks_per_source,
                )
            except Exception as e:
                err = str(e).lower()
                if "chunk_index" in err and ("not exist" in err or "invalid" in err):
                    logger.warning(
                        "Expansion: коллекция %s без поля chunk_index "
                        "(создана до миграции) — перезапрашиваю без сортировки. "
                        "Перезапустите /index для корректного порядка чанков.",
                        collection,
                    )
                    res = await self.core.milvus.client.query(
                        collection_name=collection,
                        filter=f'source == "{src}"',
                        output_fields=[
                            "text", "title", "source", "hash", "bm25_text",
                        ],
                        limit=max_chunks_per_source,
                    )
                else:
                    logger.warning(
                        "Expansion: ошибка запроса source=%s: %s", src, e,
                    )
                    continue

            # Сортировка ОДИН раз, после обоих try/except
            res_sorted = sorted(res, key=lambda x: x.get("chunk_index", 0))

            added = 0
            for hit in res_sorted:
                h = hit.get("hash")
                if h and h in seen_hashes:
                    continue
                if h:
                    seen_hashes.add(h)
                expanded.append(Document(
                    page_content=hit.get("text", ""),
                    metadata={
                        "title": hit.get("title", ""),
                        "source": src,
                        "hash": h,
                        "bm25_text": hit.get("bm25_text"),
                        "collection": collection,
                        "score": top_score,
                        "chunk_index": hit.get("chunk_index", 999),  # ★
                        "expanded": True,
                    },
                ))
                added += 1

            logger.info(
                "🪄 Expansion: source=%s, collection=%s, added=%d chunks (top_score=%.3f)",
                src, collection, added, top_score,
            )

        return expanded

    # ----------------------------------------------------------
    # Реальная индексация документов в коллекцию Milvus
    # ----------------------------------------------------------

    async def _index_collection(self, collection_name: str, docs: List[Document]):
        """Индексирует документы в указанную коллекцию (делегирует в RAGCore)."""
        total = await self.core._index_documents(collection_name, docs)
        if total == 0:
            logger.warning("_index_collection: %s — 0 фрагментов", collection_name)


    async def _search_in_collections(
            self,
            query: str,
            query_dense: List[float],
            collections: List[str],
            threshold_ratio: Optional[float] = None,
            min_absolute: Optional[float] = None,
            per_source_cap: Optional[int] = None,
            expand_sources: bool = True,
            expansion_threshold: float = 0.6,  # top-score источника для триггера
            max_expanded_sources: int = 2,  # сколько источников разворачивать
            max_chunks_per_source: int = 20,  # safety-cap на большой источник
    ) -> List[Document]:
        """
        Параллельный hybrid_search во всех коллекциях с per-collection фильтрацией.

        Для каждой коллекции:
          - берём сырые результаты (apply_filter=False)
          - считаем local_max = max(score) в этой коллекции
          - threshold = max(local_max * threshold_ratio, min_absolute)
          - оставляем только те, что >= threshold

        Затем объединяем по всем коллекциям и дедуплицируем по hash.
        Это защищает от ситуации, когда вторичная коллекция имеет низкие
        реранкер-скоры из-за составного запроса, но её документы всё равно
        нужны LLM для ответа на часть вопроса.
        """

        if threshold_ratio is None:
            threshold_ratio = self.core.config.COLLECTION_THRESHOLD_RATIO
        if min_absolute is None:
            min_absolute = self.core.config.COLLECTION_MIN_ABSOLUTE
        if per_source_cap is None:
            per_source_cap = self.core.config.PER_SOURCE_CAP

        if not collections:
            return []

        # 1. Проверяем существование и загружаем коллекции
        existing: List[str] = []
        for cname in collections:
            try:
                if await self.core.milvus.client.has_collection(cname):
                    await self.core.milvus.ensure_collection_loaded(cname)
                    existing.append(cname)
                else:
                    logger.warning("Коллекция %s не найдена — пропускаю", cname)
            except Exception as e:
                logger.exception("Ошибка проверки коллекции %s: %s", cname, e)

        if not existing:
            return []

        # 2. Параллельный поиск БЕЗ глобальной фильтрации
        tasks = [
            self.core.milvus.hybrid_search(
                query_text=query,
                query_dense=query_dense,
                fetch_k=self.core.config.FETCH_K,
                top_k=self.core.config.K,
                collection_name=cname,
                reranker_endpoint=self.core.config.RERANKER_BASE_URL,
                apply_filter=False,  # ← ключевое: фильтр применяем сами
                score_threshold=self.core.config.HYBRID_SEARCH_SCORE_THRESHOLD
            )
            for cname in existing
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 3. Per-collection relative threshold
        merged: List[Document] = []
        stats: List[str] = []
        for cname, docs in zip(existing, results):
            if isinstance(docs, Exception):
                logger.exception("Ошибка поиска в %s: %s", cname, docs)
                continue
            if not docs:
                stats.append(f"{cname}: 0/0")
                continue

            local_max = max(d.metadata.get("score", 0.0) for d in docs)
            local_threshold = max(local_max * threshold_ratio, min_absolute)

            kept = []
            for doc in docs:
                score = doc.metadata.get("score", 0.0)
                if score >= local_threshold:
                    doc.metadata["collection"] = cname
                    kept.append(doc)

            stats.append(
                f"{cname}: {len(kept)}/{len(docs)} "
                f"(max={local_max:.3f}, thr={local_threshold:.3f})"
            )
            merged.extend(kept)

        logger.info("_search_in_collections per-collection: %s", " | ".join(stats))

        if not merged:
            # Broad intent: пользователь просит «все X», reranker дал низкие
            # score — но документы, скорее всего, есть. Снижаем порог и
            # пропускаем всё, что вернул hybrid_search (top_k на коллекцию).
            if self._has_broad_intent(query):
                logger.info(
                    "Broad intent + пустой первичный поиск → мягкий порог "
                    "(принимаю всё, что вернул reranker, без min_absolute)"
                )
                # Повторяем сборку с min_absolute=0.0
                for cname, docs in zip(existing, results):
                    if isinstance(docs, Exception) or not docs:
                        continue
                    local_max = max(d.metadata.get("score", 0.0) for d in docs)
                    local_threshold = local_max * threshold_ratio  # без min_absolute
                    for doc in docs:
                        if doc.metadata.get("score", 0.0) >= local_threshold:
                            doc.metadata["collection"] = cname
                            merged.append(doc)
                    logger.info(
                        "Broad intent soft-threshold: %s kept %d/%d (thr=%.4f)",
                        cname,
                        len([d for d in merged if d.metadata.get("collection") == cname]),
                        len(docs),
                        local_threshold,
                    )

            if not merged:
                return []

        # 4. Сортировка по score (важно: cap должен применяться к отсортированному)
        merged.sort(key=lambda d: d.metadata.get("score", 0.0), reverse=True)

        # 5. Per-source cap: не более N чанков с одного файла
        source_counts: dict = {}
        capped: List[Document] = []
        dropped_by_source: dict = {}
        for d in merged:
            src = d.metadata.get("source", "N/A")
            if source_counts.get(src, 0) >= per_source_cap:
                dropped_by_source[src] = dropped_by_source.get(src, 0) + 1
                continue
            source_counts[src] = source_counts.get(src, 0) + 1
            capped.append(d)

        if dropped_by_source:
            logger.info(
                "_search_in_collections per-source cap=%d dropped: %s",
                per_source_cap,
                ", ".join(f"{src}: -{n}" for src, n in dropped_by_source.items()),
            )

        # 5.5. ── EXPANSION BY SOURCE ──────────────────────────────
        # Режим задаётся config.EXPANSION_MODE:
        #   off    — ничего не разворачиваем (старое поведение)
        #   always — по общему EXPANSION_THRESHOLD
        #   smart  — broad intent → threshold=0, narrow → EXPANSION_THRESHOLD
        if expand_sources and self.core.config.EXPANSION_MODE != "off":
            mode = self.core.config.EXPANSION_MODE
            base_threshold = self.core.config.EXPANSION_THRESHOLD
            skip = False

            if mode == "smart":
                broad = self._has_broad_intent(query)
                top_score = max((d.metadata.get("score", 0.0) for d in capped), default=0.0)
                skip_max = self.core.config.EXPANSION_SKIP_MAX_MATCHED
                skip_score = self.core.config.EXPANSION_SKIP_SCORE

                # ★ Узкий+уверенный запрос: 1 чанк, score > 0.9 → это про конкретный чанк
                if (
                        not broad
                        and skip_max > 0
                        and skip_score > 0
                        and len(capped) <= skip_max
                        and top_score >= skip_score
                ):
                    skip = True
                    logger.info(
                        "Expansion mode=smart → skip (narrow+confident: %d chunk(s), "
                        "top_score=%.3f ≥ %.2f) — документ не разворачиваю",
                        len(capped), top_score, skip_score,
                    )
                elif broad:
                    effective_threshold = 0.0
                    reason = "smart/broad-intent (thr=0.0)"
                else:
                    effective_threshold = base_threshold
                    reason = f"smart/narrow (thr={base_threshold})"
            elif mode == "always":
                effective_threshold = base_threshold
                reason = f"always (thr={base_threshold})"
            else:
                effective_threshold = base_threshold
                reason = f"fallback (thr={base_threshold})"

            if not skip:
                logger.info("Expansion mode=%s → %s", mode, reason)
                capped = await self._expand_strong_sources(
                    capped,
                    expansion_threshold=effective_threshold,
                    max_expanded_sources=self.core.config.MAX_EXPANDED_SOURCES,
                    max_chunks_per_source=self.core.config.MAX_CHUNKS_PER_SOURCE,
                )

        # 6. Дедуп по hash
        seen: set = set()
        deduped: List[Document] = []
        for d in capped:
            h = d.metadata.get("hash")
            if h and h in seen:
                continue
            if h:
                seen.add(h)
            deduped.append(d)

        return deduped


    def is_followup_query(self, query: str) -> bool:
        """
            Простая и надёжная эвристика follow-up:
            - Если текст пустой -> False
            - Возьмём первое слово (lowercase) и посмотрим, входит ли оно в набор уточняющих слов.
            """
        if not query:
            return False

        q = query.strip().lower().split()
        if not q:
            return False

        first = q[0]

        followup_first_words = {
            "какие", "какая", "какой", "какои",  # опечатки/вариации
            "ещё", "еще", "покажи", "покажите",
            "а", "подробнее", "что", "чего"
        }

        # если первое слово — из списка уточняющих — считаем follow-up
        if first in followup_first_words and len(q) <=2 :
            return True

        # короткие одно-словные вопросы, типа "какие?", "какие.", "какой?" — считаем follow-up
        if len(q) == 1 and len(first) <= 6:
            # но избегаем ловли одиночных слов, которые скорее самостоятельны:
            standalone_no = {"да", "нет", "почему", "как"}
            return first not in standalone_no

        return False


    # ----------------------------------------------------------
    # Построение Collection Router
    # ----------------------------------------------------------

    async def build_router(self):
        """
            Если в Redis есть сохранённый роутер — восстанавливаем,
            но только если ВСЕ его коллекции существуют в Milvus.
            """
        restored = await    self._load_router_state()
        if restored:
            # ★ Проверяем, что все коллекции роутера реально есть в Milvus
            try:
                existing = set(await self.core.milvus.client.list_collections())
            except Exception as e:
                logger.warning("Не удалось проверить коллекции в Milvus: %s", e)
                existing = None

            if existing is not None:
                phantom = [c for c in restored.collections if c not in existing]
                if phantom:
                    logger.warning(
                        "🗑️ Роутер ссылается на несуществующие коллекции: %s — пересобираю",
                        phantom,
                    )
                    # Сбрасываем — ниже построим заново
                    await self.redis.delete(REDIS_ROUTER_KEY)
                    async for key in self.redis.scan_iter(f"{REDIS_EMB_PREFIX}*"):
                        await self.redis.delete(key)
                    restored = None

            if restored:
                logger.info("Роутер восстановлен из Redis")
                self.router = restored
                return

        logger.info("Роутер отсутствует — создаём заново")

        root = Path(self.core.config.DATA_DIR)
        subdirs = [p for p in root.iterdir() if p.is_dir()]

        embeddings = {}

        for folder in subdirs:
            name = folder.name
            docs = self.core.load_documents(str(folder))
            if not docs:
                continue

            centroid = await build_collection_embedding(
                docs, self.core.embeddings.embed_query
            )
            embeddings[name] = centroid
            await self._save_collection_vector(name, centroid)

        self.collections = list(embeddings.keys())
        self.router = CollectionRouter(embeddings)
        logger.info("Маршрутизатор создан: %d коллекций", len(embeddings))

    # ----------------------------------------------------------
    # Выбор коллекции для запроса
    # ----------------------------------------------------------

    async def route_query(self, query: str) -> str:
        """
        Возвращает имя коллекции, куда отправить запрос.
        """

        if not self.router:
            raise RuntimeError("Router не создан: вызови build_router().")

        qvec = await self.core.embeddings.embed_query(query)
        return self.router.pick_collection(qvec)

    # ----------------------------------------------------------
    # Выполнить поиск с роутингом
    # ----------------------------------------------------------

    async def routed_search(
            self,
            query: str,
            conversation_id: Optional[str] = None,
    ) -> List[Document]:
        """Top-K routing + параллельный поиск + слияние."""
        if conversation_id is None:
            conversation_id = "global"

        if not self.router:
            await self.build_router()

        # Эмбеддинг запроса — один раз для routing и для поиска
        query_dense = await self.core._get_embedding_cached(query)
        if not query_dense:
            logger.warning("Не удалось получить эмбеддинг для запроса: %r", query)
            return []

        last = await self.load_last_used_collection(conversation_id)

        if last and self.is_followup_query(query):
            collections = [last]
            logger.info("Follow-up (conv=%s) — коллекция %s", conversation_id, last)
        else:
            collections = self.router.pick_collections(query_dense, top_k=2,  min_margin=0.1)
            logger.info("Роутер выбрал коллекции: %s", collections)
            if collections:
                await self.save_last_used_collection(collections[0], session_id=conversation_id)

        merged = await self._search_in_collections(query, query_dense, collections)
        logger.info(
            "routed_search: collections=%s, merged=%d", collections, len(merged)
        )
        # пусть _build_context сам решает, что влезает в MAX_CONTEXT_TOKENS
        # return merged[: self.core.config.K]
        return merged


    async def routed_search_with_fallback(
            self,
            query: str,
            conversation_id: Optional[str] = None,
            min_relevant: Optional[int] = None,
    ) -> List[Document]:
        """
        Как routed_search, но если после первичного поиска осталось меньше
        min_relevant документов — доискивает в остальных коллекциях.
        """
        if conversation_id is None:
            conversation_id = "global"
        # if min_relevant is None:
        #     min_relevant = max(1, self.core.config.K // 2)


        if not self.router:
            await self.build_router()

        query_dense = await self.core._get_embedding_cached(query)
        if not query_dense:
            return []

        last = await self.load_last_used_collection(conversation_id)

        if last and self.is_followup_query(query):
            collections = [last]
            logger.info("Follow-up (conv=%s) — коллекция %s", conversation_id, last)
        else:
            collections = self.router.pick_collections(query_dense, top_k=2,  min_margin=0.1)
            logger.info("Роутер выбрал коллекции: %s", collections)
            if collections:
                await self.save_last_used_collection(collections[0], session_id=conversation_id)

        merged = await self._search_in_collections(query, query_dense, collections)
        logger.info("Первичный поиск: collections=%s, docs=%d", collections, len(merged))

        if min_relevant is None:
            # Для N коллекций нужно хотя бы N документов (по одному от каждой),
            # но не меньше 2, если коллекций 1 (тогда нужен запас).
            n_collections = len(collections) if 'collections' in locals() else 1
            min_relevant = max(2, n_collections)

        if len(merged) < min_relevant:
            all_collections = list(self.router.collections.keys())
            remaining = [c for c in all_collections if c not in collections]
            if remaining:
                logger.info(
                    "Fallback: docs=%d < %d → доискиваю в %s",
                    len(merged), min_relevant, remaining,
                )
                extra = await self._search_in_collections(query, query_dense, remaining)

                # Мержим и дедуплицируем по hash
                seen = {d.metadata.get("hash") for d in merged if d.metadata.get("hash")}
                for d in extra:
                    h = d.metadata.get("hash")
                    if h and h in seen:
                        continue
                    if h:
                        seen.add(h)
                    merged.append(d)

                merged.sort(key=lambda d: d.metadata.get("score", 0.0), reverse=True)
                logger.info("Fallback завершён: docs=%d", len(merged))

        return merged[: self.core.config.K]


    # ---------------------------------------------------------
    # Redis: сохранение состояния
    # ---------------------------------------------------------
    async def save_last_used_collection(self, name: str, session_id: str = "global"):
        key = f"last_used_collection:{session_id}"
        # Сохраняем как строку
        await self.redis.set(key, name)

    async def load_last_used_collection(self, session_id: str = "global") -> Optional[str]:
        key = f"last_used_collection:{session_id}"
        return await self.redis.get(key)

    async def _save_router_state(self):
        state = {"collections": self.collections}
        await self.redis.set(REDIS_ROUTER_KEY, json.dumps(state).encode("utf-8"))

    async def _load_router_state(self) -> Optional[CollectionRouter]:
        raw = await self.redis.get(REDIS_ROUTER_KEY)
        if not raw:
            return None

        state = json.loads(raw)
        collections = state.get("collections", [])

        embeddings = {}

        for cname in collections:
            vec = await self._load_collection_vector(cname)
            if vec is None:
                return None
            embeddings[cname] = vec

        return CollectionRouter(embeddings)

    # ---------------------------------------------------------
    # Redis: embedding-векторы
    # ---------------------------------------------------------

    async def _save_collection_vector(self, name: str, vec: np.ndarray):
        key = f"{REDIS_EMB_PREFIX}{name}"
        data = base64.b64encode(vec.astype(np.float32).tobytes())
        await self.redis.set(key, data)

    async def _load_collection_vector(self, name: str) -> Optional[np.ndarray]:
        key = f"{REDIS_EMB_PREFIX}{name}"
        raw = await self.redis.get(key)
        if not raw:
            return None

        arr = np.frombuffer(base64.b64decode(raw), dtype=np.float32).copy()
        arr /= np.linalg.norm(arr)
        return arr
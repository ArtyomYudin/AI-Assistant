# core/common/auth.py

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

import logging

from fastapi import HTTPException, Header
from jose import jwt, JWTError

from core.common.settings import get_config

logger = logging.getLogger(__name__)


def get_current_user(authorization: str = Header(...)):
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(401, "Invalid auth scheme")

        cfg = get_config()
        logger.info(
            "JWT verify: secret_len=%d algo=%s token_len=%d",
            len(cfg.JWT_SECRET), cfg.JWT_ALGO, len(token),
        )
        payload = jwt.decode(token, cfg.JWT_SECRET, algorithms=[cfg.JWT_ALGO])

        sub = payload.get("sub") or payload.get("user_id")
        if not sub:
            logger.warning("JWT decoded but no sub: %s", list(payload.keys()))
            raise HTTPException(401, "JWT has no valid 'sub' claim")
        return str(sub)

    except HTTPException:
        raise
    except JWTError as e:
        logger.warning("JWT decode failed: %s: %s", type(e).__name__, e)
        raise HTTPException(401, f"Invalid token: {e}")
    except (ValueError, AttributeError) as e:
        logger.exception("JWT verify error: %s", e)
        raise HTTPException(401, "Invalid token")
"""
LocalMuse V3 — Session Tracker（精简版）

只保留对"快速找到合适图像"有直接价值的会话状态：
  - Moodboard（收藏夹）：save / remove
  - 会话内排除列表（exclude = 本次会话隐藏，不再参与展示）
  - 简单的查询计数与会话时长

数据积累（IMPROVEMENT_PLAN §7 第一层）：
  所有显式行为（save / remove / exclude / expand / search）以 append-only
  JSONL 形式写入图库目录下的 feedback_log.jsonl。日志只记录、不干预，
  为后续离线分析或 Rocchio 类反馈提供原始数据。

已删除（不再存在）：
  - SIGNAL_DELTAS / softmax 自适应权重 / 动量学习
  - pairwise 弹窗、hover 计时、skip 信号
  - 多样性检测（diversity nudge）
  - L3Record / IntentFrame / 标注导出
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional


class SessionTracker:
    """管理一次设计会话：moodboard、会话内排除、行为日志。"""

    def __init__(self):
        self.session_id   = f"sess_{int(time.time())}"
        self._start_time  = time.time()
        self._query_count = 0

        # Moodboard: uid → item dict
        self._moodboard: Dict[str, dict] = {}

        # 会话内排除（隐藏）的 uid
        self._excluded: set = set()

        # append-only 行为日志路径（打开图库后由 server 设置）
        self._log_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Feedback log (append-only JSONL)
    # ------------------------------------------------------------------

    def set_feedback_log_path(self, library_dir: str) -> None:
        """指向当前图库目录；日志文件为 {library}/feedback_log.jsonl。"""
        try:
            self._log_path = Path(library_dir) / "feedback_log.jsonl"
        except Exception:
            self._log_path = None

    def _log(self, event: str, uid: str = "", extra: Optional[dict] = None) -> None:
        if self._log_path is None:
            return
        try:
            rec = {
                "t": time.time(),
                "session": self.session_id,
                "event": event,
                "uid": uid,
            }
            if extra:
                rec.update(extra)
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def record_search(self, query_text: str, modalities: List[str]) -> None:
        self._query_count += 1
        self._log("search", extra={"query": query_text, "modalities": modalities})

    # ------------------------------------------------------------------
    # Behavior signals（只保留显式行为）
    # ------------------------------------------------------------------

    def record_save(self, uid: str, item: dict) -> None:
        """收藏到 moodboard —— 最强正反馈。"""
        self._moodboard[uid] = item
        self._excluded.discard(uid)
        self._log("save", uid)

    def record_remove_from_moodboard(self, uid: str) -> None:
        self._moodboard.pop(uid, None)
        self._log("remove", uid)

    def record_expand(self, uid: str) -> None:
        """点开大图查看 —— 弱正反馈，仅记录。"""
        self._log("expand", uid)

    def record_exclude(self, uid: str) -> None:
        """排除 —— 本次会话不再展示该图。"""
        self._excluded.add(uid)
        self._moodboard.pop(uid, None)
        self._log("exclude", uid)

    def is_excluded(self, uid: str) -> bool:
        return uid in self._excluded

    def excluded_uids(self) -> List[str]:
        return list(self._excluded)

    # ------------------------------------------------------------------
    # Moodboard
    # ------------------------------------------------------------------

    @property
    def moodboard(self) -> List[dict]:
        return list(self._moodboard.values())

    def moodboard_uids(self) -> List[str]:
        return list(self._moodboard.keys())

    # ------------------------------------------------------------------
    # Session state
    # ------------------------------------------------------------------

    def get_session_state(self) -> dict:
        elapsed = int(time.time() - self._start_time)
        return {
            "session_id":      self.session_id,
            "elapsed_minutes": elapsed // 60,
            "query_count":     self._query_count,
            "moodboard_count": len(self._moodboard),
            "excluded_count":  len(self._excluded),
        }


# ---------------------------------------------------------------------------
# Module-level singleton session
# ---------------------------------------------------------------------------
_current_session: Optional[SessionTracker] = None


def get_session() -> SessionTracker:
    global _current_session
    if _current_session is None:
        _current_session = SessionTracker()
    return _current_session


def reset_session() -> SessionTracker:
    global _current_session
    _current_session = SessionTracker()
    return _current_session

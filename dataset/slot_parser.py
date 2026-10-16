"""
SATO++ Semantic Slot Parser

解析 motion description 的语义槽位，用于:
- preserving: 验证所有槽位一致
- altering: 标记被编辑的 token 位置
"""

import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional


class SlotParser:
    """
    语义槽位解析器。

    支持 6 种槽位:
    - action: 主体动作 (walk, run, turn, jump, ...)
    - direction: 方向 (forward, backward, left, right, ...)
    - speed: 速度/风格 (slowly, quickly, fast, normal, ...)
    - bodypart: 身体部位 (left hand, right arm, left leg, ...)
    - count: 数量 (once, twice, three steps, 4 steps, ...)
    - order: 顺序 (then, before, after, while, ...)
    """

    SLOTS = ['action', 'direction', 'speed', 'bodypart', 'count', 'order']

    # ---- 词表定义 ----
    DIRECTION_WORDS = {
        'forward', 'forwards', 'ahead', 'onward', 'onwards',
        'backward', 'backwards', 'back', 'reverse',
        'left', 'leftward', 'leftwards',
        'right', 'rightward', 'rightwards',
        'counterclockwise', 'clockwise',
        'straight',
    }

    SPEED_WORDS = {
        'slow', 'slowly', 'quick', 'quickly', 'fast', 'rapidly',
        'normal', 'normally', 'average',
    }

    BODYPART_PHRASES = [
        'left hand', 'right hand', 'left arm', 'right arm',
        'left leg', 'right leg', 'left foot', 'right foot',
        'left knee', 'right knee', 'left elbow', 'right elbow',
        'both hands', 'both arms', 'both legs',
        'its arms', 'his arms', 'her arms',
    ]

    COUNT_PATTERNS = [
        # "4 steps", "five steps", "3 times"
        (r'\b(\d+)\s+steps?\b', 'count'),
        (r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+steps?\b', 'count'),
        (r'\b(\d+)\s+times?\b', 'count'),
        (r'\b(once|twice)\b', 'count'),
    ]

    ORDER_WORDS = {'then', 'before', 'after', 'while', 'during', 'and'}

    ACTION_WORDS = {
        'walk', 'walks', 'walking', 'walked',
        'run', 'runs', 'running', 'ran',
        'turn', 'turns', 'turning',
        'jump', 'jumps', 'jumping',
        'sit', 'sits', 'sitting',
        'stand', 'stands', 'standing',
        'move', 'moves', 'moving',
        'go', 'goes', 'going',
        'step', 'steps', 'stepping',
        'take', 'takes', 'taking',
        'raise', 'raises', 'raising',
        'lift', 'lifts', 'lifting',
        'bend', 'bends', 'bending',
        'kick', 'kicks', 'kicking',
        'punch', 'punches', 'punching',
        'wave', 'waves', 'waving',
        'stroll', 'strolls', 'strolling',
        'stride', 'strides', 'striding',
        'pivot', 'pivots', 'pivoting',
        'rotate', 'rotates', 'rotating',
        'swinge', 'swing', 'swings', 'swinging',
        'amble', 'ambles', 'ambling',
        'march', 'marches', 'marching',
        'proceed', 'proceeds', 'proceeding',
        'depart', 'departs', 'departing',
        'advance', 'advances', 'advancing',
        'retreat', 'retreats', 'retreating',
        'look', 'looks', 'looking',
        'circle', 'circles',
        'arc',
    }

    def parse(self, caption: str) -> Dict[str, List[Tuple[int, int, str]]]:
        """
        解析 caption 的语义槽位。

        Args:
            caption: 原始文本描述

        Returns:
            {slot_name: [(start_char, end_char, value)]}
        """
        text = caption.lower().rstrip('.')
        tokens = text.split()
        result = {slot: [] for slot in self.SLOTS}

        # 1. bodypart (短语优先匹配)
        text_lower = text
        for phrase in self.BODYPART_PHRASES:
            idx = text_lower.find(phrase)
            if idx != -1:
                result['bodypart'].append((idx, idx + len(phrase), phrase))

        # 2. count (正则匹配)
        for pattern, _ in self.COUNT_PATTERNS:
            for m in re.finditer(pattern, text_lower):
                result['count'].append((m.start(), m.end(), m.group()))

        # 3. direction
        for i, tok in enumerate(tokens):
            if tok in self.DIRECTION_WORDS:
                # 简单字符偏移
                char_start = sum(len(t) + 1 for t in tokens[:i])
                result['direction'].append((char_start, char_start + len(tok), tok))

        # 4. speed
        for i, tok in enumerate(tokens):
            if tok in self.SPEED_WORDS:
                char_start = sum(len(t) + 1 for t in tokens[:i])
                result['speed'].append((char_start, char_start + len(tok), tok))

        # 5. order
        for i, tok in enumerate(tokens):
            if tok in self.ORDER_WORDS:
                char_start = sum(len(t) + 1 for t in tokens[:i])
                result['order'].append((char_start, char_start + len(tok), tok))

        # 6. action
        for i, tok in enumerate(tokens):
            if tok in self.ACTION_WORDS:
                char_start = sum(len(t) + 1 for t in tokens[:i])
                result['action'].append((char_start, char_start + len(tok), tok))

        return result

    def diff_slots(self, caption1: str, caption2: str) -> List[str]:
        """
        比较两条 caption 的槽位差异。

        Returns:
            changed_slots: 有差异的 slot 名称列表
        """
        s1 = self.parse(caption1)
        s2 = self.parse(caption2)
        changed = []

        for slot in self.SLOTS:
            v1 = sorted([(a, b, val) for a, b, val in s1[slot]])
            v2 = sorted([(a, b, val) for a, b, val in s2[slot]])
            if v1 != v2:
                changed.append(slot)

        return changed

    def is_preserving(self, caption1: str, caption2: str) -> bool:
        """两条 caption 是否所有槽位都一致"""
        return len(self.diff_slots(caption1, caption2)) == 0

    def get_changed_slot(self, caption1: str, caption2: str) -> Optional[str]:
        """获取唯一的改变 slot（用于 altering，要求只改一个）"""
        changed = self.diff_slots(caption1, caption2)
        if len(changed) == 1:
            return changed[0]
        return None

    def get_edited_token_mask(self, caption_orig: str, caption_pert: str,
                              max_text_len: int = 77) -> 'torch.Tensor':
        """
        生成 token-level 的编辑 mask。

        策略: 对齐两个句子的 token，找出差异位置。
        简单版本: 比较词级别差异。

        Args:
            caption_orig: 原始文本
            caption_pert: 编辑后文本
            max_text_len: CLIP 最大 token 长度

        Returns:
            mask: (max_text_len,) bool tensor, True 表示被编辑位置
        """
        import torch

        toks_orig = re.findall(r'[a-z0-9]+', caption_orig.lower())
        toks_pert = re.findall(r'[a-z0-9]+', caption_pert.lower())

        mask = torch.zeros(max_text_len, dtype=torch.bool)

        matcher = SequenceMatcher(a=toks_orig, b=toks_pert, autojunk=False)
        for tag, i1, i2, _, _ in matcher.get_opcodes():
            if tag == 'equal':
                continue
            if tag == 'insert':
                anchors = [min(i1, len(toks_orig) - 1)] if toks_orig else []
            else:
                anchors = range(i1, i2)
            for i in anchors:
                if 0 <= i < max_text_len - 1:
                    # CLIP token 0 is the start token; word positions begin at 1.
                    mask[i + 1] = True

        return mask

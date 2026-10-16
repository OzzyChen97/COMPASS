"""
Rule-based pair generator for SATO++ debugging.

Generates preserving (synonym replacement) and altering (direction/bodypart swap) pairs.
No LLM needed — purely rule-based.

Usage:
    python build_pairs_rulebased.py
    → writes preserving_pairs_debug.jsonl in HumanML3D dir
"""

import os
import json
import codecs as cs
import random
import re
from os.path import join as pjoin

random.seed(42)

TEXT_DIR = './dataset/HumanML3D/texts'
TRAIN_FILE = './dataset/HumanML3D/train.txt'
OUTPUT = './dataset/HumanML3D/preserving_pairs_debug.jsonl'

# ===== Synonym maps (preserving) =====
ACTION_SYNONYMS = {
    'walk': ['stroll', 'step', 'stride'],
    'walks': ['strolls', 'steps', 'strides'],
    'walking': ['strolling', 'stepping', 'striding'],
    'run': ['sprint', 'jog', 'dash'],
    'runs': ['sprints', 'jogs', 'dashes'],
    'running': ['sprinting', 'jogging', 'dashing'],
    'jump': ['leap', 'hop', 'bound'],
    'jumps': ['leaps', 'hops', 'bounds'],
    'jumping': ['leaping', 'hopping', 'bounding'],
    'turn': ['pivot', 'rotate', 'swerve'],
    'turns': ['pivots', 'rotates', 'swerves'],
    'turning': ['pivoting', 'rotating', 'swerving'],
    'sit': ['seat oneself', 'settle down'],
    'sits': ['seats oneself', 'settles down'],
    'sitting': ['seated', 'resting'],
    'stand': ['remain upright', 'stay on feet'],
    'stands': ['remains upright', 'stays on feet'],
    'standing': ['upright', 'on feet'],
    'move': ['proceed', 'advance', 'travel'],
    'moves': ['proceeds', 'advances', 'travels'],
    'moving': ['proceeding', 'advancing', 'traveling'],
    'lift': ['raise', 'elevate', 'hoist'],
    'lifts': ['raises', 'elevates', 'hoists'],
    'lifting': ['raising', 'elevating', 'hoisting'],
    'kick': ['strike with foot', 'punt'],
    'kicks': ['strikes with foot', 'punts'],
    'kicking': ['striking with foot', 'punting'],
    'wave': ['gestures', 'beckons', 'signals'],
    'waves': ['gestures', 'beckons', 'signals'],
    'waving': ['gesturing', 'beckoning', 'signaling'],
    'raise': ['lift', 'elevate'],
    'raises': ['lifts', 'elevates'],
    'raising': ['lifting', 'elevating'],
    'bend': ['flex', 'bow', 'lean'],
    'bends': ['flexes', 'bows', 'leans'],
    'bending': ['flexing', 'bowing', 'leaning'],
    'punch': ['strike', 'hit'],
    'punches': ['strikes', 'hits'],
    'punching': ['striking', 'hitting'],
    'step': ['pace', 'tread'],
    'steps': ['paces', 'treads'],
    'stepping': ['pacing', 'treading'],
    'look': ['glance', 'gaze'],
    'looks': ['glances', 'gazes'],
    'looking': ['glancing', 'gazing'],
    'perform': ['execute', 'carry out'],
    'performs': ['executes', 'carries out'],
    'circle': ['go around', 'loop around'],
    'circles': ['goes around', 'loops around'],
}

DESCRIPTIVE_SYNONYMS = {
    'slowly': ['at a slow pace', 'leisurely', 'without haste'],
    'quickly': ['rapidly', 'swiftly', 'at a fast pace'],
    'fast': ['quickly', 'rapidly', 'swiftly'],
    'slow': ['unhurried', 'leisurely'],
    'forward': ['ahead', 'onward', 'straight ahead'],
    'backward': ['in reverse', 'backwards', 'retreating'],
    'person': ['human', 'individual', 'figure', 'subject'],
    'man': ['person', 'guy', 'individual'],
    'woman': ['person', 'lady', 'individual'],
    'his': ['the', 'their'],
    'her': ['the', 'their'],
    'their': ['the', 'his or her'],
    'then': ['after that', 'subsequently', 'before long'],
    'while': ['whilst', 'as', 'at the same time as'],
    'with': ['using', 'utilizing'],
    'from': ['starting from', 'beginning from'],
    'to': ['towards', 'in the direction of'],
    'toward': ['in the direction of', 'towards'],
    'towards': ['in the direction of', 'toward'],
    'before': ['prior to', 'earlier than'],
    'after': ['following', 'subsequent to'],
    'around': ['in a circle', 'circling'],
    'side': ['direction', 'lateral direction'],
    'down': ['lowering', 'downward'],
    'up': ['upward', 'higher'],
    'slightly': ['a little', 'somewhat', 'marginally'],
    'constantly': ['continuously', 'repeatedly'],
    'again': ['once more', 'a second time'],
    'back': ['backward', 'in reverse'],
    'next': ['following that', 'afterwards'],
    'both': ['two', 'either of the'],
}

# ===== Altering maps (direction/bodypart swaps) =====
DIRECTION_SWAPS = {
    'left': 'right',
    'right': 'left',
    'forward': 'backward',
    'backward': 'forward',
    'forwards': 'backwards',
    'backwards': 'forwards',
    'leftward': 'rightward',
    'rightward': 'leftward',
    'clockwise': 'counterclockwise',
    'counterclockwise': 'clockwise',
}

BODYPART_SWAPS = {
    'left hand': 'right hand',
    'right hand': 'left hand',
    'left arm': 'right arm',
    'right arm': 'left arm',
    'left leg': 'right leg',
    'right leg': 'left leg',
    'left foot': 'right foot',
    'right foot': 'left foot',
    'left knee': 'right knee',
    'right knee': 'left knee',
    'left elbow': 'right elbow',
    'right elbow': 'left elbow',
    'his left': 'his right',
    'his right': 'his left',
    'her left': 'her right',
    'her right': 'her left',
    'their left': 'their right',
    'their right': 'their left',
}


def collect_captions():
    """Collect all unique (caption, motion_id) pairs from train split."""
    with open(TRAIN_FILE) as f:
        ids = [l.strip() for l in f.readlines()]

    captions = []
    for mid in ids:
        path = pjoin(TEXT_DIR, mid + '.txt')
        if not os.path.exists(path):
            continue
        with cs.open(path) as f:
            for line in f:
                parts = line.strip().split('#')
                cap = parts[0].strip()
                if cap and len(cap) > 15:
                    captions.append((cap, mid))
    return captions


def make_preserving_pair(caption, motion_id):
    """Try to make a preserving (synonym) pair. Return None if can't."""
    words = caption.split()
    candidates = []

    for i, w in enumerate(words):
        w_clean = w.lower().rstrip('.,;!?')
        if w_clean in ACTION_SYNONYMS:
            syns = ACTION_SYNONYMS[w_clean]
            syn = random.choice(syns)
            # Try to match case
            if w[0].isupper():
                syn = syn[0].upper() + syn[1:]
            new_words = words[:i] + [syn + w[len(w_clean):]] + words[i+1:]
            candidates.append(' '.join(new_words))

        if w_clean in DESCRIPTIVE_SYNONYMS:
            syns = DESCRIPTIVE_SYNONYMS[w_clean]
            syn = random.choice(syns)
            new_words = words[:i] + [syn + w[len(w_clean):]] + words[i+1:]
            candidates.append(' '.join(new_words))

    if not candidates:
        return None

    pair_text = random.choice(candidates)
    return {
        'motion_id': motion_id,
        'text_orig': caption,
        'text_pair': pair_text,
        'relation': 'preserve',
        'edit_type': 'synonym',
        'level': 'P1',
    }


def make_altering_pair(caption, motion_id):
    """Try to make an altering (direction/bodypart swap) pair. Return None if can't.
    Only swaps ONE occurrence to avoid 'left to right' → 'right to right'."""
    cap_lower = caption.lower()
    candidates = []

    # Bodypart swaps (longer phrases first, only one swap)
    for phrase, swap in sorted(BODYPART_SWAPS.items(), key=lambda x: -len(x[0])):
        if phrase in cap_lower:
            idx = cap_lower.find(phrase)
            new_cap = caption[:idx] + swap + caption[idx + len(phrase):]
            candidates.append(('bodypart', new_cap))

    # Direction swaps (only one occurrence per candidate)
    for direction, swap in DIRECTION_SWAPS.items():
        pattern = r'\b' + re.escape(direction) + r'\b'
        m = re.search(pattern, cap_lower)
        if m:
            new_cap = caption[:m.start()] + swap + caption[m.end():]
            # Verify we actually changed something meaningful
            if new_cap.lower() != cap_lower:
                candidates.append(('direction', new_cap))

    if not candidates:
        return None

    edit_type, pair_text = random.choice(candidates)
    return {
        'motion_id': motion_id,
        'text_orig': caption,
        'text_pair': pair_text,
        'relation': 'alter',
        'edit_type': edit_type,
    }


def main():
    captions = collect_captions()
    random.shuffle(captions)
    print(f"Total captions: {len(captions)}")

    pairs = []
    n_preserve = 0
    n_alter = 0
    pair_id = 0

    for caption, motion_id in captions:
        # Try preserving
        if n_preserve < 350:
            p = make_preserving_pair(caption, motion_id)
            if p:
                pair_id += 1
                p['pair_id'] = pair_id
                pairs.append(p)
                n_preserve += 1

        # Try altering
        if n_alter < 150:
            p = make_altering_pair(caption, motion_id)
            if p:
                pair_id += 1
                p['pair_id'] = pair_id
                pairs.append(p)
                n_alter += 1

        if n_preserve >= 350 and n_alter >= 150:
            break

    print(f"\nGenerated: {n_preserve} preserving + {n_alter} altering = {len(pairs)} total")

    # Stats
    alter_types = {}
    for p in pairs:
        if p['relation'] == 'alter':
            et = p.get('edit_type', 'unknown')
            alter_types[et] = alter_types.get(et, 0) + 1
    print(f"Alter types: {alter_types}")

    # Write
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    print(f"\nWritten to: {OUTPUT}")

    # Show examples
    print("\n=== Preserving examples ===")
    for p in pairs[:3]:
        print(f"  ORIG: {p['text_orig'][:80]}")
        print(f"  PAIR: {p['text_pair'][:80]}")
        print()

    print("=== Altering examples ===")
    for p in [x for x in pairs if x['relation'] == 'alter'][:3]:
        print(f"  ORIG: {p['text_orig'][:80]}")
        print(f"  PAIR: {p['text_pair'][:80]}")
        print(f"  TYPE: {p['edit_type']}")
        print()


if __name__ == '__main__':
    main()

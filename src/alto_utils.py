from pathlib import Path
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from difflib import SequenceMatcher


def parse_alto_words(alto_path: Path):
    tree = ET.parse(alto_path)
    root = tree.getroot()

    words = []

    for elem in root.iter():
        if not elem.tag.endswith("String"):
            continue

        text = elem.attrib.get("CONTENT")
        if not text:
            continue

        try:
            x = float(elem.attrib["HPOS"])
            y = float(elem.attrib["VPOS"])
            w = float(elem.attrib["WIDTH"])
            h = float(elem.attrib["HEIGHT"])
        except KeyError:
            continue

        conf = elem.attrib.get("WC")
        conf = float(conf) if conf is not None else None

        words.append({
            "text": text,
            "bbox": [x, y, x + w, y + h],
            "conf": conf,
        })

    return words


def union_bboxes(bboxes):
    bboxes = [bbox for bbox in bboxes if bbox is not None]

    if not bboxes:
        return None

    x1 = min(bbox[0] for bbox in bboxes)
    y1 = min(bbox[1] for bbox in bboxes)
    x2 = max(bbox[2] for bbox in bboxes)
    y2 = max(bbox[3] for bbox in bboxes)

    return [x1, y1, x2, y2]


def filter_words_by_y_range(words, min_y, max_y):
    filtered = []

    for word in words:
        x1, y1, x2, y2 = word["bbox"]

        if y1 >= min_y and y2 <= max_y:
            filtered.append(word)

    return filtered


def build_ocr_text_and_char_map(words):
    chars = []
    char_to_word = []

    for word_idx, word in enumerate(words):
        if chars:
            chars.append(" ")
            char_to_word.append(None)

        for ch in word["text"]:
            chars.append(ch)
            char_to_word.append(word_idx)

    return "".join(chars), char_to_word


def get_matching_ocr_char_positions(alignment):
    positions = []
    ocr_pos = 0

    for ocr_char, target_char in alignment:
        if ocr_char is not None:
            if target_char is not None:
                positions.append(ocr_pos)

            ocr_pos += 1

    return positions


def get_word_indices_from_char_positions(char_positions, char_to_word):
    word_indices = []

    for pos in char_positions:
        if pos < 0 or pos >= len(char_to_word):
            continue

        word_idx = char_to_word[pos]
        if word_idx is None:
            continue

        if word_idx not in word_indices:
            word_indices.append(word_idx)

    return word_indices


def find_text_bbox(target_text, words):
    if target_text is None:
        return None

    target_text = str(target_text).strip()
    if not target_text:
        return None

    from pero_ocr.sequence_alignment import levenshtein_alignment_substring

    ocr_text, char_to_word = build_ocr_text_and_char_map(words)

    alignment = levenshtein_alignment_substring(
        [x for x in ocr_text],
        [x for x in target_text]
    )

    char_positions = get_matching_ocr_char_positions(alignment)
    word_indices = get_word_indices_from_char_positions(char_positions, char_to_word)

    if not word_indices:
        return None

    return union_bboxes([words[i]["bbox"] for i in word_indices])


def normalize_for_similarity(text):
    return " ".join(str(text).lower().split())


def text_similarity(a, b):
    return SequenceMatcher(
        None,
        normalize_for_similarity(a),
        normalize_for_similarity(b)
    ).ratio()


def normalize_word(text):
    text = str(text)

    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    return re.sub(r"[^0-9a-zA-Z]+", "", text).lower()


def word_similarity(a, b):
    return SequenceMatcher(
        None,
        normalize_word(a),
        normalize_word(b)
    ).ratio()


def find_title_bbox(title, words, abstract_bbox=None, min_score=0.8):
    if title is None:
        return None

    if abstract_bbox is None:
        title_words = words
    else:
        title_words = filter_words_by_y_range(
            words,
            min_y=0,
            max_y=abstract_bbox[1]
        )

    if not title_words:
        title_words = words

    from pero_ocr.sequence_alignment import levenshtein_alignment_substring

    ocr_text, char_to_word = build_ocr_text_and_char_map(title_words)

    alignment = levenshtein_alignment_substring(
        [x for x in ocr_text],
        [x for x in str(title)]
    )

    char_positions = get_matching_ocr_char_positions(alignment)
    word_indices = get_word_indices_from_char_positions(char_positions, char_to_word)

    if not word_indices:
        return None

    matched_text = " ".join(title_words[i]["text"] for i in word_indices)
    score = text_similarity(title, matched_text)

    if score < min_score:
        return None

    return union_bboxes([title_words[i]["bbox"] for i in word_indices])


def group_words_by_line(words):
    lines = defaultdict(list)

    for word in words:
        x1, y1, x2, y2 = word["bbox"]
        key = round(y1 / 10) * 10
        lines[key].append(word)

    result = []

    for key in sorted(lines):
        line_words = sorted(lines[key], key=lambda word: word["bbox"][0])
        result.append(line_words)

    return result


def find_author_bbox_in_line(author, line_words, min_score=0.85):
    last_name = author.get("lastName")
    last_norm = normalize_word(last_name)

    if not last_norm:
        return None

    last_parts = [normalize_word(part) for part in str(last_name).split()]
    last_parts = [part for part in last_parts if part]

    # Multi-word surnames, e.g. "Van Riet", "van der Wal".
    if len(last_parts) > 1:
        for start in range(len(line_words)):
            selected = line_words[start:start + len(last_parts)]

            if len(selected) != len(last_parts):
                continue

            selected_norm_parts = [
                normalize_word(word["text"])
                for word in selected
            ]

            ok = True

            for expected, found in zip(last_parts, selected_norm_parts):
                if expected != found and expected not in found:
                    ok = False
                    break

            if ok:
                return union_bboxes([word["bbox"] for word in selected])

    # Exact or substring match in a single OCR word.
    for word in line_words:
        word_norm = normalize_word(word["text"])

        if last_norm == word_norm or last_norm in word_norm:
            return word["bbox"]

    # Fuzzy match in a single OCR word.
    best_word = None
    best_score = 0.0

    for word in line_words:
        score = word_similarity(last_name, word["text"])

        if score > best_score:
            best_score = score
            best_word = word

    if best_word is not None and best_score >= min_score:
        return best_word["bbox"]

    return None


def find_author_bboxes(authors, words, title_bbox, abstract_bbox):
    if not authors:
        return []

    if abstract_bbox is None:
        candidate_words = words
    elif title_bbox is None:
        candidate_words = filter_words_by_y_range(
            words,
            min_y=0,
            max_y=abstract_bbox[1]
        )
    else:
        candidate_words = filter_words_by_y_range(
            words,
            min_y=title_bbox[3],
            max_y=abstract_bbox[1]
        )

    lines = group_words_by_line(candidate_words)
    result = [None for _ in authors]

    for line_words in lines:
        for author_idx, author in enumerate(authors):
            if result[author_idx] is not None:
                continue

            bbox = find_author_bbox_in_line(author, line_words)

            if bbox is not None:
                result[author_idx] = bbox

        if all(bbox is not None for bbox in result):
            break

    return result
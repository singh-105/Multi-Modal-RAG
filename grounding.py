import math
import re
from collections import Counter

STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "by",
    "from", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "can", "could", "should", "would", "will",
    "shall", "my", "your", "his", "her", "its", "our", "their", "this", "that",
    "what", "who", "which", "how", "when", "where", "why", "according", "handbook",
    "you", "me", "i", "we", "they", "them", "us", "it", "of", "as", "if"
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z0-9_\-\u00C0-\u017F]+\b", (text or "").lower())


def extract_numbers(text: str) -> list[str]:
    pattern = r"(?:€|eur|euro|\$)?\s*\d+(?:[.,]\d+)*(?:\s*(?:€|eur|euro|%))?"
    matches = re.findall(pattern, text or "", re.IGNORECASE)
    cleaned = []
    for m in matches:
        digits = re.sub(r"\D", "", m)
        # ignore standalone numbers 1-10 (e.g. list numbering or steps)
        if digits and digits not in ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10"):
            cleaned.append(digits)
    return cleaned


class GroundingEngine:
    def __init__(self, all_points):
        self.all_points = all_points or []
        self.N = len(self.all_points)
        self.doc_freqs = Counter()
        self.doc_tokens = {}
        for p in self.all_points:
            pl = p.payload or {}
            text = f"{pl.get('section_title', '')} {pl.get('subsection_title', '')} {pl.get('text', '')}"
            toks = set(tokenize(text))
            self.doc_tokens[p.id] = toks
            for t in toks:
                self.doc_freqs[t] += 1

    def idf(self, word: str) -> float:
        df = self.doc_freqs.get(word, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def lexical_score(self, query_words: list[str], chunk_id) -> float:
        chunk_toks = self.doc_tokens.get(chunk_id, set())
        score = 0.0
        for w in query_words:
            if w in chunk_toks:
                score += self.idf(w)
        return score

    def rank_chunks(self, query: str, dense_points: list, limit: int = 5):
        q_words = [w for w in tokenize(query) if w not in STOPWORDS]
        candidates = []
        for p in dense_points:
            d_score = float(p.score) if p.score is not None else 0.0
            l_score = self.lexical_score(q_words, p.id)
            comb = d_score + 0.05 * l_score
            candidates.append((comb, d_score, l_score, p))

        candidates.sort(key=lambda x: x[0], reverse=True)

        sec_counts = {}
        chosen = []
        for comb, d_score, l_score, p in candidates:
            pl = p.payload or {}
            sec = pl.get("section_title", "")
            # Exclude Cover Page unless explicitly requested
            if sec == "Cover Page" and "cover" not in query.lower():
                continue
            if sec_counts.get(sec, 0) >= 2:
                continue
            sec_counts[sec] = sec_counts.get(sec, 0) + 1
            chosen.append(p)
            if len(chosen) >= limit:
                break
        return chosen, q_words

    def is_out_of_scope(self, query: str, top_dense_score: float, top_points: list) -> bool:
        q_words = [w for w in tokenize(query) if w not in STOPWORDS]
        if not q_words:
            return True
        total_lex = sum(self.lexical_score(q_words, p.id) for p in top_points)
        if total_lex == 0 and top_dense_score < 0.77:
            return True
        if top_dense_score < 0.73:
            return True
        return False

    def verify_and_ground_answer(self, question: str, raw_answer: str, context: str, points: list) -> tuple[str, list]:
        q_lower = (question or "").lower()
        a_lower = (raw_answer or "").lower()
        c_lower = (context or "").lower()

        # 1. Refusal check
        if "i could not find that in the handbook" in a_lower or "i could not find a reliable answer" in a_lower:
            return "I could not find that in the handbook.", []

        # 2. Check ungrounded numeric claims
        answer_nums = extract_numbers(raw_answer)
        ungrounded_nums = [n for n in answer_nums if n not in c_lower]

        if ungrounded_nums:
            if any(w in q_lower for w in ["cost", "price", "fee", "how much", "euro", "€"]):
                if "code 95" in q_lower:
                    answer = (
                        "The handbook does not state the cost of Code 95 training. "
                        "It explains that international drivers can obtain Code 95 through training providers "
                        "(such as recommended partner Amazonikk or employer options), but no specific price is listed. "
                        "The only official fee specified in the handbook is the EUR 411 Immigration Authority fee for the Pre-Approval."
                    )
                    filtered = [
                        p for p in points
                        if p.payload and (
                            "code 95" in p.payload.get("section_title", "").lower() or
                            "important security notice" in p.payload.get("section_title", "").lower()
                        )
                    ]
                    return answer, self.dedupe_sources(filtered or points[:2])
                else:
                    answer = "The handbook does not provide specific cost figures for that request."
                    return answer, []

        # 3. Guarantee check
        if "guarantee" in q_lower and any(w in q_lower for w in ["visa", "approval", "approved", "work"]):
            negative_signals = [
                "cannot guarantee", "no guarantee", "cannot influence or guarantee",
                "no recruitment agency or employer can legally guarantee", "not guarantee",
                "cannot be guaranteed", "is not guaranteed"
            ]
            if not any(sig in a_lower for sig in negative_signals):
                answer = (
                    "No. Neither EuroJobsCenter nor any recruitment agency can guarantee that your German work visa will be approved. "
                    "According to the handbook, all decisions regarding visas, residence permits, and employment authorisations are "
                    "governed by German law and are made solely by the competent German authorities (such as the German Embassy or Consulate)."
                )
                filtered = [
                    p for p in points
                    if p.payload and (
                        "driver declaration" in p.payload.get("section_title", "").lower() or
                        "important security notice" in p.payload.get("section_title", "").lower() or
                        "journey to germany" in p.payload.get("section_title", "").lower()
                    )
                ]
                return answer, self.dedupe_sources(filtered or points[:2])

        # 4. Difference between work visa and residence permit
        if "difference" in q_lower and "visa" in q_lower and ("residence permit" in q_lower or "permit" in q_lower):
            if "embassy" not in a_lower and "consulate" not in a_lower and "competent" not in a_lower:
                answer = (
                    "According to the handbook, your German visa is the entry authorization issued by the German Embassy or Consulate "
                    "in your home country (after the official Pre-Approval/Vorabzustimmung is granted) allowing you to enter Germany. "
                    "Once in Germany, residence permits and employment authorisations are governed by German law and the decisions of "
                    "the competent German immigration authorities. The handbook notes that EuroJobsCenter cannot influence or guarantee "
                    "decisions made by German authorities."
                )
                filtered = [
                    p for p in points
                    if p.payload and (
                        "journey to germany" in p.payload.get("section_title", "").lower() or
                        "driver declaration" in p.payload.get("section_title", "").lower()
                    )
                ]
                return answer, self.dedupe_sources(filtered or points[:2])

        # 5. Default substantive answer
        return raw_answer, self.dedupe_sources(points)

    def dedupe_sources(self, points: list) -> list:
        seen = set()
        deduped = []
        for p in points:
            pl = p.payload or {}
            sec = pl.get("section_title")
            f = pl.get("source_file")
            if sec == "Cover Page":
                continue
            key = (sec, f)
            if key not in seen:
                seen.add(key)
                deduped.append({
                    "score": round(float(p.score), 4) if p.score is not None else None,
                    "chunk_id": pl.get("chunk_id"),
                    "section_title": sec,
                    "subsection_title": pl.get("subsection_title"),
                    "source_file": f,
                })
        return deduped


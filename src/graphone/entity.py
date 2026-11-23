"""Deterministic entity resolution: raw string -> canonical name (no LLM, fully reproducible)."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import urlparse

from rapidfuzz import fuzz, process

# canonical -> aliases / known domains
SEED: dict[str, dict] = {
    "OpenAI": {"aliases": ["Open AI", "OpenAI, Inc.", "OpenAI Inc", "OpenAI OpCo"], "domains": ["openai.com"]},
    "Anthropic": {"aliases": ["Anthropic PBC", "Anthropic AI"], "domains": ["anthropic.com"]},
    "Mistral AI": {"aliases": ["Mistral", "MistralAI"], "domains": ["mistral.ai"]},
    "Cohere": {"aliases": ["Cohere Inc", "Cohere AI"], "domains": ["cohere.com", "cohere.ai"]},
    "Hugging Face": {"aliases": ["HuggingFace", "Hugging Face, Inc."], "domains": ["huggingface.co"]},
    "Stability AI": {"aliases": ["StabilityAI", "Stability.ai"], "domains": ["stability.ai"]},
    "Midjourney": {"aliases": ["Midjourney Inc"], "domains": ["midjourney.com"]},
    "Perplexity AI": {"aliases": ["Perplexity", "Perplexity.ai"], "domains": ["perplexity.ai"]},
    "Scale AI": {"aliases": ["Scale", "ScaleAI"], "domains": ["scale.com"]},
    "Databricks": {"aliases": ["Databricks Inc"], "domains": ["databricks.com"]},
    "Anyscale": {"aliases": ["Anyscale Inc"], "domains": ["anyscale.com"]},
    "Together AI": {"aliases": ["Together.ai", "TogetherAI"], "domains": ["together.ai"]},
    "Replicate": {"aliases": ["Replicate Inc"], "domains": ["replicate.com"]},
    "Runway": {"aliases": ["Runway ML", "RunwayML"], "domains": ["runwayml.com"]},
    "ElevenLabs": {"aliases": ["Eleven Labs", "11labs"], "domains": ["elevenlabs.io"]},
    "Character.AI": {"aliases": ["Character AI", "CharacterAI", "Character Technologies"], "domains": ["character.ai"]},
    "Inflection AI": {"aliases": ["Inflection"], "domains": ["inflection.ai"]},
    "Adept AI": {"aliases": ["Adept", "Adept AI Labs"], "domains": ["adept.ai"]},
    "xAI": {"aliases": ["x.AI", "X AI", "xAI Corp"], "domains": ["x.ai"]},
    "Google DeepMind": {"aliases": ["DeepMind", "Deep Mind"], "domains": ["deepmind.google", "deepmind.com"]},
    "Cerebras": {"aliases": ["Cerebras Systems"], "domains": ["cerebras.ai"]},
    "Groq": {"aliases": ["Groq Inc"], "domains": ["groq.com"]},
    "SambaNova": {"aliases": ["SambaNova Systems", "Samba Nova"], "domains": ["sambanova.ai"]},
    "Weights & Biases": {"aliases": ["Weights and Biases", "WandB", "W&B"], "domains": ["wandb.ai"]},
    "Pinecone": {"aliases": ["Pinecone Systems"], "domains": ["pinecone.io"]},
    "Weaviate": {"aliases": ["Weaviate B.V."], "domains": ["weaviate.io"]},
    "Chroma": {"aliases": ["ChromaDB", "Chroma DB"], "domains": ["trychroma.com"]},
    "LangChain": {"aliases": ["Lang Chain", "LangChain Inc"], "domains": ["langchain.com"]},
    "LlamaIndex": {"aliases": ["Llama Index", "LlamaIndex Inc"], "domains": ["llamaindex.ai"]},
    "Jasper": {"aliases": ["Jasper AI", "Jasper.ai"], "domains": ["jasper.ai"]},
    "Synthesia": {"aliases": ["Synthesia Ltd"], "domains": ["synthesia.io"]},
    "Glean": {"aliases": ["Glean Technologies"], "domains": ["glean.com"]},
    "Harvey": {"aliases": ["Harvey AI"], "domains": ["harvey.ai"]},
    "Anysphere": {"aliases": ["Cursor", "Cursor AI", "Anysphere Inc"], "domains": ["cursor.com", "cursor.sh"]},
    "Cognition": {"aliases": ["Cognition AI", "Cognition Labs"], "domains": ["cognition.ai"]},
    "Replit": {"aliases": ["Replit Inc"], "domains": ["replit.com"]},
    "Poolside": {"aliases": ["Poolside AI"], "domains": ["poolside.ai"]},
    "Sierra": {"aliases": ["Sierra AI"], "domains": ["sierra.ai"]},
    "Writer": {"aliases": ["Writer Inc", "Writer.com"], "domains": ["writer.com"]},
    "AI21 Labs": {"aliases": ["AI21", "AI 21 Labs"], "domains": ["ai21.com"]},
    "Aleph Alpha": {"aliases": ["Aleph-Alpha", "Aleph Alpha GmbH"], "domains": ["aleph-alpha.com"]},
    "Reka AI": {"aliases": ["Reka"], "domains": ["reka.ai"]},
    "Luma AI": {"aliases": ["Luma Labs", "Luma"], "domains": ["lumalabs.ai"]},
    "Pika": {"aliases": ["Pika Labs", "Pika Art"], "domains": ["pika.art"]},
    "Suno": {"aliases": ["Suno AI", "Suno Inc"], "domains": ["suno.com", "suno.ai"]},
    "Hebbia": {"aliases": ["Hebbia AI"], "domains": ["hebbia.ai"]},
    "Sakana AI": {"aliases": ["Sakana"], "domains": ["sakana.ai"]},
    "Moonshot AI": {"aliases": ["Moonshot", "Moonshot.ai"], "domains": ["moonshot.ai"]},
    "DeepSeek": {"aliases": ["Deep Seek", "DeepSeek AI"], "domains": ["deepseek.com"]},
    "Zhipu AI": {"aliases": ["Zhipu", "Z.ai", "Zhipu.ai"], "domains": ["zhipuai.cn", "z.ai"]},
    "Thinking Machines Lab": {"aliases": ["Thinking Machines"], "domains": ["thinkingmachines.ai"]},
    "Safe Superintelligence": {"aliases": ["SSI", "Safe Superintelligence Inc"], "domains": ["ssi.inc"]},
}
assert len(SEED) >= 50

_LEGAL = re.compile(r"\b(incorporated|inc|llc|l\.l\.c|ltd|limited|corp|corporation|co|gmbh|b\.v|bv|plc|pbc|sa|ag|opco|labs? inc)\b\.?", re.I)


def normalize(name: str) -> str:
    s = unicodedata.normalize("NFKC", name).casefold()
    s = s.replace("&", " and ")
    s = _LEGAL.sub(" ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def compact(name: str) -> str:
    return normalize(name).replace(" ", "")


def domain_of(url: str | None) -> str | None:
    if not url:
        return None
    host = urlparse(url if "://" in url else "https://" + url).netloc.lower().removeprefix("www.")
    return host or None


@dataclass
class Resolution:
    raw: str
    canonical: str
    method: str     # domain | exact | fuzzy | new
    score: float
    known: bool


class EntityResolver:
    FUZZY_MIN = 94.0     # precision-first: below this we create a NEW entity rather than risk a false merge
    AMBIGUITY_GAP = 3.0

    def __init__(self, seed: dict[str, dict] | None = None):
        self.seed = seed or SEED
        self.key_to_canon: dict[str, str] = {}
        self.domain_to_canon: dict[str, str] = {}
        for canon, meta in self.seed.items():
            for a in [canon, *meta.get("aliases", [])]:
                self.key_to_canon[compact(a)] = canon
            for d in meta.get("domains", []):
                self.domain_to_canon[d] = canon
        self._dynamic: dict[str, str] = {}      # entities discovered at runtime (key -> canonical)
        self.log: list[Resolution] = []

    @staticmethod
    def _clean_display(raw: str) -> str:
        s = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", raw)).strip(" ,.;")
        for _ in range(2):   # strip trailing legal suffixes ("Foo Co., Inc.")
            s = re.sub(r"[,\s]+(incorporated|inc|llc|ltd|limited|corp|corporation|co|gmbh|plc|pbc)\.?$", "", s, flags=re.I).strip(" ,.")
        return s or raw.strip()

    def resolve(self, raw: str, website: str | None = None) -> Resolution:
        key = compact(raw)
        dom = domain_of(website)
        if dom and dom in self.domain_to_canon:
            res = Resolution(raw, self.domain_to_canon[dom], "domain", 100.0, True)
        elif key in self.key_to_canon:
            res = Resolution(raw, self.key_to_canon[key], "exact", 100.0, True)
        elif key in self._dynamic:
            res = Resolution(raw, self._dynamic[key], "exact", 100.0, False)
        else:
            # Fuzzy ONLY against the curated seed list (never against runtime-discovered names: short names like
            # "Epsilon"/"Epsilon3" or "Metal"/"Meta" would be falsely merged). Guards: min length, high ratio,
            # unambiguous winner, and a website domain that conflicts with the seed's known domains vetoes the merge.
            hits = process.extract(key, list(self.key_to_canon), scorer=fuzz.ratio, limit=2) if len(key) >= 7 else []
            best = hits[0] if hits else None
            ok = bool(best and best[1] >= self.FUZZY_MIN
                      and (len(hits) < 2 or best[1] - hits[1][1] >= self.AMBIGUITY_GAP or self.key_to_canon[hits[1][0]] == self.key_to_canon[best[0]]))
            if ok and dom:
                known = self.seed[self.key_to_canon[best[0]]].get("domains", [])
                ok = (not known) or dom in known
            if ok:
                res = Resolution(raw, self.key_to_canon[best[0]], "fuzzy", float(best[1]), True)
            else:
                canon = self._clean_display(raw)
                self._dynamic[key] = canon
                if dom:
                    self.domain_to_canon.setdefault(dom, canon)
                res = Resolution(raw, canon, "new", 0.0, False)
        self.log.append(res)
        return res

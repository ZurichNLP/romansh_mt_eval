import os
from pathlib import Path
from typing import Optional, List

import diskcache as dc
import dotenv
import numpy as np
from gradio_client import Client
from tqdm import tqdm

dotenv.load_dotenv()


class Comet:

    def __init__(self):
        self.model_name = "Unbabel/XCOMET-XL"
        # try:
        #     self.client = Client(os.environ["COMET_API_URL"])
        # except Exception as e:
        #     print(e)
        self.client = None
        self.cache_dir = Path(__file__).parent / ".comet_client_cache"
        self.cache = dc.Cache(str(self.cache_dir), expire=None, size_limit=int(40e10), cull_limit=0, eviction_policy='none')

    def segment_score(self, src: Optional[str], mt: str, ref: Optional[str]):
        cache_key = (self.model_name, src, mt, ref)
        if cache_key in self.cache:
            return float(self.cache[cache_key])
        if self.client is None:
            return 0.
        if not mt or not mt.strip():
            return 0.
        score, _ = self.client.predict(
            src=src,
            mt=mt,
            ref=ref,
            model_name=self.model_name,
            api_name="/run_inference_with_model"
        )
        self.cache[cache_key] = score
        return float(score)

    def corpus_score(self, src: List[Optional[str]], mt: List[str], ref: List[Optional[str]]):
        scores = []
        for s, m, r in tqdm(list(zip(src, mt, ref)), desc="Scoring segments"):
            score = self.segment_score(s, m, r)
            scores.append(score)
        try:
            return float(np.mean(scores))
        except TypeError as e:
            print(f"Error calculating mean score: {e}")
            return 0

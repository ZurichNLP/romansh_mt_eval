
class Pipeline:
    def __init__(self, name: str, steps: list[tuple[str,str]], record_src_step: int = 0):
        """
        name            → the `lp` tag you'll use when storing  
        steps           → list of (src_lang, tgt_lang) pairs, e.g.
                           [("de","en"), ("en","rm")]  
        record_src_step → which intermediate output to treat as “source”
                           (0=original input, 1=after first step, …)
        """
        self.name = name
        self.steps = steps
        self.record_src_step = record_src_step
    

    @staticmethod
    def build_pipeline(model_name: str, src_lang: str, tgt_lang: str, pivot_lang: str | None = None, src_variety_code: str | None = None, variety_src: str | None = None) -> "Pipeline":
        """
        Builds either a direct or pivot translation pipeline for a specific model and language pair.
        Uses `src_variety_code` for naming if provided, otherwise uses `src_lang`.
        """
        pipelines = []
        name_src = src_variety_code if src_variety_code else src_lang
        if pivot_lang:
            # Pipeline 1: Source to Pivot
            pipelines.append(Pipeline(
                name=f"{model_name}-{variety_src + '-' if variety_src is not None else ''}pivot-{name_src}-to-{pivot_lang}",
                steps=[(src_lang, pivot_lang)],
                record_src_step=0
            ))
            # Pipeline 2: Pivot to Target. The source for this dataset will be the intermediate translation.
            pipelines.append(Pipeline(
                name=f"{model_name}-{variety_src + '-' if variety_src is not None else ''}pivot-{pivot_lang}-to-{tgt_lang}",
                steps=[(src_lang, pivot_lang), (pivot_lang, tgt_lang)],
                record_src_step=1
            ))
        else:
            steps = [(src_lang, tgt_lang)]
            record_src_step = 0
            name = f"{model_name}-direct-{name_src}-to-{tgt_lang}"
            pipelines.append(Pipeline(
                name=name,
                steps=steps,
                record_src_step=record_src_step
            ))
        return pipelines

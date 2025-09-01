import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class MadladTranslator:
    """Encapsulates model, tokenizer, and translation logic."""
    def __init__(self, model_name, device=None):
        print(f"Loading model: {model_name}…")
        kwargs = {}
        if torch.cuda.is_available():
            kwargs = dict(
                torch_dtype=torch.float32, 
                device_map="auto", 
                use_safetensors=True,
            )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"→ loaded on {self.model.device}")


    def translate(self, text, target_lang, num_beams=5, max_new_tokens=128):
        """
        Translates `text` to `target_lang` using beam search.
        target_lang: ISO-639-1 code, e.g. "rm" for Romansh.
        """
        # 1) Build the language-prefix token, e.g. "<2rm>"
        lang_token = f"<2{target_lang}>"
        # 2) Either force it as BOS or simply prepend to the input:
        #    Here we prepend to the text so we don't rely on get_lang_id().
        input_str = f"{lang_token} {text}"
        # 3) Tokenize (with padding/truncation) and move to the model's device
        inputs = self.tokenizer(
            input_str,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.model.device)
        # 4) Generate with beam search
        outputs = self.model.generate(
            **inputs,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            early_stopping=True
        )
        # 5) Decode and return the single best hypothesis
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


    def translate_batch(
        self,
        texts: list[str],
        target_lang: str,
        num_beams=5,
        max_new_tokens: int = 128
    ) -> list[str]:
        """
        Translates a batch of strings to target_lang using beam search.
        Returns a list of decoded hypotheses, preserving input order.
        """
        # Build language prefixes
        lang_token = f"<2{target_lang}>"
        input_strs = [f"{lang_token} {t}" for t in texts]

        # Tokenize batch, pad/truncate to max_length=512
        inputs = self.tokenizer(
            input_strs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        # Generate in batch
        outputs = self.model.generate(
            **inputs,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            early_stopping=True,
        )
        # Decode each sequence
        return [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs]

from __future__ import annotations
import re

def clean_sql(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```sql\s*|^```\s*|```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
    if ";" in text:
        text = text.split(";", 1)[0]
    return text.strip()

class Generator:
    """Lazy-loads Qwen2.5-Coder + optional LoRA adapter. Used on GPU/Colab."""
    def __init__(self, base: str, adapter: str | None = None, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(base)
        self.model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.float16, device_map=device)
        if adapter:
            # Colab preinstalls an old torchao that makes PEFT's LoRA dispatcher raise
            # a version check, even though we never use torchao. Neutralize the check.
            try:
                import peft.tuners.lora.torchao as _lt
                _lt.is_torchao_available = lambda *a, **k: False
            except Exception:
                pass
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def generate(self, prompt: str, n: int = 8, temperature: float = 0.8,
                 max_new_tokens: int = 256, max_time: float = 300.0) -> list[str]:
        import torch
        msgs = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, do_sample=True, temperature=temperature,
                num_return_sequences=n, max_new_tokens=max_new_tokens,
                max_time=max_time,  # wall-clock backstop so one prompt can't stall the run
                pad_token_id=self.tok.eos_token_id)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return [clean_sql(self.tok.decode(g, skip_special_tokens=True)) for g in gen]

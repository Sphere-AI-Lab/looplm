from typing import Dict, List

from transformers import AutoTokenizer


class PackedTokenizer:
    def __init__(self, tokenizer_name_or_path: str, seq_len: int, add_eos: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.seq_len = seq_len
        self.add_eos = add_eos

    def encode_text(self, text: str) -> List[int]:
        # Pretrain uses sequence packing afterwards, so very long raw rows are valid.
        # Disable tokenizer length warnings to avoid noisy logs for long documents.
        ids = self.tokenizer.encode(text, add_special_tokens=False, verbose=False)
        if self.add_eos and self.tokenizer.eos_token_id is not None:
            ids = ids + [self.tokenizer.eos_token_id]
        return ids

    def pack_samples(self, texts: List[str]) -> List[Dict[str, List[int]]]:
        packed: List[Dict[str, List[int]]] = []
        buf: List[int] = []
        for text in texts:
            ids = self.encode_text(text)
            buf.extend(ids)
            while len(buf) >= self.seq_len:
                chunk = buf[: self.seq_len]
                buf = buf[self.seq_len :]
                packed.append({"input_ids": chunk, "labels": chunk.copy()})
        return packed

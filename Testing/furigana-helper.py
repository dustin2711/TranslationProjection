# import sys

# print(sys.path)


# from dataclasses import dataclass
# from fugashi.fugashi import UnidicNode
# from fugashi import Tagger
# from jamdict import Jamdict
# from typing import List, Optional
# import torch
# from transformers import AutoModelForMaskedLM, AutoTokenizer
# from sentence_transformers import SentenceTransformer, util
# from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForMaskedLM

# # Load BERT for contextual understanding
# tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese")
# model = AutoModelForMaskedLM.from_pretrained("tohoku-nlp/bert-base-japanese")

# jam = Jamdict(r"C:\Users\sens\Desktop\FingernailProjection\Testing\jamdict.db\jamdict.db")


# @dataclass
# class Meaning:
#     gloss: str
#     pos: str


# @dataclass
# class MarkedTerm:
#     marked_symbol: str
#     """ token.surface"""
#     reading: str
#     """ token.feature.kana"""
#     # total_word: str
#     # """ token.feature.pos1"""
#     meanings: List[Meaning]
#     """Gained through jamdict"""


# def find_term_for_symbol_in_sentence(sentence: str, symbol: str) -> Optional[MarkedTerm]:
#     tagger = Tagger()
#     tokens: list[UnidicNode] = list(tagger(sentence))

#     # Go through all sentence tokens and find the one that contains the symbol
#     for token in tokens:
#         if symbol in token.surface:
#             result = jam.lookup(token.surface)
#             return MarkedTerm(
#                 token.surface,
#                 token.feature.kana,
#                 # token.feature.pos1, # buggy as hell
#                 [
#                     Meaning(sense.gloss, sense.pos)
#                     for entry in result.entries
#                     for sense in entry.senses
#                 ],
#             )

#     return None


# def get_reading(sentence: str, symbol: str) -> str:
#     """Gets the most propable meaning of a symbol (kanji, kana) in a sentence."""
#     # 1. Get all possible meanings
#     if not (term := find_term_for_symbol_in_sentence(sentence, symbol)):
#         raise ValueError(f"Kanji '{symbol}' not found in sentence.")

#     return term.reading


# meaning = get_reading("二の砲曲か丘像全てを観測司育て", "砲")

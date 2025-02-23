import helper
from enums import TranslationDirection
from text_detection import TextDetection
import openai
import json
from stopwatch import Stopwatch
import numpy as np


client = openai.OpenAI()

def get_openai_translation(word: str, sentence: str, direction: TranslationDirection = TranslationDirection.ENGLISH_TO_JAPANESE
) -> str:
    """Gets a list of mappings from English to Japanese."""
    global client

    # print(f"Getting translations from OpenAI for {word} in {sentence}")

    direction = (
        "English to Japanese"
        if direction == TranslationDirection.ENGLISH_TO_JAPANESE
        else "Japanese to English"
    )
#     prompt = f"""Translate the word '{word}' in the following sentence from {direction}:
# {sentence}
# The translation should make sence within the context. Respond only with the word."""

    prompt = f"""If a Japanese reader would read the following sentence: {sentence}
And the reader wouldnt know the word '{word}', how would you translate literally while still making sence in the context? Respond only with the word itself without adding reading."""

    completion = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}]
    )
    output = completion.choices[0].message.content.replace("\\'", "'")
    return output

stopwatch = Stopwatch()

translations = {}
times_ms = []

sentences = [
    "For many centuries, the question of how our minds work was left to theologians and philosophers.",
    "But at the beginning of the twentieth century, a new science, experimental psychology emerged, in which the speculative theories of the past were confirmed or disproved by the scientific method.",
    "At the forefront of this research was J B Watson.",
    "His area of interest was the origin of human emotions.",
    "Do we learn them, or do we have them when we are born?",
    "In particular, Watson wanted to study fear, and was prepared to go to whatever lengths to study his theory.",
]

for sentence in sentences:
    print(f"sentence:\n{sentence}")
    for word in sentence.split(' '):
        stopwatch.restart()

        translation = get_openai_translation(word, sentence)
        ms = stopwatch.milliseconds

        translations[word] = translation
        times_ms.append(ms)

        print(f"{ms}: {word} => {translation}")

median = np.median(values) # 424 und 453
print(f"median = {median}")
print(translations)

collected_translations = {
# A) Ganzen Text auf einmal übersetzt:
#Give me the json dictionary with translations from english to japanese, 
# with in-context meaningful translations 
    "for": "〜の間",
    "many": "多くの",
    "centuries": "世紀",
    "the": "その",
    "question": "問題",
    "of": "の",
    "how": "どのように",
    "our": "私たちの",
    "minds": "心",
    "work": "働く",
    "was": "〜だった",
    "left": "任された",
    "to": "に",
    "theologians": "神学者",
    "and": "と",
    "philosophers": "哲学者",
    
    "but": "しかし",
    "at": "〜において",
    "beginning": "初め",
    "twentieth": "20番目の",
    "century": "世紀",
    "a": "一つの",
    "new": "新しい",
    "science": "科学",
    "experimental": "実験的な",
    "psychology": "心理学",
    "emerged": "登場した",
    "in": "の中で",
    "which": "どの",
    "speculative": "推測的な",
    "theories": "理論",
    "past": "過去の",
    "were": "〜だった",
    "confirmed": "確認された",
    "or": "または",
    "disproved": "否定された",
    "by": "〜によって",
    "scientific": "科学的な",
    "method": "方法",
    
    "forefront": "最前線",
    "this": "この",
    "research": "研究",
    "J": "J",
    "B": "B",
    "Watson": "ワトソン",
    
    "area": "分野",
    "interest": "関心",
    "origin": "起源",
    "human": "人間の",
    "emotions": "感情",
    
    "do": "するのか",
    "we": "私たちは",
    "learn": "学ぶ",
    "them": "それらを",
    "have": "持つ",
    "when": "〜の時",
    "are": "〜である",
    "born": "生まれる",
    
    "particular": "特に",
    "wanted": "望んだ",
    "study": "研究する",
    "fear": "恐怖",
    "prepared": "準備した",
    "go": "行う",
    "whatever": "どんなことでも",
    "lengths": "手段",
    "his": "彼の",
    "theory": "理論",

# B)
# Prompt: Stück für Stück mit "sence within the context"
# Translate the word '{word}' in the following sentence from {direction}:
# {sentence}
# The translation should make sence within the context. Respond only with the word.
    "For": "何世紀もの間",
    "many": "何世紀にも",
    "centuries": "何世紀も",
    "the": "その",
    "question": "問題",
    "of": "が",
    "how": "どのように",
    "our": "私たちの",
    "minds": "精神",
    "work": "働く",
    "left": "任せられました",
    "theologians": "神学者",
    "and": "および",
    "philosophers": "哲学者",
    "But": "しかし",
    "at": "に",
    "beginning": "初頭",
    "twentieth": "20世紀",
    "century": "世紀",
    "a": "新しい",
    "new": "新しい",
    "science": "科学",
    "experimental": "実験的な",
    "psychology": "心理学",
    "emerged": "現れた",
    "in": "で",
    "which": "において",
    "speculative": "仮説的な",
    "theories": "理論",
    "past": "過去",
    "were": "確認されました",
    "confirmed": "確認された",
    "or": "または",
    "disproved": "反証された",
    "by": "によって",
    "scientific": "科学的",
    "method": "方法",
    "At": "最前線",
    "forefront": "最前線",
    "this": "この",
    "research": "研究",
    "J": "ジェー",
    "B": "ビー",
    "Watson": "ワトソン",
    "His": "彼の",
    "area": "領域",
    "interest": "関心",
    "origin": "起源",
    "human": "人間",
    "emotions": "感情",
    "Do": "する",
    "we": "私たち",
    "learn": "学ぶ",
    "them": "それら",
    "have": "持って生まれる",
    "when": "とき",
    "are": "生まれる",
    "born": "生まれる",
    "In": "特に",
    "particular": "特に",
    "wanted": "望んだ",
    "to": "ために",
    "study": "研究する",
    "fear": "恐怖",
    "was": "でした",
    "prepared": "準備ができていました",
    "go": "行く",
    "whatever": "どんな",
    "lengths": "どんな手段",
    "his": "彼の",
    "theory": "理論",
# C) Literally mit Sinn im Kontext
# Prompt:
# If a Japanese reader would read the following sentence:
# For many centuries, the question of how our minds work was left to theologians and philosophers.
# and wouldnt know the word "century", how would you translate literally while still making 
# sence in the context? Respond only with the word itself without adding reading..
  "For": "Many",
  "many": "多くの",
  "centuries": "世紀",
  "the": "定冠詞",
  "question": "問題",
  "of": "の",
  "how": "どのように",
  "our": "人間の",
  "minds": "精神",
  "work": "働く",
  "was": "でした",
  "left": "任せられていた",
  "to": "に対して",
  "theologians": "神学者",
  "and": "および",
  "philosophers": "哲学者",
  "But": "しかし",
  "at": "において",
  "beginning": "初頭",
  "twentieth": "20世紀",
  "century": "世紀",
  "a": "一つの",
  "new": "新しい",
  "science": "科学",
  "experimental": "実験的",
  "psychology": "心理学",
  "emerged": "現れた",
  "in": "で",
  "which": "どちら",
  "speculative": "推測的な",
  "theories": "仮説",
  "past": "過去",
  "were": "確認されました",
  "confirmed": "確認された",
  "or": "または",
  "disproved": "誤りと証明される",
  "by": "によって",
  "scientific": "科学的",
  "method": "方法",
  "At": "先頭",
  "forefront": "最前線",
  "this": "この",
  "research": "リサーチ",
  "J": "ジェイ",
  "B": "ビー",
  "Watson": "ワトソン",
  "His": "彼の",
  "area": "領域",
  "interest": "関心",
  "origin": "起源",
  "human": "人間の",
  "emotions": "感情",
  "Do": "する",
  "we": "私たち",
  "learn": "学ぶ",
  "them": "それら",
  "have": "持つ",
  "when": "時に",
  "are": "生まれる",
  "born": "生まれる",
  "In": "特に",
  "particular": "特に",
  "wanted": "望んでいた",
  "study": "研究",
  "fear": "恐怖",
  "prepared": "用意した",
  "go": "行く",
  "whatever": "どんな",
  "lengths": "手段",
  "his": "彼の",
  "theory": "理論"
}
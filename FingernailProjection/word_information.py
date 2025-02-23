from dataclasses import dataclass
from fugashi import Tagger
from typing import List, Optional

# from fugashi.fugashi import UnidicNode
# import jaconv
# from transformers import AutoModelForMaskedLM, AutoTokenizer
# from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForMaskedLM
from jamdict import Jamdict
from config import Config
from enums import TranslationDirection
from furigana import split_furigana

# import MeCab

from helper import join_string
from text_detection import TextDetection

# Load BERT for contextual understanding
# tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese")
# model = AutoModelForMaskedLM.from_pretrained("tohoku-nlp/bert-base-japanese")

jam = Jamdict(r"C:\Users\sens\Desktop\FingernailProjection\Testing\jamdict\jamdict.db")
tagger = Tagger()


@dataclass
class Meaning:
    gloss: str
    pos: str


meanings_by_word_and_sentence = {}


@dataclass
class WordInformation:
    """Contains information of the word like the reading and meanings."""

    text: str
    """ token.surface"""
    reading: str
    """ token.feature.kana"""
    meanings: List[Meaning]
    """Gained through jamdict"""

    @staticmethod
    def create(
        word: str,
        sentence: str,
        starting_character_index: int,
        character_count: int,
        lookup_meaning: bool = True,
    ) -> Optional["WordInformation"]:
        """
        Gets information about the word in the sentence using the sentence context.
        ToDo: Implement context consideration.
        """
        word_end_index = starting_character_index + character_count

        # Check for valid indices
        if word != sentence[starting_character_index:word_end_index]:
            print("Word information got invalid data.")
            return None

        if reading_tuples := split_furigana(word):
            reading = join_string(reading_tuples, " ", lambda it: it[-1])
        else:
            return None

        pair = (word, sentence)
        if not (meanings := meanings_by_word_and_sentence.get(pair)):
            # Looks up a Japanese word, this is very slow, ~30 ms
            result = jam.lookup(word)
            meanings = (
                [
                    Meaning(sense.gloss, sense.pos)
                    for entry in result.entries
                    for sense in entry.senses
                ]
                if result.entries
                else []
            )

            meanings_by_word_and_sentence[pair] = meanings

        return WordInformation(
            text=word,
            reading=reading,
            meanings=meanings,
        )

    @staticmethod
    def create_text(detection: TextDetection, config: Config) -> Optional[str]:

        word = detection.text
        sentence = detection.parent.text

        if config.translation_direction == TranslationDirection.JAPANESE_TO_ENGLISH:
            if info := WordInformation.create(
                word,
                sentence,
                detection.text_index,
                detection.text_length,
            ):
                return None

            return (
                (
                    info.meanings[0].gloss[0] if len(info.meanings) > 0 else info.text
                )  # the text can be '?', in that case there is no meaning
                if config.show_translation_no_reading
                else info.reading
            )

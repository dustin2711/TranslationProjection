import helper
from enums import TranslationDirection
from hand_contour_detector import TranslationDirection
from matlike_utils import TranslationDirection
from text_detection import TextDetection


def get_openai_translations_by_paragraph(
    paragraph_detection: TextDetection, direction: TranslationDirection
) -> list[str]:
    return get_openai_translations(
        [detection.text for detection in paragraph_detection.children], direction
    )


def get_openai_translations(
    sentence_words: list[str], direction: TranslationDirection
) -> dict[str, str]:
    """Gets a list of mappings from English to Japanese."""

    sentence = helper.join_string(sentence_words, " | ")

    print(f"Getting translations from OpenAI for: {sentence}")

    direction = (
        "English to Japanese"
        if direction == TranslationDirection.ENGLISH_TO_JAPANESE
        else "Japanese to English"
    )
    prompt = f"""Translate each term separated by ' | ' of the following sentence from {direction}.
{sentence}

Provide the translation as a JSON object in this format:

{{
"UntranslatedTerm1": "TranslatedTerm1",
"UntranslatedTerm2": "TranslatedTerm2"
}}

The translation should make sence within the context. Ensure accurate meanings for each translated term. Respond with the JSON object only, without any additional formatting like code block markers."""

    import openai
    import json

    client = openai.OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}]
    )
    json_output = completion.choices[0].message.content.replace("\\'", "'")
    json_dict = json.loads(json_output)

    return json_dict

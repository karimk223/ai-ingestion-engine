"""Translation routing and extraction ordering, without live API/DB writes."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import httpx
from openai import BadRequestError

from app.config import settings
from app.services import extraction_service, graph_service, translation_service as translation
from app.services import graphrag_service as questions


ENGLISH = "The finance department must approve refunds before payment."
FRENCH = "Le service financier doit approuver les remboursements avant le paiement."
ARABIC = "يجب أن توافق الإدارة المالية على استرداد المبلغ قبل الدفع."


def response(data, finish_reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content=data if isinstance(data, str)
                                else json.dumps(data, ensure_ascii=False)),
    )])


class TranslationTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = patch.object(translation, "CACHE_DIR", Path(self.directory.name))
        self.cache.start()
        self.addCleanup(self.cache.stop)

    def test_language_detection_for_supported_languages(self):
        for text, language in ((ENGLISH, "en"), (FRENCH, "fr"), (ARABIC, "ar")):
            with self.subTest(language=language):
                self.assertIn(language, translation._source_languages(text))

    def test_long_text_split_loses_no_characters(self):
        for text in (ENGLISH * 100, FRENCH * 100, ARABIC * 100, "x" * 701):
            pieces = translation._split_text(text, 90)
            self.assertEqual("".join(pieces), text)
            self.assertTrue(all(0 < len(piece) <= 90 for piece in pieces))
        with self.assertRaises(ValueError):
            translation._split_text(ENGLISH, 0)
        self.assertEqual(translation._split_text("abc", 1), ["a", "b", "c"])

    def test_english_and_numbers_never_open_groq_client(self):
        original = [ENGLISH, "2026 / NCD-118 / 7,500"]
        with patch.object(translation, "OpenAI") as client:
            result = translation.prepare_segments_for_extraction(original)
        self.assertEqual(result["texts"], original)
        self.assertEqual(result["translated"], [False, False])
        client.assert_not_called()

    def test_foreign_sections_translated_and_second_run_uses_cache(self):
        with patch.object(settings, "GROQ_API_KEY", "test-key"), \
             patch.object(translation, "OpenAI") as factory, \
             patch.object(translation, "_translate_part", return_value=ENGLISH) as translate:
            first = translation.prepare_segments_for_extraction([FRENCH, ARABIC])
            self.assertEqual(first["texts"], [ENGLISH, ENGLISH])
            self.assertEqual(first["summary"]["parts_translated"], 2)
            self.assertEqual(translate.call_count, 2)
            factory.reset_mock()
            translate.reset_mock()
            second = translation.prepare_segments_for_extraction([FRENCH, ARABIC])
            self.assertEqual(second["summary"]["cached_parts"], 2)
            factory.assert_not_called()
            translate.assert_not_called()

    def test_every_part_of_long_foreign_text_is_processed(self):
        original = (FRENCH + "\n") * 20
        received = []
        def translate_part(client, text):
            received.append(text)
            return ENGLISH
        with patch.object(settings, "TRANSLATION_BATCH_CHARS", 160), \
             patch.object(settings, "GROQ_API_KEY", "test-key"), \
             patch.object(translation, "_cached_translation", return_value=None), \
             patch.object(translation, "OpenAI"), \
             patch.object(translation, "_translate_part", side_effect=translate_part):
            translation.prepare_segments_for_extraction([original])
        self.assertEqual("".join(received), original)

    def test_missing_key_stops_foreign_translation(self):
        with patch.object(settings, "GROQ_API_KEY", ""):
            with self.assertRaisesRegex(RuntimeError, "GROQ_API_KEY"):
                translation.prepare_segments_for_extraction([FRENCH])

    def test_mixed_language_section_translates_the_whole_section(self):
        source = ENGLISH + "\n" + FRENCH
        with patch.object(settings, "GROQ_API_KEY", "test-key"), \
             patch.object(translation, "OpenAI"), \
             patch.object(translation, "_translate_part", return_value=ENGLISH) as translate:
            result = translation.prepare_segments_for_extraction([source])
        self.assertEqual(set(result["segment_languages"][0]), {"en", "fr"})
        self.assertEqual(translate.call_args.args[1], source)

    def test_partial_or_non_english_translation_rejected(self):
        client = MagicMock()
        for data, finish in ((ENGLISH, "length"), (FRENCH, "stop"),
                             ("", "stop"), ([], "stop")):
            client.chat.completions.create.return_value = response(data, finish)
            with self.subTest(data=data, finish=finish), self.assertRaises(RuntimeError):
                translation._translate_part(client, FRENCH)

    def test_corrupt_cache_is_not_reused(self):
        path = translation._cache_path(FRENCH)
        path.write_text("{broken", encoding="utf-8")
        self.assertIsNone(translation._cached_translation(path))

    def test_cache_is_specific_to_source_and_model(self):
        first = translation._cache_path(FRENCH)
        self.assertNotEqual(first, translation._cache_path(ARABIC))
        with patch.object(settings, "TRANSLATION_MODEL", "another-model"):
            self.assertNotEqual(first, translation._cache_path(FRENCH))


class FilePipelineTests(unittest.TestCase):
    def test_all_sections_prepared_before_local_extraction_and_metadata_preserved(self):
        chunks = [
            {"chunk_index": 1, "page_number": 2, "text": ARABIC, "file_name": "mixed.pdf"},
            {"chunk_index": 0, "page_number": 1, "text": FRENCH, "file_name": "mixed.pdf"},
        ]
        original = [dict(chunk) for chunk in chunks]
        prepared = {"texts": [ENGLISH, ENGLISH], "summary": {},
                    "segment_languages": [["fr"], ["ar"]], "translated": [True, True]}
        events = []
        def prepare(segments):
            events.append("prepare")
            self.assertEqual(segments, [FRENCH, ARABIC])
            return prepared
        def extract(segments):
            events.append("extract")
            self.assertEqual(segments, [ENGLISH, ENGLISH])
            return {"triples": []}
        with patch.object(extraction_service, "get_chunks_by_file_id", return_value=chunks), \
             patch.object(extraction_service, "prepare_segments_for_extraction", side_effect=prepare), \
             patch.object(extraction_service, "extract_triples", side_effect=extract):
            report = extraction_service.extract_from_file("example")
        self.assertEqual(events, ["prepare", "extract"])
        self.assertEqual(chunks, original)
        self.assertEqual(report["prepared_chunks"][0]["page_number"], 1)
        self.assertEqual(report["prepared_chunks"][0]["original_text"], FRENCH)
        self.assertEqual(report["prepared_chunks"][0]["english_text"], ENGLISH)

    def test_translation_failure_leaves_graph_untouched(self):
        with patch.object(extraction_service, "get_chunks_by_file_id", return_value=[
            {"text": FRENCH, "file_name": "example.pdf", "chunk_index": 0},
        ]), patch.object(extraction_service, "prepare_segments_for_extraction",
                         side_effect=RuntimeError("translation failed")), \
             patch.object(extraction_service, "extract_triples") as extract, \
             patch.object(graph_service.graph_store, "clear_file") as clear, \
             patch.object(graph_service.graph_store, "load_triples") as load:
            with self.assertRaisesRegex(RuntimeError, "translation failed"):
                graph_service.build_graph_for_file("example")
        extract.assert_not_called()
        clear.assert_not_called()
        load.assert_not_called()


class QuestionTests(unittest.TestCase):
    def test_translation_and_entities_required_and_list_return_preserved(self):
        with patch.object(settings, "GROQ_API_KEY", "test-key"), \
             patch.object(questions, "OpenAI") as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.completions.create.return_value = response({
                "english_question": "Who approves equipment purchases?",
                "entities": [" Equipment Purchase "],
            })
            self.assertEqual(questions._extract_question_entities("من يوافق على شراء المعدات؟"),
                             ["Equipment Purchase"])
            schema = client.chat.completions.create.call_args.kwargs[
                "response_format"]["json_schema"]["schema"]
            self.assertEqual(schema["required"], ["english_question", "entities"])

    def test_missing_translation_rejected(self):
        with patch.object(settings, "GROQ_API_KEY", "test-key"), \
             patch.object(questions, "OpenAI") as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.completions.create.return_value = response({"entities": ["Wrong"]})
            self.assertEqual(questions._extract_question_entities("Bonjour"), [])

    def test_provider_invalid_json_retried_once(self):
        error = BadRequestError(
            "Invalid JSON", response=httpx.Response(
                400, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
            ), body={"code": "json_validate_failed"},
        )
        with patch.object(settings, "GROQ_API_KEY", "test-key"), \
             patch.object(questions, "OpenAI") as factory:
            create = factory.return_value.__enter__.return_value.chat.completions.create
            create.side_effect = [error, response({
                "english_question": "Who validates supplier registration?",
                "entities": ["Supplier Registration"],
            })]
            self.assertEqual(questions._extract_question_entities("Qui valide l'inscription ?"),
                             ["Supplier Registration"])
            self.assertEqual(create.call_count, 2)


if __name__ == "__main__":
    unittest.main()

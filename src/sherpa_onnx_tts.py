import sys
import os
from pathlib import Path
import sherpa_onnx
import soundfile as sf
from loguru import logger
from tts_interface import TTSInterface
from tts_model_utils import download_kokoro_model, verify_kokoro_model

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


class TTSEngine(TTSInterface):
    def __init__(
        self,
        model_type="vits",  # Add model type parameter: "vits" or "kokoro"
        # VITS parameters (existing)
        vits_model="",
        vits_lexicon="",
        vits_tokens="",
        vits_data_dir="",
        vits_dict_dir="",
        # Kokoro parameters (new)
        kokoro_model="",
        kokoro_voices="",
        kokoro_tokens="",
        kokoro_data_dir="",
        kokoro_dict_dir="",
        kokoro_lexicon="",
        kokoro_lang="",
        # Common parameters
        tts_rule_fsts="",
        max_num_sentences=2,
        sid=0,
        provider="cpu",
        num_threads=1,
        speed=1.0,
        debug=False,
    ):
        self.model_type = model_type

        # VITS parameters
        self.vits_model = vits_model
        self.vits_lexicon = vits_lexicon
        self.vits_tokens = vits_tokens
        self.vits_data_dir = vits_data_dir
        self.vits_dict_dir = vits_dict_dir

        # Kokoro parameters
        self.kokoro_model = kokoro_model
        self.kokoro_voices = kokoro_voices
        self.kokoro_tokens = kokoro_tokens
        self.kokoro_data_dir = kokoro_data_dir
        self.kokoro_dict_dir = kokoro_dict_dir
        self.kokoro_lexicon = kokoro_lexicon
        self.kokoro_lang = kokoro_lang

        # Common parameters
        self.tts_rule_fsts = tts_rule_fsts
        self.max_num_sentences = max_num_sentences
        self.sid = sid
        self.provider = provider
        self.num_threads = num_threads
        self.speed = speed
        self.debug = debug

        self.file_extension = "wav"
        self.new_audio_dir = "cache"

        if not os.path.exists(self.new_audio_dir):
            os.makedirs(self.new_audio_dir)

        self.tts = self.initialize_tts()

    def initialize_tts(self):
        """
        Initialize the sherpa-onnx TTS engine for both VITS and Kokoro models.
        Automatically downloads missing models if needed.
        """
        # Check for auto-download of Kokoro models
        if self.model_type.lower() == "kokoro":
            self._ensure_kokoro_model_available()

        if self.model_type.lower() == "kokoro":
            # Configure Kokoro model
            tts_config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                        model=self.kokoro_model,
                        voices=self.kokoro_voices,
                        tokens=self.kokoro_tokens,
                        data_dir=self.kokoro_data_dir,
                        dict_dir=self.kokoro_dict_dir,
                        lexicon=self.kokoro_lexicon,
                        length_scale=self.speed,  # Kokoro uses length_scale instead of speed
                    ),
                    provider=self.provider,
                    debug=self.debug,
                    num_threads=self.num_threads,
                ),
                rule_fsts=self.tts_rule_fsts,
                max_num_sentences=self.max_num_sentences,
            )
        elif self.model_type.lower() == "vits":
            # Configure VITS model (your original configuration)
            tts_config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=self.vits_model,
                        lexicon=self.vits_lexicon,
                        data_dir=self.vits_data_dir,
                        dict_dir=self.vits_dict_dir,
                        tokens=self.vits_tokens,
                    ),
                    provider=self.provider,
                    debug=self.debug,
                    num_threads=self.num_threads,
                ),
                rule_fsts=self.tts_rule_fsts,
                max_num_sentences=self.max_num_sentences,
            )
        else:
            raise ValueError(
                f"Unsupported model type: {self.model_type}. Use 'vits' or 'kokoro'."
            )

        # Validate the configuration
        if not tts_config.validate():
            raise ValueError("Please check your sherpa-onnx TTS config")

        # Create and return the sherpa-onnx OfflineTts object
        return sherpa_onnx.OfflineTts(tts_config)

    def _ensure_kokoro_model_available(self):
        """
        Ensure Kokoro model files are available. Download if missing.
        """
        if not self.kokoro_model:
            logger.warning("⚠️ No Kokoro model path specified")
            return

        model_path = Path(self.kokoro_model)
        model_dir = model_path.parent

        # Check if model files exist
        if (
            model_path.exists()
            and Path(self.kokoro_tokens).exists()
            and Path(self.kokoro_voices).exists()
        ):
            if verify_kokoro_model(model_dir):
                logger.info(f"✅ Kokoro model verified: {model_dir}")
                return

        # Try to determine model name from path for auto-download
        model_name = model_dir.name
        logger.info(f"🔍 Checking for downloadable Kokoro model: {model_name}")

        # Attempt auto-download
        try:
            downloaded_path = download_kokoro_model(model_name, model_dir.parent)
            if downloaded_path:
                logger.success(
                    f"🎉 Successfully downloaded Kokoro model: {downloaded_path}"
                )
                # Update paths to point to downloaded model
                self.kokoro_model = str(downloaded_path / "model.onnx")
                self.kokoro_tokens = str(downloaded_path / "tokens.txt")
                self.kokoro_voices = str(downloaded_path / "voices.bin")
                if not self.kokoro_data_dir:
                    self.kokoro_data_dir = str(downloaded_path / "espeak-ng-data")
                return
            else:
                logger.warning(f"⚠️ Could not auto-download model {model_name}")
        except Exception as e:
            logger.warning(f"⚠️ Auto-download failed for {model_name}: {e}")

        # Check if files exist after potential download
        missing_files = []
        for file_path, description in [
            (self.kokoro_model, "model.onnx"),
            (self.kokoro_tokens, "tokens.txt"),
            (self.kokoro_voices, "voices.bin"),
        ]:
            if not file_path or not Path(file_path).exists():
                missing_files.append(f"{description} ({file_path})")

        if missing_files:
            logger.error(f"❌ Missing Kokoro model files: {missing_files}")
            logger.info(
                "💡 Please ensure model files are available or check the download URLs in tts_model_utils.py"
            )

    def _normalize_unicode_punctuation(self, text: str) -> str:
        """
        Normalize Unicode punctuation characters to their ASCII equivalents.

        Many LLM outputs contain typographic (curly) quotes, em-dashes, and
        other Unicode punctuation that espeak-ng may vocalize literally as
        symbol names (e.g. "left double quotation mark"). This function
        converts them to plain ASCII so the TTS engine reads the text naturally.

        Args:
            text (str): The input text to normalize.

        Returns:
            str: Text with Unicode punctuation replaced by ASCII equivalents.
        """
        replacements = [
            # Curly / smart double quotes → straight double quote
            ("\u201c", '"'),  # " LEFT DOUBLE QUOTATION MARK
            ("\u201d", '"'),  # " RIGHT DOUBLE QUOTATION MARK
            ("\u201e", '"'),  # „ DOUBLE LOW-9 QUOTATION MARK
            ("\u00ab", '"'),  # « LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
            ("\u00bb", '"'),  # » RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
            # Curly / smart single quotes → straight apostrophe
            ("\u2018", "'"),  # ' LEFT SINGLE QUOTATION MARK
            ("\u2019", "'"),  # ' RIGHT SINGLE QUOTATION MARK
            ("\u201a", "'"),  # ‚ SINGLE LOW-9 QUOTATION MARK
            ("\u2039", "'"),  # ‹ SINGLE LEFT-POINTING ANGLE QUOTATION MARK
            ("\u203a", "'"),  # › SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
            # Dashes → hyphen or comma-space for natural speech rhythm
            ("\u2014", ", "),  # — EM DASH  (e.g. "it was clear — we had won" → ", ")
            ("\u2013", "-"),   # – EN DASH
            ("\u2012", "-"),   # ‒ FIGURE DASH
            ("\u2015", "-"),   # ― HORIZONTAL BAR
            # Ellipsis character → three dots (then _convert_ellipsis_to_periods handles it)
            ("\u2026", "..."),  # … HORIZONTAL ELLIPSIS
            # Other common typographic symbols
            ("\u2022", ""),    # • BULLET → remove
            ("\u00b7", ""),    # · MIDDLE DOT → remove
            ("\u2032", "'"),   # ′ PRIME (feet/minutes) → apostrophe
            ("\u2033", '"'),   # ″ DOUBLE PRIME (inches/seconds) → quote
        ]
        for src, dst in replacements:
            text = text.replace(src, dst)
        return text

    def _filter_asterisks(self, text: str) -> str:
        """
        Remove asterisk characters while preserving the emphasized text content.

        Handles *, **, ***, etc. while preserving the text inside them, so
        LLM markdown emphasis is spoken without the word "asterisk".

        Args:
            text (str): The input text.

        Returns:
            str: Text with asterisk characters removed but content preserved.
        """
        import re

        filtered_text = re.sub(r"\*+([^*]*?)\*+", r"\1", text)
        filtered_text = re.sub(r"\*+", "", filtered_text)
        filtered_text = re.sub(r"\s+", " ", filtered_text).strip()
        return filtered_text

    def _filter_nested(self, text: str, left: str, right: str) -> str:
        """
        Remove all content inside a pair of delimiter characters, handling nesting.

        Args:
            text (str): The input text.
            left (str): The opening delimiter (e.g. '[', '(', '<').
            right (str): The closing delimiter (e.g. ']', ')', '>').

        Returns:
            str: Text with all content inside the delimiters removed.
        """
        import re

        if not isinstance(text, str) or not text:
            return text
        result = []
        depth = 0
        for char in text:
            if char == left:
                depth += 1
            elif char == right:
                if depth > 0:
                    depth -= 1
            else:
                if depth == 0:
                    result.append(char)
        filtered_text = "".join(result)
        filtered_text = re.sub(r"\s+", " ", filtered_text).strip()
        return filtered_text

    def _filter_brackets(self, text: str) -> str:
        """Remove all content inside square brackets [like this].

        Args:
            text (str): The input text.

        Returns:
            str: Text with bracket-enclosed content removed.
        """
        return self._filter_nested(text, "[", "]")

    def _filter_parentheses(self, text: str) -> str:
        """Remove all content inside parentheses (like this).

        Args:
            text (str): The input text.

        Returns:
            str: Text with parentheses-enclosed content removed.
        """
        return self._filter_nested(text, "(", ")")

    def _filter_angle_brackets(self, text: str) -> str:
        """Remove all content inside angle brackets <like this>.

        Args:
            text (str): The input text.

        Returns:
            str: Text with angle-bracket-enclosed content removed.
        """
        return self._filter_nested(text, "<", ">")

    def _convert_ellipsis_to_periods(self, text: str) -> str:
        """
        Convert ellipsis sequences ('...') to a single period for cleaner TTS chunking.

        Collapses two or more consecutive dots into one period, avoiding unnatural
        triple-pause artefacts in espeak-ng.  Does NOT normalize spaces around all
        periods, as that would mangle abbreviation dots such as Ph.D.

        Args:
            text (str): The input text.

        Returns:
            str: Text with multi-dot ellipsis replaced by a single period.
        """
        import re

        processed_text = re.sub(r"\.{2,}", ".", text)
        processed_text = re.sub(r"\s+", " ", processed_text).strip()
        return processed_text

    def _preprocess_abbreviations(self, text: str) -> str:
        """
        Preprocess text to prevent abbreviations from being treated as sentence boundaries.

        Removes or expands periods from common abbreviations so sherpa-onnx does not
        interpret them as sentence-ending periods, which would cause unnatural pauses
        mid-phrase (e.g. "Mr. Smith" would otherwise pause after "Mr.").

        Args:
            text (str): Input text that may contain abbreviations with periods.

        Returns:
            str: Text with abbreviation periods removed or expanded.
        """
        import re

        # Title and honorific abbreviations — strip the trailing period so the
        # sentence splitter does not treat them as sentence boundaries.
        # Pattern: word boundary + title + literal dot + one-or-more whitespace.
        titles = [
            "Mr", "Mrs", "Ms", "Mz", "Dr", "Prof", "Rev", "Sr", "Jr",
            "Gen", "Lt", "Sgt", "Cpl", "Pvt", "Cpt", "Capt", "Maj", "Col",
            "Brig", "Adm", "Cmdr", "Det", "Insp",
            "Gov", "Pres", "Sec", "Rep", "Sen",
            "St",  # Saint
        ]
        for title in titles:
            text = re.sub(
                rf"\b{title}\.\s+",
                f"{title} ",
                text,
                flags=re.IGNORECASE,
            )

        # Latin abbreviations — expand to spoken equivalents.
        text = re.sub(r"\be\.g\.\s*", "for example, ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bi\.e\.\s*", "that is, ", text, flags=re.IGNORECASE)
        text = re.sub(r"\betc\.(?=\s|$)", "etc", text, flags=re.IGNORECASE)
        text = re.sub(r"\bvs\.\s*", "versus ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bapprox\.\s*", "approximately ", text, flags=re.IGNORECASE)

        # Academic degrees — collapse dotted initialism to plain uppercase.
        text = re.sub(r"\bPh\.D\.?", "PhD", text)
        text = re.sub(r"\bM\.D\.?", "MD", text)
        text = re.sub(r"\bB\.A\.?", "BA", text)
        text = re.sub(r"\bM\.A\.?", "MA", text)
        text = re.sub(r"\bB\.S\.?", "BS", text)
        text = re.sub(r"\bM\.S\.?", "MS", text)
        text = re.sub(r"\bM\.B\.A\.?", "MBA", text)

        return text

    def _preprocess_text_for_contractions(self, text: str) -> str:
        """
        Preprocess text to ensure contractions are properly formatted.

        Args:
            text (str): Input text that may contain contractions.

        Returns:
            str: Text with contractions properly formatted.
        """
        import re

        # Normalize common contraction variations to standard forms
        # Handle cases where contractions might be split or malformed
        text = re.sub(r"\bi\s*'\s*ll\b", "i'll", text, flags=re.IGNORECASE)
        text = re.sub(r"\bi\s*'\s*m\b", "i'm", text, flags=re.IGNORECASE)
        text = re.sub(r"\bi\s*'\s*ve\b", "i've", text, flags=re.IGNORECASE)
        text = re.sub(r"\bi\s*'\s*d\b", "i'd", text, flags=re.IGNORECASE)
        text = re.sub(r"\byou\s*'\s*re\b", "you're", text, flags=re.IGNORECASE)
        text = re.sub(r"\byou\s*'\s*ll\b", "you'll", text, flags=re.IGNORECASE)
        text = re.sub(r"\byou\s*'\s*ve\b", "you've", text, flags=re.IGNORECASE)
        text = re.sub(r"\byou\s*'\s*d\b", "you'd", text, flags=re.IGNORECASE)
        text = re.sub(r"\bdon\s*'\s*t\b", "don't", text, flags=re.IGNORECASE)
        text = re.sub(r"\bcan\s*'\s*t\b", "can't", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwon\s*'\s*t\b", "won't", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwouldn\s*'\s*t\b", "wouldn't", text, flags=re.IGNORECASE)
        text = re.sub(r"\bcouldn\s*'\s*t\b", "couldn't", text, flags=re.IGNORECASE)
        text = re.sub(r"\bshouldn\s*'\s*t\b", "shouldn't", text, flags=re.IGNORECASE)
        text = re.sub(r"\bisn\s*'\s*t\b", "isn't", text, flags=re.IGNORECASE)
        text = re.sub(r"\baren\s*'\s*t\b", "aren't", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwasn\s*'\s*t\b", "wasn't", text, flags=re.IGNORECASE)
        text = re.sub(r"\bweren\s*'\s*t\b", "weren't", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhasn\s*'\s*t\b", "hasn't", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhaven\s*'\s*t\b", "haven't", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhadn\s*'\s*t\b", "hadn't", text, flags=re.IGNORECASE)
        text = re.sub(r"\bdoesn\s*'\s*t\b", "doesn't", text, flags=re.IGNORECASE)
        text = re.sub(r"\bdidn\s*'\s*t\b", "didn't", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhe\s*'\s*s\b", "he's", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhe\s*'\s*ll\b", "he'll", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhe\s*'\s*d\b", "he'd", text, flags=re.IGNORECASE)
        text = re.sub(r"\bshe\s*'\s*s\b", "she's", text, flags=re.IGNORECASE)
        text = re.sub(r"\bshe\s*'\s*ll\b", "she'll", text, flags=re.IGNORECASE)
        text = re.sub(r"\bshe\s*'\s*d\b", "she'd", text, flags=re.IGNORECASE)
        text = re.sub(r"\bit\s*'\s*s\b", "it's", text, flags=re.IGNORECASE)
        text = re.sub(r"\bit\s*'\s*ll\b", "it'll", text, flags=re.IGNORECASE)
        text = re.sub(r"\bit\s*'\s*d\b", "it'd", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwe\s*'\s*re\b", "we're", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwe\s*'\s*ll\b", "we'll", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwe\s*'\s*ve\b", "we've", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwe\s*'\s*d\b", "we'd", text, flags=re.IGNORECASE)
        text = re.sub(r"\bthey\s*'\s*re\b", "they're", text, flags=re.IGNORECASE)
        text = re.sub(r"\bthey\s*'\s*ll\b", "they'll", text, flags=re.IGNORECASE)
        text = re.sub(r"\bthey\s*'\s*ve\b", "they've", text, flags=re.IGNORECASE)
        text = re.sub(r"\bthey\s*'\s*d\b", "they'd", text, flags=re.IGNORECASE)
        text = re.sub(r"\bthat\s*'\s*s\b", "that's", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwhat\s*'\s*s\b", "what's", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwho\s*'\s*s\b", "who's", text, flags=re.IGNORECASE)
        text = re.sub(r"\bthere\s*'\s*s\b", "there's", text, flags=re.IGNORECASE)
        text = re.sub(r"\blet\s*'\s*s\b", "let's", text, flags=re.IGNORECASE)

        return text

    def generate_audio(self, text, file_name_no_ext=None):
        """
        Generate speech audio file using sherpa-onnx TTS.
        Works for both VITS and Kokoro models.

        Parameters:
            text (str): The text to speak.
            file_name_no_ext (str, optional): Name of the file without extension.

        Returns:
            str: The path to the generated audio file.
        """
        # Layer 1 — structural / Unicode filters (order matters):
        #   1. Normalize Unicode punctuation (curly quotes, em-dashes, Unicode ellipsis …)
        #      MUST run before ellipsis conversion so '…' → '...' is handled.
        text = self._normalize_unicode_punctuation(text)
        #   2. Strip LLM markdown emphasis (*bold*, **bold**) — keep the text inside.
        text = self._filter_asterisks(text)
        #   3. Remove content inside brackets / parentheses / angle brackets entirely
        #      (stage directions, aside notes, XML-like tags from LLM output).
        text = self._filter_brackets(text)
        text = self._filter_parentheses(text)
        text = self._filter_angle_brackets(text)
        #   4. Collapse '...' → '.' for better TTS sentence chunking.
        text = self._convert_ellipsis_to_periods(text)

        # Layer 2 — engine-level fixes applied just before sherpa-onnx sees the text:
        #   5. Expand abbreviation periods so the sentence splitter doesn't pause mid-phrase.
        text = self._preprocess_abbreviations(text)
        #   6. Re-join split contractions (e.g. "don ' t" → "don't") from upstream tokenizers.
        text = self._preprocess_text_for_contractions(text)
        logger.debug(f"🔤 Preprocessed text: '{text}'")

        file_name = self.generate_cache_file_name(file_name_no_ext, self.file_extension)

        try:
            if self.model_type.lower() == "kokoro":
                # For Kokoro, the speed is controlled by length_scale in the model config
                # So we just pass sid parameter
                audio = self.tts.generate(text, sid=self.sid)
            else:
                # For VITS, use the original parameters
                audio = self.tts.generate(text, sid=self.sid, speed=self.speed)

            if len(audio.samples) == 0:
                logger.error(
                    "Error in generating audios. Please read previous error messages."
                )
                return None

            sf.write(
                file_name,
                audio.samples,
                samplerate=audio.sample_rate,
                subtype="PCM_16",
            )

            return file_name

        except Exception as e:
            logger.critical(f"\nError: sherpa-onnx unable to generate audio: {e}")
            return None

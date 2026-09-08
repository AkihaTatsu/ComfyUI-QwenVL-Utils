import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

try:
    from llama_cpp import Llama, llama_cpp
    from llama_cpp.llama_chat_format import MTMDChatHandler
except ImportError:
    MTMDChatHandler = None


def load_gguf_backend():
    package_name = "_qwenvl_test_lib"
    lib_dir = Path(__file__).resolve().parents[1] / "lib"

    package = types.ModuleType(package_name)
    package.__path__ = [str(lib_dir)]
    sys.modules[package_name] = package

    settings = types.ModuleType(f"{package_name}.settings")
    settings.GGUF_VL_CATALOG = {}
    settings.SYSTEM_PROMPTS = {}
    sys.modules[settings.__name__] = settings

    model_utils = types.ModuleType(f"{package_name}.model_utils")
    model_utils.get_gguf_base_dir = lambda: Path(".")
    model_utils.safe_dirname = str
    model_utils.download_gguf_file = lambda *args, **kwargs: None
    model_utils.filter_kwargs_for_callable = lambda callable_, values: values
    sys.modules[model_utils.__name__] = model_utils

    media = types.ModuleType(f"{package_name}.media")
    media.tensor_to_base64_png = lambda image: ""
    media.sample_video_frames = lambda video, frame_count: []
    sys.modules[media.__name__] = media

    device = types.ModuleType(f"{package_name}.device")
    device.pick_device = lambda choice: choice
    device.clear_memory = lambda: None
    sys.modules[device.__name__] = device

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy_utils = types.ModuleType("comfy.utils")
    comfy_utils.ProgressBar = lambda total: types.SimpleNamespace(update_absolute=lambda *args: None)
    model_management = types.ModuleType("comfy.model_management")
    model_management.processing_interrupted = lambda: False
    model_management.throw_exception_if_processing_interrupted = lambda: None
    comfy.utils = comfy_utils
    comfy.model_management = model_management
    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = comfy_utils
    sys.modules["comfy.model_management"] = model_management

    module_name = f"{package_name}.gguf_backend"
    spec = importlib.util.spec_from_file_location(module_name, lib_dir / "gguf_backend.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


gguf_backend = load_gguf_backend()


CONTEXT_ERROR = "Llama.eval(decode): Failed completely even with batch size 1."


class FakeHandler:
    def __init__(self, text_tokens=128, image_tokens=(2048,)):
        self.text_tokens = text_tokens
        self.image_tokens = image_tokens
        self._init_mtmd_context = Mock()
        self._mtmd_cpp = types.SimpleNamespace(
            mtmd_input_chunks_free=Mock(), mtmd_bitmap_free=Mock(),
        )
        self.chunks = object()
        self.bitmaps = [object() for _ in image_tokens]
        self.close = Mock()

    def _process_mtmd_prompt(self, **kwargs):
        return [1] * (self.text_tokens + sum(self.image_tokens)), [], self.chunks, self.bitmaps


class FakeLlama(gguf_backend._GGUFGenerationLimit):
    def __init__(self, chunks=("complete",), error=None, n_ctx=8192, finish="stop"):
        self.chunks = chunks
        self.error = error
        self.context_size = n_ctx
        self.finish = finish
        self.calls = []
        self.close = Mock()
        self.stream_closed = False

    def n_ctx(self):
        return self.context_size

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        try:
            for content in self.chunks:
                yield {"choices": [{"delta": {"content": content}}]}
            if self.error is not None:
                raise self.error
            yield {"choices": [{"delta": {}, "finish_reason": self.finish}]}
        finally:
            self.stream_closed = True


def configured_backend(llm=None, handler=None):
    backend = gguf_backend.GGUFModelBackend()
    backend.llm = llm if llm is not None else FakeLlama()
    backend.chat_handler = handler
    backend.load_model = Mock()
    return backend


def run_backend(backend, keep_model_loaded=True, **kwargs):
    arguments = dict(
        max_tokens=16384, enable_thinking=None, image=None, video=None,
    )
    arguments.update(kwargs)
    return backend.run(
        model_name="test.gguf",
        preset_prompt="describe",
        custom_prompt="",
        frame_count=1,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.2,
        seed=1,
        keep_model_loaded=keep_model_loaded,
        device="cuda",
        **arguments,
    )


class GGUFContextRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.progress = Mock()
        self.tqdm_patch = patch.object(gguf_backend, "tqdm", return_value=self.progress)
        self.tqdm_patch.start()
        self.addCleanup(self.tqdm_patch.stop)

    def test_streaming_context_error_preserves_partial_and_closes_progress(self):
        llm = FakeLlama(chunks=("partial", " output"), error=RuntimeError(CONTEXT_ERROR))
        backend = configured_backend(llm)
        with self.assertLogs(level="WARNING") as logs:
            self.assertEqual(run_backend(backend), ("partial output",))
        self.assertIn("truncated response", "\n".join(logs.output))
        backend.load_model.assert_called_once()
        self.assertEqual(len(llm.calls), 1)
        self.assertTrue(llm.stream_closed)
        self.progress.close.assert_called_once()
        llm.close.assert_called_once()
        self.assertIsNone(backend.llm)

    def test_context_exhaustion_before_output_returns_empty(self):
        llm = FakeLlama(chunks=(), error=RuntimeError(CONTEXT_ERROR))
        backend = configured_backend(llm)
        with self.assertLogs(level="WARNING") as logs:
            self.assertEqual(run_backend(backend), ("",))
        self.assertIn("empty response", "\n".join(logs.output))
        self.assertIsNone(backend.llm)

    def test_prompt_fills_or_exceeds_context_returns_empty_without_generation(self):
        for input_tokens in (8192, 8193):
            with self.subTest(input_tokens=input_tokens):
                handler = FakeHandler(text_tokens=input_tokens, image_tokens=())
                llm = FakeLlama()
                backend = configured_backend(llm, handler)
                with self.assertLogs(level="WARNING") as logs:
                    self.assertEqual(run_backend(backend), ("",))
                self.assertIn(f"Input uses {input_tokens}", "\n".join(logs.output))
                self.assertEqual(llm.calls, [])
                handler._mtmd_cpp.mtmd_input_chunks_free.assert_called_once_with(handler.chunks)
                handler.close.assert_called_once()
                self.assertIsNone(backend.chat_handler)

    def test_context_error_without_comfy_still_preserves_partial(self):
        llm = FakeLlama(chunks=("部分", "输出🙂"), error=RuntimeError(CONTEXT_ERROR))
        with patch.object(gguf_backend, "_COMFY", False), self.assertLogs(level="WARNING"):
            self.assertEqual(run_backend(configured_backend(llm)), ("部分输出🙂",))
        self.assertTrue(llm.calls[0]["stream"])
        self.assertTrue(llm.stream_closed)

    def test_non_context_runtime_error_is_not_retried(self):
        for message in ("CUDA out of memory", "kv_cache corrupted", "llama_decode failed: code -3"):
            with self.subTest(message=message):
                llm = FakeLlama(error=RuntimeError(message))
                backend = configured_backend(llm)
                with self.assertRaisesRegex(RuntimeError, message):
                    run_backend(backend)
                backend.load_model.assert_called_once()
                self.assertIsNone(backend.llm)
                llm.close.assert_called_once()
                self.assertTrue(llm.stream_closed)

    def test_context_error_variants_return_partial(self):
        for message in (
            "decode: failed to find a memory slot for batch of size 1",
            "No KV slot available",
            "Prompt exceeds n_ctx",
            "Requested tokens (8193) exceed context window of 8192",
            "Llama.eval: Context Shift is explicitly disabled. You MUST increase n_ctx",
        ):
            with self.subTest(message=message), self.assertLogs(level="WARNING"):
                llm = FakeLlama(error=RuntimeError(message))
                self.assertEqual(run_backend(configured_backend(llm)), ("complete",))

    def test_cancellation_is_not_returned_as_success(self):
        class Interrupted(Exception):
            pass

        llm = FakeLlama()
        backend = configured_backend(llm)
        interrupt = gguf_backend.comfy.model_management
        with patch.object(interrupt, "throw_exception_if_processing_interrupted", side_effect=[None, Interrupted()]):
            with self.assertRaises(Interrupted):
                run_backend(backend)
        self.assertIsNone(backend.llm)
        self.assertTrue(llm.stream_closed)
        self.progress.close.assert_called_once()

    def test_early_stop_preserves_loaded_model(self):
        llm = FakeLlama()
        backend = configured_backend(llm)
        self.assertEqual(run_backend(backend), ("complete",))
        self.assertIs(backend.llm, llm)
        llm.close.assert_not_called()
        self.assertTrue(llm.stream_closed)

    def test_keep_model_loaded_false_still_clears_backend(self):
        llm = FakeLlama()
        backend = configured_backend(llm)
        self.assertEqual(run_backend(backend, keep_model_loaded=False), ("complete",))
        self.assertIsNone(backend.llm)
        llm.close.assert_called_once()

    def test_next_request_loads_fresh_after_context_error(self):
        first = FakeLlama(error=RuntimeError(CONTEXT_ERROR))
        backend = configured_backend(first)
        with self.assertLogs(level="WARNING"):
            run_backend(backend)
        second = FakeLlama(chunks=("fresh",))

        def load_again(*args, **kwargs):
            self.assertIsNone(backend.llm)
            backend.llm = second

        backend.load_model.side_effect = load_again
        self.assertEqual(run_backend(backend), ("fresh",))
        self.assertEqual(len(first.calls), 1)
        self.assertEqual(len(second.calls), 1)

    def test_context_limit_is_independent_of_requested_text_budget(self):
        for text_tokens, image_tokens, n_ctx, requested, expected, reason in (
            (128, (2048,), 8192, 1024, 1024, "max_tokens"),
            (128, (2048,), 8192, 16384, 6016, "context"),
            (128, (1024, 2048), 8192, 16384, 4992, "context"),
            (128, (1024,) * 4, 8192, 16384, 3968, "context"),
            (128, (2048,), 4096, 16384, 1920, "context"),
            (8191, (), 8192, 64, 1, "context"),
            (8128, (), 8192, 64, 64, "max_tokens"),
        ):
            with self.subTest(text=text_tokens, images=image_tokens, n_ctx=n_ctx, requested=requested):
                handler = FakeHandler(text_tokens, image_tokens)
                llm = FakeLlama(n_ctx=n_ctx, finish="length")
                backend = configured_backend(llm, handler)
                with self.assertLogs(level="WARNING") as logs:
                    self.assertEqual(run_backend(backend, max_tokens=requested), ("complete",))
                self.assertEqual(llm.calls[0]["max_tokens"], expected)
                self.assertIn(f"at {reason} limit", "\n".join(logs.output))
                self.assertIn(f"requested max_tokens={requested}", "\n".join(logs.output))
                handler._mtmd_cpp.mtmd_input_chunks_free.assert_called_once_with(handler.chunks)
                self.assertEqual(handler._mtmd_cpp.mtmd_bitmap_free.call_count, len(image_tokens))

    def test_chunk_and_character_counts_do_not_control_text_budget(self):
        chunks = ("中文🙂", " text", "<think>", "思考", "</think>", "回答")
        llm = FakeLlama(chunks=chunks)
        backend = configured_backend(llm)
        self.assertEqual(run_backend(backend, max_tokens=2, enable_thinking=True), ("".join(chunks),))
        self.assertEqual(llm.calls[0]["max_tokens"], 2)
        # Fake chunks deliberately outnumber the token budget: the adapter
        # must delegate the limit to the generator, never truncate by chunks.

    def test_thinking_cleanup_does_not_refill_generation_budget(self):
        llm = FakeLlama(chunks=("<think>reasoning</think>", "answer"), finish="length")
        with self.assertLogs(level="WARNING"):
            self.assertEqual(run_backend(configured_backend(llm), max_tokens=64), ("answer",))
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.calls[0]["max_tokens"], 64)

    def test_non_mtmd_handler_keeps_original_generation_limit(self):
        llm = FakeLlama()
        handler = types.SimpleNamespace(close=Mock())
        run_backend(configured_backend(llm, handler), max_tokens=1024)
        self.assertEqual(llm.calls[0]["max_tokens"], 1024)


@unittest.skipIf(MTMDChatHandler is None, "llama-cpp-python MTMD not installed")
class InstalledMTMDTests(unittest.TestCase):
    def test_utf8_buffering_cannot_exceed_sampled_text_token_limit(self):
        class CompletionSource:
            def __init__(self, pieces):
                self._model = types.SimpleNamespace(
                    vocab=None, token_bos=lambda: 99, token_eos=lambda: 98,
                    token_sep=lambda: -1, token_fim_pre=lambda: -1,
                    token_fim_mid=lambda: -1, token_fim_suf=lambda: -1,
                    get_add_sep=lambda: False,
                )
                self.metadata = {}
                self.spm_infill = False
                self.model_path = "test"
                self.verbose = False
                self._n_ctx = 128
                self.cache = None
                self._seed = 1
                self.sampled = []
                self.sampler_closed = False
                self.token_bos = lambda: 99
                self.pieces = pieces

            def generate(self, tokens, **kwargs):
                try:
                    for token in range(len(self.pieces)):
                        self.sampled.append(token)
                        yield token
                finally:
                    self.sampler_closed = True

            def detokenize(self, tokens, prev_tokens=None):
                return b"".join(self.pieces[token] for token in tokens)

        class LimitedCompletion(gguf_backend._GGUFGenerationLimit, CompletionSource):
            pass

        for pieces, limit, expected in (
            ([b"abc", b"\xe4", b"\xbd", b"\xa0", b"end"], 2, "abc"),
            ([b"<think>", b"reasoning", b"</think>", b"answer", b"extra"], 4, "<think>reasoning</think>answer"),
            ([b"<think>", b"reasoning", b"</think>", b"answer"], 2, "<think>reasoning"),
            (["中文".encode(), "🙂".encode(), b"extra"], 2, "中文🙂"),
        ):
            with self.subTest(pieces=pieces, limit=limit):
                llm = LimitedCompletion(pieces)
                llm.set_generation_limit(limit)
                with patch.object(llama_cpp, "llama_token_is_eog", return_value=False):
                    stream = Llama._create_completion(llm, prompt=[99], max_tokens=limit, stream=True)
                    result = list(stream)
                self.assertEqual(len(llm.sampled), limit)
                self.assertEqual("".join(chunk["choices"][0]["text"] for chunk in result), expected)
                self.assertTrue(llm.sampler_closed)
                self.assertEqual(result[-1]["choices"][0]["finish_reason"], "length")

    def test_actual_handler_position_undercount_cannot_overrun_context(self):
        text_kind, image_kind, audio_kind = 0, 1, 2
        chunks = [types.SimpleNamespace(tokens=[1] * 128),
                  types.SimpleNamespace(count=2048, positions=48),
                  types.SimpleNamespace(tokens=[2])]

        class LedgerLlama(FakeLlama):
            def __init__(self):
                super().__init__()
                self.n_tokens = 0
                self.physical_tokens = 0
                self.input_ids = np.zeros(8192, dtype=np.int32)
                self.verbose = False
                self.is_hybrid = False
                self.n_batch = 512
                self._ctx = types.SimpleNamespace(ctx=None)
                self.completion_args = None

            def longest_token_prefix(self, *args):
                return 0

            def eval(self, tokens):
                self.input_ids[self.n_tokens:self.n_tokens + len(tokens)] = tokens
                self.n_tokens += len(tokens)
                self.physical_tokens += len(tokens)

            def create_chat_completion(self, **kwargs):
                return handler(llama=self, **kwargs)

            def create_completion(self, **kwargs):
                self.completion_args = kwargs
                # Exercise the real handler's handoff without native inference.
                return iter([{
                    "id": "test", "model": "test", "created": 0,
                    "choices": [{"text": "safe", "logprobs": None, "finish_reason": None}],
                }, {
                    "id": "test", "model": "test", "created": 0,
                    "choices": [{"text": "", "logprobs": None, "finish_reason": "length"}],
                }])

        llm = LedgerLlama()

        def tokens_text(chunk, count):
            count._obj.value = len(chunk.tokens)
            return chunk.tokens

        def eval_image(mtmd, ctx, chunk, pos, seq, batch, logits, out):
            llm.physical_tokens += chunk.count
            out._obj.value = pos.value + chunk.positions
            return 0

        handler = object.__new__(MTMDChatHandler)
        handler.verbose = False
        handler.mtmd_ctx = object()
        handler._init_mtmd_context = lambda _: None
        native = types.SimpleNamespace(
            mtmd_input_chunk_type=types.SimpleNamespace(
                MTMD_INPUT_CHUNK_TYPE_TEXT=text_kind, MTMD_INPUT_CHUNK_TYPE_IMAGE=image_kind,
                MTMD_INPUT_CHUNK_TYPE_AUDIO=audio_kind,
            ),
            mtmd_input_chunk_get_tokens_text=tokens_text,
            mtmd_input_chunk_get_n_tokens=lambda chunk: chunk.count,
            mtmd_helper_eval_chunk_single=eval_image,
            mtmd_input_chunks_free=Mock(), mtmd_bitmap_free=Mock(),
        )
        handler._mtmd_cpp = native
        spans = [(0, 128, chunks[0], text_kind, None),
                 (128, 2176, chunks[1], image_kind, -100),
                 (2176, 2177, chunks[2], text_kind, None)]
        handler._process_mtmd_prompt = Mock(side_effect=lambda **kwargs: (
            [1] * 128 + [-100] * 2048 + [2], spans, object(), [object()],
        ))
        backend = configured_backend(llm, handler)
        try:
            with patch.object(gguf_backend, "tqdm", return_value=Mock()), self.assertLogs(level="WARNING"):
                self.assertEqual(run_backend(backend), ("safe",))
            self.assertEqual(llm.physical_tokens, 2177)
            self.assertEqual(len(llm.completion_args["prompt"]), 177)
            self.assertEqual(llm.completion_args["max_tokens"], 6015)
            self.assertEqual(native.mtmd_input_chunks_free.call_count, 2)
            self.assertEqual(native.mtmd_bitmap_free.call_count, 2)
        finally:
            handler.mtmd_ctx = None


if __name__ == "__main__":
    unittest.main()

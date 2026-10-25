"""Vocabulary handle for a segment whose weights live in the keep-head runner."""

class LlamaVocabulary:
    def __init__(self, path):
        from llama_cpp import llama_cpp
        from llama_cpp._internals import LlamaModel
        params = llama_cpp.llama_model_default_params()
        params.vocab_only = True
        params.n_gpu_layers = 0
        self._model = LlamaModel(path_model=path, params=params, verbose=False)
        self.metadata = self._model.metadata()

    def tokenize(self, text, add_bos=True, special=False):
        return self._model.tokenize(text, add_bos=add_bos, special=special)

    def detokenize(self, tokens, special=False):
        return self._model.detokenize(tokens, special=special)

    def token_eos(self):
        return self._model.token_eos()

    def close(self):
        self._model.close()

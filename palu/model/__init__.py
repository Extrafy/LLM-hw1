#llama
from .svd_llama import (
    PaluLlamaConfig,
    PaluLlamaForCausalLM
)

#mistral
from .svd_mistral import (
    PaluMistralConfig,
    PaluMistralForCausalLM
)

# Qwen2 was added after the pinned transformers 4.37.2 release.
import transformers
if hasattr(transformers, "Qwen2ForCausalLM"):
    from .svd_qwen import PaluQwen2Config, PaluQwen2ForCausalLM

#modules
from .modules import (
    HeadwiseLowRankModule
)

#TODO Mistral



AVAILABLE_MODELS = {
    'llama': {
        'config': PaluLlamaConfig,
        'ModelForCausalLM': PaluLlamaForCausalLM
    },
    'mistral': {
        'config': PaluMistralConfig,
        'ModelForCausalLM': PaluMistralForCausalLM
    },
}
if hasattr(transformers, "Qwen2ForCausalLM"):
    AVAILABLE_MODELS['qwen2'] = {
        'config': PaluQwen2Config,
        'ModelForCausalLM': PaluQwen2ForCausalLM,
    }



# OpenAI compatible API base address
# Official OpenAI: https://api.openai.com
# Locally deployed model service (such as llama.cpp/vllm): for example http://localhost:8000/v1
# Third-party services (such as Alibaba Cloud Tongyi Qianwen, Baidu Wenxinyiyan and other compatible interfaces): fill in according to the service document
OPENAI_BASE_URL = ""


# API access key
# Official OpenAI source: Get API keys from https://platform.openai.com/account/api-keys
# Third-party services: Obtained from the console of the corresponding platform.
# Local service (if no key is required):
OPENAI_API_KEY = ""  # your API key


# Default model name
# Must be compatible with models supported by the API service (e.g., officially supported gpt-3.5-turbo, gpt-4, etc.)
DEFAULT_MODEL = "gpt-3.5-turbo"

# Maximum number of tokens to generate a response by default
# Note: Different models have maximum context limits (e.g., gpt-3.5-turbo defaults to 4096 tokens).
DEFAULT_MAX_TOKENS = 512



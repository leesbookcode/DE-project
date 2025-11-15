# config.py
# 注意：此文件包含敏感信息（如API密钥），请勿提交到代码仓库（已在.gitignore中忽略）

# OpenAI兼容API的基础地址
# 官方OpenAI：https://api.openai.com
# 本地部署的模型服务（如llama.cpp/vllm）：例如 http://localhost:8000/v1
# 第三方服务（如阿里云通义千问、百度文心一言等兼容接口）：根据服务文档填写
OPENAI_BASE_URL = ""

# API访问密钥
# 官方OpenAI：从 https://platform.openai.com/account/api-keys 获取
# 第三方服务：从对应平台的控制台获取
# 本地服务（如无需密钥）：可设为空字符串 ""
OPENAI_API_KEY = ""  # 替换为你的真实API密钥

# 默认使用的模型名称
# 需与API服务支持的模型匹配（如官方支持 gpt-3.5-turbo、gpt-4 等）
DEFAULT_MODEL = "gpt-3.5-turbo"

# 默认生成回复的最大令牌数（tokens）
# 注意：不同模型有最大上下文限制（如gpt-3.5-turbo默认4096 tokens）
DEFAULT_MAX_TOKENS = 4096

# 新增：知识库存放路径
KNOWLEDGE_BASE_PATH = "./knowledge_base"  # 项目根目录下的knowledge_base文件夹
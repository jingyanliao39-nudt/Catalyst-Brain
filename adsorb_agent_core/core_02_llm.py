import os
from langchain_openai import ChatOpenAI 
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI as OpenAI_Claude
from project_paths import PROJECT_ROOT, env_key

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

openapi_key = env_key("OPENAI_API_KEY")
anthropic_key = env_key("ANTHROPIC_API_KEY")
deepseek_key = env_key("DEEPSEEK_API_KEY")
qwen_key = env_key("QWEN_API_KEY")
gpt_5_key = env_key("GPT5_API_KEY", openapi_key)
Gemini_3_key = env_key("GEMINI_API_KEY")
cloude_key = env_key("CLAUDE_API_KEY", anthropic_key)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OPENAI_API_KEY"] = openapi_key
os.environ['ANTHROPIC_API_KEY'] = anthropic_key
os.environ['DEEPSEEK_API_KEY'] = deepseek_key

def _init_llm(agent_settings):
    """Initialize LLM with optional timeout/retry settings.

    Avoids long hangs on network/API calls by honoring:
    - agent_settings.llm_timeout (seconds)
    - agent_settings.llm_max_retries
    """

    provider = agent_settings.get("provider")
    version = agent_settings.get("version")
    timeout = agent_settings.get("llm_timeout", 120)
    max_retries = agent_settings.get("llm_max_retries", 2)
    temperature = agent_settings.get("llm_temperature", 0)
    base_url = agent_settings.get("base_url")

    common_kwargs = {
        "timeout": timeout,
        "max_retries": max_retries,
        "temperature": temperature,
        "max_tokens": 4096,
    }

    if base_url:
        common_kwargs["base_url"] = base_url

    if provider == "openai":
        try:
            return ChatOpenAI(model=version, **common_kwargs)
        except TypeError:
            # Fallback in case some arguments are not supported by the installed version
            if base_url:
                 return ChatOpenAI(model=version, base_url=base_url)
            return ChatOpenAI(model=version)
    if provider == "anthropic":
        try:
            return ChatAnthropic(model=version, **common_kwargs)
        except TypeError:
            return ChatAnthropic(model=version)
    if provider == "deepseek":
        try:
            return ChatDeepSeek(model=version, **common_kwargs)
        except TypeError:
            return ChatDeepSeek(model=version)

    if provider == "gpt5":
        # Uses gpt_5_key with ChatOpenAI structure
        kwargs = common_kwargs.copy()
        kwargs["api_key"] = gpt_5_key
        try:
            return ChatOpenAI(model=version, **kwargs)
        except TypeError:
             # Strip unavailable kwargs if needed, though they generally support it
            return ChatOpenAI(model=version, api_key=gpt_5_key, base_url=base_url if base_url else None)

    if provider == "claude":
        kwargs = common_kwargs.copy()
        kwargs["api_key"] = cloude_key
        try:
            return OpenAI_Claude(model=version, **kwargs)
        except TypeError:
            return OpenAI_Claude(model=version, api_key=cloude_key, base_url=base_url if base_url else None)

    if provider == "gemini":
        # Uses Gemini_3_key with ChatOpenAI structure (for compatible endpoints)
        kwargs = common_kwargs.copy()
        kwargs["api_key"] = Gemini_3_key
        try:
            return ChatOpenAI(model=version, **kwargs)
        except TypeError:
            return ChatOpenAI(model=version, api_key=Gemini_3_key, base_url=base_url if base_url else None)

    if provider == "qwen":
        # Uses qwen_key with Dashscope OpenAI compatible endpoint
        kwargs = common_kwargs.copy()
        kwargs["api_key"] = qwen_key
        # Ensure we don't accidentally use huiyan-ai for qwen
        base_url_qwen = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        try:
            return ChatOpenAI(model=version, base_url=base_url_qwen, **kwargs)
        except TypeError:
            return ChatOpenAI(model=version, api_key=qwen_key, base_url=base_url_qwen)

    raise ValueError(f"Unknown provider: {provider}")

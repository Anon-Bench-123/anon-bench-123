"""
Shared Bedrock timeout monkey-patch.

Bedrock Converse API can take >60s for long responses (4096 tokens).
The default botocore read_timeout is 60s which causes ReadTimeoutError.
This patch increases it before inspect-ai creates the Bedrock client.
"""


def patch_bedrock_timeout(read_timeout: int = 600) -> None:
    """
    Patch the inspect-ai Bedrock provider to use a higher read timeout.

    Parameters
    ----------
    read_timeout : int
        Read timeout in seconds for Bedrock API calls. Default 600s (10 min).
    """
    from inspect_ai.model._providers import bedrock as bedrock_module

    _original_generate = bedrock_module.BedrockAPI.generate

    async def _patched_generate(self, *args, **kwargs):
        from botocore.config import Config

        _original_config_init = Config.__init__

        def _config_init_with_timeout(config_self, *c_args, **c_kwargs):
            c_kwargs.setdefault("read_timeout", read_timeout)
            c_kwargs.setdefault("connect_timeout", 60)
            return _original_config_init(config_self, *c_args, **c_kwargs)

        Config.__init__ = _config_init_with_timeout
        try:
            return await _original_generate(self, *args, **kwargs)
        finally:
            Config.__init__ = _original_config_init

    bedrock_module.BedrockAPI.generate = _patched_generate

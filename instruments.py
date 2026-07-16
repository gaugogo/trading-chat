"""
instruments.py — Trading instrument definitions

Mỗi instrument có cấu hình: symbol, decimals, spot URL, DeepSeek model, v.v.
Có thể override qua environment variables:
  - DEEPSEEK_MODEL: override model cho tất cả instruments
  - DEEPSEEK_THINKING: 0/1 để bật/tắt thinking cho tất cả
"""

import os

INSTRUMENTS = {
    "xau": {
        "id": "xau",
        "symbol": "GC=F",
        "symbol_encoded": "GC%3DF",
        "display_name": "XAUUSD (Gold)",
        "decimals": 2,
        "has_spot": True,
        "spot_url": "https://www.investing.com/currencies/xau-usd",
        "spot_label": "XAUUSD Spot",
        "prompt_instrument": "XAUUSD",
        "prompt_analyst_type": "forex and commodities",
        "deepseek_model": "deepseek-v4-pro",
        "deepseek_thinking": True,
        "has_smc": True,
        "file_prefix": "xauusd",
    },
    "btc": {
        "id": "btc",
        "symbol": "BTC-USD",
        "symbol_encoded": "BTC-USD",
        "display_name": "BTC/USD (Bitcoin)",
        "decimals": 2,
        "has_spot": False,
        "prompt_instrument": "BTC/USD",
        "prompt_analyst_type": "cryptocurrency",
        "deepseek_model": "deepseek-chat",
        "deepseek_thinking": False,
        "has_smc": True,
        "file_prefix": "btc",
    },
    "gbp": {
        "id": "gbp",
        "symbol": "GBPUSD=X",
        "symbol_encoded": "GBPUSD%3DX",
        "display_name": "GBP/USD (Cable)",
        "decimals": 5,
        "has_spot": True,
        "spot_url": "https://www.investing.com/currencies/gbp-usd",
        "spot_label": "GBP/USD Spot",
        "prompt_instrument": "GBP/USD",
        "prompt_analyst_type": "forex",
        "deepseek_model": "deepseek-chat",
        "deepseek_thinking": False,
        "has_smc": True,
        "file_prefix": "gbpusd",
    },
}


def apply_env_overrides() -> None:
    """Override instrument config from environment variables.

    Supports:
      DEEPSEEK_MODEL=deepseek-chat          → override model cho tất cả instruments
      DEEPSEEK_THINKING=1                   → override thinking cho tất cả
    """
    env_model = os.environ.get("DEEPSEEK_MODEL", "").strip()
    env_thinking = os.environ.get("DEEPSEEK_THINKING", "").strip()

    for instr_id, cfg in INSTRUMENTS.items():
        if env_model:
            old_model = cfg.get("deepseek_model", "unknown")
            cfg["deepseek_model"] = env_model
            if old_model != env_model:
                pass  # Would log if logger available
        if env_thinking == "1":
            cfg["deepseek_thinking"] = True
        elif env_thinking == "0":
            cfg["deepseek_thinking"] = False


# Apply overrides on import
apply_env_overrides()

"""
配置管理模块
支持YAML/JSON配置，热加载
"""

import json
import os
import time
from typing import Dict, Any, Optional
from pathlib import Path

CONFIG_DIR = Path(__file__).parent
CONFIG_FILE = CONFIG_DIR / 'config.json'


# 默认配置
DEFAULT_CONFIG = {
    # 服务器配置
    'server': {
        'host': '0.0.0.0',
        'port': 5002,
        'debug': False,
        'threaded': True,
    },
    
    # 缓存配置
    'cache': {
        'realtime': 10,        # 实时行情 TTL (秒)
        'technical': 60,       # 技术指标 TTL
        'fundamental': 300,    # 基本面 TTL
        'industry': 600,       # 行业数据 TTL
        'kline': 300,          # K线数据 TTL
        'strategy': 120,       # 策略结果 TTL
    },
    
    # 数据源配置
    'datasources': {
        'tencent': {
            'enabled': True,
            'timeout': 10,
            'retry_count': 3,
        },
        'eastmoney': {
            'enabled': True,
            'timeout': 10,
            'retry_count': 3,
        },
    },
    
    # 策略配置
    'strategy': {
        'max_position': 100,
        'min_stop_loss': 5,
        'max_stop_loss': 20,
        'take_profit_ratio': 2,
        'trailing_stop_pct': 5,
        'atr_multiplier_stop': 2.0,
        'atr_multiplier_profit': 3.0,
        'kelly_fraction': 0.5,
        'kelly_max_position': 0.25,
        'time_stop_days': 60,
    },
    
    # 信号权重
    'signal_weights': {
        'support_bounce': 1.2,
        'volume_breakout': 1.0,
        'oversold_recovery': 0.8,
        'profit_taking': 1.3,
        'stop_loss': 1.5,
        'resistance_rejection': 0.9,
        'multi_cycle_resonance': 1.4,
        'trend_following': 1.2,
        'momentum_continuation': 1.1,
    },
    
    # 默认股票
    'default_stock': {
        'code': 'sz300620',
        'name': '光库科技',
        'industry': '光通信',
        'cost_basis': 120,
    },
    
    # 日志配置
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        'file': 'logs/stock_analyzer.log',
        'max_bytes': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5,
    },
    
    # 监控配置
    'monitor': {
        'interval': 60,
        'max_iterations': 0,  # 0 = 无限
        'alert_interval': 30,
    },
}


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self._config_file = config_file or str(CONFIG_FILE)
        self._config = dict(DEFAULT_CONFIG)
        self._last_load = 0
        self._hot_reload_interval = 30  # 30秒检查一次
    
    def load(self, force: bool = False) -> Dict:
        """加载配置"""
        now = time.time()
        if not force and now - self._last_load < self._hot_reload_interval:
            return self._config
        
        try:
            if os.path.exists(self._config_file):
                with open(self._config_file, 'r') as f:
                    user_config = json.load(f)
                # 合并配置
                self._merge_config(self._config, user_config)
        except Exception as e:
            print(f"[Config] 加载配置失败: {e}")
        
        self._last_load = now
        return self._config
    
    def save(self):
        """保存配置到文件"""
        try:
            with open(self._config_file, 'w') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Config] 保存配置失败: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        self.load()
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """设置配置项"""
        self.load()
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self.save()
    
    def _merge_config(self, base: Dict, override: Dict):
        """递归合并配置"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def reload(self):
        """强制重新加载配置"""
        self.load(force=True)


# 全局配置实例
config = ConfigManager()

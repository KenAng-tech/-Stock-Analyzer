"""
资金流向热力图生成模块
生成行业/板块资金流向热力图
"""

import random
from typing import Dict, List


class HeatmapGenerator:
    """热力图生成器"""
    
    def __init__(self):
        # 行业板块定义
        self.industries = {
            'AI硬件': ['半导体', '光模块', 'AI芯片', '服务器'],
            '新能源': ['光伏', '储能', '锂电池', '风电'],
            '科技': ['通信设备', '计算机', '软件服务', '物联网'],
            '金融': ['银行', '证券', '保险', '多元金融'],
            '医药': ['制药', '医疗器械', '生物制品', '中药'],
            '消费': ['食品饮料', '家电', '零售', '旅游'],
            '能源': ['石油石化', '煤炭', '电力', '氢能'],
            '制造': ['汽车', '机械', '化工', '建材'],
            '地产': ['房地产开发', '建筑装饰', '房地产服务'],
            '传媒': ['影视院线', '出版发行', '广告营销']
        }
        
        # 颜色映射
        self.color_map = {
            'strong_inflow': '#10b981',  # 强流入 - 绿色
            'moderate_inflow': '#34d399',  # 中流入 - 浅绿
            'neutral': '#fbbf24',  # 中性 - 黄色
            'moderate_outflow': '#f97316',  # 中流出 - 橙色
            'strong_outflow': '#ef4444'  # 强流出 - 红色
        }
    
    def generate_industry_heatmap(self) -> Dict:
        """生成行业资金流向热力图"""
        heatmap = {}
        
        for industry, sectors in self.industries.items():
            heatmap[industry] = {
                'total_flow': 0,
                'sectors': {}
            }
            
            for sector in sectors:
                # 模拟资金流向（-100 到 +100 亿元）
                flow = random.uniform(-100, 100)
                heatmap[industry]['total_flow'] += flow
                
                # 确定流向等级
                if flow > 50:
                    level = 'strong_inflow'
                elif flow > 10:
                    level = 'moderate_inflow'
                elif flow > -10:
                    level = 'neutral'
                elif flow > -50:
                    level = 'moderate_outflow'
                else:
                    level = 'strong_outflow'
                
                heatmap[industry]['sectors'][sector] = {
                    'flow': round(flow, 2),
                    'level': level,
                    'color': self.color_map[level]
                }
            
            # 行业整体流向
            total = heatmap[industry]['total_flow']
            if total > 100:
                heatmap[industry]['level'] = 'strong_inflow'
            elif total > 20:
                heatmap[industry]['level'] = 'moderate_inflow'
            elif total > -20:
                heatmap[industry]['level'] = 'neutral'
            elif total > -100:
                heatmap[industry]['level'] = 'moderate_outflow'
            else:
                heatmap[industry]['level'] = 'strong_outflow'
            
            heatmap[industry]['color'] = self.color_map[heatmap[industry]['level']]
        
        return heatmap
    
    def generate_sector_heatmap(self, stock_code: str) -> Dict:
        """生成个股所属板块热力图"""
        # 模拟数据
        sectors = ['主板', '创业板', '科创板', '中小板']
        heatmap = {}
        
        for sector in sectors:
            flow = random.uniform(-50, 100)
            
            if flow > 30:
                level = 'strong_inflow'
            elif flow > 10:
                level = 'moderate_inflow'
            elif flow > -10:
                level = 'neutral'
            elif flow > -30:
                level = 'moderate_outflow'
            else:
                level = 'strong_outflow'
            
            heatmap[sector] = {
                'flow': round(flow, 2),
                'level': level,
                'color': self.color_map[level]
            }
        
        return heatmap
    
    def get_heatmap_css(self) -> str:
        """获取热力图CSS样式"""
        return """
        .heatmap-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            padding: 15px;
        }
        
        .heatmap-item {
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            transition: transform 0.3s;
            cursor: pointer;
        }
        
        .heatmap-item:hover {
            transform: scale(1.05);
        }
        
        .heatmap-item.strong_inflow { background: #d1fae5; border: 2px solid #10b981; }
        .heatmap-item.moderate_inflow { background: #ecfdf5; border: 2px solid #34d399; }
        .heatmap-item.neutral { background: #fef3c7; border: 2px solid #fbbf24; }
        .heatmap-item.moderate_outflow { background: #ffedd5; border: 2px solid #f97316; }
        .heatmap-item.strong_outflow { background: #fee2e2; border: 2px solid #ef4444; }
        
        .heatmap-sector {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 5px;
            margin-top: 10px;
        }
        
        .heatmap-sector-item {
            padding: 5px;
            border-radius: 4px;
            font-size: 11px;
            text-align: center;
        }
        
        .heatmap-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .heatmap-value {
            font-size: 14px;
            font-weight: 700;
        }
        
        .heatmap-level {
            font-size: 11px;
            padding: 2px 6px;
            border-radius: 4px;
            background: rgba(0,0,0,0.1);
        }
        """

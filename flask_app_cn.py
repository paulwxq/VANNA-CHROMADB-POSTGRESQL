"""
Flask应用入口文件（中文版），使用直接中文提示词实现
"""
from vanna.flask import VannaFlaskApp
import app_config
from vanna_llm_factory import create_vanna_instance

# 检查中文提示词设置
if not hasattr(app_config, 'USE_CHINESE_PROMPTS'):
    print("添加USE_CHINESE_PROMPTS配置到app_config...")
    app_config.USE_CHINESE_PROMPTS = True
else:
    # 尊重ext_config.py中的设置，不强制覆盖
    print(f"使用ext_config.py中的USE_CHINESE_PROMPTS设置: {app_config.USE_CHINESE_PROMPTS}")

print(f"中文提示词状态: {'已启用' if app_config.USE_CHINESE_PROMPTS else '未启用'}")

# 使用直接实现版工厂函数创建Vanna实例
vn = create_vanna_instance()

app = VannaFlaskApp(
    vn,
    title="智能数据问答平台",
    subtitle="让 AI 为你写 SQL",
    chart=False,
    allow_llm_to_see_data=True
)

# 运行Flask应用
print("正在启动Flask应用...")
app.run(host="0.0.0.0", port=8084) 
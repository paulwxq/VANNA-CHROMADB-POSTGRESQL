from vanna.flask import VannaFlaskApp
from vanna_llm_factory import create_vanna_instance

vn = create_vanna_instance()

# 实例化 VannaFlaskApp
app = VannaFlaskApp(
    vn,
    title="智能数据问答平台",
    subtitle="让 AI 为你写 SQL",
    chart=True,
    allow_llm_to_see_data=True,
    ask_results_correct=True,
    followup_questions=True,
    debug=True
)

print("Flask应用正在启动.")
print("访问地址: http://localhost:8084")
app.run(host="0.0.0.0", port=8084, debug=True)
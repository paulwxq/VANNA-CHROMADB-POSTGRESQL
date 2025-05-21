from vanna.flask import VannaFlaskApp
from vanna_llm_factory import create_vanna_instance
from flask import request, jsonify
import pandas as pd

vn = create_vanna_instance()

# 实例化 VannaFlaskApp
app = VannaFlaskApp(
    vn,
    title="辞图智能数据问答平台",
    logo = "https://www.citupro.com/img/logo-black-2.png",
    subtitle="让 AI 为你写 SQL",
    chart=True,
    allow_llm_to_see_data=True,
    ask_results_correct=True,
    followup_questions=True,
    debug=True
)

# 添加一个API，给前端使用：
@app.flask_app.route('/api/v0/ask', methods=['POST'])
def ask_full():
    req = request.get_json(force=True)
    question = req.get("question", None)
    if not question:
        return jsonify({"type": "error", "error": "No question provided"}), 400

    sql, df, fig = vn.ask(
        question=question,
        print_results=False,
        visualize=True,
        allow_llm_to_see_data=True
    )

    rows, columns = [], []
    if isinstance(df, pd.DataFrame) and not df.empty:
        rows = df.head(1000).to_dict(orient="records")
        columns = list(df.columns)

    return jsonify({
        "sql": sql,
        "rows": rows,
        "columns": columns
    })


print("正在启动Flask应用...")
app.run(host="0.0.0.0", port=8084, debug=True)
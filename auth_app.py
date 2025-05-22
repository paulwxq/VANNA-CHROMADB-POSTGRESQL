from vanna.flask import VannaFlaskApp
from vanna_llm_factory import create_vanna_instance
from common import SimpleUserAuth
import flask

# 创建vanna实例
vn = create_vanna_instance()

# 创建认证实例
auth = SimpleUserAuth(
    users_file="users.json",
    create_default_admin=True
)

# 实例化 VannaFlaskApp 并添加认证
app = VannaFlaskApp(
    vn,
    title="智能数据问答平台",
    subtitle="让 AI 为你写 SQL",
    auth=auth,  # 添加认证
    chart=True,
    allow_llm_to_see_data=True,
    ask_results_correct=True,
    followup_questions=True,
    debug=True
)

# 添加用户管理路由（可选）
@app.flask_app.route('/admin/add_user', methods=['GET', 'POST'])
def add_user():
    """管理员添加用户页面"""
    # 检查当前用户是否为管理员
    current_user = auth.get_user(flask.request)
    if not current_user or current_user.get("role") != "admin":
        return "权限不足", 403
    
    if flask.request.method == 'POST':
        username = flask.request.form.get('username', '').strip()
        email = flask.request.form.get('email', '').strip()
        password = flask.request.form.get('password', '')
        role = flask.request.form.get('role', 'user')
        
        success, message = auth.add_user(username, email, password, role)
        if success:
            return f"<h2>成功</h2><p>{message}</p><a href='/admin/add_user'>继续添加</a> | <a href='/'>返回首页</a>"
        else:
            return f"<h2>错误</h2><p>{message}</p><a href='/admin/add_user'>重试</a>"
    
    # GET请求，显示添加用户表单
    return '''
    <div style="max-width: 500px; margin: 50px auto; padding: 20px;">
        <h2>添加新用户</h2>
        <form method="POST">
            <div style="margin-bottom: 15px;">
                <label>用户名:</label>
                <input type="text" name="username" required style="width: 100%; padding: 5px;">
            </div>
            <div style="margin-bottom: 15px;">
                <label>邮箱:</label>
                <input type="email" name="email" required style="width: 100%; padding: 5px;">
            </div>
            <div style="margin-bottom: 15px;">
                <label>密码:</label>
                <input type="password" name="password" required style="width: 100%; padding: 5px;">
            </div>
            <div style="margin-bottom: 15px;">
                <label>角色:</label>
                <select name="role" style="width: 100%; padding: 5px;">
                    <option value="user">普通用户</option>
                    <option value="admin">管理员</option>
                </select>
            </div>
            <button type="submit" style="background: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer;">
                添加用户
            </button>
        </form>
        <p><a href="/">返回首页</a></p>
    </div>
    '''

print("Flask应用正在启动...")
print("访问地址: http://localhost:8084")
print("默认管理员账户: admin / admin")
print("管理员可以访问: http://localhost:8084/admin/add_user 添加新用户")
app.run(host="0.0.0.0", port=8084, debug=True)
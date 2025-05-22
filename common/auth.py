# auth.py - 用户认证模块
from vanna.flask.auth import AuthInterface
import flask
import hashlib
import json
import os

class SimpleUserAuth(AuthInterface):
    """简单的用户认证系统"""
    
    def __init__(self, users_file="users.json", create_default_admin=True):
        """
        初始化认证系统
        
        Args:
            users_file: 用户数据文件路径
            create_default_admin: 是否创建默认管理员账户
        """
        self.users_file = users_file
        self.users = self._load_users()
        
        # 如果没有用户且需要创建默认管理员，则创建一个
        if create_default_admin and not self.users:
            self._create_default_admin()
    
    def _load_users(self):
        """从文件加载用户数据"""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载用户文件出错: {e}")
                return []
        return []
    
    def _save_users(self):
        """保存用户数据到文件"""
        try:
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump(self.users, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存用户文件出错: {e}")
    
    def _hash_password(self, password):
        """密码哈希"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _create_default_admin(self):
        """创建默认管理员账户"""
        admin_user = {
            "username": "admin",
            "email": "admin@example.com", 
            "password": self._hash_password("admin"),
            "role": "admin",
            "active": True
        }
        self.users.append(admin_user)
        self._save_users()
        print("已创建默认管理员账户 - 用户名: admin, 密码: admin")
    
    def get_user(self, flask_request) -> any:
        """从请求中获取当前用户"""
        username = flask_request.cookies.get('user')
        if username:
            # 查找用户
            for user in self.users:
                if user.get("username") == username:
                    return user
        return None
    
    def is_logged_in(self, user: any) -> bool:
        """检查用户是否已登录"""
        return user is not None and user.get("active", False)
    
    def override_config_for_user(self, user: any, config: dict) -> dict:
        """根据用户角色覆盖配置"""
        if user and user.get("role") == "admin":
            # 管理员有更多权限
            config["show_training_data"] = True
            config["allow_sql_editing"] = True
        else:
            # 普通用户限制权限
            config["show_training_data"] = False
            config["allow_sql_editing"] = False
        return config
    
    def login_form(self) -> str:
        """返回登录表单HTML"""
        return '''
        <div class="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
            <div class="max-w-md w-full space-y-8">
                <div>
                    <h2 class="mt-6 text-center text-3xl font-extrabold text-gray-900">
                        登录到智能数据问答平台
                    </h2>
                    <p class="mt-2 text-center text-sm text-gray-600">
                        请输入您的凭据以访问系统
                    </p>
                </div>
                <form class="mt-8 space-y-6" action="/auth/login" method="POST">
                    <div class="rounded-md shadow-sm -space-y-px">
                        <div>
                            <label for="username" class="sr-only">用户名</label>
                            <input id="username" name="username" type="text" required 
                                   class="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-t-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm" 
                                   placeholder="用户名">
                        </div>
                        <div>
                            <label for="password" class="sr-only">密码</label>
                            <input id="password" name="password" type="password" required 
                                   class="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-b-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm" 
                                   placeholder="密码">
                        </div>
                    </div>

                    <div class="flex justify-end space-x-2 mt-6">
                        <button type="reset" 
                                class="py-2 px-4 border border-gray-300 text-sm font-normal rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500">
                            重置
                        </button>
                        <button type="submit" 
                                class="py-2 px-4 border border-transparent text-sm font-normal rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
                            登录
                        </button>
                    </div>
                </form>
            </div>
        </div>
        '''
    
    def login_handler(self, flask_request) -> str:
        """处理登录请求"""
        username = flask_request.form.get('username', '').strip()
        password = flask_request.form.get('password', '')
        
        if not username or not password:
            return self._login_error("用户名和密码不能为空")
        
        # 验证用户凭据
        hashed_password = self._hash_password(password)
        for user in self.users:
            if (user.get("username") == username and 
                user.get("password") == hashed_password and 
                user.get("active", False)):
                
                # 登录成功，设置cookie并重定向
                response = flask.make_response()
                response.set_cookie('user', username, max_age=24*60*60)  # 24小时有效
                response.headers['Location'] = '/'
                response.status_code = 302
                return response
        
        return self._login_error("用户名或密码错误")
    
    def _login_error(self, error_msg):
        """返回登录错误页面"""
        return f'''
        <div class="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
            <div class="max-w-md w-full space-y-8">
                <div class="text-center">
                    <h2 class="text-2xl font-bold text-red-600">登录失败</h2>
                    <p class="mt-2 text-gray-600">{error_msg}</p>
                    <a href="/auth/login" class="mt-4 inline-block bg-indigo-600 text-white px-4 py-2 rounded hover:bg-indigo-700">
                        重新登录
                    </a>
                </div>
            </div>
        </div>
        '''
    
    def logout_handler(self, flask_request) -> str:
        """处理登出请求"""
        response = flask.make_response()
        response.delete_cookie('user')
        response.headers['Location'] = '/auth/login'
        response.status_code = 302
        return response
    
    def callback_handler(self, flask_request) -> str:
        """处理认证回调请求
        
        由于我们使用的是简单的用户名/密码认证，这个方法可能不会被调用
        但作为抽象方法，我们必须实现它
        """
        # 简单实现，返回重定向到登录页面
        response = flask.make_response()
        response.headers['Location'] = '/auth/login'
        response.status_code = 302
        return response
    
    def add_user(self, username, email, password, role="user"):
        """添加新用户"""
        # 检查用户名是否已存在
        for user in self.users:
            if user.get("username") == username:
                return False, "用户名已存在"
        
        # 添加新用户
        new_user = {
            "username": username,
            "email": email,
            "password": self._hash_password(password),
            "role": role,
            "active": True
        }
        self.users.append(new_user)
        self._save_users()
        return True, "用户添加成功" 
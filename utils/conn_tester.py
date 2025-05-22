"""
连接测试工具模块
提供测试各种数据库和模型连接的工具函数
"""
import os
import sys
from pathlib import Path
import time
from typing import Dict, Any, Optional, Tuple, Union, List

# 添加项目根目录到路径
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
sys.path.append(project_root)

import app_config

def test_embedding_connection() -> Dict[str, Any]:
    """
    测试嵌入模型连接和配置是否正确
    
    Returns:
        dict: 测试结果，包括成功/失败状态、错误消息等
    """
    try:
        from embedding_function import get_embedding_function
        
        # 获取嵌入函数实例
        print("正在测试嵌入模型连接...")
        embedding_function = get_embedding_function()
        
        # 测试连接
        test_result = embedding_function.test_connection("测试连接")
        
        if test_result["success"]:
            print(f"嵌入模型连接测试成功!")
            if "警告" in test_result.get("message", ""):
                print(test_result["message"])
                print(f"建议将app_config.py中的EMBEDDING_CONFIG['embedding_dimension']修改为{test_result['actual_dimension']}")
        else:
            print(f"嵌入模型连接测试失败: {test_result['message']}")
            
        return test_result
        
    except Exception as e:
        error_message = f"无法测试嵌入模型连接: {str(e)}"
        print(error_message)
        return {
            "success": False,
            "message": error_message
        }

def test_vector_db_connection() -> Dict[str, Any]:
    """
    测试向量数据库连接
    支持ChromaDB和PgVector
    
    Returns:
        dict: 测试结果，包括成功/失败状态
    """
    result = {
        "success": False,
        "db_type": "",
        "message": "",
        "details": {}
    }
    
    try:
        # 检查向量数据库类型
        vector_db_type = app_config.VECTOR_DB_TYPE.lower()
        result["db_type"] = vector_db_type
        
        print(f"正在测试{vector_db_type}向量数据库连接...")
        
        if vector_db_type == "chromadb":
            # ChromaDB测试逻辑
            try:
                import chromadb
                result["details"]["version"] = chromadb.__version__
                
                # 检查ChromaDB文件
                chromadb_file = check_chromadb_file()
                result["details"]["file_exists"] = chromadb_file["exists"]
                result["details"]["file_path"] = chromadb_file["path"]
                result["details"]["file_size"] = chromadb_file["size"]
                
                # 尝试初始化客户端
                client = chromadb.PersistentClient(path=project_root)
                collections = client.list_collections()
                result["details"]["collections"] = len(collections)
                result["details"]["collections_names"] = [c.name for c in collections]
                
                result["success"] = True
                result["message"] = "ChromaDB连接成功"
                print(f"ChromaDB连接测试成功，找到{len(collections)}个集合")
                
            except Exception as e:
                result["success"] = False
                result["message"] = f"ChromaDB连接失败: {str(e)}"
                print(result["message"])
        
        elif vector_db_type == "pgvector":
            # PgVector测试逻辑
            try:
                from sqlalchemy import create_engine, text
                
                # 从配置中获取连接信息
                db_cfg = app_config.PGVECTOR_CONFIG
                connection_string = (
                    f"postgresql://{db_cfg['user']}:{db_cfg['password']}@"
                    f"{db_cfg['host']}:{db_cfg['port']}/{db_cfg['dbname']}"
                )
                
                result["details"]["host"] = db_cfg["host"]
                result["details"]["port"] = db_cfg["port"]
                result["details"]["dbname"] = db_cfg["dbname"]
                
                # 尝试连接数据库
                engine = create_engine(connection_string)
                with engine.connect() as conn:
                    # 检查pgvector扩展是否已安装
                    vector_ext = conn.execute(text("SELECT * FROM pg_extension WHERE extname = 'vector'")).fetchone()
                    result["details"]["vector_extension"] = bool(vector_ext)
                    
                    # 检查表是否存在
                    tables_query = text("""
                    SELECT tablename FROM pg_tables 
                    WHERE schemaname = 'public' 
                    AND tablename LIKE 'langchain_pg_%'
                    """)
                    tables = [row[0] for row in conn.execute(tables_query).fetchall()]
                    result["details"]["tables"] = tables
                    
                    if not vector_ext:
                        result["success"] = False
                        result["message"] = "PgVector数据库连接成功，但未找到vector扩展"
                        print(result["message"])
                    else:
                        result["success"] = True
                        result["message"] = "PgVector数据库连接成功"
                        print(f"PgVector数据库连接测试成功，找到表: {', '.join(tables)}")
            
            except Exception as e:
                result["success"] = False
                result["message"] = f"PgVector数据库连接失败: {str(e)}"
                print(result["message"])
        
        else:
            result["success"] = False
            result["message"] = f"未知的向量数据库类型: {vector_db_type}"
            print(result["message"])
    
    except Exception as e:
        result["success"] = False
        result["message"] = f"测试向量数据库连接时出错: {str(e)}"
        print(result["message"])
    
    return result

def test_llm_connection() -> Dict[str, Any]:
    """
    测试大模型连接
    支持DeepSeek和Qwen
    
    Returns:
        dict: 测试结果，包括成功/失败状态
    """
    result = {
        "success": False,
        "model_type": "",
        "message": "",
        "details": {}
    }
    
    try:
        # 检查模型类型
        model_type = app_config.MODEL_TYPE.lower()
        result["model_type"] = model_type
        
        print(f"正在测试{model_type}大模型连接...")
        
        if model_type == "deepseek":
            # DeepSeek模型测试
            config = app_config.DEEPSEEK_CONFIG.copy()
            result["details"]["model"] = config.get("model", "未指定")
            
            if not config.get("api_key"):
                result["success"] = False
                result["message"] = "DeepSeek API密钥未设置或为空"
                print(result["message"])
                return result
            
            # 执行简单测试
            from openai import OpenAI
            client = OpenAI(
                api_key=config["api_key"], 
                base_url="https://api.deepseek.com/v1"
            )
            
            response = client.chat.completions.create(
                model=config["model"],
                messages=[{"role": "user", "content": "Hello, this is a connection test."}],
                max_tokens=10
            )
            
            if response and hasattr(response, "choices") and len(response.choices) > 0:
                result["success"] = True
                result["message"] = "DeepSeek模型连接成功"
                result["details"]["response"] = response.choices[0].message.content
                print(f"DeepSeek模型连接测试成功")
            else:
                result["success"] = False
                result["message"] = "DeepSeek模型连接失败: 响应异常"
                print(result["message"])
            
        elif model_type == "qwen":
            # Qwen模型测试
            config = app_config.QWEN_CONFIG.copy()
            result["details"]["model"] = config.get("model", "未指定")
            
            if not config.get("api_key"):
                result["success"] = False
                result["message"] = "Qwen API密钥未设置或为空"
                print(result["message"])
                return result
            
            # 执行简单测试
            from openai import OpenAI
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            if "base_url" in config:
                base_url = config["base_url"]
                
            client = OpenAI(
                api_key=config["api_key"],
                base_url=base_url
            )
            
            response = client.chat.completions.create(
                model=config["model"],
                messages=[{"role": "user", "content": "Hello, this is a connection test."}],
                max_tokens=10
            )
            
            if response and hasattr(response, "choices") and len(response.choices) > 0:
                result["success"] = True
                result["message"] = "Qwen模型连接成功"
                result["details"]["response"] = response.choices[0].message.content
                print(f"Qwen模型连接测试成功")
            else:
                result["success"] = False
                result["message"] = "Qwen模型连接失败: 响应异常"
                print(result["message"])
        
        else:
            result["success"] = False
            result["message"] = f"未知的模型类型: {model_type}"
            print(result["message"])
    
    except Exception as e:
        result["success"] = False
        result["message"] = f"测试大模型连接时出错: {str(e)}"
        print(result["message"])
    
    return result

def test_app_db_connection() -> Dict[str, Any]:
    """
    测试应用数据库连接
    
    Returns:
        dict: 测试结果，包括成功/失败状态
    """
    result = {
        "success": False,
        "message": "",
        "details": {}
    }
    
    try:
        # 检查是否有应用数据库配置
        if not hasattr(app_config, "APP_DB_CONFIG"):
            result["success"] = False
            result["message"] = "应用数据库配置不存在(APP_DB_CONFIG未定义)"
            print(result["message"])
            return result
        
        db_config = app_config.APP_DB_CONFIG
        result["details"]["host"] = db_config["host"]
        result["details"]["port"] = db_config["port"]
        result["details"]["dbname"] = db_config["dbname"]
        
        print(f"正在测试应用数据库连接 {db_config['host']}:{db_config['port']}/{db_config['dbname']}...")
        
        # 尝试连接数据库
        from sqlalchemy import create_engine, text
        
        connection_string = (
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
        )
        
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            # 执行简单查询检查连接
            db_version = conn.execute(text("SELECT version();")).scalar()
            result["details"]["version"] = db_version
            
            # 获取表信息
            tables_query = text("""
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public'
            """)
            tables = [row[0] for row in conn.execute(tables_query).fetchall()]
            result["details"]["tables_count"] = len(tables)
            
            result["success"] = True
            result["message"] = "应用数据库连接成功"
            print(f"应用数据库连接测试成功，找到{len(tables)}个表")
    
    except Exception as e:
        result["success"] = False
        result["message"] = f"应用数据库连接失败: {str(e)}"
        print(result["message"])
    
    return result

def check_chromadb_file() -> Dict[str, Any]:
    """
    检查ChromaDB文件是否存在
    
    Returns:
        dict: 检查结果，包括是否存在、路径和大小
    """
    result = {
        "exists": False,
        "path": "",
        "size": 0
    }
    
    # ChromaDB默认文件名
    chroma_file = "chroma.sqlite3"
    
    # 使用项目根目录作为ChromaDB文件路径
    db_file_path = os.path.join(project_root, chroma_file)
    result["path"] = db_file_path
    
    if os.path.exists(db_file_path):
        file_size = os.path.getsize(db_file_path) / 1024  # KB
        result["exists"] = True
        result["size"] = file_size
        print(f"ChromaDB文件存在: {db_file_path} (大小: {file_size:.2f} KB)")
    else:
        print(f"ChromaDB文件不存在: {db_file_path}")
    
    return result

def test_all_connections() -> Dict[str, Dict[str, Any]]:
    """
    测试所有连接
    
    Returns:
        dict: 所有测试结果的集合
    """
    results = {}
    
    # 测试嵌入模型连接
    results["embedding"] = test_embedding_connection()
    
    # 测试向量数据库连接
    results["vector_db"] = test_vector_db_connection()
    
    # 测试大模型连接
    results["llm"] = test_llm_connection()
    
    # 测试应用数据库连接
    results["app_db"] = test_app_db_connection()
    
    # 检查ChromaDB文件
    results["chromadb_file"] = check_chromadb_file()
    
    return results

if __name__ == "__main__":
    """
    作为独立脚本运行时的入口点
    """
    print("\n===== 开始全面连接测试 =====\n")
    
    results = test_all_connections()
    
    print("\n===== 测试结果汇总 =====")
    
    # 打印嵌入模型测试结果
    embedding_result = results["embedding"]
    print(f"\n1. 嵌入模型连接: {'成功' if embedding_result['success'] else '失败'}")
    print(f"   消息: {embedding_result.get('message', '')}")
    
    # 打印向量数据库测试结果
    vector_db_result = results["vector_db"]
    print(f"\n2. 向量数据库({vector_db_result.get('db_type', '')})连接: {'成功' if vector_db_result['success'] else '失败'}")
    print(f"   消息: {vector_db_result.get('message', '')}")
    
    # 打印大模型测试结果
    llm_result = results["llm"]
    print(f"\n3. 大模型({llm_result.get('model_type', '')})连接: {'成功' if llm_result['success'] else '失败'}")
    print(f"   消息: {llm_result.get('message', '')}")
    
    # 打印应用数据库测试结果
    app_db_result = results["app_db"]
    print(f"\n4. 应用数据库连接: {'成功' if app_db_result['success'] else '失败'}")
    print(f"   消息: {app_db_result.get('message', '')}")
    
    # 打印ChromaDB文件检查结果
    chromadb_file_result = results["chromadb_file"]
    print(f"\n5. ChromaDB文件检查: {'存在' if chromadb_file_result['exists'] else '不存在'}")
    if chromadb_file_result["exists"]:
        print(f"   路径: {chromadb_file_result['path']}")
        print(f"   大小: {chromadb_file_result['size']:.2f} KB")
    
    print("\n===== 测试完成 =====") 
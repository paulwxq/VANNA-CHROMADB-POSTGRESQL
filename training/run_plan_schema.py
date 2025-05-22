# run_plan.py
import os
import sys
import argparse
from pathlib import Path
import pandas as pd

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app_config

# 从vanna_trainer.py导入所需函数
from vanna_trainer import (
    train_ddl,
    train_documentation,
    train_sql_example,
    train_question_sql_pair,
    flush_training,
    shutdown_trainer
)

def check_embedding_model_connection():
    """检查嵌入模型连接是否可用    
    如果无法连接到嵌入模型，则终止程序执行    
    Returns:
        bool: 连接成功返回True，否则终止程序
    """
    # 尝试导入新工具包
    try:
        from utils.conn_tester import test_embedding_connection
        print("正在使用连接测试工具检查嵌入模型连接...")
    except ImportError:
        # 回退到原始实现
        from embedding_function import test_embedding_connection
        print("正在检查嵌入模型连接...")
    
    # 使用测试函数进行连接测试
    test_result = test_embedding_connection()
    
    if test_result["success"]:
        print(f"可以继续训练过程。")
        return True
    else:
        print(f"\n错误: 无法连接到嵌入模型: {test_result['message']}")
        print("训练过程终止。请检查配置和API服务可用性。")
        sys.exit(1)

def run_training_plan():
    """
    执行Vanna的training plan功能
    
    1. 从业务数据库获取表结构信息
    2. 使用get_training_plan_generic生成训练计划
    3. 应用训练计划
    """
    print("\n===== 开始执行Vanna Training Plan =====")
    
    # 创建Vanna实例
    from vanna_llm_factory import create_vanna_instance
    vn = create_vanna_instance()
    
    try:
        # 获取数据库表结构信息
        print("\n===== 正在从数据库获取表结构信息 =====")
        # SELECT * FROM INFORMATION_SCHEMA.COLUMNS;
        df_information_schema = vn.run_sql("SELECT * FROM information_schema.columns WHERE table_schema = 'public';")
                
        
        if df_information_schema is None or df_information_schema.empty:
            print("错误: 无法获取数据库表结构信息")
            return False
            
        print(f"成功获取表结构信息，共 {len(df_information_schema)} 行")
        
        # 生成训练计划
        print("\n===== 正在生成训练计划 =====")
        plan = vn.get_training_plan_generic(df_information_schema)
        
        if plan is None:
            print("错误: 无法生成训练计划")
            return False
            
        print(f"成功生成训练计划")
        
        # 打印训练计划概要
        if hasattr(plan, '_plan') and hasattr(plan._plan, '__len__'):
            item_types = {}
            for item in plan._plan:
                item_type = getattr(item, 'item_type', 'unknown')
                if item_type in item_types:
                    item_types[item_type] += 1
                else:
                    item_types[item_type] = 1
            
            print("\n===== 训练计划概要 =====")
            for item_type, count in item_types.items():
                print(f"{item_type}: {count} 项")
        
        # 应用训练计划
        print("\n===== 正在应用训练计划 =====")
        vn.train(plan=plan)
        
        # 刷新和关闭批处理器
        print("\n===== 训练完成，处理剩余批次 =====")
        flush_training()
        shutdown_trainer()
        
        # 验证数据是否成功写入
        print("\n===== 验证训练数据 =====")
        training_data = vn.get_training_data()
        if training_data is not None and not training_data.empty:
            print(f"已从向量数据库中检索到 {len(training_data)} 条训练数据进行验证。")
        elif training_data is not None and training_data.empty:
            print("在向量数据库中未找到任何训练数据。")
        else: # training_data is None
            print("无法从Vanna获取训练数据 (可能返回了None)。请检查连接和Vanna实现。")
            
        return True
            
    except Exception as e:
        print(f"执行训练计划时出错: {e}")
        return False

def main():
    """主函数：配置和运行训练流程"""
   
    # 设置正确的项目根目录路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 检查嵌入模型连接
    check_embedding_model_connection()
    
    # 打印向量数据库相关信息
    try:
        # 检查向量数据库类型
        vector_db_type = app_config.VECTOR_DB_TYPE.lower()
        
        if vector_db_type == "chromadb":
            # ChromaDB相关检查逻辑
            try:
                import chromadb
                chroma_version = chromadb.__version__
            except ImportError:
                chroma_version = "未知"
            
            # 尝试查看当前使用的ChromaDB文件
            chroma_file = "chroma.sqlite3"  # 默认文件名
            
            # 使用项目根目录作为ChromaDB文件路径
            db_file_path = os.path.join(project_root, chroma_file)

            if os.path.exists(db_file_path):
                file_size = os.path.getsize(db_file_path) / 1024  # KB
                print(f"\n===== ChromaDB数据库: {os.path.abspath(db_file_path)} (大小: {file_size:.2f} KB) =====")
            else:
                print(f"\n===== 未找到ChromaDB数据库文件于: {os.path.abspath(db_file_path)} =====")
                
            # 打印ChromaDB版本
            print(f"===== ChromaDB客户端库版本: {chroma_version} =====\n")
        
        elif vector_db_type == "pgvector":
            # PgVector相关检查逻辑
            try:
                import psycopg
                from sqlalchemy import create_engine, text
                
                # 从配置中获取连接信息
                db_cfg = app_config.PGVECTOR_CONFIG
                connection_string = (
                    f"postgresql://{db_cfg['user']}:{db_cfg['password']}@"
                    f"{db_cfg['host']}:{db_cfg['port']}/{db_cfg['dbname']}"
                )
                
                # 尝试连接数据库
                engine = create_engine(connection_string)
                with engine.connect() as conn:
                    # 检查pgvector扩展是否已安装
                    result = conn.execute(text("SELECT * FROM pg_extension WHERE extname = 'vector'")).fetchone()
                    if result:
                        print(f"\n===== PgVector数据库连接成功 =====")
                        print(f"数据库: {db_cfg['host']}:{db_cfg['port']}/{db_cfg['dbname']}")
                        print(f"pgvector扩展已安装")
                    else:
                        print(f"\n===== 警告: PgVector数据库连接成功，但未找到vector扩展 =====")
                        print(f"请在PostgreSQL中安装pgvector扩展: CREATE EXTENSION vector;")
            except Exception as e:
                print(f"\n===== 无法连接到PgVector数据库: {e} =====")
                print(f"请检查app_config.py中的PGVECTOR_CONFIG配置")
        else:
            print(f"\n===== 未知的向量数据库类型: {vector_db_type} =====")
    except Exception as e:
        print(f"\n===== 无法获取向量数据库信息: {e} =====\n")
    
    # 执行训练计划
    success = run_training_plan()
    
    if success:
        print("\n===== Training Plan 执行成功 =====")
    else:
        print("\n===== Training Plan 执行失败 =====")
    
    # 输出embedding模型信息
    print("\n===== Embedding模型信息 =====")
    print(f"模型名称: {app_config.EMBEDDING_CONFIG.get('model_name')}")
    print(f"向量维度: {app_config.EMBEDDING_CONFIG.get('embedding_dimension')}")
    print(f"API服务: {app_config.EMBEDDING_CONFIG.get('base_url')}")
    # 打印向量数据库信息
    if vector_db_type == "chromadb":
        chroma_display_path = os.path.abspath(project_root)
        print(f"向量数据库: ChromaDB ({chroma_display_path})")
    else:
        print(f"向量数据库: PgVector ({app_config.PGVECTOR_CONFIG.get('host')}:{app_config.PGVECTOR_CONFIG.get('port')}/{app_config.PGVECTOR_CONFIG.get('dbname')})")
    print("===== 执行流程完成 =====\n")

if __name__ == "__main__":
    main() 
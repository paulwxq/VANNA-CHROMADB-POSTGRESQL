# run_plan_file.py
import os
import sys
import argparse
from pathlib import Path

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

def process_txt_files(data_path):
    """
    处理指定路径下的所有.txt文件，并构建训练计划
    
    Args:
        data_path (str): 包含.txt文件的目录路径
        
    Returns:
        tuple: (plan, success) - 训练计划和是否成功处理了至少一个文件
    """
    print(f"\n===== 扫描.txt文件目录: {os.path.abspath(data_path)} =====")
    
    # 检查目录是否存在
    if not os.path.exists(data_path):
        print(f"错误: 目录不存在: {data_path}")
        return None, False
    
    # 初始化TrainingPlan
    # 从vanna模块获取训练计划类型
    from vanna.types import TrainingPlan
    plan = TrainingPlan(plan=[])

    
    file_count = 0
    
    # 递归遍历目录中的所有文件
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith('.txt'):
                file_path = os.path.join(root, file)
                file_count += 1
                
                try:
                    # 读取.txt文件内容
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    if not content:
                        print(f"警告: 文件 {file_path} 为空，已跳过")
                        continue
                    
                    print(f"处理文件: {file_path}")
                    
                    # 分析文件名决定训练类型
                    if "ddl" in file.lower():
                        # 作为DDL处理
                        print(f"  将作为DDL处理")
                        plan.add_ddl(content)
                    elif "doc" in file.lower() or "document" in file.lower():
                        # 作为文档处理
                        print(f"  将作为文档处理")
                        plan.add_documentation(content)
                    else:
                        # 默认作为文档处理
                        print(f"  将作为通用文档处理")
                        plan.add_documentation(content)
                    
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")
    
    if file_count == 0:
        print(f"警告: 在目录 {data_path} 中未找到任何.txt文件")
        return None, False
    
    print(f"\n共找到 {file_count} 个.txt文件")
    return plan, file_count > 0

def run_training_plan_from_files():
    """
    从.txt文件构建并执行Vanna的training plan功能
    
    1. 从training/data目录读取.txt文件
    2. 构建训练计划
    3. 应用训练计划
    """
    print("\n===== 开始执行基于文件的Vanna Training Plan =====")
    
    # 创建Vanna实例
    from vanna_llm_factory import create_vanna_instance
    vn = create_vanna_instance()
    
    try:
        # 获取training/data目录路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(project_root, 'training', 'data')
        
        # 处理.txt文件并构建训练计划
        plan, success = process_txt_files(data_path)
        
        if plan is None:
            print("\n===== 未能从文件构建训练计划，训练过程终止 =====")
            return False
        
        if not success:
            print("\n===== 未能成功处理任何文件，训练过程终止 =====")
            return False
            
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
    success = run_training_plan_from_files()
    
    if success:
        print("\n===== 基于文件的Training Plan执行成功 =====")
    else:
        print("\n===== 基于文件的Training Plan执行失败 =====")
    
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
    # 检查training/data目录是否存在，如果不存在则创建
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'training', 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"已创建training/data目录: {data_dir}")
    
    main() 
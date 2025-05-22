# run_training.py
import os
import time
import re
import json
import sys
import requests
import pandas as pd
import argparse
from pathlib import Path
from sqlalchemy import create_engine


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
    from embedding_function import test_embedding_connection

    print("正在检查嵌入模型连接...")
    
    # 使用专门的测试函数进行连接测试
    test_result = test_embedding_connection()
    
    if test_result["success"]:
        print(f"可以继续训练过程。")
        return True
    else:
        print(f"\n错误: 无法连接到嵌入模型: {test_result['message']}")
        print("训练过程终止。请检查配置和API服务可用性。")
        sys.exit(1)

def read_file_by_delimiter(filepath, delimiter="---"):
    """通用读取：将文件按分隔符切片为多个段落"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    blocks = [block.strip() for block in content.split(delimiter) if block.strip()]
    return blocks

def read_markdown_file_by_sections(filepath):
    """专门用于Markdown文件：按标题(#、##、###)分割文档
    
    Args:
        filepath (str): Markdown文件路径
        
    Returns:
        list: 分割后的Markdown章节列表
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 确定文件是否为Markdown
    is_markdown = filepath.lower().endswith('.md') or filepath.lower().endswith('.markdown')
    
    if not is_markdown:
        # 非Markdown文件使用默认的---分隔
        return read_file_by_delimiter(filepath, "---")
    
    # 直接按照标题级别分割内容，处理#、##和###
    sections = []
    
    # 匹配所有级别的标题（#、##或###开头）
    header_pattern = r'(?:^|\n)((?:#|##|###)[^#].*?)(?=\n(?:#|##|###)[^#]|\Z)'
    all_sections = re.findall(header_pattern, content, re.DOTALL)
    
    for section in all_sections:
        section = section.strip()
        if section:
            sections.append(section)
    
    # 处理没有匹配到标题的情况
    if not sections and content.strip():
        sections = [content.strip()]
        
    return sections

def train_ddl_statements(ddl_file):
    """训练DDL语句
    Args:
        ddl_file (str): DDL文件路径
    """
    print(f"开始训练 DDL: {ddl_file}")
    if not os.path.exists(ddl_file):
        print(f"DDL 文件不存在: {ddl_file}")
        return
    for idx, ddl in enumerate(read_file_by_delimiter(ddl_file, ";"), start=1):
        try:
            print(f"\n DDL 训练 {idx}")
            train_ddl(ddl)
        except Exception as e:
            print(f"错误：DDL #{idx} - {e}")

def train_documentation_blocks(doc_file):
    """训练文档块
    Args:
        doc_file (str): 文档文件路径
    """
    print(f"开始训练 文档: {doc_file}")
    if not os.path.exists(doc_file):
        print(f"文档文件不存在: {doc_file}")
        return
    
    # 检查是否为Markdown文件
    is_markdown = doc_file.lower().endswith('.md') or doc_file.lower().endswith('.markdown')
    
    if is_markdown:
        # 使用Markdown专用分割器
        sections = read_markdown_file_by_sections(doc_file)
        print(f" Markdown文档已分割为 {len(sections)} 个章节")
        
        for idx, section in enumerate(sections, start=1):
            try:
                section_title = section.split('\n', 1)[0].strip()
                print(f"\n Markdown章节训练 {idx}: {section_title}")
                
                # 检查部分长度并提供警告
                if len(section) > 2000:
                    print(f" 章节 {idx} 长度为 {len(section)} 字符，接近API限制(2048)")
                
                train_documentation(section)
            except Exception as e:
                print(f" 错误：章节 #{idx} - {e}")
    else:
        # 非Markdown文件使用传统的---分隔
        for idx, doc in enumerate(read_file_by_delimiter(doc_file, "---"), start=1):
            try:
                print(f"\n 文档训练 {idx}")
                train_documentation(doc)
            except Exception as e:
                print(f" 错误：文档 #{idx} - {e}")

def train_sql_examples(sql_file):
    """训练SQL示例
    Args:
        sql_file (str): SQL示例文件路径
    """
    print(f" 开始训练 SQL 示例: {sql_file}")
    if not os.path.exists(sql_file):
        print(f" SQL 示例文件不存在: {sql_file}")
        return
    for idx, sql in enumerate(read_file_by_delimiter(sql_file, ";"), start=1):
        try:
            print(f"\n SQL 示例训练 {idx}")
            train_sql_example(sql)
        except Exception as e:
            print(f" 错误：SQL #{idx} - {e}")

def train_question_sql_pairs(qs_file):
    """训练问答对
    Args:
        qs_file (str): 问答对文件路径
    """
    print(f" 开始训练 问答对: {qs_file}")
    if not os.path.exists(qs_file):
        print(f" 问答文件不存在: {qs_file}")
        return
    try:
        with open(qs_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for idx, line in enumerate(lines, start=1):
            if "::" not in line:
                continue
            question, sql = line.strip().split("::", 1)
            print(f"\n 问答训练 {idx}")
            train_question_sql_pair(question.strip(), sql.strip())
    except Exception as e:
        print(f" 错误：问答训练 - {e}")

def train_formatted_question_sql_pairs(formatted_file):
    """训练格式化的问答对文件
    支持两种格式：
    1. Question: xxx\nSQL: xxx (单行SQL)
    2. Question: xxx\nSQL:\nxxx\nxxx (多行SQL)
    
    Args:
        formatted_file (str): 格式化问答对文件路径
    """
    print(f" 开始训练 格式化问答对: {formatted_file}")
    if not os.path.exists(formatted_file):
        print(f" 格式化问答文件不存在: {formatted_file}")
        return
    
    # 读取整个文件内容
    with open(formatted_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 按双空行分割不同的问答对
    # 使用更精确的分隔符，避免误识别
    pairs = []
    blocks = content.split("\n\nQuestion:")
    
    # 处理第一块（可能没有前导的"\n\nQuestion:"）
    first_block = blocks[0]
    if first_block.strip().startswith("Question:"):
        pairs.append(first_block.strip())
    elif "Question:" in first_block:
        # 处理文件开头没有Question:的情况
        question_start = first_block.find("Question:")
        pairs.append(first_block[question_start:].strip())
    
    # 处理其余块
    for block in blocks[1:]:
        pairs.append("Question:" + block.strip())
    
    # 处理每个问答对
    successfully_processed = 0
    for idx, pair in enumerate(pairs, start=1):
        try:
            if "Question:" not in pair or "SQL:" not in pair:
                print(f" 跳过不符合格式的对 #{idx}")
                continue
                
            # 提取问题部分
            question_start = pair.find("Question:") + len("Question:")
            sql_start = pair.find("SQL:", question_start)
            
            if sql_start == -1:
                print(f" SQL部分未找到，跳过对 #{idx}")
                continue
                
            question = pair[question_start:sql_start].strip()
            
            # 提取SQL部分（支持多行）
            sql_part = pair[sql_start + len("SQL:"):].strip()
            
            # 检查是否存在下一个Question标记（防止解析错误）
            next_question = pair.find("Question:", sql_start)
            if next_question != -1:
                sql_part = pair[sql_start + len("SQL:"):next_question].strip()
            
            if not question or not sql_part:
                print(f" 问题或SQL为空，跳过对 #{idx}")
                continue
            
            # 训练问答对
            print(f"\n格式化问答训练 {idx}")
            print(f"问题: {question}")
            print(f"SQL: {sql_part}")
            train_question_sql_pair(question, sql_part)
            successfully_processed += 1
            
        except Exception as e:
            print(f" 错误：格式化问答训练对 #{idx} - {e}")
    
    print(f"格式化问答训练完成，共成功处理 {successfully_processed} 对问答（总计 {len(pairs)} 对）")

def train_json_question_sql_pairs(json_file):
    """训练JSON格式的问答对
    
    Args:
        json_file (str): JSON格式问答对文件路径
    """
    print(f" 开始训练 JSON格式问答对: {json_file}")
    if not os.path.exists(json_file):
        print(f" JSON问答文件不存在: {json_file}")
        return
    
    try:
        # 读取JSON文件
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 确保数据是列表格式
        if not isinstance(data, list):
            print(f" 错误: JSON文件格式不正确，应为问答对列表")
            return
            
        successfully_processed = 0
        for idx, pair in enumerate(data, start=1):
            try:
                # 检查问答对格式
                if not isinstance(pair, dict) or "question" not in pair or "sql" not in pair:
                    print(f" 跳过不符合格式的对 #{idx}")
                    continue
                
                question = pair["question"].strip()
                sql = pair["sql"].strip()
                
                if not question or not sql:
                    print(f" 问题或SQL为空，跳过对 #{idx}")
                    continue
                
                # 训练问答对
                print(f"\n JSON格式问答训练 {idx}")
                print(f"问题: {question}")
                print(f"SQL: {sql}")
                train_question_sql_pair(question, sql)
                successfully_processed += 1
                
            except Exception as e:
                print(f" 错误：JSON问答训练对 #{idx} - {e}")
        
        print(f"JSON格式问答训练完成，共成功处理 {successfully_processed} 对问答（总计 {len(data)} 对）")
        
    except json.JSONDecodeError as e:
        print(f" 错误：JSON解析失败 - {e}")
    except Exception as e:
        print(f" 错误：处理JSON问答训练 - {e}")

def process_training_files(data_path):
    """处理指定路径下的所有训练文件
    
    Args:
        data_path (str): 训练数据目录路径
    """
    print(f"\n===== 扫描训练数据目录: {os.path.abspath(data_path)} =====")
    
    # 检查目录是否存在
    if not os.path.exists(data_path):
        print(f"错误: 训练数据目录不存在: {data_path}")
        return False
    
    # 初始化统计计数器
    stats = {
        "ddl": 0,
        "documentation": 0,
        "sql_example": 0,
        "question_sql_formatted": 0,
        "question_sql_json": 0
    }
    
    # 递归遍历目录中的所有文件
    for root, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_lower = file.lower()
            
            # 根据文件类型调用相应的处理函数
            try:
                if file_lower.endswith(".ddl"):
                    print(f"\n处理DDL文件: {file_path}")
                    train_ddl_statements(file_path)
                    stats["ddl"] += 1
                    
                elif file_lower.endswith(".md") or file_lower.endswith(".markdown"):
                    print(f"\n处理文档文件: {file_path}")
                    train_documentation_blocks(file_path)
                    stats["documentation"] += 1
                    
                elif file_lower.endswith("_pair.json") or file_lower.endswith("_pairs.json"):
                    print(f"\n处理JSON问答对文件: {file_path}")
                    train_json_question_sql_pairs(file_path)
                    stats["question_sql_json"] += 1
                    
                elif file_lower.endswith("_sql_pair.sql") or file_lower.endswith("_sql_pairs.sql"):
                    print(f"\n处理格式化问答对文件: {file_path}")
                    train_formatted_question_sql_pairs(file_path)
                    stats["question_sql_formatted"] += 1
                    
                elif file_lower.endswith(".sql") and not (file_lower.endswith("_sql_pair.sql") or file_lower.endswith("_sql_pairs.sql")):
                    print(f"\n处理SQL示例文件: {file_path}")
                    train_sql_examples(file_path)
                    stats["sql_example"] += 1
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
    
    # 打印处理统计
    print("\n===== 训练文件处理统计 =====")
    print(f"DDL文件: {stats['ddl']}个")
    print(f"文档文件: {stats['documentation']}个")
    print(f"SQL示例文件: {stats['sql_example']}个")
    print(f"格式化问答对文件: {stats['question_sql_formatted']}个")
    print(f"JSON问答对文件: {stats['question_sql_json']}个")
    
    total_files = sum(stats.values())
    if total_files == 0:
        print(f"警告: 在目录 {data_path} 中未找到任何可训练的文件")
        return False
        
    return True

def main():
    """主函数：配置和运行训练流程"""
    
    # 先导入所需模块
    import os
    import app_config
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练Vanna NL2SQL模型')
    parser.add_argument('--data_path', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
                        help='训练数据目录路径 (默认: training/data)')
    args = parser.parse_args()
    
    # 使用Path对象处理路径以确保跨平台兼容性
    data_path = Path(args.data_path)
    
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
    
    # 处理训练文件
    process_successful = process_training_files(data_path)
    
    if process_successful:
        # 训练结束，刷新和关闭批处理器
        print("\n===== 训练完成，处理剩余批次 =====")
        flush_training()
        shutdown_trainer()
        
        # 验证数据是否成功写入
        print("\n===== 验证训练数据 =====")
        from vanna_llm_factory import create_vanna_instance
        vn = create_vanna_instance()
        
        # 根据向量数据库类型执行不同的验证逻辑
        # 由于已确定只使用ChromaDB，简化这部分逻辑
        try:
            training_data = vn.get_training_data()
            if training_data is not None and not training_data.empty:
                # get_training_data 内部通常会打印数量，这里可以补充一个总结
                print(f"已从ChromaDB中检索到 {len(training_data)} 条训练数据进行验证。")
            elif training_data is not None and training_data.empty:
                 print("在ChromaDB中未找到任何训练数据。")
            else: # training_data is None
                print("无法从Vanna获取训练数据 (可能返回了None)。请检查连接和Vanna实现。")

        except Exception as e:
            print(f"验证训练数据失败: {e}")
            print("请检查ChromaDB连接和表结构。")
    else:
        print("\n===== 未能找到或处理任何训练文件，训练过程终止 =====")
    
    # 输出embedding模型信息
    print("\n===== Embedding模型信息 =====")
    print(f"模型名称: {app_config.EMBEDDING_CONFIG.get('model_name')}")
    print(f"向量维度: {app_config.EMBEDDING_CONFIG.get('embedding_dimension')}")
    print(f"API服务: {app_config.EMBEDDING_CONFIG.get('base_url')}")
    # 打印ChromaDB路径信息
    chroma_display_path = os.path.abspath(project_root)
    print(f"向量数据库: ChromaDB ({chroma_display_path})")
    print("===== 训练流程完成 =====\n")

if __name__ == "__main__":
    main() 
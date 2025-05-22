# reset_vectordb.py
import os
import sys
import argparse
from pathlib import Path
from sqlalchemy import create_engine, text

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app_config

def reset_pgvector_tables():
    """
    重置(清空)PgVector数据库中的embedding表
    
    Returns:
        bool: 操作是否成功
    """
    # 检查向量数据库类型
    if not hasattr(app_config, 'VECTOR_DB_TYPE') or app_config.VECTOR_DB_TYPE.lower() != 'pgvector':
        print("错误: 当前配置的向量数据库类型不是pgvector")
        print(f"当前配置: {getattr(app_config, 'VECTOR_DB_TYPE', '未设置')}")
        return False
    
    # 获取PgVector连接信息
    if not hasattr(app_config, 'PGVECTOR_CONFIG'):
        print("错误: 缺少PGVECTOR_CONFIG配置")
        return False
    
    try:
        # 从配置中获取连接信息
        db_cfg = app_config.PGVECTOR_CONFIG
        connection_string = (
            f"postgresql://{db_cfg['user']}:{db_cfg['password']}@"
            f"{db_cfg['host']}:{db_cfg['port']}/{db_cfg['dbname']}"
        )
        
        # 显示将要操作的数据库
        print(f"\n===== 即将重置PgVector数据库中的embedding表 =====")
        print(f"数据库: {db_cfg['host']}:{db_cfg['port']}/{db_cfg['dbname']}")
        print(f"表: langchain_pg_embedding")
        
        # 创建数据库连接
        print("正在连接到数据库...")
        engine = create_engine(connection_string)
        
        with engine.connect() as conn:
            # 开始事务
            with conn.begin():
                # 检查表是否存在
                print("检查表是否存在...")
                check_query = text("""
                SELECT EXISTS (
                    SELECT FROM pg_tables 
                    WHERE schemaname = 'public' 
                    AND tablename = 'langchain_pg_embedding'
                )
                """)
                result = conn.execute(check_query).scalar()
                
                if not result:
                    print("警告: langchain_pg_embedding表不存在，跳过清空操作")
                    return False
                
                # 查找外键约束
                print("查找外键约束...")
                constraint_query = text("""
                SELECT conname
                FROM pg_constraint
                WHERE conrelid = 'langchain_pg_embedding'::regclass
                AND contype = 'f'
                """)
                
                constraint_result = conn.execute(constraint_query).fetchall()
                constraint_exists = False
                
                # 检查是否找到约束
                if constraint_result:
                    for constraint in constraint_result:
                        constraint_name = constraint[0]
                        print(f"找到外键约束: {constraint_name}")
                        
                        # 如果是langchain_pg_embedding_collection_id_fkey约束，则删除
                        if constraint_name == 'langchain_pg_embedding_collection_id_fkey':
                            constraint_exists = True
                            print("正在删除外键约束: langchain_pg_embedding_collection_id_fkey...")
                            drop_constraint_query = text("""
                            ALTER TABLE langchain_pg_embedding
                            DROP CONSTRAINT langchain_pg_embedding_collection_id_fkey
                            """)
                            conn.execute(drop_constraint_query)
                            print("外键约束已删除")
                else:
                    print("未找到外键约束")
                
                # 清空langchain_pg_embedding表
                print("正在清空langchain_pg_embedding表...")
                truncate_query = text("TRUNCATE TABLE langchain_pg_embedding")
                conn.execute(truncate_query)
                print("langchain_pg_embedding表已清空")
                
                # 无论之前是否存在约束，都检查并确保约束存在
                print("检查外键约束是否存在...")
                check_constraint_query = text("""
                SELECT EXISTS (
                    SELECT FROM pg_constraint
                    WHERE conrelid = 'langchain_pg_embedding'::regclass
                    AND conname = 'langchain_pg_embedding_collection_id_fkey'
                    AND contype = 'f'
                )
                """)
                constraint_exists_now = conn.execute(check_constraint_query).scalar()
                
                if not constraint_exists_now:
                    print("外键约束不存在，正在创建...")
                    add_constraint_query = text("""
                    ALTER TABLE langchain_pg_embedding
                    ADD CONSTRAINT langchain_pg_embedding_collection_id_fkey
                    FOREIGN KEY (collection_id)
                    REFERENCES langchain_pg_collection(uuid)
                    ON DELETE CASCADE
                    """)
                    conn.execute(add_constraint_query)
                    print("外键约束已创建")
                else:
                    print("外键约束已存在，无需创建")
                
                print("所有操作已完成")
                return True
                
    except Exception as e:
        print(f"重置表时出错: {e}")
        return False

def main():
    """主函数：执行重置操作"""
    
    # 执行重置操作
    success = reset_pgvector_tables()
    
    if success:
        print("\n===== PgVector数据库embedding表重置成功 =====")
    else:
        print("\n===== PgVector数据库embedding表重置失败 =====")
    
if __name__ == "__main__":
    main() 
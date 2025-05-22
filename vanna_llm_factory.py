"""
Vanna LLM 工厂文件，支持 ChromaDB 和 PGVector，自动组合 LLM 和 VectorStore。
"""
from vanna.chromadb import ChromaDB_VectorStore
from customqianwen.Custom_QianwenAI_chat import QianWenAI_Chat
from customdeepseek.custom_deepseek_chat import DeepSeekChat
from custompgvector.custom_pgvector import PG_VectorStore
from embedding_function import get_embedding_function
import app_config
import os

def CustomVannaDynamic(vectorstore_cls, llm_cls):
    class _CustomVanna(vectorstore_cls, llm_cls):
        def __init__(self, config=None):
            vectorstore_cls.__init__(self, config=config)
            llm_cls.__init__(self, config=config)
    _CustomVanna.__name__ = f"CustomVanna_{vectorstore_cls.__name__}_{llm_cls.__name__}"
    return _CustomVanna

def create_vanna_instance(config_module=None):
    """
    工厂函数：根据配置创建并初始化一个Vanna实例，支持 ChromaDB 和 PGVector。
    """
    if config_module is None:
        config_module = app_config

    model_type = config_module.MODEL_TYPE.lower()
    vector_db_type = getattr(config_module, "VECTOR_DB_TYPE", "chromadb").lower()

    # 选择 LLM 类
    if model_type == "deepseek":
        config = config_module.DEEPSEEK_CONFIG.copy()
        llm_cls = DeepSeekChat
        if not config.get("api_key"):
            print(f"\n错误: DeepSeek API密钥未设置或为空")
            print(f"请在.env文件中设置DEEPSEEK_API_KEY环境变量")
            import sys; sys.exit(1)
        print(f"创建DeepSeek模型实例，使用模型: {config['model']}")
    elif model_type == "qwen":
        config = config_module.QWEN_CONFIG.copy()
        llm_cls = QianWenAI_Chat
        if not config.get("api_key"):
            print(f"\n错误: Qwen API密钥未设置或为空")
            print(f"请在.env文件中设置QWEN_API_KEY环境变量")
            import sys; sys.exit(1)
        print(f"创建Qwen模型实例，使用模型: {config['model']}")
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 选择向量数据库类与配置
    if vector_db_type == "chromadb":
        vectorstore_cls = ChromaDB_VectorStore
        project_root = os.path.dirname(os.path.abspath(__file__))
        config["path"] = project_root
        print(f"已配置使用ChromaDB作为向量数据库，路径：{project_root}")
    elif vector_db_type == "pgvector":
        vectorstore_cls = PG_VectorStore
        db_cfg = config_module.PGVECTOR_CONFIG
        connection_string = (
            f"postgresql://{db_cfg['user']}:{db_cfg['password']}@"
            f"{db_cfg['host']}:{db_cfg['port']}/{db_cfg['dbname']}"
        )
        config["connection_string"] = connection_string
        print(f"已配置使用PGVector作为向量数据库：{connection_string}")
    else:
        raise ValueError(f"不支持的向量数据库类型: {vector_db_type}")

    # 配置embedding function
    embedding_function = get_embedding_function()
    config["embedding_function"] = embedding_function
    print(f"已配置嵌入模型: {config_module.EMBEDDING_CONFIG['model_name']}, 维度: {config_module.EMBEDDING_CONFIG['embedding_dimension']}")

    # 动态组合实例化
    VannaClass = CustomVannaDynamic(vectorstore_cls, llm_cls)
    vn = VannaClass(config=config)

    # 连接业务数据库（用 kwargs，支持各种参数）
    if hasattr(config_module, "APP_DB_CONFIG"):
        vn.connect_to_postgres(**config_module.APP_DB_CONFIG)
        print(f"已连接到业务数据库: "
              f"{config_module.APP_DB_CONFIG['host']}:"
              f"{config_module.APP_DB_CONFIG['port']}/"
              f"{config_module.APP_DB_CONFIG['dbname']}")
    else:
        print("未配置业务数据库连接，跳过 connect_to_postgres。")

    return vn

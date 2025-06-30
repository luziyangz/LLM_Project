# RAG_cy 项目 SOP 文档
## 📋 项目概述
RAG_cy 是一个基于检索增强生成（Retrieval-Augmented Generation）技术的企业财报智能问答系统。该系统能够自动解析PDF格式的企业财报，构建向量数据库，并通过自然语言问答的方式为用户提供精准的财报信息检索服务。

## 🎯 项目背景
### 业务需求
- 信息检索效率低 ：传统的财报分析需要人工逐页查找信息，效率低下
- 数据处理复杂 ：财报文档结构复杂，包含大量表格、图表和文本信息
- 知识获取门槛高 ：需要专业知识才能准确理解和分析财报内容
- 实时性要求 ：投资决策需要快速获取关键财务指标和业务信息
### 技术挑战
- PDF文档解析的准确性和完整性
- 大规模文本的高效向量化存储
- 多模态信息（文本+表格）的统一处理
- 检索结果的相关性和准确性优化
## 🏗️ 系统架构设计
### 核心设计思想
1. 模块化架构 ：采用松耦合的模块化设计，每个组件职责单一，便于维护和扩展
2. 流水线处理 ：将复杂的RAG流程分解为多个阶段，支持并行处理和错误恢复
3. 多策略融合 ：结合向量检索、BM25检索和LLM重排序，提升检索精度
4. 可配置化 ：通过配置类统一管理系统参数，支持不同场景的灵活配置
### 系统流程图
```
PDF财报 → PDF解析 → Markdown转换 → 文本分
块 → 向量化 → 向量数据库
                                        
                            ↓
用户问题 → 问题理解 → 混合检索 → 结果重排 → 
上下文构建 → LLM生成 → 答案输出
```
## 🔧 核心组件详解
### 1. Pipeline.py - 主流程控制器
功能 ：统一管理整个RAG系统的执行流程

关键特性 ：

- 支持配置化的流程控制
- 自动文件格式转换（JSON→CSV）
- 并行PDF处理能力
- 灵活的输出文件命名策略
核心配置类 ：

```
@dataclass
class RunConfig:
    use_serialized_tables: bool = 
    False      # 是否使用序列化表格
    parent_document_retrieval: bool = 
    False  # 父文档检索
    use_vector_dbs: bool = 
    True             # 向量数据库检索
    llm_reranking: bool = 
    False             # LLM重排序
    top_n_retrieval: int = 
    10               # 检索结果数量
    parallel_requests: int = 
    1              # 并行请求数
    api_provider: str = 
    "dashscope"         # API提供商
    answering_model: str = 
    "qwen-turbo-latest" # 回答模型
```
### 2. pdf_mineru.py - PDF解析引擎
功能 ：将PDF财报转换为结构化的Markdown格式

技术特点 ：

- 基于MinerU API的云端解析服务
- 支持复杂表格和图表的准确识别
- 异步任务处理机制
- 自动文件下载和解压
### 3. text_splitter.py - 文本分块器
功能 ：将长文档智能分割为适合向量化的文本块

分块策略 ：

- 基于语义边界的智能分割
- 保持上下文连贯性
- 支持表格和文本的混合处理
- 元数据保留和传递
### 4. ingestion.py - 向量数据库构建器
功能 ：将文本块转换为向量并构建FAISS索引

核心技术 ：

```
class VectorDBIngestor:
    def _get_embeddings(self, text, 
    model="text-embedding-v1"):
        # 使用DashScope API获取文本嵌入向量
        # 支持批量处理和错误重试
        
    def create_faiss_index_from_reports
    (self, reports, output_dir):
        # 构建FAISS向量索引
        # 支持增量更新和持久化存储
```
### 5. questions_processing.py - 问题处理器
功能 ：处理用户问题并生成高质量答案

处理流程 ：

1. 问题理解 ：分析用户意图和关键信息
2. 混合检索 ：结合向量检索和BM25检索
3. 结果重排 ：使用LLM对检索结果进行重新排序
4. 上下文构建 ：组织相关信息形成完整上下文
5. 答案生成 ：调用大语言模型生成最终答案
### 6. retrieval.py - 检索引擎
功能 ：实现多种检索策略的统一接口

检索策略 ：

- 向量检索 ：基于语义相似度的密集检索
- BM25检索 ：基于关键词匹配的稀疏检索
- 混合检索 ：融合多种检索结果
- 父文档检索 ：返回包含更多上下文的父文档片段
## 🚀 部署和使用指南
### 环境准备
1. Python环境 ：Python 3.8+
2. 依赖安装 ：
```
pip install -r requirements_simple.txt
```
3. API配置 ：
```
# 在.env文件中配置
DASHSCOPE_API_KEY=your_dashscope_api_key
```
### 快速启动
1. 数据准备 ：
   
   - 将PDF财报放入 data/pdf_reports/ 目录
   - 准备问题文件 questions.json
   - 配置企业信息文件 subset.csv
2. 运行流水线 ：
```
from Pipeline import Pipeline, RunConfig
from pathlib import Path

# 配置运行参数
config = RunConfig(
    llm_reranking=True,
    parallel_requests=4,
    answering_model="qwen-turbo-latest"
)

# 创建流水线实例
pipeline = Pipeline(root_path=Path("./
data"), run_config=config)

# 执行完整流程
pipeline.export_reports_to_markdown('财
报文件.pdf')
pipeline.chunk_reports()
pipeline.create_vector_dbs()
pipeline.process_questions()
```
3. Web界面启动 ：
```
streamlit run src/streamlit_app.py
```
### 测试验证
```
# 运行测试套件
cd tests
run_tests.bat
```
## 📊 性能优化策略
### 1. 并行处理优化
- PDF解析支持多进程并行
- 向量化支持批量处理
- API请求支持并发控制
### 2. 内存管理优化
- 分批处理大文件
- 及时释放临时数据
- 使用生成器减少内存占用
### 3. 检索精度优化
- 多策略检索融合
- LLM重排序机制
- 父文档检索扩展上下文
## 🔍 关键代码解析
### 混合检索实现
```
class HybridRetriever:
    def retrieve(self, query: str, 
    top_k: int = 10):
        # 向量检索
        vector_results = self.
        vector_retriever.retrieve
        (query, top_k)
        
        # BM25检索
        bm25_results = self.
        bm25_retriever.retrieve(query, 
        top_k)
        
        # 结果融合
        combined_results = self.
        _combine_results(
            vector_results, bm25_results
        )
        
        # LLM重排序
        if self.llm_reranking:
            combined_results = self.
            _llm_rerank(
                query, combined_results
            )
            
        return combined_results[:top_k]
```
### 向量数据库构建
```
def create_faiss_index_from_reports
(self, reports, output_dir):
    all_embeddings = []
    all_metadata = []
    
    for report in tqdm(reports, 
    desc="Processing reports"):
        # 获取文本嵌入
        embeddings = self.
        _get_embeddings(report
        ['chunks'])
        all_embeddings.extend
        (embeddings)
        
        # 保存元数据
        for i, chunk in enumerate(report
        ['chunks']):
            metadata = {
                'company_name': report
                ['company_name'],
                'chunk_id': i,
                'content': chunk
            }
            all_metadata.append
            (metadata)
    
    # 构建FAISS索引
    embeddings_array = np.array
    (all_embeddings).astype('float32')
    index = faiss.IndexFlatIP
    (embeddings_array.shape[1])
    index.add(embeddings_array)
    
    # 保存索引和元数据
    faiss.write_index(index, str
    (output_dir / "index.faiss"))
    with open(output_dir / "metadata.
    pkl", 'wb') as f:
        pickle.dump(all_metadata, f)
```
## 🛠️ 故障排查指南
### 常见问题及解决方案
1. PDF解析失败
   
   - 检查API密钥配置
   - 验证网络连接
   - 确认PDF文件格式
2. 向量化错误
   
   - 检查文本编码格式
   - 验证API调用限制
   - 确认内存使用情况
3. 检索结果不准确
   
   - 调整检索参数
   - 优化文本分块策略
   - 启用LLM重排序
## 📈 扩展方向
### 短期优化
- 支持更多PDF解析引擎
- 增加更多向量模型选择
- 优化检索算法性能
### 长期规划
- 支持多模态信息检索
- 集成知识图谱技术
- 构建领域专用模型
- 实现实时数据更新

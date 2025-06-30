import streamlit as st
import json
import os
from pathlib import Path
import sys

# 添加src目录到Python路径
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from Pipeline import Pipeline, RunConfig, PipelineConfig
from questions_processing import QuestionsProcessor

# 页面配置
st.set_page_config(
    page_title="🚀 RAG Challenge 2 - RTX 5080 Powered",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.question-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #667eea;
    margin: 1rem 0;
}
.answer-box {
    background-color: #e8f5e8;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #28a745;
    margin: 1rem 0;
}
.error-box {
    background-color: #f8d7da;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #dc3545;
    margin: 1rem 0;
}
.info-box {
    background-color: #d1ecf1;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #17a2b8;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# 主标题
st.markdown("""
<div class="main-header">
    <h1>🚀 RAG Challenge 2 - RTX 5080 Powered</h1>
    <p>基于开源RAG系统，由RTX 5080 GPU加速</p>
    <p>📊 支持公司财报检索 • 🔥 问题答案 • LLM智能推理 • GPT-4o</p>
</div>
""", unsafe_allow_html=True)

# 初始化session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'history' not in st.session_state:
    st.session_state.history = []

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 系统配置")
    
    # 数据路径配置
    st.subheader("📁 数据路径")
    root_path = st.text_input(
        "数据集根目录",
        value=r"E:\code\AIProjectCode\trae_code\project\企业知识库\data",
        help="包含PDF报告、向量数据库等的根目录"
    )
    
    # 模型配置
    st.subheader("🤖 模型配置")
    api_provider = st.selectbox(
        "API提供商",
        ["dashscope", "openai"],
        index=0
    )
    
    answering_model = st.selectbox(
        "回答模型",
        ["qwen-turbo-latest", "gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06"],
        index=0
    )
    
    # 检索配置
    st.subheader("🔍 检索配置")
    top_n_retrieval = st.slider("检索文档数量", 5, 20, 10)
    llm_reranking = st.checkbox("启用LLM重排", value=True)
    parent_document_retrieval = st.checkbox("启用父文档检索", value=True)
    
    # 初始化按钮
    if st.button("🚀 初始化系统", type="primary"):
        try:
            with st.spinner("正在初始化系统..."):
                # 创建运行配置
                run_config = RunConfig(
                    parent_document_retrieval=parent_document_retrieval,
                    llm_reranking=llm_reranking,
                    top_n_retrieval=top_n_retrieval,
                    parallel_requests=1,
                    api_provider=api_provider,
                    answering_model=answering_model,
                    submission_file=False
                )
                
                # 创建Pipeline实例
                pipeline = Pipeline(Path(root_path), run_config=run_config)
                
                # 创建QuestionsProcessor实例
                processor = QuestionsProcessor(
                    vector_db_dir=pipeline.paths.vector_db_dir,
                    documents_dir=pipeline.paths.documents_dir,
                    new_challenge_pipeline=True,
                    subset_path=pipeline.paths.subset_path,
                    parent_document_retrieval=parent_document_retrieval,
                    llm_reranking=llm_reranking,
                    top_n_retrieval=top_n_retrieval,
                    parallel_requests=1,
                    api_provider=api_provider,
                    answering_model=answering_model
                )
                
                st.session_state.pipeline = pipeline
                st.session_state.processor = processor
                
            st.success("✅ 系统初始化成功！")
        except Exception as e:
            st.error(f"❌ 系统初始化失败: {str(e)}")

# 主界面
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 智能问答")
    
    # 问题输入
    question_text = st.text_area(
        "请输入您的问题",
        placeholder="例如：TSX_U 的CEO是谁？",
        height=100,
        help="请输入关于公司财报的问题，系统将基于RAG技术为您提供准确答案"
    )
    
    # 问题类型选择
    question_kind = st.selectbox(
        "问题类型",
        ["string", "number", "boolean"],
        index=0,
        help="选择期望的答案类型"
    )
    
    # 提交按钮
    col_submit, col_clear = st.columns([1, 1])
    
    with col_submit:
        if st.button("🔍 获取答案", type="primary", disabled=st.session_state.processor is None):
            if not question_text.strip():
                st.warning("⚠️ 请输入问题")
            else:
                try:
                    with st.spinner("🤔 正在思考中..."):
                        # 调用processor处理单个问题
                        answer_dict = st.session_state.processor.process_question(
                            question=question_text,
                            schema=question_kind
                        )
                        
                        # 添加到历史记录
                        st.session_state.history.append({
                            "question": question_text,
                            "kind": question_kind,
                            "answer": answer_dict.get("final_answer", "N/A"),
                            "references": answer_dict.get("references", []),
                            "reasoning": answer_dict.get("step_by_step_analysis", "")
                        })
                        
                    # 显示答案
                    st.markdown("""
                    <div class="answer-box">
                        <h4>💡 答案</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write(f"**答案**: {answer_dict.get('final_answer', 'N/A')}")
                    
                    # 显示推理过程
                    if answer_dict.get("step_by_step_analysis"):
                        with st.expander("🧠 推理过程"):
                            st.write(answer_dict["step_by_step_analysis"])
                    
                    # 显示参考页面
                    if answer_dict.get("references"):
                        with st.expander("📄 参考页面"):
                            for ref in answer_dict["references"]:
                                st.write(f"- 文档: {ref.get('pdf_sha1', 'Unknown')}")
                                st.write(f"  页码: {ref.get('page_index', 'Unknown')}")
                    
                except Exception as e:
                    st.markdown("""
                    <div class="error-box">
                        <h4>❌ 处理错误</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.error(f"处理问题时出错: {str(e)}")
    
    with col_clear:
        if st.button("🗑️ 清空历史"):
            st.session_state.history = []
            st.success("历史记录已清空")

with col2:
    st.header("📊 检索结果")
    
    # 系统状态
    if st.session_state.processor is None:
        st.markdown("""
        <div class="info-box">
            <h4>ℹ️ 系统状态</h4>
            <p>请先在左侧配置并初始化系统</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <h4>✅ 系统就绪</h4>
            <p>系统已初始化，可以开始问答</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 显示配置信息
    if st.session_state.processor:
        st.subheader("⚙️ 当前配置")
        config_info = {
            "API提供商": api_provider,
            "模型": answering_model,
            "检索数量": top_n_retrieval,
            "LLM重排": "✅" if llm_reranking else "❌",
            "父文档检索": "✅" if parent_document_retrieval else "❌"
        }
        
        for key, value in config_info.items():
            st.write(f"**{key}**: {value}")

# 历史记录
if st.session_state.history:
    st.header("📝 问答历史")
    
    for i, item in enumerate(reversed(st.session_state.history)):
        with st.expander(f"问题 {len(st.session_state.history) - i}: {item['question'][:50]}..."):
            st.write(f"**问题**: {item['question']}")
            st.write(f"**类型**: {item['kind']}")
            st.write(f"**答案**: {item['answer']}")
            
            if item['reasoning']:
                st.write("**推理过程**:")
                st.write(item['reasoning'])
            
            if item['references']:
                st.write("**参考文档**:")
                for ref in item['references']:
                    st.write(f"- {ref.get('pdf_sha1', 'Unknown')} (页码: {ref.get('page_index', 'Unknown')})")

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🚀 RAG Challenge 2 - Powered by RTX 5080 GPU | 基于开源RAG技术构建</p>
</div>
""", unsafe_allow_html=True)
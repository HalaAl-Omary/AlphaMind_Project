import streamlit as st
# IMPORT THE LOGIC FUNCTIONS
from server import agent_report_logic, get_market_data_logic

st.set_page_config(page_title="AlphaMind Pro", layout="wide")

st.title("📊 AlphaMind Pro: Local Agentic RAG")
st.info("System Status: Running on Ollama (Llama3) + MCP Server")

with st.sidebar:
    symbol = st.text_input("Stock Ticker", "NVDA")
    query = st.text_area("Analysis Goal", "Explain if the current market context suggests a Bull or Bear strategy for this stock.")
    run_btn = st.button("Execute Agent Pipeline", type="primary")

if run_btn:
    with st.status("Agents working...", expanded=True) as status:
        st.write("Fetching Market Data...")
        # USE THE LOGIC FUNCTION
        basic_data = get_market_data_logic(symbol)
        
        st.write("Querying RAG Knowledge Base and Running Analysis...")
        # USE THE LOGIC FUNCTION
        report = agent_report_logic(symbol, query)
        
        status.update(label="Analysis Complete!", state="complete")

    st.subheader("Final Agent Report")
    st.markdown(report)
    
    with st.expander("View Raw Market Data"):
        st.write(basic_data)
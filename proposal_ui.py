import streamlit as st
from proposal import write_proposal, select_best_proposal

st.title("Upwork Proposal Generator")

job_text = st.text_area("Paste the raw job data here:", height=300)

if st.button("Submit"):
    with st.spinner("Generating proposals..."):
        proposals = [write_proposal(job_text) for _ in range(3)]
        best_result = select_best_proposal(job_text, proposals[0], proposals[1], proposals[2])
        best_proposal_content = ''
        if isinstance(best_result, dict):
            best_proposal_content = best_result.get('best_proposal', proposals[0])
        elif isinstance(best_result, str):
            best_proposal_content = best_result
        else:
            best_proposal_content = proposals[0]
        st.subheader("Generated Proposals:")
        for i, proposal in enumerate(proposals, 1):
            st.markdown(f"**Proposal {i}:**\n{proposal}")
        st.subheader("Best Proposal:")
        st.markdown(best_proposal_content)

import streamlit as st
import time
from proposal import write_proposal, select_best_proposal

st.title("Upwork Proposal Generator")
tabs = st.tabs(["Write Proposal", "Better Proposal"])

with tabs[0]:
    st.header("Write Proposal")
    if 'last_submit_time_wp' not in st.session_state:
        st.session_state['last_submit_time_wp'] = 0
    if 'disable_submit_wp' not in st.session_state:
        st.session_state['disable_submit_wp'] = False
    job_text_wp = st.text_area("Paste the raw job data here:", key="wp_text", height=300)
    current_time = time.time()
    if st.session_state['disable_submit_wp'] and current_time - st.session_state['last_submit_time_wp'] < 60:
        st.warning(f"Please wait {int(60 - (current_time - st.session_state['last_submit_time_wp']))} seconds before submitting again.")
        st.button("Submit", disabled=True, key="wp_submit")
    else:
        if st.button("Submit", key="wp_submit"):
            st.session_state['last_submit_time_wp'] = time.time()
            st.session_state['disable_submit_wp'] = True
            with st.spinner("Request Received...") as spinner:
                time.sleep(1)
                st.info("Analyzing Job...")
                time.sleep(2)
                st.success("Job Analyzed.")
                time.sleep(1)
                st.info("Writing Proposal...")
                time.sleep(1)
                proposal = write_proposal(job_text_wp)
                st.success("Proposal Written.")
                time.sleep(1)
                st.success("Done!")
                st.subheader("Written Proposal:")
                st.markdown(proposal)

with tabs[1]:
    st.header("Better Proposal")
    st.info("This feature is currently under development.")
    st.button("Submit", disabled=True, key="bp_submit")
    st.caption("Use the 'Write Proposal' tab to generate proposals.")
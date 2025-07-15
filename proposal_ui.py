import streamlit as st
import time
from proposal import write_proposal
from vector_storage import update_proposal_history_by_id

st.title("Upwork Proposal Generator")

# --- Auth Logic ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    password = st.text_input("Enter password to use the app:", type="password")
    
    if password:
        if password == st.secrets["APP_PASSWORD"]:
            st.success("Access granted.")
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# --- Main App ---
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

            with st.spinner("Request received..."):
                time.sleep(1)
                st.info("Analyzing job...")
                time.sleep(2)
                st.success("Job analyzed.")
                time.sleep(1)
                st.info("Writing proposal...")
                time.sleep(1)

                proposal, proposal_id = write_proposal(job_text_wp)

                st.success("Proposal written.")
                time.sleep(1)
                st.success("Done.")

                # Store proposal and id in session_state
                st.session_state['last_proposal_id'] = proposal_id
                st.session_state['last_proposal'] = proposal

    # --- Display Proposal and Feedback only if proposal exists ---
    if st.session_state.get('last_proposal_id') and st.session_state.get('last_proposal'):
        st.subheader("Written Proposal")
        st.markdown(st.session_state['last_proposal'])

        # --- Copy to Clipboard Button ---
        escaped_proposal = st.session_state['last_proposal'].replace("\\", "\\\\").replace("`", "\\`").replace("\n", "\\n").replace("\"", "\\\"")
        copy_button = f"""
            <script>
            function copyProposal() {{
                const text = `{escaped_proposal}`;
                navigator.clipboard.writeText(text).then(function() {{
                    alert("Proposal copied to clipboard!");
                }}, function(err) {{
                    alert("Error copying to clipboard: " + err);
                }});
            }}
            </script>
            <button onclick=\"copyProposal()\">📋 Copy Proposal</button>
        """
        st.components.v1.html(
            """
            <style>
            button {
                background-color: #4CAF50;
                color: white;
                padding: 6px;
                font-size: 16px;
                width: 100%;
                border: none;
                cursor: pointer;
                border-radius: 5px;
            }
            </style>
            """ + copy_button, 
            height=40
        )
        st.markdown("---")

        # --- Comment and Rating Input ---
        st.subheader("Feedback on Proposal")
        comment = st.text_area("Add your comment:", key="wp_comment")
        rating = st.slider("Rate this proposal (1=Poor, 5=Excellent):", min_value=1, max_value=5, value=5, key="wp_rating")
        if st.button("Submit Feedback", key="wp_feedback"):
            proposal_id = st.session_state.get('last_proposal_id')
            if proposal_id:
                success = update_proposal_history_by_id(proposal_id, comment, rating)
                if success:
                    st.success("Feedback saved!")
                else:
                    st.error("Could not save feedback. Please try again.")
            else:
                st.error("No proposal to update. Please generate a proposal first.")

with tabs[1]:
    st.header("Better Proposal")
    st.info("This feature is under development.")
    st.button("Submit", disabled=True, key="bp_submit")
    st.caption("Use the 'Write Proposal' tab to generate proposals.")

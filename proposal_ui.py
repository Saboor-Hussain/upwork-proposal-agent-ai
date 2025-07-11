import streamlit as st
import time
from proposal import write_proposal, select_best_proposal

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

                proposal = write_proposal(job_text_wp)

                st.success("Proposal written.")
                time.sleep(1)
                st.success("Done.")

                # --- Display Proposal ---
                st.subheader("Written Proposal")
                st.code(proposal, language="markdown", width=1000)

                # --- Copy to Clipboard Button ---
                escaped_proposal = proposal.replace("`", "\\`").replace("\\", "\\\\")
                copy_code = f"""
                    <script>
                    function copyToClipboard(text) {{
                        navigator.clipboard.writeText(text).then(function() {{
                            alert("Proposal copied to clipboard.");
                        }}, function(err) {{
                            alert("Failed to copy: " + err);
                        }});
                    }}
                    </script>

                    <button onclick="copyToClipboard(`{escaped_proposal}`)">Copy Proposal</button>
                """

with tabs[1]:
    st.header("Better Proposal")
    st.info("This feature is under development.")
    st.button("Submit", disabled=True, key="bp_submit")
    st.caption("Use the 'Write Proposal' tab to generate proposals.")

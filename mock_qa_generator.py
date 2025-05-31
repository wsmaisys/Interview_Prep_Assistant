import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
import time
import os

# Configure page settings for better UX
st.set_page_config(
    page_title="Interview Prep Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Main app styling */
    .main {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        background-attachment: fixed;
    }
    
    /* Text visibility fixes */
    .stApp {
        color: #f1f5f9;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
    }
    
    p, div, span {
        color: #e2e8f0 !important;
    }
    
    /* Container styling */
    .block-container {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid #475569;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        margin-top: 2rem;
    }
    
    /* Input field styling */
    .stSelectbox > div > div {
        background-color: #1e293b !important;
        border: 2px solid #475569 !important;
        border-radius: 8px;
        color: #f1f5f9 !important;
    }
    
    .stSelectbox label {
        color: #f1f5f9 !important;
    }
    
    .stTextInput > div > div > input {
        background-color: #1e293b !important;
        border: 2px solid #475569 !important;
        border-radius: 8px;
        color: #f1f5f9 !important;
    }
    
    .stTextInput label {
        color: #f1f5f9 !important;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #1e293b !important;
        border: 2px solid #475569 !important;
        border-radius: 8px;
        color: #f1f5f9 !important;
    }
    
    .stTextArea label {
        color: #f1f5f9 !important;
    }
    
    /* Radio button styling */
    .stRadio label {
        color: #f1f5f9 !important;
    }
    
    .stRadio > div {
        background-color: transparent;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #059669, #047857);
        color: white !important;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(5, 150, 105, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(5, 150, 105, 0.4);
        background: linear-gradient(45deg, #047857, #065f46);
    }
    
    /* Footer styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: linear-gradient(90deg, #1e293b, #334155);
        color: #f1f5f9;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.3);
        border-top: 1px solid #475569;
    }
    
    /* Success message styling */
    .success-message {
        background: linear-gradient(45deg, #059669, #10b981);
        color: white !important;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(5, 150, 105, 0.2);
    }
    
    /* Question container styling */
    .question-container {
        background: #1e293b;
        border: 1px solid #475569;
        border-left: 4px solid #3b82f6;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        color: #f1f5f9 !important;
    }
    
    .question-container * {
        color: #e2e8f0 !important;
    }
    
    /* Sidebar styling if needed */
    .css-1d391kg {
        background-color: #0f172a;
    }
    
    /* Markdown styling */
    .stMarkdown {
        color: #e2e8f0 !important;
    }
    
    /* Help text styling */
    .stHelp {
        color: #94a3b8 !important;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }
    
    /* Info/Error message styling */
    .stAlert {
        background-color: #1e293b !important;
        border: 1px solid #475569 !important;
        color: #f1f5f9 !important;
    }
    
    /* Columns and dividers */
    hr {
        border-color: #475569 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for caching
@st.cache_resource
def initialize_llm():
    """Initialize and cache the LLM instance"""
    try:
        # Try to get API key from environment variables first (for local development)
        mistral_api_key = os.getenv("MISTRALAI_API_KEY")
        
        # Fallback to Streamlit secrets (for deployment)
        if not mistral_api_key:
            try:
                mistral_api_key = st.secrets["MISTRALAI_API_KEY"]
            except (KeyError, FileNotFoundError):
                pass
        
        # Check if API key exists
        if not mistral_api_key or mistral_api_key.strip() == "":
            st.error("🔑 **Missing API Key**: MISTRALAI_API_KEY not found.")
            st.markdown("""
            **🔧 For Local Development:**
            1. Create a `.env` file in your project root
            2. Add: `MISTRALAI_API_KEY=your-api-key-here`
            3. Get your API key from [Mistral AI Console](https://console.mistral.ai/)
            
            **🔧 For Streamlit Cloud Deployment:**
            1. Add your API key in Streamlit secrets
            2. Format: `MISTRALAI_API_KEY = "your-api-key-here"`
            """)
            st.stop()
        
        # Debug info for local testing (remove in production)
        api_key_source = "Environment Variable (.env)" if os.getenv("MISTRALAI_API_KEY") else "Streamlit Secrets"
        st.success(f"✅ **API Key Loaded** from: {api_key_source}")
        
        return ChatMistralAI(
            temperature=0.7, 
            model="mistral-small-latest",
            api_key=mistral_api_key.strip()  # Changed from mistral_api_key parameter
        )
        
    except Exception as e:
        st.error(f"🔑 **LLM Initialization Error**: {str(e)}")
        st.markdown("""
        **🔧 Troubleshooting Steps:**
        1. **Check your .env file**: Ensure it contains `MISTRALAI_API_KEY=your-key`
        2. **Verify API key**: Go to [Mistral AI Console](https://console.mistral.ai/)
        3. **Check file location**: .env should be in the same folder as your Python script
        4. **Install python-dotenv**: `pip install python-dotenv`
        5. **Restart the app** after making changes
        """)
        st.stop()

@st.cache_resource
def get_prompt_template():
    """Create and cache the prompt template"""
    return PromptTemplate(
        input_variables=["job_title", "round_type", "experience_years", "include_answers", "candidate_background"],
        template="""
You are an expert interview coach specializing in preparing candidates for {job_title} roles.

Generate exactly 5 {round_type} interview questions suitable for a candidate with {experience_years} years of experience.

Candidate background: {candidate_background}

{include_answers}

Format your response clearly with:
**Question 1:** [question]

**Question 2:** [question]

etc.

If model answers are requested, provide them after each question like this:
**Model Answer:** [detailed answer]

Use a supportive tone that builds confidence. Focus on practical, real-world scenarios relevant to current industry practices.
"""
    )

# Initialize components with better error handling
try:
    llm = initialize_llm()
    prompt_template = get_prompt_template()
except Exception as e:
    st.error(f"🚨 **Initialization Failed**: {str(e)}")
    st.stop()

# Header section
st.markdown("# 🎯 Interview Prep Assistant")
st.markdown("### 💫 *Ace your next interview with confidence and preparation*")
st.markdown("---")

# Create columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### 🚀 Let's get you interview-ready!")
    
    # Job role input
    job_title = st.text_input(
        "🎭 What role are you interviewing for?", 
        value="Machine Learning Engineer",
        help="Be specific! e.g., 'Senior Frontend Developer', 'Data Scientist'"
    )
    
    # Interview round selection
    round_type = st.selectbox(
        "📋 Select Interview Round", 
        [
            "HR Round", 
            "Technical Round", 
            "Manager Round", 
            "Telephonic Screening", 
            "Remote Screening",
            "Final Round"
        ],
        help="Different rounds focus on different aspects of your candidacy"
    )

with col2:
    st.markdown("#### ⚙️ Customize Your Experience")
    
    # Experience level
    experience_years = st.selectbox(
        "⏰ Your Experience Level", 
        [
            "Fresh Graduate (0 years)",
            "Entry Level (1-2 years)", 
            "Mid Level (3-5 years)",
            "Senior Level (5-8 years)", 
            "Expert Level (8+ years)"
        ],
        help="This helps tailor question complexity"
    )
    
    # Model answers preference
    include_answers = st.radio(
        "💡 Include model answers?", 
        ["Yes, show me sample answers", "No, just the questions"],
        help="Model answers help you prepare better responses"
    )

# Background section
st.markdown("#### 📝 Tell us about yourself")
candidate_background = st.text_area(
    "Share your background, skills, and recent projects", 
    value="Completed a Data Science Diploma, worked on prompt-tuning projects...",
    height=100,
    help="This helps personalize questions to your experience"
)

# Generate button with loading state
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    if st.button("✨ Generate My Interview Questions", type="primary"):
        if not job_title.strip():
            st.error("🚨 Please enter a job role to continue")
        elif not candidate_background.strip():
            st.error("🚨 Please share your background to get personalized questions")
        else:
            # Show loading spinner
            with st.spinner("🔍 Crafting personalized questions for you..."):
                try:
                    # Process include_answers for the prompt
                    answers_instruction = (
                        "After each question, provide a comprehensive model answer that demonstrates strong communication skills and technical competence." 
                        if "Yes" in include_answers 
                        else "Only provide the questions without model answers."
                    )
                    
                    # Extract years for processing
                    exp_years = experience_years.split('(')[1].split(')')[0] if '(' in experience_years else experience_years
                    
                    # Generate response using modern RunnableSequence (prompt | llm)
                    chain = prompt_template | llm | StrOutputParser()
                    result = chain.invoke({
                        "job_title": job_title,
                        "round_type": round_type,
                        "experience_years": exp_years,
                        "include_answers": answers_instruction,
                        "candidate_background": candidate_background
                    })
                    
                    # Get the response text - this should work with the corrected setup
                    response_text = result
                    
                    # Success message
                    st.markdown("""
                    <div class="success-message">
                        🎉 <strong>Great!</strong> Your personalized interview questions are ready. Take your time to review and practice!
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display results with proper markdown rendering
                    st.markdown("### 📚 Your Personalized Interview Questions")
                    st.markdown(f"**Role:** {job_title} | **Round:** {round_type} | **Experience:** {experience_years}")
                    st.markdown("---")
                    
                    # Render the response as markdown for proper formatting
                    st.markdown(response_text)
                    
                    # Additional tips
                    st.markdown("---")
                    st.markdown("### 💪 Quick Prep Tips")
                    tips_col1, tips_col2 = st.columns(2)
                    
                    with tips_col1:
                        st.markdown("""
                        **🎯 Before the Interview:**
                        - Practice your answers out loud
                        - Research the company thoroughly
                        - Prepare 2-3 questions to ask them
                        """)
                    
                    with tips_col2:
                        st.markdown("""
                        **⚡ During the Interview:**
                        - Take a moment to think before answering
                        - Use the STAR method for behavioral questions
                        - Show enthusiasm and ask clarifying questions
                        """)
                
                except KeyError:
                    st.error("🔑 **API Key Missing**: MISTRALAI_API_KEY not found in Streamlit secrets.")
                    st.markdown("""
                    **🔧 How to fix this:**
                    1. Create a `.streamlit/secrets.toml` file in your project root
                    2. Add your API key: `MISTRALAI_API_KEY = "your-api-key-here"`
                    3. Get your API key from [Mistral AI Console](https://console.mistral.ai/)
                    4. Restart your Streamlit app
                    """)
                except Exception as e:
                    error_msg = str(e).lower()
                    if "401" in error_msg or "unauthorized" in error_msg:
                        st.error("🔑 **Authentication Failed**: Your Mistral API key is invalid or expired.")
                        st.markdown("""
                        **🔧 Steps to fix:**
                        1. **Verify your API key** at [Mistral AI Console](https://console.mistral.ai/)
                        2. **Check billing status** - ensure your account has credits
                        3. **Generate a new key** if the current one is expired
                        4. **Update your secrets.toml** with the new key
                        5. **Restart the app**
                        """)
                    elif "rate" in error_msg or "quota" in error_msg:
                        st.error("⏱️ **Rate Limit Exceeded**: Too many requests to Mistral API.")
                        st.info("💡 Wait a moment and try again, or check your API usage limits.")
                    else:
                        st.error("Please add your MISTRALAI_API_KEY to Streamlit secrets.")
                    st.markdown("""
                    **To fix this:**
                    1. Go to your Streamlit app settings
                    2. Add your Mistral API key in secrets as: `MISTRALAI_API_KEY = "your-key-here"`
                    3. Restart your app
                    """)

# Motivational section
st.markdown("---")
st.markdown("### 🌟 You've Got This!")
motivation_col1, motivation_col2, motivation_col3 = st.columns(3)

with motivation_col1:
    st.markdown("""
    **🧠 Confidence**
    
    Remember, you're qualified for this role. Trust your preparation and experience.
    """)

with motivation_col2:
    st.markdown("""
    **💬 Communication**
    
    Practice makes perfect. The more you rehearse, the more natural you'll sound.
    """)

with motivation_col3:
    st.markdown("""
    **🎯 Success**
    
    Every interview is a learning opportunity. You're already on the path to success!
    """)

# Footer
st.markdown("""
<div class="footer">
    ❤️ Created by Waseem M Ansari, response served from Mistral AI 🤖
</div>
""", unsafe_allow_html=True)

# Add some spacing for the footer
st.markdown("<br><br>", unsafe_allow_html=True)
import os
import sqlite3
import hashlib
import json
import PIL.Image
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, wait_random
from pydantic import BaseModel, Field

# --- PYDANTIC SCHEMAS FOR STRUCTURED OUTPUTS ---
class ExtractionSchema(BaseModel):
    answers: list[str] = Field(description="Array of exactly 4 strings")

class StudyMethodSchema(BaseModel):
    method: str
    reasoning: str

class MethodListSchema(BaseModel):
    methods: list[StudyMethodSchema]

class TaskArchitectSchema(BaseModel):
    task_text: str
    requires_math: bool
    answer_key: str

class AssessmentSchema(BaseModel):
    question_text: str
    requires_math: bool
    answer_key: str
    
class CaseSchema(BaseModel):
    case_text: str
    requires_math: bool
    answer_key: str

st.set_page_config(page_title="Evidence-Based Faculty Coach", layout="wide")
load_dotenv()

# ==========================================
# 1. DATABASE & FILE HANDLING INITIALIZATION
# ==========================================
def init_db():
    conn = sqlite3.connect('course_data.db')
    
    # --- RESILIENCE UPGRADE: WAL Mode & Busy Timeout ---
    # Allows simultaneous read/writes to prevent "database is locked" crashes
    conn.execute('PRAGMA journal_mode=WAL;')
    conn.execute('PRAGMA busy_timeout=5000;')
    
    cursor = conn.cursor()
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS prep_tracker_v2 (student_id_hash TEXT PRIMARY KEY, q1_importance TEXT, q2_prior_knowledge TEXT, q3_objectives TEXT, q4_confusing TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS post_class_tracker (student_id_hash TEXT PRIMARY KEY, summary_paragraph TEXT, struggled_remember TEXT, struggled_understand TEXT, extra_time_materials TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS mastery_dashboard (student_id_hash TEXT, objective TEXT, status TEXT, PRIMARY KEY (student_id_hash, objective))''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS study_session_log (id INTEGER PRIMARY KEY AUTOINCREMENT, student_id_hash TEXT, smart_s TEXT, smart_m TEXT, smart_a TEXT, smart_r TEXT, smart_t INTEGER, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS assessment_log (id INTEGER PRIMARY KEY AUTOINCREMENT, student_id_hash TEXT, objective TEXT, question TEXT, student_answer TEXT, ai_feedback TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS mastery_history (id INTEGER PRIMARY KEY AUTOINCREMENT, student_id_hash TEXT, objective TEXT, old_status TEXT, new_status TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    conn.commit()
    conn.close()

init_db()

BASE_PATH = r"Reference_docs"

def safe_load(filename):
    filepath = os.path.join(BASE_PATH, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return f"[Notice: {filename} was not found in the reference folder.]"

if "client" not in st.session_state:
    st.session_state.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ==========================================
# 2. MODEL ROUTING & CONTEXT CACHING
# ==========================================
HEAVY_MODEL = "gemini-3.5-flash"       
LIGHT_MODEL = "gemini-3.1-flash-lite"  

@st.cache_resource(show_spinner=False)
def get_master_cache():
    try:
        temp_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        slides_text = safe_load("Nutrition_Slides.md")
        handout_text = safe_load("Nutrition_objectives.md")
        textbook_text = safe_load("CH_10_summary.md") + "\n" + safe_load("CH_11_summary.md") + "\n" + safe_load("CH_12_summary.md")
        ebls_text = safe_load("EBLS_summary.md")
        
        master_context = f"--- CLINICAL MATERIALS ---\nSlides:\n{slides_text}\nHandout:\n{handout_text}\nTextbook:\n{textbook_text}\n\n--- EBLS FRAMEWORK ---\n{ebls_text}"
        
        cache = temp_client.caches.create(
            model=HEAVY_MODEL,
            config=types.CreateCachedContentConfig(
                contents=[master_context],
                system_instruction="You are a supportive but firm pharmacy faculty member and academic coach.",
                ttl="7200s", 
            )
        )
        return cache.name
    except Exception as e:
        print(f"Context Caching failed or not supported in this region: {e}")
        return None

master_cache_name = get_master_cache()

# --- RESILIENCE UPGRADE: Exponential Backoff + Jitter ---
# Will retry automatically up to 4 times, waiting between 2 and 10 seconds (plus random jitter)
@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=10) + wait_random(0, 1), reraise=True)
def ask_coach(prompt, image=None, model=HEAVY_MODEL, use_cache=False, response_schema=None, require_json=False):
    """Centralized LLM Router for managing caching, retries, and fallbacks."""
    contents = [prompt]
    if image is not None:
        contents.append(image)
        
    config_args = {"temperature": 0.0}
    
    # 1. ENFORCING STRUCTURED JSON OUTPUTS
    if response_schema:
        config_args["response_mime_type"] = "application/json"
        config_args["response_schema"] = response_schema
    elif require_json:
        config_args["response_mime_type"] = "application/json"
    
    if use_cache:
        if master_cache_name and model == HEAVY_MODEL:
            config_args["cached_content"] = master_cache_name
        else:
            slides_text = safe_load("Nutrition_Slides.md")
            handout_text = safe_load("Nutrition_objectives.md")
            textbook_text = safe_load("CH_10_summary.md") + "\n" + safe_load("CH_11_summary.md") + "\n" + safe_load("CH_12_summary.md")
            ebls_text = safe_load("EBLS_summary.md")
            fallback_context = f"\n\n--- CLINICAL MATERIALS ---\nSlides:\n{slides_text}\nHandout:\n{handout_text}\nTextbook:\n{textbook_text}\n\n--- EBLS FRAMEWORK ---\n{ebls_text}"
            contents[0] = contents[0] + fallback_context

    return st.session_state.client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(**config_args)
    )

st.title("🍏 Clinical Nutrition: EN & PN Architecture")

# ==========================================
# SECURE LOGIN & DYNAMIC OBJECTIVE LOADING
# ==========================================
raw_student_id = st.text_input("Enter your MCPHS Student ID (e.g., JD1234):").strip().upper()

if not raw_student_id:
    st.info("Please enter your Student ID to access your personalized study plan.")
    st.stop()

hashed_id = hashlib.sha256(raw_student_id.encode()).hexdigest()
st.success("Identity verified and anonymized.")

# --- FETCH PERSISTED TAB 1 & 2 DATA ---
conn = sqlite3.connect('course_data.db')
cursor = conn.cursor()

cursor.execute("SELECT q1_importance, q2_prior_knowledge, q3_objectives, q4_confusing FROM prep_tracker_v2 WHERE student_id_hash = ?", (hashed_id,))
t1_saved = cursor.fetchone()
t1_vals = t1_saved if t1_saved else ("", "", "", "")

cursor.execute("SELECT summary_paragraph, struggled_remember, struggled_understand, extra_time_materials FROM post_class_tracker WHERE student_id_hash = ?", (hashed_id,))
t2_saved = cursor.fetchone()
t2_vals = t2_saved if t2_saved else ("", "", "", "")

conn.close()

@st.cache_data(show_spinner=False)
def load_course_objectives():
    try:
        handout_text = safe_load("Nutrition_objectives.md")
        extraction_prompt = f"""
        You are an expert curriculum informatics parser and instructional designer. Your objective is to ingest raw academic text and extract the learning objectives into a clean, machine-readable JSON dictionary.
        
        Strict Rules:
        1. Identify the overarching goals/headers as "Learning Objectives" (the keys of the dictionary).
        2. Identify the specific bullet points underneath each header as "Performance Criteria" (the array of strings for each key).
        3. Do not extract non-pedagogical boilerplate.
        
        --- HANDOUT TEXT ---
        {handout_text}
        """
        response = ask_coach(extraction_prompt, model=LIGHT_MODEL, use_cache=False, require_json=True)
        return json.loads(response.text)
    except Exception as e:
        return {"API Error - Could not load dynamic syllabus.": ["Check connection and logs."]}

with st.spinner("Syncing with course syllabus..."):
    course_objectives_dict = load_course_objectives()

conn = sqlite3.connect('course_data.db')
cursor = conn.cursor()
for macro, micros in course_objectives_dict.items():
    for micro in micros:
        full_objective = f"{macro}: {micro}"
        cursor.execute('INSERT OR IGNORE INTO mastery_dashboard (student_id_hash, objective, status) VALUES (?, ?, ?)', (hashed_id, full_objective, '🔵 Untested'))
conn.commit()
conn.close()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Part 1: Pre-Class", "Part 2: Brain Dump", "Part 3: Active Recall Session", "Part 4: Interleaved Practice Exams", "Part 5: Mastery & Coach Audit"])

# ==========================================
# TAB 1: PRE-CLASS PREPARATION (PRIMING)
# ==========================================
with tab1:
    st.header("Step 1: Priming for Learning")
    st.markdown("Establish your baseline before lecture. Type your answers OR snap a photo of your handwritten notebook.")
    
    if t1_saved:
        st.info("💡 Your previous Pre-Class Prep has been loaded. You can edit it or submit again.")
    
    if "t1_cam_active" not in st.session_state: st.session_state.t1_cam_active = False
    if "t1_photo" not in st.session_state: st.session_state.t1_photo = None

    # 2. MEDICAL HANDWRITING OCR - INITIALIZING TEXT AREA STATES
    if "t1_q1" not in st.session_state: st.session_state.t1_q1 = t1_vals[0]
    if "t1_q2" not in st.session_state: st.session_state.t1_q2 = t1_vals[1]
    if "t1_q3" not in st.session_state: st.session_state.t1_q3 = t1_vals[2]
    if "t1_q4" not in st.session_state: st.session_state.t1_q4 = t1_vals[3]

    st.markdown("### Option A: Type it out")
    with st.form("prep_form"):
        # Bind text areas to session state so OCR can safely populate them
        q1 = st.text_area("1) What will I be learning and why is it important?", key="t1_q1")
        q2 = st.text_area("2) What do I already know about this material?", key="t1_q2")
        q3 = st.text_area("3) What are the learning objectives / goals for this class?", key="t1_q3")
        q4 = st.text_area("4) What is new or confusing in the notes or reading assignment?", key="t1_q4")
        
        if st.form_submit_button("📤 Submit Typed Prep"):
            if q1.strip() and q2.strip() and q3.strip() and q4.strip():
                try:
                    conn = sqlite3.connect('course_data.db')
                    conn.execute('PRAGMA busy_timeout=5000;')
                    cursor = conn.cursor()
                    cursor.execute('REPLACE INTO prep_tracker_v2 (student_id_hash, q1_importance, q2_prior_knowledge, q3_objectives, q4_confusing) VALUES (?, ?, ?, ?, ?)', (hashed_id, q1, q2, q3, q4))
                    conn.commit()
                    conn.close()
                    st.success("Your pre-class prep has been securely saved!")
                    with st.spinner("Faculty Coach is reviewing your notes..."):
                        feedback_prompt = f"Student Pre-class prep. Goals: {q1} | Prior: {q2} | Obj: {q3} | Confusing: {q4}. Evaluate the submission. Do not penalize for incorrect facts at this stage. Instead, act as a supportive coach, explicitly identify the 2 or 3 concepts they are struggling with, and tell them to pay close attention when the professor covers those specific topics in lecture today."
                        feedback_res = ask_coach(feedback_prompt, model=LIGHT_MODEL)
                        st.info(f"🩺 **Coach Note:** {feedback_res.text}")
                except Exception as e:
                    st.warning(f"⚠️ Network error. Please try clicking submit again.")
            else:
                st.warning("Please fill out all four text boxes.")

    st.markdown("### Option B: Upload Handwritten Worksheet")
    
    if not st.session_state.t1_cam_active and st.session_state.t1_photo is None:
        if st.button("📸 Open Webcam", key="t1_open"):
            st.session_state.t1_cam_active = True
            st.rerun()

    elif st.session_state.t1_cam_active:
        if st.button("❌ Turn Camera Off", key="t1_close"):
            st.session_state.t1_cam_active = False
            st.rerun()
        
        t1_cam = st.camera_input("Hold your notebook up to the webcam:", key="t1_cam_input")
        if t1_cam is not None:
            st.session_state.t1_photo = t1_cam
            st.session_state.t1_cam_active = False
            st.rerun()

    if st.session_state.t1_photo is not None:
        st.success("✅ Photo captured successfully! (Hardware camera is now OFF)")
        st.image(st.session_state.t1_photo, width=400)
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📤 Transcribe Handwritten Worksheet", key="t1_submit_photo"):
                with st.spinner("Coach is transcribing your handwriting..."):
                    try:
                        img = PIL.Image.open(st.session_state.t1_photo)
                        extract_prompt = "Read this handwritten pre-class worksheet. Extract answers for: 1. Importance, 2. Prior Knowledge, 3. Objectives, 4. Confusing Topics."
                        ext_res = ask_coach(extract_prompt, image=img, model=LIGHT_MODEL, response_schema=ExtractionSchema)
                        extracted_data = json.loads(ext_res.text)["answers"]
                        
                        # Populate the text boxes and stop hardware bypass, allowing user editing!
                        st.session_state.t1_q1 = extracted_data[0] if len(extracted_data) > 0 else ""
                        st.session_state.t1_q2 = extracted_data[1] if len(extracted_data) > 1 else ""
                        st.session_state.t1_q3 = extracted_data[2] if len(extracted_data) > 2 else ""
                        st.session_state.t1_q4 = extracted_data[3] if len(extracted_data) > 3 else ""
                        st.session_state.t1_photo = None
                        
                        st.success("Handwriting transcribed! Please review and edit your answers in the text boxes above, then click 'Submit Typed Prep'.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"⚠️ Could not read the handwriting cleanly. Please try typing your answers. ({e})")
        with c2:
            if st.button("🗑️ Retake Photo", key="t1_retake"):
                st.session_state.t1_photo = None
                st.session_state.t1_cam_active = True
                st.rerun()

# ==========================================
# TAB 2: POST-CLASS BRAIN DUMP (POSTVIEWING)
# ==========================================
with tab2:
    st.header("Step 2: Post-Class Brain Dump")
    st.markdown("Close your textbook and slides. Type your recall below or write it on paper and snap a photo.")
    
    if t2_saved:
        st.info("💡 Your previous Brain Dump has been loaded. You can edit it or submit again.")
        
    if "t2_cam_active" not in st.session_state: st.session_state.t2_cam_active = False
    if "t2_photo" not in st.session_state: st.session_state.t2_photo = None

    if "t2_q1" not in st.session_state: st.session_state.t2_q1 = t2_vals[0]
    if "t2_q2" not in st.session_state: st.session_state.t2_q2 = t2_vals[1]
    if "t2_q3" not in st.session_state: st.session_state.t2_q3 = t2_vals[2]
    if "t2_q4" not in st.session_state: st.session_state.t2_q4 = t2_vals[3]

    st.markdown("### Option A: Type it out")
    with st.form("brain_dump_form"):
        bd_q1 = st.text_area("1) Without the aid of notes, write a paragraph summarizing the main concepts learned.", key="t2_q1")
        bd_q2 = st.text_area("2) Identify major topics you struggled to remember.", key="t2_q2")
        bd_q3 = st.text_area("3) List topics that you struggled to understand.", key="t2_q3")
        bd_q4 = st.text_area("4) Identify tables, figures, etc. that require extra time.", key="t2_q4")
        
        if st.form_submit_button("📤 Submit Typed Brain Dump"):
            if bd_q1.strip() and bd_q2.strip() and bd_q3.strip() and bd_q4.strip():
                try:
                    conn = sqlite3.connect('course_data.db')
                    conn.execute('PRAGMA busy_timeout=5000;')
                    cursor = conn.cursor()
                    cursor.execute('REPLACE INTO post_class_tracker (student_id_hash, summary_paragraph, struggled_remember, struggled_understand, extra_time_materials) VALUES (?, ?, ?, ?, ?)', (hashed_id, bd_q1, bd_q2, bd_q3, bd_q4))
                    conn.commit()
                    conn.close()
                    st.success("Brain dump saved!")
                    with st.spinner("Coach is analyzing your recall..."):
                        feedback_prompt = f"""
                        Student brain dump: Summary: {bd_q1} | Remember: {bd_q2} | Understand: {bd_q3} | Needs time: {bd_q4}.
                        Analyze this brain dump against the Slides in the CLINICAL MATERIALS provided in your context. 
                        Output your feedback in three distinct bulleted lists: 
                        🟢 Mastered (fully captured concepts), 
                        🟡 Partial (concepts they mentioned but lacked detail on), 
                        🔴 Critical Omissions (major concepts from the lecture they completely forgot to write down).
                        """
                        feedback_res = ask_coach(feedback_prompt, model=HEAVY_MODEL, use_cache=True)
                        st.info(f"🧠 **Coach Review:**\n{feedback_res.text}")
                except Exception as e:
                    st.warning(f"⚠️ Network Error. Please click submit again.")
            else:
                st.warning("Please fill out all four text boxes.")

    st.markdown("### Option B: Upload Handwritten Brain Dump")
    
    if not st.session_state.t2_cam_active and st.session_state.t2_photo is None:
        if st.button("📸 Open Webcam", key="t2_open"):
            st.session_state.t2_cam_active = True
            st.rerun()

    elif st.session_state.t2_cam_active:
        if st.button("❌ Turn Camera Off", key="t2_close"):
            st.session_state.t2_cam_active = False
            st.rerun()
            
        t2_cam = st.camera_input("Hold your notebook up to the webcam:", key="t2_cam_input")
        if t2_cam is not None:
            st.session_state.t2_photo = t2_cam
            st.session_state.t2_cam_active = False
            st.rerun()

    if st.session_state.t2_photo is not None:
        st.success("✅ Photo captured successfully! (Hardware camera is now OFF)")
        st.image(st.session_state.t2_photo, width=400)
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📤 Transcribe Handwritten Brain Dump", key="t2_submit_photo"):
                with st.spinner("Coach is transcribing your brain dump..."):
                    try:
                        img = PIL.Image.open(st.session_state.t2_photo)
                        extract_prompt = "Read this handwritten brain dump. Extract text for: 1. Summary, 2. Remember, 3. Understand, 4. Needs extra time."
                        ext_res = ask_coach(extract_prompt, image=img, model=LIGHT_MODEL, response_schema=ExtractionSchema)
                        extracted_data = json.loads(ext_res.text)["answers"]
                        
                        st.session_state.t2_q1 = extracted_data[0] if len(extracted_data) > 0 else ""
                        st.session_state.t2_q2 = extracted_data[1] if len(extracted_data) > 1 else ""
                        st.session_state.t2_q3 = extracted_data[2] if len(extracted_data) > 2 else ""
                        st.session_state.t2_q4 = extracted_data[3] if len(extracted_data) > 3 else ""
                        st.session_state.t2_photo = None 

                        st.success("Handwriting transcribed! Please review and edit your answers in the text boxes above, then click 'Submit Typed Brain Dump'.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"⚠️ Could not read the handwriting cleanly. Please try typing your answers. ({e})")
        with c2:
            if st.button("🗑️ Retake Photo", key="t2_retake"):
                st.session_state.t2_photo = None
                st.session_state.t2_cam_active = True
                st.rerun()

# ==========================================
# TAB 3: ACTIVE RECALL SESSION
# ==========================================
with tab3:
    head_col1, head_col2 = st.columns([4, 1])
    with head_col1:
        st.header("Active Recall Session")
    
    if "study_step" not in st.session_state: st.session_state.study_step = "intake"
    
    with head_col2:
        if st.session_state.study_step != "intake":
            if st.button("🔄 Reset Session", use_container_width=True, key="global_reset_t3"):
                st.session_state.study_step = "intake"
                st.session_state.current_question = None
                st.session_state.t3_task_requires_math = False
                st.session_state.t3_task_answer_key = "N/A"
                st.session_state.t3_case_text = None
                st.session_state.t3_requires_math = False
                st.session_state.t3_hidden_answer_key = "N/A"
                st.session_state.t3_exec_photo = None
                st.session_state.t3_case_photo = None
                st.rerun()

    if "suggested_methods" not in st.session_state: st.session_state.suggested_methods = []
    if "final_assignment" not in st.session_state: st.session_state.final_assignment = None
    if "current_question" not in st.session_state: st.session_state.current_question = None
    
    if "t3_task_requires_math" not in st.session_state: st.session_state.t3_task_requires_math = False
    if "t3_task_answer_key" not in st.session_state: st.session_state.t3_task_answer_key = "N/A"
    
    if "t3_case_text" not in st.session_state: st.session_state.t3_case_text = None
    if "t3_requires_math" not in st.session_state: st.session_state.t3_requires_math = False
    if "t3_hidden_answer_key" not in st.session_state: st.session_state.t3_hidden_answer_key = "N/A"
    
    if "t3_exec_cam_active" not in st.session_state: st.session_state.t3_exec_cam_active = False
    if "t3_exec_photo" not in st.session_state: st.session_state.t3_exec_photo = None
    if "t3_case_cam_active" not in st.session_state: st.session_state.t3_case_cam_active = False
    if "t3_case_photo" not in st.session_state: st.session_state.t3_case_photo = None

    if st.session_state.study_step == "intake":
        st.subheader("Step 1: Objectives and Strategy")
        
        selected_macros_t3 = st.multiselect("🎯 Select Learning Objective(s):", options=list(course_objectives_dict.keys()), key="t3_macro")
        
        criteria_options_t3 = []
        for macro in selected_macros_t3:
            for micro in course_objectives_dict[macro]:
                criteria_options_t3.append(f"{macro}: {micro}")
        
        with st.form("study_intake_form"):
            st.write("**Target Specific Performance Criteria:**")
            associated_objs = st.multiselect("Select Performance Criteria:", options=criteria_options_t3)
            
            focus_area = st.text_area("🎯 Specific focus or struggle in this section:")
            
            if st.form_submit_button("Suggest Best Study Method"):
                if not selected_macros_t3:
                    st.error("Please select at least one Learning Objective.")
                elif not associated_objs:
                    st.error("Please associate this session with at least one Performance Criterion.")
                else:
                    try:
                        suggestion_prompt = f"You are an expert learning scientist. A student is studying clinical nutrition material. Focus: {focus_area}. Objectives: {associated_objs}. Evaluate and rank ALL of the following active recall methods in descending order (from most recommended to least recommended) based on the student's parameters: [Exam question generation, focused listing, empty outline activity, minute paper, concept map, error interception drill, compare and contrast matrix]."
                        with st.spinner("Analyzing study parameters..."):
                            res = ask_coach(suggestion_prompt, model=LIGHT_MODEL, response_schema=MethodListSchema)
                            st.session_state.suggested_methods = json.loads(res.text)["methods"]
                            st.session_state.study_step = "suggestion"
                            st.session_state.saved_focus = focus_area
                            st.session_state.associated_objs = associated_objs 
                            st.rerun()
                    except Exception as e:
                        st.error("Faculty Coach is busy architecting. Please try clicking submit again.")

    elif st.session_state.study_step == "suggestion":
        st.subheader("🤖 Faculty Coach Strategy Recommendations")
        st.markdown("Here is your ranked list of evidence-based strategies. The top options will yield the highest retention based on your current struggles.")
        
        for idx, item in enumerate(st.session_state.suggested_methods):
            with st.expander(f"#{idx + 1} ✨ {item['method']}"):
                st.write(item['reasoning'])
        
        choice = st.radio("Select the method you will commit to:", [m['method'] for m in st.session_state.suggested_methods])
        
        if st.button("Lock In & Start Studying"):
            conn = sqlite3.connect('course_data.db')
            conn.execute('PRAGMA busy_timeout=5000;')
            cursor = conn.cursor()
            cursor.execute('INSERT INTO study_session_log (student_id_hash, smart_s, smart_m, smart_a, smart_r, smart_t) VALUES (?, ?, ?, ?, ?, ?)', 
                           (hashed_id, "Clinical Nutrition", str(st.session_state.associated_objs), choice, "N/A", 0))
            conn.commit()
            conn.close()
            st.session_state.final_assignment = choice
            st.session_state.study_step = "execution"
            st.rerun()

    elif st.session_state.study_step == "execution":
        st.subheader(f"🚀 Step 2: Current Task - {st.session_state.final_assignment}")
        
        if st.session_state.current_question is None:
            with st.spinner("Faculty Coach is designing your specific clinical task..."):
                instr_prompt = f"""
                You are a strict clinical pharmacy faculty coach designing a specific, actionable active-recall task.
                The student has chosen the '{st.session_state.final_assignment}' study method.
                Their selected target criteria are: {st.session_state.associated_objs}.
                Their specific confusion/focus is: {st.session_state.saved_focus}.

                CRITICAL CRITERIA LIMIT:
                From the student's selected criteria list, you MUST select a manageable subset (maximum of 2 specific objectives or performance criteria) to build this task around. Do not try to exhaust the entire list.

                CRITICAL INSTRUCTIONS:
                Do NOT give generic cheerleader advice. Construct the ACTUAL, highly-specific skeleton or prompt they need to execute right now.
                - If it's an "Empty Outline", provide high-level Roman numerals based primarily on the Slides and Handout (use Textbook for supplemental context).
                - If it's a "Concept Map", provide the central node and 3-4 specific clinical branches.
                - If it's a "Minute Paper", provide a challenging clinical comparison or drug mechanism.
                - If it's "Exam Question Generation", specify the clinical trap or calculation type.
                - If it's an "Error Interception Drill", generate a realistic clinical artifact containing 1 or 2 dangerous flaws.
                - If it's a "Compare and Contrast Matrix", generate an empty Markdown table with specific clinical rows and columns.
                
                # 3. THE MATH HALLUCINATION LOOP & CHAIN OF THOUGHT GENERATION
                - "answer_key": string (If requires_math is true, you MUST write out the mathematical formula used, plug in the variables step-by-step, and only calculate the final dose at the very end. Identify any flaws. If false, write 'N/A').
                """
                try:
                    instr_res = ask_coach(instr_prompt, model=HEAVY_MODEL, use_cache=True, response_schema=TaskArchitectSchema)
                    task_data = json.loads(instr_res.text)
                    st.session_state.current_question = task_data.get("task_text", "Error loading task text.")
                    st.session_state.t3_task_requires_math = task_data.get("requires_math", False)
                    st.session_state.t3_task_answer_key = task_data.get("answer_key", "N/A")
                except Exception as e:
                    st.session_state.current_question = f"Error parsing Coach task. Please click Reset Session and try again. Network log: {e}"

        st.info(st.session_state.current_question)

        if st.session_state.t3_task_requires_math:
             st.warning("🧮 **Clinical Math Required:** Precise calculation is needed to complete this task.")

        st.markdown("### Prove Your Mastery")
        
        typed_ans = st.text_area("⌨️ Type your submission here (Optional if uploading photo):")
        st.write("--- OR ---")
        
        if not st.session_state.t3_exec_cam_active and st.session_state.t3_exec_photo is None:
            if st.button("📸 Open Webcam for Visual Assignment", key="t3_exec_open"):
                st.session_state.t3_exec_cam_active = True
                st.rerun()

        elif st.session_state.t3_exec_cam_active:
            if st.button("❌ Turn Camera Off", key="t3_exec_close"):
                st.session_state.t3_exec_cam_active = False
                st.rerun()
                
            t3_exec_cam = st.camera_input("Capture your visual work:", key="t3_exec_input")
            if t3_exec_cam is not None:
                st.session_state.t3_exec_photo = t3_exec_cam
                st.session_state.t3_exec_cam_active = False
                st.rerun()

        if st.session_state.t3_exec_photo is not None:
            st.success("✅ Work captured! (Camera is OFF)")
            st.image(st.session_state.t3_exec_photo, width=300)
            if st.button("🗑️ Retake Photo", key="t3_exec_retake"):
                st.session_state.t3_exec_photo = None
                st.session_state.t3_exec_cam_active = True
                st.rerun()

        if st.button("📤 Submit to Faculty Coach for Review"):
            if not typed_ans.strip() and st.session_state.t3_exec_photo is None:
                st.warning("Please provide a submission.")
            else:
                try:
                    conn = sqlite3.connect('course_data.db')
                    conn.execute('PRAGMA busy_timeout=5000;')
                    cursor = conn.cursor()
                    cursor.execute("SELECT q4_confusing FROM prep_tracker_v2 WHERE student_id_hash = ?", (hashed_id,))
                    prep_data = cursor.fetchone()
                    
                    grading_prompt = f"""
                    Your objective is to evaluate the student's submission using the Evidence-Based Learning Strategies (EBLS) framework.

                    Follow these strict grading protocols:
                    1. **Clinical Accuracy:** Grade the clinical logic primarily against the Slides and Handout within the CLINICAL MATERIALS, using the Textbook materials as a supplemental backup.
                    2. **EBLS Coaching Integration:** Diagnose *where* their retrieval failed. Using the EBLS FRAMEWORK in your context, explicitly name the cognitive mechanism they failed at (e.g., Illusion of Knowing, poor priming) and prescribe a specific Phase 4 tool to fix it. Do not just give them the correct clinical answer; tell them exactly how to study to find it.
                    3. **MATH PROTOCOL:** Does this task require math/strict matching? {'YES' if st.session_state.t3_task_requires_math else 'NO'}.
                    True Answer Key: {st.session_state.t3_task_answer_key}
                    If YES, you MUST compare the student's answer strictly against the 'True Answer Key'. Do not recalculate math yourself. 
                    4. **History Context:** Address any relevant Pre-Class Confusion: {prep_data[0] if prep_data else 'None'}.

                    Task: {st.session_state.current_question}
                    Student Typed Answer: {typed_ans if typed_ans else "See attached image."}
                    """
                    
                    img = PIL.Image.open(st.session_state.t3_exec_photo) if st.session_state.t3_exec_photo else None
                    with st.spinner("Faculty Coach is analyzing your submission..."):
                        grade_response = ask_coach(grading_prompt, image=img, model=HEAVY_MODEL, use_cache=True)
                    
                    st.markdown("### Faculty Coach Feedback")
                    st.success(grade_response.text)

                    cursor.execute('INSERT INTO assessment_log (student_id_hash, objective, question, student_answer, ai_feedback) VALUES (?, ?, ?, ?, ?)', 
                                   (hashed_id, str(st.session_state.associated_objs), st.session_state.current_question, typed_ans, grade_response.text))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    st.warning(f"⚠️ Network error. Please click submit again.")

        st.divider()
        
        # --- TAB 3: CLINICAL APPLICATION WITH JSON ROUTER ---
        st.subheader("Step 3: Clinical Application")
        if st.button("Generate a Patient Case"):
            try:
                case_prompt = f"""
                Using PRIMARILY the Slides and Handout within the CLINICAL MATERIALS provided in your context (using the Textbook materials only for supplemental backup), generate ONE challenging patient vignette (case study) testing the student's criteria: {st.session_state.associated_objs}. 
                
                CRITICAL CRITERIA LIMIT:
                Select a manageable subset (maximum of 2 specific objectives or performance criteria) from their list to focus on for this specific case narrative.

                CRITICAL RULES: No generic cases. End with a clinical decision question. Do NOT provide the answer in the case text.
                
                - "answer_key": string (If requires_math is true, you MUST write out the mathematical formula used, plug in the variables step-by-step, and only calculate the final dose at the very end. If false, write 'N/A').
                """
                with st.spinner("Coach is writing a grounded patient case..."):
                    case_res = ask_coach(case_prompt, model=HEAVY_MODEL, use_cache=True, response_schema=CaseSchema)
                
                case_data = json.loads(case_res.text)
                st.session_state.t3_case_text = case_data.get("case_text", "Error loading case text.")
                st.session_state.t3_requires_math = case_data.get("requires_math", False)
                st.session_state.t3_hidden_answer_key = case_data.get("answer_key", "N/A")
            except Exception as e:
                st.warning(f"⚠️ Network Error. Please click Generate again.")

        if st.session_state.t3_case_text:
            st.info(st.session_state.t3_case_text)
            
            if st.session_state.t3_requires_math:
                st.warning("🧮 **Clinical Math Required:** Precise calculation is needed for this case.")
                case_typed = st.text_area("⌨️ Detail your mathematical steps and final dose here:")
            else:
                case_typed = st.text_area("⌨️ Type your clinical decision here:")
            
            st.write("--- OR ---")
            
            if not st.session_state.t3_case_cam_active and st.session_state.t3_case_photo is None:
                if st.button("📸 Open Webcam for Scratchpad", key="t3_case_open"):
                    st.session_state.t3_case_cam_active = True
                    st.rerun()

            elif st.session_state.t3_case_cam_active:
                if st.button("❌ Turn Camera Off", key="t3_case_close"):
                    st.session_state.t3_case_cam_active = False
                    st.rerun()
                    
                t3_case_cam = st.camera_input("Capture your care plan:", key="t3_case_input")
                if t3_case_cam is not None:
                    st.session_state.t3_case_photo = t3_case_cam
                    st.session_state.t3_case_cam_active = False
                    st.rerun()

            if st.session_state.t3_case_photo is not None:
                st.success("✅ Calculations captured! (Camera is OFF)")
                st.image(st.session_state.t3_case_photo, width=300)
                if st.button("🗑️ Retake Photo", key="t3_case_retake"):
                    st.session_state.t3_case_photo = None
                    st.session_state.t3_case_cam_active = True
                    st.rerun()

            if st.button("📤 Submit Case for Faculty Coach Review"):
                if not case_typed.strip() and st.session_state.t3_case_photo is None:
                    st.warning("Please provide a decision.")
                else:
                    try:
                        case_grade_prompt = f"""
                        Your objective is to evaluate the student's clinical decision.

                        Follow these strict grading protocols:
                        1. **Clinical Accuracy:** Grade the clinical logic primarily against the Slides and Handout within the CLINICAL MATERIALS, using the Textbook materials as a supplemental backup.
                        2. **EBLS Coaching Integration:** Diagnose *where* their clinical application failed using the EBLS FRAMEWORK provided in your context. Prescribe specific corrective action.
                        3. **MATH PROTOCOL:** Does this case require math? {'YES' if st.session_state.t3_requires_math else 'NO'}.
                        True Answer Key: {st.session_state.t3_hidden_answer_key}
                        If YES, compare the student's answer strictly against the 'True Answer Key'. Do not recalculate the math yourself. 
                        
                        Patient Case: {st.session_state.t3_case_text}
                        Student Answer: {case_typed if case_typed else 'See attached image.'}
                        """
                        img = PIL.Image.open(st.session_state.t3_case_photo) if st.session_state.t3_case_photo else None
                        
                        with st.spinner("Faculty Coach is reviewing your clinical decision..."):
                            case_eval = ask_coach(case_grade_prompt, image=img, model=HEAVY_MODEL, use_cache=True)
                        st.markdown("### Case Feedback")
                        st.success(case_eval.text)

                        conn = sqlite3.connect('course_data.db')
                        conn.execute('PRAGMA busy_timeout=5000;')
                        cursor = conn.cursor()
                        cursor.execute('INSERT INTO assessment_log (student_id_hash, objective, question, student_answer, ai_feedback) VALUES (?, ?, ?, ?, ?)', 
                                       (hashed_id, f"Clinical Case: {str(st.session_state.associated_objs)}", st.session_state.t3_case_text, case_typed, case_eval.text))
                        conn.commit()
                        conn.close()
                    except Exception as e:
                        st.error(f"⚠️ Network Error. Please click submit again.")

        st.divider()
        if st.button("🔄 Start New Study Session"):
            st.session_state.study_step = "intake"
            st.session_state.current_question = None
            st.session_state.t3_task_requires_math = False
            st.session_state.t3_task_answer_key = "N/A"
            st.session_state.t3_case_text = None
            st.session_state.t3_requires_math = False
            st.session_state.t3_hidden_answer_key = "N/A"
            st.session_state.t3_exec_photo = None
            st.session_state.t3_case_photo = None
            st.rerun()

# ==========================================
# TAB 4: INTERLEAVED PRACTICE EXAMS
# ==========================================
with tab4:
    st.header("🎯 Interleaved Practice Exams")
    st.markdown("Generate a practice question or clinical case tailored to your exact study material.")

    if "t4_question_text" not in st.session_state: st.session_state.t4_question_text = None
    if "t4_requires_math" not in st.session_state: st.session_state.t4_requires_math = False
    if "t4_hidden_answer_key" not in st.session_state: st.session_state.t4_hidden_answer_key = "N/A"
    
    if "test_me_format" not in st.session_state: st.session_state.test_me_format = None
    
    if "t4_cam_active" not in st.session_state: st.session_state.t4_cam_active = False
    if "t4_photo" not in st.session_state: st.session_state.t4_photo = None

    selected_macros_t4 = st.multiselect("🎯 Select Learning Objective(s):", options=list(course_objectives_dict.keys()), key="t4_macro")

    criteria_options_t4 = []
    for macro in selected_macros_t4:
        for micro in course_objectives_dict[macro]:
            criteria_options_t4.append(f"{macro}: {micro}")

    with st.form("test_me_intake"):
        st.write("**Target Specific Performance Criteria:**")
        test_objs = st.multiselect("Select Performance Criteria:", options=criteria_options_t4)
        
        test_confusion = st.text_area("🎯 What areas are you most confused about?")
        test_format = st.radio("📝 Select Assessment Type:", ["Multiple Choice Question", "Open-Ended Clinical Case"])

        if st.form_submit_button("Generate Assessment"):
            if not selected_macros_t4:
                st.error("Please select at least one Learning Objective.")
            elif not test_objs:
                st.error("Please associate this assessment with at least one Performance Criterion.")
            else:
                try:
                    if test_format == "Multiple Choice Question":
                        # DYNAMIC FORK 1: Strict limits for MCQs
                        obj_instruction = f"From the following list of targeted performance criteria, randomly select 1 or 2 to focus on for this assessment: {test_objs}. Do NOT try to test all of them at once. "
                        
                        gen_prompt = f"""
                        Based PRIMARILY on the Slides and Handout within the CLINICAL MATERIALS provided in your context (using the Textbook materials only as a supplemental backup), generate a board-style Multiple Choice Question (A,B,C,D). 
                        {obj_instruction} Confusion focus: '{test_confusion}'. 
                        
                        - "answer_key": string (Explicitly state the correct letter. If math is required, you MUST write out the mathematical formula used, plug in the variables step-by-step, and calculate the final dose at the very end).
                        """
                    else:
                        # DYNAMIC FORK 2: Maximum complexity for Clinical Cases
                        obj_instruction = f"The assessment MUST test concepts from these selected criteria: {test_objs}. "
                        
                        gen_prompt = f"""
                        Based PRIMARILY on the Slides and Handout within the CLINICAL MATERIALS provided in your context (using the Textbook materials only as a supplemental backup), generate a comprehensive, deep clinical vignette. 
                        {obj_instruction} Confusion focus: '{test_confusion}'. End with a series of connected clinical decision questions. 
                        
                        COMPLEXITY LEEWAY: 
                        Do not hold back on the scope. If the student selects a wide array of criteria, build a robust, multi-stage patient tracking scenario (e.g., critical illness evolution, metabolic shifts, medication conflicts, and exact compounding demands) that requires them to synthesize multiple clinical rules simultaneously.

                        - "answer_key": string (If math is required, you MUST write out the mathematical formula used, plug in the variables step-by-step, and calculate the final dose at the very end. If qualitative, the key clinical decisions required).
                        """
                    
                    with st.spinner("Coach is building your custom test..."):
                        test_res = ask_coach(gen_prompt, model=HEAVY_MODEL, use_cache=True, response_schema=AssessmentSchema)
                    
                    test_data = json.loads(test_res.text)
                    st.session_state.t4_question_text = test_data.get("question_text", "Error loading question.")
                    st.session_state.t4_requires_math = test_data.get("requires_math", False)
                    st.session_state.t4_hidden_answer_key = test_data.get("answer_key", "N/A")
                    
                    st.session_state.test_me_format = test_format
                    st.session_state.t4_photo = None
                    st.session_state.t4_cam_active = False
                except Exception as e:
                    st.error(f"⚠️ Network error or generator failure. Please try clicking Generate again.")

    if st.session_state.t4_question_text:
        st.divider()
        st.markdown("### Your Custom Assessment")
        st.info(st.session_state.t4_question_text)

        if st.session_state.test_me_format == "Multiple Choice Question":
            mcq_answer = st.radio("Select your answer:", ["A", "B", "C", "D"], key="t4_mcq_radio")
        else:
            if st.session_state.t4_requires_math:
                st.warning("🧮 **Clinical Math Required:** Precise calculation is needed for this case.")
                mcq_answer = st.text_area("⌨️ Detail your mathematical steps and final dose here:", key="t4_math_area")
            else:
                mcq_answer = st.text_area("⌨️ Type your clinical decision here:", key="t4_clinical_area")
                
            st.write("--- OR ---")
            
            if not st.session_state.t4_cam_active and st.session_state.t4_photo is None:
                if st.button("📸 Open Webcam", key="t4_open"):
                    st.session_state.t4_cam_active = True
                    st.rerun()

            elif st.session_state.t4_cam_active:
                if st.button("❌ Turn Camera Off", key="t4_close"):
                    st.session_state.t4_cam_active = False
                    st.rerun()
                    
                t4_cam = st.camera_input("Capture your paper calculations:", key="t4_cam_input")
                if t4_cam is not None:
                    st.session_state.t4_photo = t4_cam
                    st.session_state.t4_cam_active = False
                    st.rerun()

            if st.session_state.t4_photo is not None:
                st.success("✅ Calculations captured! (Camera is OFF)")
                st.image(st.session_state.t4_photo, width=300)
                if st.button("🗑️ Retake Photo", key="t4_retake"):
                    st.session_state.t4_photo = None
                    st.session_state.t4_cam_active = True
                    st.rerun()

        if st.button("📤 Submit Answer for Grading"):
            if st.session_state.test_me_format == "Open-Ended Clinical Case" and not mcq_answer.strip() and st.session_state.t4_photo is None:
                st.warning("Please provide an answer before submitting.")
            else:
                try:
                    grade_prompt = f"""
                    Evaluate the student's test submission.

                    Follow these strict grading protocols:
                    1. **Clinical Accuracy:** Grade the clinical logic primarily against the Slides and Handout within the CLINICAL MATERIALS, using the Textbook materials as a supplemental backup.
                    2. **EBLS Exam Wrapper (Phase 5):** If the student gets the answer wrong, use the "Exam Wrapper Algorithm" from the EBLS FRAMEWORK in your context to diagnose their failure point.
                    3. **MATH/LOGIC PROTOCOL:** Does this case require strict matching? {'YES' if st.session_state.t4_requires_math or st.session_state.test_me_format == 'Multiple Choice Question' else 'NO'}.
                    True Answer Key: {st.session_state.t4_hidden_answer_key}
                    If YES, you MUST compare the student's answer strictly against the 'True Answer Key'. Do not recalculate math yourself. 

                    Question: {st.session_state.t4_question_text}
                    Student Answer: {mcq_answer if mcq_answer else 'See attached image.'}
                    """
                    
                    img = PIL.Image.open(st.session_state.t4_photo) if st.session_state.t4_photo else None
                    with st.spinner("Grading..."):
                        grade_res = ask_coach(grade_prompt, image=img, model=HEAVY_MODEL, use_cache=True)
                    
                    st.markdown("### Faculty Coach Feedback")
                    st.success(grade_res.text)

                    conn = sqlite3.connect('course_data.db')
                    conn.execute('PRAGMA busy_timeout=5000;')
                    cursor = conn.cursor()
                    cursor.execute('INSERT INTO assessment_log (student_id_hash, objective, question, student_answer, ai_feedback) VALUES (?, ?, ?, ?, ?)', 
                                   (hashed_id, f"Test Me Drill: {st.session_state.test_me_format}", st.session_state.t4_question_text, mcq_answer, grade_res.text))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    st.error(f"⚠️ Network error. Please click submit again.")
        
        st.divider()
        if st.button("🔄 Clear Test and Start Over"):
            st.session_state.t4_question_text = None
            st.session_state.t4_requires_math = False
            st.session_state.t4_hidden_answer_key = "N/A"
            st.session_state.t4_photo = None
            st.session_state.t4_cam_active = False
            st.rerun()

# ==========================================
# TAB 5: MASTERY & COACH AUDIT
# ==========================================
with tab5:
    st.header("📈 Mastery & Coach Audit")
    st.markdown("Review your data, request an audit from your Coach, and update your mastery status.")

    conn = sqlite3.connect('course_data.db')

    # Self-Assessment & Coach Audit
    st.subheader("Self-Assessment & Coach Audit")
    
    selected_macros_t5 = st.multiselect("🎯 Select Learning Objective(s):", options=list(course_objectives_dict.keys()), key="audit_macro_select_t5")
    
    criteria_options_t5 = []
    for macro in selected_macros_t5:
        for micro in course_objectives_dict[macro]:
            criteria_options_t5.append(f"{macro}: {micro}")

    audit_objs = st.multiselect("Select Performance Criteria to Update:", options=criteria_options_t5, key="audit_micro_select_t5")
    
    if st.button("📊 Assess My Progress (Coach Audit)"):
        if not audit_objs:
            st.warning("Please select at least one Performance Criterion to audit.")
        else:
            try:
                conn.execute('PRAGMA busy_timeout=5000;')
                cursor = conn.cursor()
                history = []
                for obj in audit_objs:
                    cursor.execute('''
                        SELECT question, student_answer, ai_feedback 
                        FROM assessment_log 
                        WHERE student_id_hash = ? AND objective LIKE ? 
                        ORDER BY timestamp DESC LIMIT 5
                    ''', (hashed_id, f"%{obj}%"))
                    history.extend(cursor.fetchall())

                if not history:
                    st.warning("⚠️ The Coach doesn't have enough data yet. Complete a study activity or clinical case for the selected criteria first!")
                else:
                    formatted_history = "\n\n".join([f"Task: {row[0]}\nStudent Answer: {row[1]}\nPast Grade/Feedback: {row[2]}" for row in history])
                    
                    audit_prompt = f"""
                    You are evaluating a student's mastery of the following performance criteria: {audit_objs}.

                    Review their recent study history below. Based strictly on their performance, recommend a mastery level using the "Boxing Technique" rules found in the EBLS FRAMEWORK provided in your context.

                    Output format: 
                    1. Start with the color emoji and status (🔴 Needs Review, 🟡 In Progress, 🟢 Mastered).
                    2. Follow with a 2-sentence metacognitive justification based on their data. Tell them exactly what phase of the S.A.L.A.M.I. protocol they need to focus on next.

                    --- RECENT STUDY HISTORY ---
                    {formatted_history}
                    """
                    
                    with st.spinner("Faculty Coach is auditing your recent performance data..."):
                        audit_res = ask_coach(audit_prompt, model=HEAVY_MODEL, use_cache=True)
                        
                    st.info(f"**Coach Recommendation:**\n{audit_res.text}")
            except Exception as e:
                st.warning(f"⚠️ Network error loading audit. Please try again.")

    with st.form("update_mastery_form"):
        new_status = st.radio("Commit to Mastery Level:", options=['🔴 Needs Review', '🟡 In Progress', '🟢 Mastered', '🔵 Untested'])
        if st.form_submit_button("Update Dashboard"):
            if not audit_objs:
                st.error("Please select at least one Performance Criterion above.")
            else:
                conn.execute('PRAGMA busy_timeout=5000;')
                cursor = conn.cursor()
                for obj in audit_objs:
                    cursor.execute("SELECT status FROM mastery_dashboard WHERE student_id_hash = ? AND objective = ?", (hashed_id, obj))
                    result = cursor.fetchone()
                    old_status = result[0] if result else "None"
                    
                    cursor.execute('REPLACE INTO mastery_dashboard (student_id_hash, objective, status) VALUES (?, ?, ?)', (hashed_id, obj, new_status))
                    cursor.execute('INSERT INTO mastery_history (student_id_hash, objective, old_status, new_status) VALUES (?, ?, ?, ?)', (hashed_id, obj, old_status, new_status))
                conn.commit()
                st.success("Status updated! Refreshing data...")
                st.rerun()

    st.divider()

    # Status Dashboard
    st.subheader("Current Performance Criteria Mastery")
    try:
        df_dashboard = pd.read_sql_query("SELECT objective, status FROM mastery_dashboard WHERE student_id_hash = ?", conn, params=(hashed_id,))
        
        if not df_dashboard.empty:
            db_dict = dict(zip(df_dashboard['objective'], df_dashboard['status']))
            
            for macro, micros in course_objectives_dict.items():
                statuses = [db_dict.get(f"{macro}: {micro}", "🔵 Untested") for micro in micros]
                
                if "🔴 Needs Review" in statuses:
                    macro_status = "🔴"
                elif "🟡 In Progress" in statuses:
                    macro_status = "🟡"
                elif all(s == "🟢 Mastered" for s in statuses):
                    macro_status = "🟢"
                elif all(s == "🔵 Untested" for s in statuses):
                    macro_status = "🔵"
                else:
                    macro_status = "🟡"
                    
                st.markdown(f"#### {macro_status} | {macro}")
                for micro in micros:
                    m_stat = db_dict.get(f"{macro}: {micro}", "🔵 Untested")
                    st.markdown(f"- {m_stat} | {micro}")
        else:
            st.info("No mastery statuses logged yet.")
    except Exception as e:
        st.warning("⚠️ Database busy, unable to load dashboard right now. Please refresh.")

    st.divider()

    st.subheader("Mastery Updates Over Time")
    try:
        df_mastery = pd.read_sql_query("SELECT timestamp as Date, objective as 'Performance Criteria', old_status as 'Previous Status', new_status as 'Current Status' FROM mastery_history WHERE student_id_hash = ? ORDER BY timestamp DESC", conn, params=(hashed_id,))
        if not df_mastery.empty:
            df_mastery['Date'] = pd.to_datetime(df_mastery['Date']).dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(df_mastery, use_container_width=True, hide_index=True)
        else:
            st.info("No mastery updates logged yet.")
    except Exception as e:
        pass # Silently pass if DB is locked here, user can just refresh

    st.divider()

    st.subheader("Faculty Coach Feedback Ledger")
    try:
        df_assessments = pd.read_sql_query("SELECT timestamp as Date, objective as 'Performance Criteria', question as Question, ai_feedback as Feedback FROM assessment_log WHERE student_id_hash = ? ORDER BY timestamp DESC", conn, params=(hashed_id,))
        if not df_assessments.empty:
            df_assessments['Date'] = pd.to_datetime(df_assessments['Date']).dt.strftime('%Y-%m-%d %H:%M')
            for index, row in df_assessments.iterrows():
                with st.expander(f"{row['Date']} | {row['Performance Criteria']}"):
                    st.markdown(f"**Prompt/Question:** {row['Question']}")
                    st.markdown(f"**Coach Grading:** {row['Feedback']}")
        else:
            st.info("No assessments completed yet.")
    except Exception as e:
        pass

    conn.close()
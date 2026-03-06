import streamlit as st
import pandas as pd
import numpy as np
import os
import zipfile
import re
import spacy
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import docx2txt
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import google.generativeai as genai
import json
from dotenv import load_dotenv

load_dotenv()
# Set up Gemini API key
GEMINI_API_KEY =os.getenv("API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


# Set page configuration
st.set_page_config(
    page_title="Resume Screening & Ranking System",
    page_icon="📄",
    layout="wide"
)

# Create folders if they don't exist
if not os.path.exists("uploaded_resumes"):
    os.makedirs("uploaded_resumes")

if not os.path.exists("assets"):
    os.makedirs("assets")
    os.makedirs("assets/ResumeModel")
    os.makedirs("assets/ResumeModel/output")

# Function to load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        # Try to load a pre-trained model if available
        return spacy.load('en_core_web_sm')
    except:
        # If model isn't available, download it
        st.info("Downloading language model for the first time (this may take a while)")
        spacy.cli.download("en_core_web_sm")
        return spacy.load('en_core_web_sm')

nlp = load_spacy_model()

# Improved function to extract text from PDF with better structure preservation
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            
            # Extract text blocks to preserve structure
            blocks = page.get_text("blocks")
            for block in blocks:
                text += block[4] + "\n"
                
            # Add extra line break between pages
            text += "\n"
            
        # Clean up text - remove excessive newlines and spaces
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# Function to extract text from DOCX
def extract_text_from_docx(docx_file):
    try:
        text = docx2txt.process(docx_file)
        # Clean up text - remove excessive newlines and spaces
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return ""

# Enhanced function to process resumes and extract information
def process_resume(text):
    # Process with spaCy
    doc = nlp(text[:100000])  # Limit to prevent memory issues with large docs
    
    # Initialize dictionary for resume data
    resume_data = {
        "name": None,
        "email": None,
        "phone": None,
        "skills": [],
        "education": [],
        "experience": [],
        "projects": [],
        "certifications": [],
        "cgpa": None
    }
    
    # Extract email with improved pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        resume_data["email"] = emails[0]
    
    # Extract phone numbers with improved pattern
    phone_pattern = r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\d{10}'
    phones = re.findall(phone_pattern, text)
    if phones:
        resume_data["phone"] = phones[0]
    
    # Extract CGPA with improved pattern matching
    # This pattern matches common CGPA formats across many resume types
    cgpa_patterns = [
        r'(?:CGPA|GPA)(?:\s*[:of]\s|\s*[-=]?\s*)(\d+(?:\.\d+)?)',
        r'(?:CGPA|GPA)(?:\s*[:of]\s|\s*[-=]?\s*)(\d+(?:\.\d+)?)/\d+(?:\.\d+)?',
        r'(?:CGPA|GPA)(?:.?)(\d+\.\d+)(?:\s\/\s*\d+(?:\.\d+)?)?',
    ]
    
    for pattern in cgpa_patterns:
        cgpa_matches = re.findall(pattern, text, re.IGNORECASE)
        if cgpa_matches:
            try:
                # Extract just the number part if there's a fraction
                cgpa_value = cgpa_matches[0].split('/')[0] if '/' in cgpa_matches[0] else cgpa_matches[0]
                resume_data["cgpa"] = float(cgpa_value)
                break
            except:
                continue
    
    edu_section = re.search(r'(?:EDUCATION|ACADEMIC QUALIFICATIONS|QUALIFICATIONS|EDUCATIONAL BACKGROUND)[\s\S]*?(?=\n[A-Z][A-Z ]+\n|\Z)', 
                            text, re.IGNORECASE)

    if edu_section:
        edu_text = edu_section.group().strip()  # Get entire education section

        # Split into individual entries based on line breaks
        edu_lines = edu_text.split("\n")

        # Extract relevant information (degree, institution, year, CGPA, percentage)
        degree_pattern = r"(B\.?Tech|M\.?Tech|Ph\.?D|Bachelor|Master|MBA|MSc|BSc|BE|ME|Diploma|Intermediate)[^,\n]*"
        institution_pattern = r"(University|College|Institute|School)[^,\n]*"
        year_pattern = r"(\d{4}\s?[-–]\s?\d{4}|Expected\s?[A-Za-z]+\s?\d{4})"
        score_pattern = r"(CGPA[:\s]*\d+\.\d+|Percentage[:\s]*\d+\.\d+)"

        current_entry = ""
        for line in edu_lines:
            if re.search(degree_pattern, line, re.IGNORECASE):
                if current_entry:  
                    resume_data["education"].append(current_entry.strip())  # Store previous entry
                current_entry = line  # Start new entry
            elif re.search(institution_pattern, line, re.IGNORECASE) or re.search(year_pattern, line) or re.search(score_pattern, line):
                current_entry += " " + line  # Append related information to current degree entry
            else:
                continue

        # Append last entry
        if current_entry:
            resume_data["education"].append(current_entry.strip())

    
    # Extract skills with expanded keywords and improved pattern matching
    skill_keywords = [
        "python", "java", "c\\+\\+", "javascript", "typescript", "html", "css", "react", "angular", 
        "node\\.js", "django", "flask", "express", "mongodb", "mysql", "postgresql", "redis",
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "git", "devops", "ci/cd",
        "machine learning", "deep learning", "data analysis", "data science", "artificial intelligence", 
        "tensorflow", "pytorch", "keras", "nlp", "computer vision", "rust", "golang", "scala",
        "hadoop", "spark", "kafka", "tableau", "power bi", "excel", "word", "powerpoint", "sql",
        "communication", "leadership", "teamwork", "problem solving", "critical thinking",
        "agile", "scrum", "kanban", "project management", "time management", "jenkins", "jira",
        "graphql", "rest api", "microservices", "spring boot", "oop", "functional programming"
    ]
    
    # Find skills section
    skills_section = re.search(r'(?:SKILLS?|TECHNICAL SKILLS?|TECHNOLOGIES?)[^\n]\n+(.?)(?:\n\n|\n[A-Z]{2,}|\Z)', 
                            text, re.IGNORECASE | re.DOTALL)
    
    if skills_section:
        skills_text = skills_section.group(1).lower()
        # Extract skills from the dedicated section
        for skill in skill_keywords:
            if re.search(r'\b' + re.escape(skill) + r'\b', skills_text):
                if skill not in resume_data["skills"]:
                    resume_data["skills"].append(skill)
    
    # Also look for skills throughout the document
    for skill in skill_keywords:
        if re.search(r'\b' + re.escape(skill) + r'\b', text.lower()):
            if skill not in resume_data["skills"]:
                resume_data["skills"].append(skill)
    
    # Extract experience with improved pattern matching
    exp_section = re.search(r'(?:EXPERIENCE|WORK EXPERIENCE|EMPLOYMENT|WORK HISTORY)[\s\S]*?(?=\n[A-Z][A-Z ]+\n|\Z)', 
                            text, re.IGNORECASE)

    if exp_section:
        exp_text = exp_section.group().strip()  # Get full experience section

        # Split based on new lines for better structuring
        exp_lines = exp_text.split("\n")

        experiences = []
        current_exp = ""

        for line in exp_lines:
            # Check if the line contains a job title (Assuming job title starts with capital letters)
            if re.match(r"^[A-Z].{3,}", line):  
                if current_exp:
                    experiences.append(current_exp.strip())  # Store previous experience
                current_exp = line  # Start new experience
            else:
                current_exp += " " + line  # Append details to current job

        # Append the last job experience
        if current_exp:
            experiences.append(current_exp.strip())

        resume_data["experience"] = experiences
    
    # Extract certifications with improved pattern matching
    cert_section = re.search(r'(?:CERTIFICATIONS?|COURSES?|TRAINING)[\s\S]*?(?=\n[A-Z][A-Z ]+\n|\Z)', 
                             text, re.IGNORECASE)

    if cert_section:
        cert_text = cert_section.group().strip()

        # Extract certification lines and remove empty lines
        certs = [line.strip() for line in cert_text.split("\n") if line.strip()]
        
        # Remove the section heading from the list
        if certs:
            certs.pop(0)  

        resume_data["certifications"] = certs
    
    # Extract name using NER but also try alternative approaches
    # First try to extract using NER
    for ent in doc.ents:
        if ent.label_ == "PERSON" and not resume_data["name"]:
            # Check if this appears at the beginning of the document (likely the candidate's name)
            if text.find(ent.text) < 500:  # Only consider names near the start
                resume_data["name"] = ent.text
    
    # If name not found, try to use the first line that's not too long
    if not resume_data["name"]:
        first_lines = text.strip().split('\n')[:5]  # Check first 5 lines
        for line in first_lines:
            line = line.strip()
            # A name is typically short and doesn't contain special characters
            if 2 <= len(line.split()) <= 5 and re.match(r'^[A-Za-z\s.]+$', line):
                resume_data["name"] = line
                break
    
    project_section = re.search(r'(?:PROJECTS?|ACADEMIC PROJECTS?|PERSONAL PROJECTS?)[\s\S]*?(?=\n[A-Z][A-Z ]+\n|\Z)', 
                                text, re.IGNORECASE)

    if project_section:
        project_text = project_section.group().strip()  # Get full projects section

        # Split based on line breaks, ensuring project entries are grouped correctly
        project_lines = project_text.split("\n")

        projects = []
        current_project = ""

        for line in project_lines:
            # If the line starts with a likely project title (capitalized words), start a new project
            if re.match(r"^[A-Za-z].{5,}", line):  # Ensures a valid project title (avoids single words)
                if current_project:
                    projects.append(current_project.strip())  # Store previous project
                current_project = line  # Start new project
            else:
                current_project += " " + line  # Append description lines to current project

        # Append the last project
        if current_project:
            projects.append(current_project.strip())

        resume_data["projects"] = projects
    
    return resume_data

# Enhanced function to calculate match score between resume and job requirements
def calculate_match_score(resume_data, required_skills, min_cgpa, role_name):
    score = 0
    max_score = 100
    
    # Skills matching (50%)
    if required_skills:
        skills_score = 0
        matched_skills = []
        
        for skill in required_skills:
            skill_lower = skill.lower()
            # Check for exact matches and partial matches
            if any(skill_lower == s.lower() for s in resume_data["skills"]):
                skills_score += 1
                matched_skills.append(skill)
            elif any(skill_lower in s.lower() for s in resume_data["skills"]):
                skills_score += 0.5  # Partial match
                matched_skills.append(skill + " (partial)")
        
        if len(required_skills) > 0:
            skills_percentage = (skills_score / len(required_skills)) * 50
        else:
            skills_percentage = 0
        score += skills_percentage
    
    # CGPA matching (20%)
    cgpa_score = 0
    if min_cgpa and resume_data["cgpa"]:
        cgpa_score = min(resume_data["cgpa"] / min_cgpa, 1) * 20
        score += cgpa_score
    
    # Role matching (15%)
    role_score = 0
    if role_name:
        text = " ".join([
            " ".join(resume_data["experience"]) if resume_data["experience"] else "",
            " ".join(resume_data["projects"]) if resume_data["projects"] else "",
            " ".join(resume_data["education"]) if resume_data["education"] else ""
        ])
        
        # Check for exact and partial role matches
        if re.search(r'\b' + re.escape(role_name) + r'\b', text, re.IGNORECASE):
            role_score = 15
        elif any(word.lower() in text.lower() for word in role_name.split()):
            role_score = 7.5  # Partial match
        
        score += role_score
    
    # Projects and certifications (15%)
    projects_count = len(resume_data["projects"])
    certs_count = len(resume_data["certifications"])
    projects_certs_score = min(projects_count + certs_count, 5) / 5 * 15
    score += projects_certs_score
    
    # Return overall score and component scores for detailed analysis
    return {
        "total_score": round(score, 2),
        "skills_score": round(skills_percentage if 'skills_percentage' in locals() else 0, 2),
        "cgpa_score": round(cgpa_score, 2),
        "role_score": round(role_score, 2),
        "projects_certs_score": round(projects_certs_score, 2),
        "matched_skills": matched_skills if 'matched_skills' in locals() else []
    }

# Function to handle file upload
def handle_uploaded_file(uploaded_file):
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    if file_ext == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_ext in ['docx', 'doc']:
        return extract_text_from_docx(uploaded_file)
    else:
        st.warning(f"Unsupported file format: {file_ext}")
        return ""

# Enhanced function to handle zip file upload
def handle_zip_upload(zip_file):
    resume_texts = {}
    
    try:
        with zipfile.ZipFile(zip_file) as z:
            for file_name in z.namelist():
                if file_name.endswith(('.pdf', '.docx', '.doc')) and not file_name.startswith('__MACOSX'):
                    with z.open(file_name) as f:
                        content = BytesIO(f.read())
                        if file_name.endswith('.pdf'):
                            try:
                                pdf_document = fitz.open(stream=content.getvalue(), filetype="pdf")
                                text = ""
                                for page_num in range(len(pdf_document)):
                                    page = pdf_document.load_page(page_num)
                                    # Extract text blocks to preserve structure
                                    blocks = page.get_text("blocks")
                                    for block in blocks:
                                        text += block[4] + "\n"
                                    # Add extra line break between pages
                                    text += "\n"
                                
                                # Clean up text
                                text = re.sub(r'\n{3,}', '\n\n', text)
                                text = re.sub(r' {2,}', ' ', text)
                                
                                resume_texts[file_name] = text
                            except Exception as e:
                                st.error(f"Error processing {file_name}: {e}")
                        elif file_name.endswith(('.docx', '.doc')):
                            try:
                                text = docx2txt.process(content)
                                text = re.sub(r'\n{3,}', '\n\n', text)
                                text = re.sub(r' {2,}', ' ', text)
                                resume_texts[file_name] = text
                            except Exception as e:
                                st.error(f"Error processing {file_name}: {e}")
    except Exception as e:
        st.error(f"Error processing zip file: {e}")
    
    return resume_texts

# Function to create a download link for CSV file
def get_csv_download_link(df, filename="ranked_candidates.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="btn" style="background-color:#4CAF50;color:white;padding:8px 12px;text-decoration:none;border-radius:4px;">Download Ranked Candidates CSV</a>'
    return href

def analyze_with_gemini(resume_data, job_description):
    prompt = f"""
        You are an AI specialized in resume screening. Analyze the following resume and compare it to the given job description.

        **Resume Data:**
        {resume_data}

        **Job Description:**
        {job_description}

        **Scoring Guide:**
        - **Degree:** Prioritize major over degree level. More relevant degrees get higher scores.
        - **Experience:** More relevant experience gets higher scores.
        - **Technical Skills:** More matching technical skills get higher scores.
        - **Responsibilities:** More matching responsibilities get higher scores.
        - **Certificates:** Required certificates get full points; related certificates get partial points.
        - **Soft Skills:** Foreign language and leadership skills are prioritized.

        **Evaluation Criteria (0-100 Scale):**
        - **Degree:** Evaluate relevance of candidate's degree.
        - **Experience:** Assess relevance and duration of experience.
        - **Technical Skills:** Compare technical expertise to job requirements.
        - **Responsibilities:** Match past responsibilities with job expectations.
        - **Certificates:** Score based on required and related certificates.
        - **Soft Skills:** Consider foreign languages and leadership.
        - **Overall Summary:** Provide a conclusion based on the scores.

        **Output:** Provide a **well-structured text-based** summary, NOT in JSON format.
        """

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        result = response.text  # Convert response to JSON
        return result
    except Exception as e:
        return {"error": str(e)}


# Main app UI
st.title("Automated Resume Screening & Ranking System")

tab1, tab2 = st.tabs(["Resume Screening", "About the System"])

with tab1:
    st.header("Upload Resumes & Set Requirements")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Resumes")
        upload_option = st.radio("Choose upload method:", ["Individual Files", "Zip File"])
        
        resume_texts = {}
        
        if upload_option == "Individual Files":
            uploaded_files = st.file_uploader("Upload resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
            if uploaded_files:
                with st.spinner("Processing uploaded files..."):
                    for file in uploaded_files:
                        text = handle_uploaded_file(file)
                        if text:
                            resume_texts[file.name] = text
                    
                    st.success(f"Successfully processed {len(resume_texts)} resumes")
        
        else:  # Zip File
            zip_file = st.file_uploader("Upload ZIP file containing resumes", type=["zip"])
            if zip_file:
                with st.spinner("Extracting resumes from ZIP file..."):
                    resume_texts = handle_zip_upload(zip_file)
                    st.success(f"Successfully extracted {len(resume_texts)} resumes from ZIP file")
    
    with col2:
        st.subheader("Set Job Requirements")
        
        role_name = st.text_input("Job Title/Role:", placeholder="e.g., Data Scientist, Software Engineer")
        
        min_cgpa = st.number_input("Minimum CGPA Required:", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
        
        skills_input = st.text_area("Required Skills (one per line):", 
                                   placeholder="e.g.,\nPython\nMachine Learning\nSQL")
        
        required_skills = [skill.strip() for skill in skills_input.split("\n") if skill.strip()]
        
        additional_keywords = st.text_input("Additional Keywords (comma separated):", 
                                           placeholder="e.g., AWS, leadership, agile")
        
        if additional_keywords:
            additional_keywords = [k.strip() for k in additional_keywords.split(",") if k.strip()]
            required_skills.extend(additional_keywords)
    
    if st.button("Screen & Rank Resumes") and resume_texts:
        with st.spinner("Processing resumes and calculating rankings..."):
            results = []
            detailed_data = {}
            
            for filename, text in resume_texts.items():
                resume_data = process_resume(text)
                match_scores = calculate_match_score(
                    resume_data, required_skills, min_cgpa, role_name
                )
                job_description = {
                   "required_skills": required_skills,  # List of required skills
                    "min_cgpa": min_cgpa,  # Minimum CGPA requirement
                    "role_name": role_name  # Job title
                                   }
                print(resume_data)
                analysis = analyze_with_gemini(resume_data,job_description)
                # Store detailed data for expanded view
                detailed_data[filename] = {
                    "resume_data": resume_data,
                    "match_scores": match_scores,
                    "Ai_Analysis":analysis
                }
                
                results.append({
                    "Filename": filename,
                    "Name": resume_data["name"] or "Unknown",
                    "Email": resume_data["email"] or "Not found",
                    "Phone": resume_data["phone"] or "Not found",
                    "CGPA": resume_data["cgpa"] or "Not found",
                    "Skills": ", ".join(resume_data["skills"]),
                    "Match Score": match_scores["total_score"],
                    "Ai Analysis":analysis
                })
            
            # Create DataFrame and sort by match score
            df_results = pd.DataFrame(results).sort_values(by="Match Score", ascending=False)
            
            # Display results
            st.header("Ranking Results")
            
            # Display top candidates
            st.subheader("Top Candidates")
            st.dataframe(df_results, use_container_width=True)
            
            # Create download link for CSV
            st.markdown(get_csv_download_link(df_results), unsafe_allow_html=True)
            
            # Show detailed info for top candidates
            st.subheader("Top Candidates Details")
            top_candidates = df_results.head(min(5, len(df_results)))
            
            for i, (_, row) in enumerate(top_candidates.iterrows(), 1):
                filename = row['Filename']
                detail = detailed_data[filename]
                
                with st.expander(f"#{i}: {row['Name']} - Match Score: {row['Match Score']}%"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"*Contact Information:*")
                        st.write(f"- Email: {row['Email']}")
                        st.write(f"- Phone: {row['Phone']}")
                        st.write(f"*Education:*")
                        st.write(f"- CGPA: {row['CGPA']}")
                        for edu in detail["resume_data"]["education"]:
                            st.write(f"- {edu}")
                    
                    with col2:
                        st.write(f"*Match Score Details:*")
                        st.write(f"- Skills: {detail['match_scores']['skills_score']}/50")
                        st.write(f"- CGPA: {detail['match_scores']['cgpa_score']}/20")
                        st.write(f"- Role: {detail['match_scores']['role_score']}/15")
                        st.write(f"- Projects & Certs: {detail['match_scores']['projects_certs_score']}/15")
                        
                        if detail["match_scores"]["matched_skills"]:
                            st.write("*Matched Skills:*")
                            for skill in detail["match_scores"]["matched_skills"]:
                                st.write(f"- {skill}")
                    
                    # Additional sections
                    if detail["resume_data"]["projects"]:
                        st.write("*Projects:*")
                        for project in detail["resume_data"]["projects"][:3]:  # Show up to 3 projects
                            st.write(f"- {project}")
                    
                    if detail["resume_data"]["certifications"]:
                        st.write("*Certifications:*")
                        for cert in detail["resume_data"]["certifications"][:3]:  # Show up to 3 certs
                            st.write(f"- {cert}")
                    
                    if detail["resume_data"]["experience"]:
                        st.write("*Experience:*")
                        for exp in detail["resume_data"]["experience"][:3]:  # Show up to 3 experiences
                            st.write(f"- {exp}")
            col1, col2, col3 = st.columns([1, 4, 1])  # Centering effect

            with col2:  # Place graph in the middle column
             st.subheader("Match Score Distribution")
    
             fig, ax = plt.subplots(figsize=(8,4))
             df_results_sorted = df_results.sort_values(by="Match Score", ascending=False)

             ax.bar(df_results_sorted["Name"], df_results_sorted["Match Score"], color='royalblue')
             ax.set_xlabel("Applicants")
             ax.set_ylabel("Match Score (%)")
             ax.set_title("Match Scores of Applicants")
             ax.set_xticklabels(df_results_sorted["Name"], rotation=45, ha="right", fontsize=8)
             ax.set_ylim(0, 100)

             fig.tight_layout()
    
             st.pyplot(fig)
with tab2:
    st.header("About the Resume Screening System")
    
    st.write("""
    This automated resume screening and ranking system helps employers efficiently filter through job applications
    by automatically extracting key information from resumes and ranking candidates based on job requirements.
    
    ### Key Features:
    
    1. *Resume Parsing*: Extracts essential information like name, email, skills, education, and experience from 
       uploaded resumes in PDF or DOCX formats.
       
    2. *Customizable Requirements*: Allows employers to specify job-specific requirements including:
       - Minimum CGPA
       - Required skills
       - Job title/role
       - Additional keywords
       
    3. *Smart Ranking*: Calculates match scores based on multiple factors:
       - Skills matching (50%)
       - CGPA comparison (20%)
       - Role relevance (15%)
       - Projects and certifications (15%)
       
    4. *Bulk Processing*: Supports both individual resume uploads and batch processing via ZIP files.
    
    5. *Exportable Results*: Results can be downloaded as CSV files for further analysis.
    """)
    
    st.info("""
    *Note*: This system is designed to assist in the initial screening process. 
    It is recommended to review the top candidates manually for a more comprehensive evaluation.
    """)
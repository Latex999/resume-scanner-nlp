"""
Resume Parser Module

This module handles the extraction of text from various document formats and
performs advanced NLP processing to extract structured information from resumes.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Union

import spacy
import PyPDF2
import docx
import textract
from loguru import logger
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np
from dateutil import parser as date_parser

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_lg")
    logger.info("Loaded spaCy model: en_core_web_lg")
except OSError:
    logger.warning("Downloading spaCy model...")
    spacy.cli.download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")


class ResumeParser:
    """
    A class for parsing and extracting information from resumes.
    
    This class handles document conversion, text extraction, and NLP processing
    to identify key sections and information in a resume.
    """
    
    def __init__(self, skills_path: str = "data/skills_taxonomy.csv"):
        """
        Initialize the ResumeParser with the necessary resources.
        
        Args:
            skills_path: Path to the skills taxonomy CSV file
        """
        self.nlp = nlp
        self.skills = self._load_skills_taxonomy(skills_path)
        
        # Initialize BERT model for semantic understanding
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        # Regular expressions for section detection
        self.section_patterns = {
            "contact": re.compile(r"(?i)(contact|email|phone|address|location)", re.IGNORECASE),
            "summary": re.compile(r"(?i)(summary|profile|objective|about me)", re.IGNORECASE),
            "experience": re.compile(r"(?i)(experience|work|employment|job|career)", re.IGNORECASE),
            "education": re.compile(r"(?i)(education|degree|university|college|school|academic)", re.IGNORECASE),
            "skills": re.compile(r"(?i)(skills|technologies|competencies|expertise|proficiency)", re.IGNORECASE),
            "projects": re.compile(r"(?i)(projects|portfolio|works)", re.IGNORECASE),
            "certifications": re.compile(r"(?i)(certifications|certificates|credentials)", re.IGNORECASE),
            "awards": re.compile(r"(?i)(awards|honors|achievements)", re.IGNORECASE),
            "languages": re.compile(r"(?i)(languages|linguistic)", re.IGNORECASE),
            "interests": re.compile(r"(?i)(interests|hobbies|activities)", re.IGNORECASE),
        }
        
        # Date patterns for experience duration extraction
        self.date_patterns = [
            re.compile(r"(?i)(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september|oct|october|nov|november|dec|december)\.?\s+\d{4}"),
            re.compile(r"(?i)\d{1,2}/\d{4}"),
            re.compile(r"(?i)\d{4}/\d{1,2}"),
            re.compile(r"(?i)\d{4}-\d{2}"),
            re.compile(r"(?i)\d{2}-\d{4}"),
        ]
        
        # Education degree levels for qualification ranking
        self.degree_levels = {
            "high school": 1,
            "associate": 2,
            "bachelor": 3,
            "master": 4,
            "phd": 5,
            "doctorate": 5,
            "mba": 4,
        }
        
        logger.info("ResumeParser initialized successfully")

    def _load_skills_taxonomy(self, skills_path: str) -> pd.DataFrame:
        """
        Load the skills taxonomy from a CSV file.
        If file doesn't exist, return an empty DataFrame.
        
        Args:
            skills_path: Path to the skills taxonomy CSV file
            
        Returns:
            DataFrame with skills taxonomy
        """
        try:
            if os.path.exists(skills_path):
                skills_df = pd.read_csv(skills_path)
                logger.info(f"Loaded {len(skills_df)} skills from taxonomy")
                return skills_df
            else:
                logger.warning(f"Skills taxonomy file not found at {skills_path}")
                return pd.DataFrame(columns=["skill", "category", "synonyms", "level"])
        except Exception as e:
            logger.error(f"Error loading skills taxonomy: {e}")
            return pd.DataFrame(columns=["skill", "category", "synonyms", "level"])

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        try:
            text = ""
            with open(pdf_path, "rb") as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
            
            # If PyPDF2 extraction is poor, try pdfminer as backup
            if len(text.strip()) < 100:
                logger.info("Using fallback PDF extractor for better results")
                text = textract.process(pdf_path, method='pdfminer').decode("utf-8")
                
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            # Fallback to textract as a last resort
            try:
                return textract.process(pdf_path).decode("utf-8")
            except:
                return ""

    def extract_text_from_docx(self, docx_path: str) -> str:
        """
        Extract text from a DOCX file.
        
        Args:
            docx_path: Path to the DOCX file
            
        Returns:
            Extracted text as a string
        """
        try:
            doc = docx.Document(docx_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {e}")
            # Fallback to textract
            try:
                return textract.process(docx_path).decode("utf-8")
            except:
                return ""

    def extract_text(self, file_path: str) -> str:
        """
        Extract text from various file formats.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text as a string
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == ".pdf":
            return self.extract_text_from_pdf(file_path)
        elif file_extension == ".docx":
            return self.extract_text_from_docx(file_path)
        elif file_extension == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        else:
            # For other formats, try using textract
            try:
                return textract.process(file_path).decode("utf-8")
            except Exception as e:
                logger.error(f"Error extracting text from {file_path}: {e}")
                return ""

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the extracted text for better NLP analysis.
        
        Args:
            text: Raw text extracted from the document
            
        Returns:
            Preprocessed text
        """
        # Remove excess whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Fix missing spaces between words
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\-\(\)\&\/\']', ' ', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def identify_sections(self, text: str) -> Dict[str, str]:
        """
        Identify and extract different sections from the resume.
        
        Args:
            text: Preprocessed resume text
            
        Returns:
            Dictionary with section names as keys and content as values
        """
        # Split text into lines for easier section detection
        lines = text.split('\n')
        current_section = "unknown"
        sections = {section: "" for section in self.section_patterns.keys()}
        sections["unknown"] = ""
        
        for i, line in enumerate(lines):
            # Check if line could be a section header
            if len(line.strip()) < 50 and not line.strip().endswith('.'):
                section_found = False
                for section, pattern in self.section_patterns.items():
                    if pattern.search(line):
                        current_section = section
                        section_found = True
                        break
                        
                if section_found:
                    continue
            
            # Add the line to the current section
            sections[current_section] += line + "\n"
        
        # Clean up whitespace in each section
        for section in sections:
            sections[section] = sections[section].strip()
        
        return sections

    def extract_contact_info(self, text: str) -> Dict[str, str]:
        """
        Extract contact information from the resume.
        
        Args:
            text: Resume text (typically from the contact section)
            
        Returns:
            Dictionary with contact information
        """
        contact_info = {
            "name": "",
            "email": "",
            "phone": "",
            "location": "",
            "linkedin": "",
            "github": "",
            "website": ""
        }
        
        # Email pattern
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        emails = email_pattern.findall(text)
        if emails:
            contact_info["email"] = emails[0]
        
        # Phone pattern - handles various formats
        phone_pattern = re.compile(r'\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b')
        phones = phone_pattern.findall(text)
        if phones:
            contact_info["phone"] = phones[0]
        
        # LinkedIn
        linkedin_pattern = re.compile(r'linkedin\.com/in/[\w-]+')
        linkedin = linkedin_pattern.findall(text)
        if linkedin:
            contact_info["linkedin"] = linkedin[0]
        
        # GitHub
        github_pattern = re.compile(r'github\.com/[\w-]+')
        github = github_pattern.findall(text)
        if github:
            contact_info["github"] = github[0]
        
        # Website
        website_pattern = re.compile(r'https?://(?:www\.)?(?!linkedin\.com|github\.com)[\w.-]+\.[A-Za-z]{2,}[\w/-]*')
        website = website_pattern.findall(text)
        if website:
            contact_info["website"] = website[0]
        
        # Use spaCy for name and location extraction
        doc = self.nlp(text[:500])  # Process just the beginning of text for efficiency
        
        # Extract name (usually the first PERSON entity near the beginning)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                contact_info["name"] = ent.text
                break
        
        # Extract location (GPE entities - cities, states, countries)
        locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
        if locations:
            contact_info["location"] = ", ".join(locations[:2])  # Limit to 2 location entities
        
        return contact_info

    def extract_skills(self, text: str) -> List[Dict[str, str]]:
        """
        Extract skills from the resume using the skills taxonomy.
        
        Args:
            text: Resume text
            
        Returns:
            List of dictionaries with skill information
        """
        found_skills = []
        
        # If we have a skills taxonomy
        if not self.skills.empty:
            # Check for exact matches
            for _, skill_row in self.skills.iterrows():
                skill_name = skill_row["skill"].lower()
                
                # Check for exact match
                if re.search(r'\b' + re.escape(skill_name) + r'\b', text.lower()):
                    found_skills.append({
                        "name": skill_row["skill"],
                        "category": skill_row["category"],
                        "level": skill_row.get("level", "")
                    })
                    continue
                
                # Check synonyms if available
                if "synonyms" in skill_row and pd.notna(skill_row["synonyms"]):
                    synonyms = [s.strip().lower() for s in skill_row["synonyms"].split(",")]
                    for synonym in synonyms:
                        if re.search(r'\b' + re.escape(synonym) + r'\b', text.lower()):
                            found_skills.append({
                                "name": skill_row["skill"],  # Use canonical name
                                "category": skill_row["category"],
                                "level": skill_row.get("level", "")
                            })
                            break
        else:
            # Fallback: extract potential skills using NLP
            doc = self.nlp(text)
            
            # Extract noun phrases as potential skills
            noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]
            
            # Extract recognized technology entities
            tech_entities = [ent.text for ent in doc.ents if ent.label_ in ["PRODUCT", "ORG"] and len(ent.text.split()) <= 3]
            
            # Combine and deduplicate
            potential_skills = list(set(noun_phrases + tech_entities))
            
            # Filter out common non-skill phrases
            non_skill_words = ["resume", "curriculum", "vitae", "cv", "email", "phone", "address"]
            filtered_skills = [s for s in potential_skills if not any(w in s.lower() for w in non_skill_words)]
            
            # Convert to same format as taxonomy-based skills
            found_skills = [{"name": skill, "category": "Unknown", "level": ""} for skill in filtered_skills]
        
        return found_skills

    def extract_education(self, education_text: str) -> List[Dict[str, Any]]:
        """
        Extract education information from the resume.
        
        Args:
            education_text: Text from the education section
            
        Returns:
            List of dictionaries with education information
        """
        education_entries = []
        
        # Process with spaCy
        doc = self.nlp(education_text)
        
        # Extract degree keywords
        degree_keywords = [
            "Bachelor", "BS", "B.S.", "BA", "B.A.", "Master", "MS", "M.S.", "MA", "M.A.",
            "PhD", "Ph.D.", "Doctorate", "Associate", "MBA", "M.B.A.", "BSc", "B.Sc.",
            "MSc", "M.Sc.", "BBA", "B.B.A.", "BEng", "B.Eng.", "MEng", "M.Eng.",
            "Certificate", "Certification", "Diploma"
        ]
        
        # Find organizations (universities/schools)
        organizations = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        
        # Find dates
        dates = []
        for pattern in self.date_patterns:
            dates.extend(pattern.findall(education_text))
        
        # Extract education entries
        if organizations:
            # Split text into possible segments based on organizations
            segments = []
            last_pos = 0
            
            for org in organizations:
                org_pos = education_text.find(org)
                if org_pos > last_pos:
                    segments.append(education_text[last_pos:org_pos])
                segments.append(education_text[org_pos:org_pos + len(org)])
                last_pos = org_pos + len(org)
            
            if last_pos < len(education_text):
                segments.append(education_text[last_pos:])
            
            # Combine segments into education entries
            current_entry = {"institution": "", "degree": "", "field": "", "dates": "", "gpa": ""}
            
            for segment in segments:
                # Check if this segment contains an organization
                is_org = any(org in segment for org in organizations)
                
                if is_org:
                    # If we have a previous entry, save it
                    if current_entry["institution"]:
                        education_entries.append(current_entry)
                        current_entry = {"institution": "", "degree": "", "field": "", "dates": "", "gpa": ""}
                    
                    # Set current institution
                    current_entry["institution"] = segment.strip()
                else:
                    # Check for degree
                    for keyword in degree_keywords:
                        if keyword.lower() in segment.lower():
                            degree_index = segment.lower().find(keyword.lower())
                            end_index = segment.find(",", degree_index)
                            if end_index == -1:
                                end_index = segment.find(".", degree_index)
                            if end_index == -1:
                                end_index = len(segment)
                            
                            degree_text = segment[degree_index:end_index].strip()
                            current_entry["degree"] = degree_text
                            
                            # Try to extract field of study
                            field_match = re.search(r'in\s+([^,\.]+)', segment[end_index:])
                            if field_match:
                                current_entry["field"] = field_match.group(1).strip()
                    
                    # Check for dates
                    for date in dates:
                        if date in segment:
                            if current_entry["dates"]:
                                current_entry["dates"] += " - " + date
                            else:
                                current_entry["dates"] = date
                    
                    # Check for GPA
                    gpa_match = re.search(r'GPA:?\s*([\d\.]+)', segment, re.IGNORECASE)
                    if gpa_match:
                        current_entry["gpa"] = gpa_match.group(1)
            
            # Add the last entry
            if current_entry["institution"]:
                education_entries.append(current_entry)
        
        return education_entries

    def extract_experience(self, experience_text: str) -> List[Dict[str, Any]]:
        """
        Extract work experience information from the resume.
        
        Args:
            experience_text: Text from the experience section
            
        Returns:
            List of dictionaries with experience information
        """
        experience_entries = []
        
        # Process with spaCy
        doc = self.nlp(experience_text)
        
        # Find organizations (companies)
        organizations = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        
        # Find dates
        dates = []
        for pattern in self.date_patterns:
            dates.extend(pattern.findall(experience_text))
        
        # Find job titles
        job_title_keywords = [
            "Engineer", "Developer", "Manager", "Director", "Analyst", "Specialist",
            "Consultant", "Coordinator", "Assistant", "Associate", "Lead", "Head",
            "Architect", "Administrator", "Designer", "Supervisor", "Officer",
            "Intern", "President", "CEO", "CTO", "CFO", "COO", "VP"
        ]
        
        # Split into paragraphs (likely job entries)
        paragraphs = re.split(r'\n\s*\n', experience_text)
        
        for para in paragraphs:
            if len(para.strip()) < 20:  # Skip very short paragraphs
                continue
                
            entry = {
                "title": "",
                "company": "",
                "location": "",
                "start_date": "",
                "end_date": "",
                "duration": "",
                "description": para.strip()
            }
            
            # Extract company
            for org in organizations:
                if org in para:
                    entry["company"] = org
                    break
            
            # Extract job title
            title_candidates = []
            para_lines = para.split('\n')
            for line in para_lines[:2]:  # Check first two lines for title
                for keyword in job_title_keywords:
                    if keyword in line:
                        title_match = re.search(r'([^,\n\.\|]+' + re.escape(keyword) + r'[^,\n\.\|]*)', line)
                        if title_match:
                            title_candidates.append(title_match.group(1).strip())
            
            if title_candidates:
                # Choose the longest title as it's likely the most detailed
                entry["title"] = max(title_candidates, key=len)
            
            # Extract dates
            para_dates = []
            for pattern in self.date_patterns:
                para_dates.extend(pattern.findall(para))
                
            if len(para_dates) >= 2:
                entry["start_date"] = para_dates[0]
                entry["end_date"] = para_dates[1]
                
                # Calculate duration
                try:
                    start = date_parser.parse(para_dates[0], fuzzy=True)
                    
                    # Check if end date is "Present" or similar
                    if "present" in para.lower() or "current" in para.lower():
                        import datetime
                        end = datetime.datetime.now()
                    else:
                        end = date_parser.parse(para_dates[1], fuzzy=True)
                    
                    # Calculate duration in months
                    duration_months = (end.year - start.year) * 12 + end.month - start.month
                    if duration_months < 12:
                        entry["duration"] = f"{duration_months} months"
                    else:
                        years = duration_months // 12
                        months = duration_months % 12
                        if months == 0:
                            entry["duration"] = f"{years} years"
                        else:
                            entry["duration"] = f"{years} years, {months} months"
                except:
                    entry["duration"] = ""
            
            # Extract location
            location_entities = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"] and ent.text in para]
            if location_entities:
                entry["location"] = location_entities[0]
            
            experience_entries.append(entry)
        
        return experience_entries

    def extract_projects(self, projects_text: str) -> List[Dict[str, str]]:
        """
        Extract project information from the resume.
        
        Args:
            projects_text: Text from the projects section
            
        Returns:
            List of dictionaries with project information
        """
        projects = []
        
        # Split into paragraphs (likely project entries)
        paragraphs = re.split(r'\n\s*\n', projects_text)
        
        for para in paragraphs:
            if len(para.strip()) < 20:  # Skip very short paragraphs
                continue
                
            project = {
                "name": "",
                "description": para.strip(),
                "technologies": []
            }
            
            # Try to extract project name from first line
            lines = para.split('\n')
            if lines:
                # The first line is likely the project name
                project["name"] = lines[0].strip()
            
            # Extract technologies mentioned
            doc = self.nlp(para)
            
            # Look for skills within the project description
            project_skills = self.extract_skills(para)
            project["technologies"] = [skill["name"] for skill in project_skills]
            
            projects.append(project)
        
        return projects

    def generate_embeddings(self, text: str) -> np.ndarray:
        """
        Generate BERT embeddings for text.
        
        Args:
            text: Input text
            
        Returns:
            Numpy array with embeddings
        """
        # Tokenize and get BERT embeddings
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling - take attention mask into account for averaging
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Return numpy array
        return (sum_embeddings / sum_mask).numpy()[0]

    def parse_resume(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a resume file and extract structured information.
        
        Args:
            file_path: Path to the resume file
            
        Returns:
            Dictionary with parsed resume information
        """
        # Extract text from document
        raw_text = self.extract_text(file_path)
        
        # Check if extraction was successful
        if not raw_text:
            logger.error(f"Failed to extract text from {file_path}")
            return {"error": f"Failed to extract text from {file_path}"}
        
        # Preprocess text
        processed_text = self.preprocess_text(raw_text)
        
        # Identify sections
        sections = self.identify_sections(processed_text)
        
        # Extract information from each section
        parsed_resume = {
            "contact_info": self.extract_contact_info(sections.get("contact", "") or raw_text[:1000]),
            "summary": sections.get("summary", ""),
            "skills": self.extract_skills(processed_text),
            "education": self.extract_education(sections.get("education", "")),
            "experience": self.extract_experience(sections.get("experience", "")),
            "projects": self.extract_projects(sections.get("projects", "")),
            "certifications": sections.get("certifications", ""),
            "languages": sections.get("languages", ""),
            "interests": sections.get("interests", ""),
            # Include parsed sections for reference
            "sections": sections,
            # Generate embeddings for the entire resume
            "embeddings": self.generate_embeddings(processed_text).tolist()
        }
        
        return parsed_resume
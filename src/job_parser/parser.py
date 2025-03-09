"""
Job Description Parser Module

This module handles the parsing and analysis of job descriptions,
extracting key requirements, skills, experience levels, and other
relevant information.
"""

import re
from typing import Dict, List, Set, Any, Tuple, Optional

import spacy
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from loguru import logger

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_lg")
    logger.info("Loaded spaCy model: en_core_web_lg")
except OSError:
    logger.warning("Downloading spaCy model...")
    spacy.cli.download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")


class JobDescriptionParser:
    """
    A class for parsing and analyzing job descriptions.
    
    This class extracts structured information from job descriptions,
    including required skills, experience levels, education requirements,
    and other key details.
    """
    
    def __init__(self, skills_path: str = "data/skills_taxonomy.csv"):
        """
        Initialize the JobDescriptionParser with necessary resources.
        
        Args:
            skills_path: Path to the skills taxonomy CSV file
        """
        self.nlp = nlp
        self.skills = self._load_skills_taxonomy(skills_path)
        
        # Initialize BERT model for semantic understanding
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        # Regular expressions for requirement detection
        self.requirement_patterns = {
            "required": re.compile(r'(?i)(required|must have|essential|needed|necessary|requirements?|qualifications?|you have|you possess|minimum|what you.+bring)', re.IGNORECASE),
            "preferred": re.compile(r'(?i)(preferred|nice to have|desired|bonus|plus|beneficial|ideally|advantage|we.+like)', re.IGNORECASE),
            "responsibilities": re.compile(r'(?i)(responsibilities|duties|what you.+do|role|the job|position|job description|you will|day to day)', re.IGNORECASE),
            "benefits": re.compile(r'(?i)(benefits|perks|we offer|compensation|salary|package|we provide)', re.IGNORECASE),
            "company": re.compile(r'(?i)(about us|our company|who we are|the company|our team|our mission)', re.IGNORECASE),
        }
        
        # Patterns for education requirements
        self.education_patterns = {
            "degree": re.compile(r'(?i)(bachelor\'?s?|master\'?s?|phd|doctorate|degree|bs|ba|ms|ma|mba|associate\'?s?)', re.IGNORECASE),
            "field": re.compile(r'(?i)(computer science|engineering|business|marketing|finance|accounting|economics|psychology|biology|chemistry|physics|mathematics|math|law|education)', re.IGNORECASE),
            "level": re.compile(r'(?i)(high school|college|university|graduate|undergraduate|postgraduate)', re.IGNORECASE),
        }
        
        # Patterns for experience requirements
        self.experience_patterns = {
            "years": re.compile(r'(?i)(\d+\+?|\d+\s*\-\s*\d+|\w+\s+to\s+\w+|\w+)\s+(year|yr)s?(\s+of\s+experience|\s+experience)?', re.IGNORECASE),
            "level": re.compile(r'(?i)(entry[\s\-]level|junior|mid[\s\-]level|senior|lead|principal|staff|director|executive)', re.IGNORECASE),
        }
        
        # Keywords indicating requirement importance
        self.importance_keywords = {
            "must_have": ["required", "must", "essential", "necessary", "need", "critical", "crucial", "important"],
            "good_to_have": ["preferred", "desired", "nice", "plus", "bonus", "beneficial", "ideally", "advantage"],
        }
        
        # Keywords for positive and negative sentiment in job descriptions
        self.positive_keywords = [
            "opportunity", "growth", "innovative", "creative", "flexible", "dynamic", 
            "collaborative", "friendly", "supportive", "diverse", "inclusive", 
            "modern", "cutting-edge", "state-of-the-art", "exciting", "challenging",
            "rewarding", "fun", "balanced", "remote", "hybrid", "competitive"
        ]
        
        self.negative_keywords = [
            "demanding", "fast-paced", "pressure", "strict", "rigid", "deadline", 
            "tough", "intense", "stressful", "competitive", "long hours", "overtime", 
            "weekend", "on-call", "travel"
        ]
        
        logger.info("JobDescriptionParser initialized successfully")
    
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
            import os
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
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the job description text for better analysis.
        
        Args:
            text: Raw job description text
            
        Returns:
            Preprocessed text
        """
        # Remove excess whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common issues
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
        Identify and extract different sections from the job description.
        
        Args:
            text: Preprocessed job description text
            
        Returns:
            Dictionary with section names as keys and content as values
        """
        # Split text into lines for easier section detection
        lines = text.split('\n')
        current_section = "unknown"
        sections = {section: "" for section in self.requirement_patterns.keys()}
        sections["unknown"] = ""
        
        for i, line in enumerate(lines):
            # Check if line could be a section header
            if len(line.strip()) < 60 and not line.strip().endswith('.'):
                section_found = False
                for section, pattern in self.requirement_patterns.items():
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
    
    def extract_skills(self, text: str) -> List[Dict[str, str]]:
        """
        Extract required skills from the job description.
        
        Args:
            text: Job description text
            
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
                    # Determine if it's required or preferred
                    skill_context = self._get_context(text, skill_name, window_size=150)
                    importance = self._determine_importance(skill_context)
                    
                    found_skills.append({
                        "name": skill_row["skill"],
                        "category": skill_row["category"],
                        "level": skill_row.get("level", ""),
                        "importance": importance
                    })
                    continue
                
                # Check synonyms if available
                if "synonyms" in skill_row and pd.notna(skill_row["synonyms"]):
                    synonyms = [s.strip().lower() for s in skill_row["synonyms"].split(",")]
                    for synonym in synonyms:
                        if re.search(r'\b' + re.escape(synonym) + r'\b', text.lower()):
                            # Determine if it's required or preferred
                            skill_context = self._get_context(text, synonym, window_size=150)
                            importance = self._determine_importance(skill_context)
                            
                            found_skills.append({
                                "name": skill_row["skill"],  # Use canonical name
                                "category": skill_row["category"],
                                "level": skill_row.get("level", ""),
                                "importance": importance
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
            non_skill_words = ["job", "description", "requirements", "qualifications", "responsibilities", "application"]
            filtered_skills = [s for s in potential_skills if not any(w in s.lower() for w in non_skill_words)]
            
            # Determine importance for each skill
            for skill in filtered_skills:
                skill_context = self._get_context(text, skill, window_size=150)
                importance = self._determine_importance(skill_context)
                
                found_skills.append({
                    "name": skill,
                    "category": "Unknown",
                    "level": "",
                    "importance": importance
                })
        
        return found_skills
    
    def _get_context(self, text: str, keyword: str, window_size: int = 100) -> str:
        """
        Extract context around a keyword.
        
        Args:
            text: Full text
            keyword: Keyword to find context for
            window_size: Size of context window in characters
            
        Returns:
            Context string
        """
        keyword_pos = text.lower().find(keyword.lower())
        if keyword_pos == -1:
            return ""
        
        start = max(0, keyword_pos - window_size)
        end = min(len(text), keyword_pos + len(keyword) + window_size)
        
        return text[start:end]
    
    def _determine_importance(self, context: str) -> str:
        """
        Determine if a skill is required or preferred based on context.
        
        Args:
            context: Context text around the skill
            
        Returns:
            "must_have" or "good_to_have"
        """
        context_lower = context.lower()
        
        # Check for must-have indicators
        for keyword in self.importance_keywords["must_have"]:
            if keyword in context_lower:
                return "must_have"
        
        # Check for good-to-have indicators
        for keyword in self.importance_keywords["good_to_have"]:
            if keyword in context_lower:
                return "good_to_have"
        
        # Default to required
        return "must_have"
    
    def extract_education_requirements(self, text: str) -> Dict[str, Any]:
        """
        Extract education requirements from the job description.
        
        Args:
            text: Job description text
            
        Returns:
            Dictionary with education requirement information
        """
        education_info = {
            "required": False,
            "degrees": [],
            "fields": [],
            "level": "",
            "raw_text": ""
        }
        
        # Look for degree requirements
        degree_matches = self.education_patterns["degree"].findall(text)
        if degree_matches:
            education_info["required"] = True
            education_info["degrees"] = list(set(degree_matches))
        
        # Look for field requirements
        field_matches = self.education_patterns["field"].findall(text)
        if field_matches:
            education_info["fields"] = list(set(field_matches))
        
        # Look for education level
        level_matches = self.education_patterns["level"].findall(text)
        if level_matches:
            education_info["level"] = level_matches[0]
        
        # Extract raw text about education
        education_contexts = []
        
        # Get context around degree mentions
        for degree in degree_matches:
            context = self._get_context(text, degree, window_size=200)
            education_contexts.append(context)
        
        # Get context around education level mentions
        for level in level_matches:
            context = self._get_context(text, level, window_size=200)
            education_contexts.append(context)
        
        if education_contexts:
            education_info["raw_text"] = " ... ".join(education_contexts)
        
        return education_info
    
    def extract_experience_requirements(self, text: str) -> Dict[str, Any]:
        """
        Extract experience requirements from the job description.
        
        Args:
            text: Job description text
            
        Returns:
            Dictionary with experience requirement information
        """
        experience_info = {
            "required": False,
            "min_years": 0,
            "max_years": 0,
            "level": "",
            "raw_text": ""
        }
        
        # Look for years of experience
        years_matches = self.experience_patterns["years"].findall(text)
        if years_matches:
            experience_info["required"] = True
            
            # Process the first match - typically the most relevant
            match = years_matches[0]
            years_text = match[0].lower()
            
            # Handle numeric ranges
            if "-" in years_text:
                parts = years_text.split("-")
                try:
                    experience_info["min_years"] = int(parts[0].strip())
                    experience_info["max_years"] = int(parts[1].strip())
                except:
                    pass
            # Handle "X+" format
            elif "+" in years_text:
                try:
                    experience_info["min_years"] = int(years_text.replace("+", "").strip())
                    experience_info["max_years"] = 99  # Effectively no upper limit
                except:
                    pass
            # Handle simple numeric values
            elif years_text.isdigit():
                try:
                    experience_info["min_years"] = int(years_text)
                    experience_info["max_years"] = int(years_text)
                except:
                    pass
            # Handle text values
            else:
                if "entry" in years_text or "junior" in years_text:
                    experience_info["min_years"] = 0
                    experience_info["max_years"] = 2
                elif "mid" in years_text:
                    experience_info["min_years"] = 2
                    experience_info["max_years"] = 5
                elif "senior" in years_text or "experienced" in years_text:
                    experience_info["min_years"] = 5
                    experience_info["max_years"] = 99
        
        # Look for experience level
        level_matches = self.experience_patterns["level"].findall(text)
        if level_matches:
            experience_info["level"] = level_matches[0]
        
        # Extract raw text about experience
        experience_contexts = []
        
        # Get context around years mentions
        for match in years_matches:
            years_text = match[0]
            context = self._get_context(text, years_text, window_size=200)
            experience_contexts.append(context)
        
        # Get context around level mentions
        for level in level_matches:
            context = self._get_context(text, level, window_size=200)
            experience_contexts.append(context)
        
        if experience_contexts:
            experience_info["raw_text"] = " ... ".join(experience_contexts)
        
        return experience_info
    
    def extract_job_details(self, text: str) -> Dict[str, Any]:
        """
        Extract basic job details from the job description.
        
        Args:
            text: Job description text
            
        Returns:
            Dictionary with job details
        """
        job_details = {
            "title": "",
            "company": "",
            "location": "",
            "job_type": "",
            "seniority": ""
        }
        
        # Process with spaCy
        doc = self.nlp(text[:1000])  # Process just the beginning for job title and company
        
        # Extract job title (usually at the beginning)
        title_candidates = []
        for i, sent in enumerate(doc.sents):
            if i > 3:  # Only check first few sentences
                break
            for token in sent:
                if token.pos_ == "NOUN" and token.text.lower() in [
                    "engineer", "developer", "manager", "analyst", "designer",
                    "specialist", "consultant", "coordinator", "director", "lead"
                ]:
                    # Get noun phrase containing this job title
                    for chunk in sent.noun_chunks:
                        if token.i >= chunk.start and token.i < chunk.end:
                            title_candidates.append(chunk.text)
        
        if title_candidates:
            job_details["title"] = max(title_candidates, key=len)
        
        # Extract company name
        org_entities = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        if org_entities:
            job_details["company"] = org_entities[0]
        
        # Extract location
        loc_entities = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
        if loc_entities:
            job_details["location"] = loc_entities[0]
        
        # Extract job type
        job_type_patterns = [
            (re.compile(r'\b(full[\s\-]time|fulltime)\b', re.IGNORECASE), "Full-time"),
            (re.compile(r'\b(part[\s\-]time|parttime)\b', re.IGNORECASE), "Part-time"),
            (re.compile(r'\b(contract|contractual)\b', re.IGNORECASE), "Contract"),
            (re.compile(r'\b(freelance)\b', re.IGNORECASE), "Freelance"),
            (re.compile(r'\b(internship|intern)\b', re.IGNORECASE), "Internship"),
            (re.compile(r'\b(temporary|temp)\b', re.IGNORECASE), "Temporary"),
        ]
        
        for pattern, job_type in job_type_patterns:
            if pattern.search(text):
                job_details["job_type"] = job_type
                break
        
        # Extract seniority level
        seniority_patterns = [
            (re.compile(r'\b(entry[\s\-]level|junior)\b', re.IGNORECASE), "Entry-level"),
            (re.compile(r'\b(mid[\s\-]level|intermediate)\b', re.IGNORECASE), "Mid-level"),
            (re.compile(r'\b(senior|experienced|lead|principal)\b', re.IGNORECASE), "Senior"),
            (re.compile(r'\b(manager|director|head)\b', re.IGNORECASE), "Manager"),
            (re.compile(r'\b(executive|ceo|cto|cfo|vp)\b', re.IGNORECASE), "Executive"),
        ]
        
        for pattern, seniority in seniority_patterns:
            if pattern.search(text):
                job_details["seniority"] = seniority
                break
        
        return job_details
    
    def extract_company_culture(self, text: str) -> Dict[str, Any]:
        """
        Extract information about company culture and work environment.
        
        Args:
            text: Job description text
            
        Returns:
            Dictionary with company culture information
        """
        culture_info = {
            "positive_traits": [],
            "negative_traits": [],
            "work_environment": "",
            "diversity_inclusion": False,
            "remote_work": False,
            "sentiment_score": 0.0
        }
        
        # Check for positive keywords
        for keyword in self.positive_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text.lower()):
                culture_info["positive_traits"].append(keyword)
        
        # Check for negative keywords
        for keyword in self.negative_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text.lower()):
                culture_info["negative_traits"].append(keyword)
        
        # Check for diversity and inclusion mentions
        diversity_pattern = re.compile(r'\b(diversity|inclusion|equal opportunity|minorities|gender|race|disability|veteran)\b', re.IGNORECASE)
        if diversity_pattern.search(text):
            culture_info["diversity_inclusion"] = True
        
        # Check for remote work mentions
        remote_pattern = re.compile(r'\b(remote|work from home|wfh|telecommute|virtual|flexible location)\b', re.IGNORECASE)
        if remote_pattern.search(text):
            culture_info["remote_work"] = True
        
        # Determine work environment based on keywords
        environment_keywords = {
            "Fast-paced": re.compile(r'\b(fast[\s\-]paced|fast paced|dynamic|quickly|rapid)\b', re.IGNORECASE),
            "Collaborative": re.compile(r'\b(collaborative|team|together|cooperation)\b', re.IGNORECASE),
            "Innovative": re.compile(r'\b(innovative|cutting[\s\-]edge|innovation|creative|creativity)\b', re.IGNORECASE),
            "Structured": re.compile(r'\b(structured|process|methodical|organized)\b', re.IGNORECASE),
            "Flexible": re.compile(r'\b(flexible|adaptable|agile)\b', re.IGNORECASE),
        }
        
        for env, pattern in environment_keywords.items():
            if pattern.search(text):
                if culture_info["work_environment"]:
                    culture_info["work_environment"] += ", " + env
                else:
                    culture_info["work_environment"] = env
        
        # Calculate sentiment score (-1.0 to 1.0)
        positive_count = len(culture_info["positive_traits"])
        negative_count = len(culture_info["negative_traits"])
        total_traits = positive_count + negative_count
        
        if total_traits > 0:
            culture_info["sentiment_score"] = (positive_count - negative_count) / total_traits
        
        return culture_info
    
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
    
    def parse_job_description(self, text: str) -> Dict[str, Any]:
        """
        Parse a job description and extract structured information.
        
        Args:
            text: Job description text
            
        Returns:
            Dictionary with parsed job information
        """
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Identify sections
        sections = self.identify_sections(processed_text)
        
        # Extract information
        skills = self.extract_skills(processed_text)
        education = self.extract_education_requirements(processed_text)
        experience = self.extract_experience_requirements(processed_text)
        job_details = self.extract_job_details(processed_text)
        company_culture = self.extract_company_culture(processed_text)
        
        # Create structured representation
        parsed_job = {
            "details": job_details,
            "skills": skills,
            "education": education,
            "experience": experience,
            "culture": company_culture,
            "sections": sections,
            "embeddings": self.generate_embeddings(processed_text).tolist()
        }
        
        return parsed_job
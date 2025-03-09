"""
Resume to Job Description Matcher Module

This module handles the comparison of parsed resumes against job descriptions
to calculate match scores and provide detailed analysis of compatibility.
"""

import re
from typing import Dict, List, Any, Tuple, Optional, Set, Union
from datetime import datetime
import math

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dateutil import parser as date_parser
from loguru import logger

class ResumeMatcher:
    """
    A class that compares parsed resumes with job descriptions and
    calculates match scores based on multiple dimensions of compatibility.
    
    This class uses both keyword matching and semantic similarity
    to provide a comprehensive assessment of resume-job compatibility.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the matcher with customizable scoring weights.
        
        Args:
            weights: Dictionary of weights for different matching criteria
        """
        # Default weights for different matching criteria
        self.weights = {
            "skills_match": 0.35,
            "experience_match": 0.25,
            "education_match": 0.15,
            "semantic_similarity": 0.15,
            "keyword_match": 0.10,
        }
        
        # Override with custom weights if provided
        if weights:
            self.weights.update(weights)
        
        # Normalize weights to ensure they sum to 1.0
        weight_sum = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= weight_sum
        
        logger.info(f"ResumeMatcher initialized with weights: {self.weights}")
    
    def calculate_skills_match(
        self, resume_skills: List[Dict[str, str]], job_skills: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Calculate skill match score between resume and job description.
        
        Args:
            resume_skills: List of skills from the resume
            job_skills: List of skills from the job description
            
        Returns:
            Dictionary with skill match results
        """
        # Extract skill names for easier comparison
        resume_skill_names = [skill["name"].lower() for skill in resume_skills]
        job_skill_names_all = [skill["name"].lower() for skill in job_skills]
        
        # Separate must-have and good-to-have skills
        must_have_skills = [skill["name"].lower() for skill in job_skills if skill.get("importance") == "must_have"]
        good_to_have_skills = [skill["name"].lower() for skill in job_skills if skill.get("importance") == "good_to_have"]
        
        # If importance is not specified, assume all skills are must-have
        if not must_have_skills and not good_to_have_skills:
            must_have_skills = job_skill_names_all
        
        # Calculate matches
        must_have_matches = [skill for skill in must_have_skills if skill in resume_skill_names]
        good_to_have_matches = [skill for skill in good_to_have_skills if skill in resume_skill_names]
        
        # Calculate scores
        must_have_score = len(must_have_matches) / max(len(must_have_skills), 1)
        good_to_have_score = len(good_to_have_matches) / max(len(good_to_have_skills), 1) if good_to_have_skills else 1.0
        
        # Overall score: 80% for must-have and 20% for good-to-have
        overall_score = (must_have_score * 0.8) + (good_to_have_score * 0.2)
        
        # Identify missing skills
        missing_must_have = [skill for skill in must_have_skills if skill not in resume_skill_names]
        missing_good_to_have = [skill for skill in good_to_have_skills if skill not in resume_skill_names]
        
        # Construct result
        result = {
            "score": overall_score,
            "must_have_match": must_have_score,
            "good_to_have_match": good_to_have_score,
            "matched_skills": sorted(must_have_matches + good_to_have_matches),
            "missing_must_have": sorted(missing_must_have),
            "missing_good_to_have": sorted(missing_good_to_have),
            "has_critical_skills": must_have_score >= 0.7,  # At least 70% of must-have skills
        }
        
        return result
    
    def calculate_experience_match(
        self, resume_experience: List[Dict[str, Any]], job_experience: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate experience match score between resume and job requirements.
        
        Args:
            resume_experience: List of experience entries from the resume
            job_experience: Experience requirements from the job description
            
        Returns:
            Dictionary with experience match results
        """
        # Initialize result
        result = {
            "score": 0.0,
            "total_years": 0,
            "meets_minimum": False,
            "experience_level_match": False,
            "recency_score": 0.0,
            "domain_relevance": 0.0,
            "explanation": ""
        }
        
        # If job doesn't require experience
        if not job_experience.get("required", False) or job_experience.get("min_years", 0) == 0:
            result["score"] = 1.0
            result["meets_minimum"] = True
            result["experience_level_match"] = True
            result["explanation"] = "No specific experience requirements."
            return result
        
        # Calculate total years of experience
        total_years = 0
        for entry in resume_experience:
            # Try to extract duration in years
            if entry.get("duration"):
                duration_text = entry["duration"]
                years_match = re.search(r'(\d+)\s*years?', duration_text)
                if years_match:
                    total_years += int(years_match.group(1))
        
        result["total_years"] = total_years
        
        # Calculate if minimum experience is met
        min_years_required = job_experience.get("min_years", 0)
        result["meets_minimum"] = total_years >= min_years_required
        
        # Calculate experience score based on required years
        if min_years_required > 0:
            # Scale score based on how close the candidate is to the required experience
            if total_years >= min_years_required:
                max_years = job_experience.get("max_years", min_years_required * 2)
                # Score is 0.7 if meeting minimum, up to 1.0 if exceeding maximum
                years_score = 0.7 + 0.3 * min(1.0, (total_years - min_years_required) / (max_years - min_years_required))
            else:
                # Score scales from 0 to 0.7 based on how close they are to minimum
                years_score = 0.7 * (total_years / min_years_required)
        else:
            years_score = 1.0
        
        # Calculate experience level match
        job_level = job_experience.get("level", "").lower()
        
        if job_level:
            # Map experience years to levels
            level_mapping = {
                "entry": (0, 2),
                "junior": (0, 2),
                "mid": (2, 5),
                "mid-level": (2, 5),
                "intermediate": (2, 5),
                "senior": (5, 10),
                "lead": (7, 15),
                "principal": (10, 20),
                "manager": (5, 15),
                "director": (8, 20),
                "executive": (10, 25),
            }
            
            for level_key, (min_level_years, max_level_years) in level_mapping.items():
                if level_key in job_level and min_level_years <= total_years <= max_level_years:
                    result["experience_level_match"] = True
                    break
        else:
            # If no specific level mentioned, consider it a match if years requirement is met
            result["experience_level_match"] = result["meets_minimum"]
        
        # Calculate recency score based on how recent the most relevant experience is
        recency_score = 0.0
        relevant_keywords = set()
        
        # Extract keywords from job description text
        if job_experience.get("raw_text"):
            # Extract nouns and noun phrases as keywords
            import re
            from collections import Counter
            
            # Simple extraction of potential keywords
            words = re.findall(r'\b[A-Za-z][A-Za-z-]+\b', job_experience["raw_text"].lower())
            word_counts = Counter(words)
            
            # Filter out common words
            common_words = {"experience", "year", "years", "month", "months", "work", "working", "job", "position", "required", "preferred"}
            relevant_keywords = {word for word, count in word_counts.items() if count > 1 and word not in common_words}
        
        # If we have some keywords to check for relevance
        if relevant_keywords:
            most_recent_match_years = float('inf')
            
            for entry in resume_experience:
                entry_text = f"{entry.get('title', '')} {entry.get('company', '')} {entry.get('description', '')}"
                entry_text = entry_text.lower()
                
                # Check if this experience entry matches job keywords
                matches = sum(1 for keyword in relevant_keywords if keyword in entry_text)
                relevance_ratio = matches / len(relevant_keywords)
                
                if relevance_ratio > 0.2:  # At least 20% keyword match
                    # Calculate how many years ago this experience was
                    end_date = entry.get("end_date", "")
                    years_ago = 0
                    
                    if end_date:
                        if "present" in end_date.lower() or "current" in end_date.lower():
                            years_ago = 0
                        else:
                            try:
                                end_date_obj = date_parser.parse(end_date, fuzzy=True)
                                years_ago = (datetime.now() - end_date_obj).days / 365.0
                            except:
                                pass
                    
                    # Update if this is the most recent relevant experience
                    if years_ago < most_recent_match_years:
                        most_recent_match_years = years_ago
            
            # Calculate recency score - higher if recent experience
            if most_recent_match_years < float('inf'):
                recency_score = max(0, 1 - (most_recent_match_years / 10))  # Scale over 10 years
            
            result["recency_score"] = recency_score
        else:
            # If no keywords to match, assume neutral recency score
            result["recency_score"] = 0.5
        
        # Calculate domain relevance by looking for industry or domain keywords
        # This is a simplified approach - in a real system this would be more sophisticated
        domain_relevance = 0.5  # Neutral default
        result["domain_relevance"] = domain_relevance
        
        # Final experience score combines years, level match, recency and domain relevance
        experience_score = (
            years_score * 0.5 +
            (1.0 if result["experience_level_match"] else 0.0) * 0.2 +
            recency_score * 0.2 +
            domain_relevance * 0.1
        )
        
        result["score"] = experience_score
        
        # Add explanation
        if result["meets_minimum"] and result["experience_level_match"]:
            result["explanation"] = f"Meets the required {min_years_required}+ years of experience and level requirements."
        elif result["meets_minimum"]:
            result["explanation"] = f"Meets the required {min_years_required}+ years of experience but may not match the level requirements."
        elif result["experience_level_match"]:
            result["explanation"] = f"Matches the experience level but has fewer than the required {min_years_required} years of experience."
        else:
            result["explanation"] = f"Does not meet the minimum {min_years_required} years of experience requirement."
        
        return result
    
    def calculate_education_match(
        self, resume_education: List[Dict[str, Any]], job_education: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate education match score between resume and job requirements.
        
        Args:
            resume_education: List of education entries from the resume
            job_education: Education requirements from the job description
            
        Returns:
            Dictionary with education match results
        """
        # Initialize result
        result = {
            "score": 0.0,
            "has_required_degree": False,
            "has_relevant_field": False,
            "degree_level_match": "",
            "explanation": ""
        }
        
        # If job doesn't require specific education
        if not job_education.get("required", False) or not job_education.get("degrees"):
            result["score"] = 1.0
            result["has_required_degree"] = True
            result["explanation"] = "No specific education requirements."
            return result
        
        # Define degree levels for comparison
        degree_levels = {
            "high school": 1,
            "associate": 2,
            "bachelor": 3,
            "bs": 3,
            "ba": 3,
            "b.s.": 3,
            "b.a.": 3,
            "master": 4,
            "ms": 4,
            "ma": 4,
            "m.s.": 4,
            "m.a.": 4,
            "mba": 4,
            "m.b.a.": 4,
            "phd": 5,
            "ph.d.": 5,
            "doctorate": 5,
        }
        
        # Get required degrees from job
        required_degrees = [degree.lower() for degree in job_education.get("degrees", [])]
        required_fields = [field.lower() for field in job_education.get("fields", [])]
        
        # Check candidate's highest degree level
        highest_degree_level = 0
        highest_degree = ""
        has_relevant_field = False
        
        for edu in resume_education:
            degree = edu.get("degree", "").lower()
            field = edu.get("field", "").lower()
            
            # Check degree level
            for degree_keyword, level in degree_levels.items():
                if degree_keyword in degree and level > highest_degree_level:
                    highest_degree_level = level
                    highest_degree = degree
            
            # Check field match
            if required_fields:
                for req_field in required_fields:
                    if req_field in field:
                        has_relevant_field = True
                        break
        
        # Determine if candidate has required degree
        has_required_degree = False
        required_level = 0
        
        for req_degree in required_degrees:
            # Get level of required degree
            for degree_keyword, level in degree_levels.items():
                if degree_keyword in req_degree:
                    required_level = max(required_level, level)
            
            # Check if candidate's degree matches by keyword
            if any(degree_keyword in highest_degree for degree_keyword in req_degree.split()):
                has_required_degree = True
                break
        
        # If no specific match but candidate's degree level meets or exceeds required level
        if not has_required_degree and highest_degree_level >= required_level and required_level > 0:
            has_required_degree = True
        
        result["has_required_degree"] = has_required_degree
        result["has_relevant_field"] = has_relevant_field
        
        # Determine degree level match description
        if highest_degree_level < required_level:
            result["degree_level_match"] = "below_requirements"
        elif highest_degree_level == required_level:
            result["degree_level_match"] = "meets_requirements"
        else:
            result["degree_level_match"] = "exceeds_requirements"
        
        # Calculate education score
        if has_required_degree:
            degree_score = 0.7  # Base score for having required degree
            
            # Bonus for exceeding requirements
            if result["degree_level_match"] == "exceeds_requirements":
                degree_score += 0.1
            
            # Bonus for relevant field
            if has_relevant_field:
                degree_score += 0.2
            elif not required_fields:  # If no specific field required
                degree_score += 0.2
        else:
            # Partial credit based on how close they are to requirements
            degree_score = 0.5 * (highest_degree_level / max(required_level, 1))
            
            # Small bonus for relevant field even without required degree
            if has_relevant_field:
                degree_score += 0.1
        
        result["score"] = degree_score
        
        # Add explanation
        if has_required_degree and has_relevant_field:
            result["explanation"] = "Has the required degree in a relevant field."
        elif has_required_degree:
            result["explanation"] = "Has the required degree level, but not in a specified field."
        elif has_relevant_field:
            result["explanation"] = "Has education in a relevant field, but not at the required degree level."
        else:
            result["explanation"] = "Does not meet the education requirements."
        
        return result
    
    def calculate_semantic_similarity(
        self, resume_embeddings: List[float], job_embeddings: List[float]
    ) -> float:
        """
        Calculate semantic similarity between resume and job description
        using their vector embeddings.
        
        Args:
            resume_embeddings: Vector embeddings of the resume
            job_embeddings: Vector embeddings of the job description
            
        Returns:
            Similarity score between 0 and 1
        """
        # Convert to numpy arrays
        resume_vector = np.array(resume_embeddings).reshape(1, -1)
        job_vector = np.array(job_embeddings).reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(resume_vector, job_vector)[0][0]
        
        # Normalize to 0-1 range (cosine similarity is between -1 and 1)
        normalized_score = (similarity + 1) / 2
        
        return normalized_score
    
    def calculate_keyword_match(
        self, resume_text: str, job_description: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate keyword match score by looking for job-specific terms in the resume.
        
        Args:
            resume_text: Full text of the resume
            job_description: Parsed job description
            
        Returns:
            Dictionary with keyword match results
        """
        # Initialize result
        result = {
            "score": 0.0,
            "matched_keywords": [],
            "sections_matched": {}
        }
        
        # Extract keywords from job description sections
        keywords = set()
        section_keywords = {}
        
        # Process each section
        for section_name, section_text in job_description.get("sections", {}).items():
            if not section_text:
                continue
                
            # Skip non-essential sections
            if section_name in ["company", "benefits", "unknown"]:
                continue
            
            # Simple keyword extraction - could be improved with more sophisticated NLP
            section_words = re.findall(r'\b[A-Za-z][A-Za-z-]{2,}\b', section_text.lower())
            
            # Remove common English stopwords
            stopwords = {"and", "the", "with", "for", "this", "that", "you", "will", "your", "our", "we", "are", "have"}
            filtered_words = [word for word in section_words if word not in stopwords]
            
            # Get word frequency in this section
            from collections import Counter
            word_counts = Counter(filtered_words)
            
            # Extract top keywords (appearing more than once)
            section_keywords[section_name] = {word for word, count in word_counts.items() if count > 1}
            keywords.update(section_keywords[section_name])
        
        # Add job title keywords explicitly
        job_title = job_description.get("details", {}).get("title", "").lower()
        title_words = re.findall(r'\b[A-Za-z][A-Za-z-]{2,}\b', job_title)
        keywords.update(title_words)
        
        # Check for keyword matches in resume
        resume_text_lower = resume_text.lower()
        matched_keywords = {keyword for keyword in keywords if keyword in resume_text_lower}
        
        # Calculate section matches
        sections_matched = {}
        for section_name, section_words in section_keywords.items():
            if section_words:
                matches = section_words.intersection(matched_keywords)
                match_ratio = len(matches) / len(section_words)
                sections_matched[section_name] = match_ratio
        
        # Calculate overall keyword match score
        if keywords:
            keyword_score = len(matched_keywords) / len(keywords)
        else:
            keyword_score = 0.0
        
        result["score"] = keyword_score
        result["matched_keywords"] = sorted(matched_keywords)
        result["sections_matched"] = sections_matched
        
        return result
    
    def calculate_match_score(
        self, resume: Dict[str, Any], job_description: Dict[str, Any], detailed: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate overall match score between a resume and job description.
        
        Args:
            resume: Parsed resume dictionary
            job_description: Parsed job description dictionary
            detailed: Whether to include detailed analysis in the result
            
        Returns:
            Dictionary with match results and analysis
        """
        # Initialize results
        match_results = {
            "overall_score": 0.0,
            "summary": "",
            "recommendation": "",
        }
        
        detailed_scores = {}
        
        # 1. Skills Match
        skills_match = self.calculate_skills_match(
            resume.get("skills", []), 
            job_description.get("skills", [])
        )
        detailed_scores["skills_match"] = skills_match
        
        # 2. Experience Match
        experience_match = self.calculate_experience_match(
            resume.get("experience", []), 
            job_description.get("experience", {})
        )
        detailed_scores["experience_match"] = experience_match
        
        # 3. Education Match
        education_match = self.calculate_education_match(
            resume.get("education", []), 
            job_description.get("education", {})
        )
        detailed_scores["education_match"] = education_match
        
        # 4. Semantic Similarity
        semantic_similarity = self.calculate_semantic_similarity(
            resume.get("embeddings", []), 
            job_description.get("embeddings", [])
        )
        detailed_scores["semantic_similarity"] = semantic_similarity
        
        # 5. Keyword Match
        full_resume_text = ""
        for section_text in resume.get("sections", {}).values():
            full_resume_text += section_text + " "
        
        keyword_match = self.calculate_keyword_match(
            full_resume_text,
            job_description
        )
        detailed_scores["keyword_match"] = keyword_match
        
        # Calculate overall score using weights
        overall_score = (
            skills_match["score"] * self.weights["skills_match"] +
            experience_match["score"] * self.weights["experience_match"] +
            education_match["score"] * self.weights["education_match"] +
            semantic_similarity * self.weights["semantic_similarity"] +
            keyword_match["score"] * self.weights["keyword_match"]
        )
        
        match_results["overall_score"] = overall_score
        
        # Add detailed scores if requested
        if detailed:
            match_results["detailed_scores"] = detailed_scores
        
        # Generate summary and recommendation
        match_results["summary"] = self._generate_summary(overall_score, detailed_scores)
        match_results["recommendation"] = self._generate_recommendation(overall_score, detailed_scores)
        
        # Add skills gap analysis
        match_results["skills_gap"] = {
            "missing_must_have": skills_match.get("missing_must_have", []),
            "missing_good_to_have": skills_match.get("missing_good_to_have", []),
        }
        
        return match_results
    
    def _generate_summary(self, overall_score: float, detailed_scores: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of the match results.
        
        Args:
            overall_score: Overall match score
            detailed_scores: Detailed scores for different criteria
            
        Returns:
            Summary string
        """
        score_percent = int(overall_score * 100)
        
        if score_percent >= 90:
            match_quality = "excellent"
        elif score_percent >= 80:
            match_quality = "strong"
        elif score_percent >= 70:
            match_quality = "good"
        elif score_percent >= 60:
            match_quality = "moderate"
        elif score_percent >= 50:
            match_quality = "fair"
        else:
            match_quality = "poor"
        
        # Create summary based on scores
        summary = f"This resume has a {match_quality} match ({score_percent}%) with the job description. "
        
        # Add highlights based on detailed scores
        skills_match = detailed_scores.get("skills_match", {})
        experience_match = detailed_scores.get("experience_match", {})
        education_match = detailed_scores.get("education_match", {})
        
        # Skills highlight
        skills_score = int(skills_match.get("score", 0) * 100)
        matched_skills_count = len(skills_match.get("matched_skills", []))
        missing_must_have_count = len(skills_match.get("missing_must_have", []))
        
        if skills_score >= 80:
            summary += f"The candidate has {matched_skills_count} of the required skills. "
        elif missing_must_have_count > 0:
            summary += f"The candidate is missing {missing_must_have_count} required skills. "
        
        # Experience highlight
        meets_min_experience = experience_match.get("meets_minimum", False)
        exp_level_match = experience_match.get("experience_level_match", False)
        
        if meets_min_experience and exp_level_match:
            summary += "Their experience level meets the job requirements. "
        elif not meets_min_experience:
            summary += "They may not have enough experience for this role. "
        
        # Education highlight
        has_required_degree = education_match.get("has_required_degree", False)
        has_relevant_field = education_match.get("has_relevant_field", False)
        
        if has_required_degree and has_relevant_field:
            summary += "Their education is a good match for this position."
        elif not has_required_degree and education_match.get("score", 0) < 0.5:
            summary += "Their education may not meet the job requirements."
        
        return summary
    
    def _generate_recommendation(self, overall_score: float, detailed_scores: Dict[str, Any]) -> str:
        """
        Generate a recommendation based on the match results.
        
        Args:
            overall_score: Overall match score
            detailed_scores: Detailed scores for different criteria
            
        Returns:
            Recommendation string
        """
        score_percent = int(overall_score * 100)
        
        # Decision thresholds
        if score_percent >= 80:
            recommendation = "Strongly consider interviewing this candidate."
        elif score_percent >= 70:
            recommendation = "Consider interviewing this candidate."
        elif score_percent >= 60:
            recommendation = "This candidate may be worth considering, but note the missing requirements."
        elif score_percent >= 50:
            recommendation = "This candidate is below the preferred qualifications but could be considered if the applicant pool is limited."
        else:
            recommendation = "This candidate does not appear to be a good match for this position."
        
        # Add specific improvement suggestions
        skills_match = detailed_scores.get("skills_match", {})
        missing_must_have = skills_match.get("missing_must_have", [])
        
        if missing_must_have:
            if len(missing_must_have) <= 3:
                recommendation += f" The candidate should develop skills in: {', '.join(missing_must_have)}."
            else:
                recommendation += f" The candidate is missing several required skills including: {', '.join(missing_must_have[:3])}."
        
        return recommendation
    
    def get_ats_compatibility_score(self, resume: Dict[str, Any], job_description: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate how well the resume is optimized for Applicant Tracking Systems (ATS).
        
        Args:
            resume: Parsed resume dictionary
            job_description: Parsed job description dictionary
            
        Returns:
            Dictionary with ATS compatibility results
        """
        ats_results = {
            "score": 0.0,
            "issues": [],
            "suggestions": []
        }
        
        issues = []
        suggestions = []
        
        # Check for proper contact information
        if not resume.get("contact_info", {}).get("email"):
            issues.append("Missing email address")
            suggestions.append("Add an email address to your contact information")
        
        if not resume.get("contact_info", {}).get("phone"):
            issues.append("Missing phone number")
            suggestions.append("Add a phone number to your contact information")
        
        # Check for proper section headers
        required_sections = ["experience", "education", "skills"]
        missing_sections = [section for section in required_sections if not resume.get("sections", {}).get(section)]
        
        if missing_sections:
            issues.append(f"Missing clear section headers for: {', '.join(missing_sections)}")
            suggestions.append(f"Add clear headings for: {', '.join(missing_sections)}")
        
        # Check for keyword optimization
        job_skills = [skill["name"].lower() for skill in job_description.get("skills", [])]
        resume_text = " ".join(resume.get("sections", {}).values()).lower()
        
        missing_keywords = [skill for skill in job_skills if skill not in resume_text]
        if len(missing_keywords) > len(job_skills) / 2:
            issues.append("Resume not well-optimized for job keywords")
            suggestions.append("Customize your resume with more job-specific keywords and skills")
        
        # Check for proper formatting (simple heuristics)
        has_bullets = False
        for section in resume.get("sections", {}).values():
            if "•" in section or "*" in section or "-" in section:
                has_bullets = True
                break
        
        if not has_bullets:
            issues.append("Lack of bullet points for easy scanning")
            suggestions.append("Use bullet points to highlight achievements and responsibilities")
        
        # Calculate ATS score based on issues
        if not issues:
            ats_score = 1.0
        else:
            ats_score = max(0.0, 1.0 - (len(issues) * 0.1))  # Deduct 10% per issue
        
        ats_results["score"] = ats_score
        ats_results["issues"] = issues
        ats_results["suggestions"] = suggestions
        
        return ats_results
    
    def generate_improvement_suggestions(
        self, resume: Dict[str, Any], job_description: Dict[str, Any], match_results: Dict[str, Any]
    ) -> List[str]:
        """
        Generate actionable suggestions to improve the resume for this job.
        
        Args:
            resume: Parsed resume dictionary
            job_description: Parsed job description dictionary
            match_results: Match results from calculate_match_score
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # 1. Address missing skills
        missing_must_have = match_results.get("skills_gap", {}).get("missing_must_have", [])
        missing_good_to_have = match_results.get("skills_gap", {}).get("missing_good_to_have", [])
        
        if missing_must_have:
            suggestions.append(f"Add the following critical skills if you have them: {', '.join(missing_must_have[:5])}")
        
        if missing_good_to_have and len(suggestions) < 5:
            suggestions.append(f"Consider adding these beneficial skills if you have them: {', '.join(missing_good_to_have[:3])}")
        
        # 2. Address experience gaps
        experience_match = match_results.get("detailed_scores", {}).get("experience_match", {})
        
        if not experience_match.get("meets_minimum", False):
            job_exp = job_description.get("experience", {})
            min_years = job_exp.get("min_years", 0)
            suggestions.append(f"Highlight any additional experience to meet the {min_years}+ years requirement")
        
        # 3. Address education gaps
        education_match = match_results.get("detailed_scores", {}).get("education_match", {})
        
        if not education_match.get("has_required_degree", True) and job_description.get("education", {}).get("required", False):
            degree_text = ", ".join(job_description.get("education", {}).get("degrees", []))
            if degree_text:
                suggestions.append(f"Emphasize any education relevant to the {degree_text} requirement")
        
        # 4. Keyword optimization
        keyword_match = match_results.get("detailed_scores", {}).get("keyword_match", {})
        job_title = job_description.get("details", {}).get("title", "")
        
        if keyword_match.get("score", 0) < 0.6 and job_title:
            suggestions.append(f"Optimize your resume with keywords from the job description, especially related to '{job_title}'")
        
        # 5. Section-specific improvements
        if "responsibilities" in keyword_match.get("sections_matched", {}) and keyword_match["sections_matched"]["responsibilities"] < 0.5:
            suggestions.append("Align your experience descriptions with the job responsibilities")
        
        # 6. ATS optimization
        ats_results = self.get_ats_compatibility_score(resume, job_description)
        suggestions.extend(ats_results.get("suggestions", [])[:2])  # Add up to 2 ATS suggestions
        
        return suggestions[:5]  # Limit to top 5 suggestions
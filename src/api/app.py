"""
Resume Scanner API Service

This module provides a REST API for the resume scanning and matching functionality.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from loguru import logger

# Import our modules
from src.resume_parser.parser import ResumeParser
from src.job_parser.parser import JobDescriptionParser
from src.matcher.matcher import ResumeMatcher

# Create FastAPI app
app = FastAPI(
    title="Resume Scanner API",
    description="API for scanning resumes and matching them with job descriptions",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our parsers and matcher
resume_parser = ResumeParser()
job_parser = JobDescriptionParser()
matcher = ResumeMatcher()

# Create a temporary directory for uploaded files
UPLOAD_DIR = Path(tempfile.gettempdir()) / "resume_scanner_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Pydantic models for API requests and responses
class JobDescriptionRequest(BaseModel):
    text: str
    title: Optional[str] = None
    company: Optional[str] = None

class JobDescriptionResponse(BaseModel):
    job_id: str
    parsed_data: Dict[str, Any]

class MatchRequest(BaseModel):
    resume_id: str
    job_id: str

class MatchResponse(BaseModel):
    match_score: float
    analysis: Dict[str, Any]

class EnhancementSuggestions(BaseModel):
    suggestions: List[str]

class ParsedResume(BaseModel):
    resume_id: str
    parsed_data: Dict[str, Any]

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "Resume Scanner API is running"}

@app.post("/resume/parse", response_model=ParsedResume)
async def parse_resume(resume_file: UploadFile = File(...)):
    """
    Parse a resume file and extract structured information.
    
    Args:
        resume_file: The resume file to parse (PDF, DOCX, etc.)
        
    Returns:
        Parsed resume data
    """
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / resume_file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(resume_file.file, f)
        
        # Parse resume
        parsed_data = resume_parser.parse_resume(str(file_path))
        
        # Generate a unique ID for this resume (using filename for simplicity)
        resume_id = f"resume_{hash(resume_file.filename)}"
        
        return {
            "resume_id": resume_id,
            "parsed_data": parsed_data
        }
    except Exception as e:
        logger.error(f"Error parsing resume: {e}")
        raise HTTPException(status_code=500, detail=f"Error parsing resume: {str(e)}")

@app.post("/job/parse", response_model=JobDescriptionResponse)
async def parse_job_description(job_description: JobDescriptionRequest):
    """
    Parse a job description and extract structured information.
    
    Args:
        job_description: Job description text and metadata
        
    Returns:
        Parsed job description data
    """
    try:
        # Parse job description
        parsed_data = job_parser.parse_job_description(job_description.text)
        
        # Add metadata if provided
        if job_description.title:
            parsed_data["details"]["title"] = job_description.title
        
        if job_description.company:
            parsed_data["details"]["company"] = job_description.company
        
        # Generate a unique ID for this job description
        job_id = f"job_{hash(job_description.text[:100])}"
        
        return {
            "job_id": job_id,
            "parsed_data": parsed_data
        }
    except Exception as e:
        logger.error(f"Error parsing job description: {e}")
        raise HTTPException(status_code=500, detail=f"Error parsing job description: {str(e)}")

@app.post("/match", response_model=MatchResponse)
async def match_resume_to_job(
    resume_data: Dict[str, Any],
    job_data: Dict[str, Any]
):
    """
    Match a parsed resume against a parsed job description.
    
    Args:
        resume_data: Parsed resume data
        job_data: Parsed job description data
        
    Returns:
        Match score and detailed analysis
    """
    try:
        # Calculate match score
        match_results = matcher.calculate_match_score(resume_data, job_data)
        
        return {
            "match_score": match_results["overall_score"],
            "analysis": match_results
        }
    except Exception as e:
        logger.error(f"Error matching resume to job: {e}")
        raise HTTPException(status_code=500, detail=f"Error matching resume to job: {str(e)}")

@app.post("/enhance", response_model=EnhancementSuggestions)
async def get_enhancement_suggestions(
    resume_data: Dict[str, Any],
    job_data: Dict[str, Any]
):
    """
    Get suggestions for enhancing a resume to better match a job description.
    
    Args:
        resume_data: Parsed resume data
        job_data: Parsed job description data
        
    Returns:
        List of enhancement suggestions
    """
    try:
        # Calculate match score
        match_results = matcher.calculate_match_score(resume_data, job_data)
        
        # Generate suggestions
        suggestions = matcher.generate_improvement_suggestions(resume_data, job_data, match_results)
        
        return {"suggestions": suggestions}
    except Exception as e:
        logger.error(f"Error generating enhancement suggestions: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating enhancement suggestions: {str(e)}")

@app.post("/ats-score", response_model=Dict[str, Any])
async def get_ats_compatibility_score(
    resume_data: Dict[str, Any],
    job_data: Dict[str, Any]
):
    """
    Calculate how well a resume is optimized for ATS systems.
    
    Args:
        resume_data: Parsed resume data
        job_data: Parsed job description data
        
    Returns:
        ATS compatibility score and issues
    """
    try:
        # Calculate ATS score
        ats_results = matcher.get_ats_compatibility_score(resume_data, job_data)
        
        return ats_results
    except Exception as e:
        logger.error(f"Error calculating ATS score: {e}")
        raise HTTPException(status_code=500, detail=f"Error calculating ATS score: {str(e)}")

@app.post("/batch-match", response_model=List[Dict[str, Any]])
async def batch_match_resumes(
    job_data: Dict[str, Any],
    resume_data_list: List[Dict[str, Any]]
):
    """
    Match multiple resumes against a job description and rank them.
    
    Args:
        job_data: Parsed job description data
        resume_data_list: List of parsed resume data
        
    Returns:
        Ranked list of match results
    """
    try:
        results = []
        
        for resume_data in resume_data_list:
            # Calculate match score
            match_results = matcher.calculate_match_score(resume_data, job_data, detailed=False)
            
            # Add to results
            results.append({
                "resume_id": resume_data.get("id", "unknown"),
                "match_score": match_results["overall_score"],
                "summary": match_results["summary"],
                "recommendation": match_results["recommendation"]
            })
        
        # Sort by match score (descending)
        results.sort(key=lambda x: x["match_score"], reverse=True)
        
        return results
    except Exception as e:
        logger.error(f"Error batch matching resumes: {e}")
        raise HTTPException(status_code=500, detail=f"Error batch matching resumes: {str(e)}")

@app.on_event("shutdown")
def cleanup():
    """Clean up temporary files on shutdown."""
    try:
        shutil.rmtree(UPLOAD_DIR)
    except:
        pass

# Mount the static files directory (if we have a frontend)
try:
    app.mount("/ui", StaticFiles(directory="static", html=True), name="static")
except:
    logger.warning("Static directory not found, UI will not be available")

if __name__ == "__main__":
    # Run the API server
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)